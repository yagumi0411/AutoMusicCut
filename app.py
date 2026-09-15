"""AutoMusicCut Web 服务（本地工具）。

启动: .venv/Scripts/python.exe app.py [--port 8765]  或双击 start.bat
打开: http://localhost:8765

API 一览:
  GET  /api/browse?path=xxx       浏览目录（选视频）
  POST /api/analyze               启动检测任务（后台线程 + 轮询进度）
  GET  /api/task/{id}             查询任务状态
  GET  /api/candidates            读取候选清单
  POST /api/candidates/add        把候选段加入最终清单
  GET  /api/songs                 读取最终清单
  POST /api/songs                 保存最终清单
  POST /api/recognize             识别单个时间段的歌名（后台线程）
  POST /api/recognize-all         依次识别全部候选段歌名（后台线程）
  POST /api/cut                   按最终清单剪切（后台线程）
  GET  /api/stream?path=xxx       视频流（支持 Range，供播放器跳转）
  GET  /api/open?path=xxx         在资源管理器中打开路径
"""

import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from amc.cut import cut as do_cut
from amc.cut import parse_songlist, sanitize_filename
from amc.detect import (
    DEFAULT_MODEL_DIR, MERGE_GAP, WIN_SEC, WIN_THRESH, detect,
)
from amc.recognize import recognize_segment
from amc.timefmt import format_time, parse_time

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".flv", ".mkv", ".ts", ".mov", ".avi", ".wmv", ".m4v"}
PLAYABLE_EXTS = {".mp4", ".m4v", ".webm"}  # 浏览器内嵌播放支持的容器

app = FastAPI(title="AutoMusicCut")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# ---- 任务表: task_id -> {status, message, result, error} ----
TASKS = {}
TASKS_LOCK = threading.Lock()


def _new_task():
    task_id = uuid.uuid4().hex[:12]
    with TASKS_LOCK:
        TASKS[task_id] = {"status": "running", "messages": [], "result": None,
                          "error": None, "started": time.time()}
    return task_id


def _run_task(task_id, fn):
    def wrapper():
        try:
            result = fn()
            with TASKS_LOCK:
                TASKS[task_id]["status"] = "done"
                TASKS[task_id]["result"] = result
        except Exception as e:  # noqa: BLE001
            with TASKS_LOCK:
                TASKS[task_id]["status"] = "error"
                TASKS[task_id]["error"] = str(e)
    threading.Thread(target=wrapper, daemon=True).start()


def _task_progress(task_id):
    """返回一个把日志写入任务的回调。"""
    def cb(msg):
        with TASKS_LOCK:
            TASKS[task_id]["messages"].append(msg)
    return cb


# ---- 会话目录：每个视频一个子文件夹 ----
CURRENT_VIDEO_REQ = None            # 前端最近一次「选视频/分析」的请求锁（并发保护）


def _session_dir(video):
    """当前视频的会话目录根：work/<视频名>/（temp 与 result 在下面）。"""
    stem = sanitize_filename(Path(video).stem) or "video"
    return ROOT / "work" / stem


def _session_paths(video):
    """返回当前视频的会话目录结构，确保目录存在。

    work/<视频名>/temp/   ← candidates.txt、songs.txt、16k 音轨等中间文件
    work/<视频名>/result/ ← 剪切输出的视频
    """
    base = _session_dir(video)
    temp, result = base / "temp", base / "result"
    temp.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    return {
        "temp": temp, "result": result,
        "candidates": temp / "candidates.txt",
        "songs": temp / "songs.txt",
        "audio": temp / "audio16k.wav",
    }


def _current_video():
    """当前会话视频的绝对路径。优先用最近的 select/analyze 状态，
    其次用 work/last_video.txt 兜底；都没有返回 None。"""
    if CURRENT_VIDEO_REQ:
        return CURRENT_VIDEO_REQ
    last = ROOT / "work" / "last_video.txt"
    if last.exists():
        return last.read_text(encoding="utf-8").strip()
    return None


def _current_session_paths(video=""):
    """解析当前视频的会话目录；video 传了就用它，否则用当前状态。"""
    if not video:
        cur = _current_video()
        if cur:
            video = cur
        else:
            return None
    return _session_paths(video)


def _current_candidates_path(video=""):
    sess = _current_session_paths(video)
    return sess["candidates"] if sess else ROOT / "candidates.txt"


def _current_songs_path(video=""):
    sess = _current_session_paths(video)
    return sess["songs"] if sess else ROOT / "songs.txt"


# ---- 目录浏览 ----

@app.get("/api/browse")
def browse(path: str = Query(default="")):
    p = Path(path) if path else ROOT
    if not p.exists():
        raise HTTPException(404, f"路径不存在: {p}")
    if not p.is_dir():
        raise HTTPException(400, f"不是目录: {p}")
    try:
        entries = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
                "is_video": child.is_file() and child.suffix.lower() in VIDEO_EXTS,
                "size": child.stat().st_size if child.is_file() else None,
            })
    except PermissionError:
        raise HTTPException(403, "无权限访问该目录")
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "entries": entries}


# ---- 任务查询 ----

@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    with TASKS_LOCK:
        t = TASKS.get(task_id)
        if t is None:
            raise HTTPException(404, "任务不存在")
        return {k: t[k] for k in ("status", "messages", "result", "error")}


# ---- 分析 ----

@app.post("/api/analyze")
def api_analyze(payload: dict):
    video = payload.get("video", "")
    if not video or not Path(video).is_file():
        raise HTTPException(400, "视频路径无效")
    global CURRENT_VIDEO_REQ
    CURRENT_VIDEO_REQ = str(Path(video))
    sess = _session_paths(video)
    task_id = _new_task()
    log = _task_progress(task_id)

    def run():
        detect(
            video=video,
            work_dir=str(sess["temp"]),
            candidates_path=str(sess["candidates"]),
            device=payload.get("device", "cpu"),
            min_duration=payload.get("min_duration", 15.0),
            win_sec=payload.get("win_sec", WIN_SEC),
            win_thresh=payload.get("win_thresh", WIN_THRESH),
            merge_gap=payload.get("merge_gap", MERGE_GAP),
            progress_cb=log,
        )
        return {"candidates_path": str(sess["candidates"])}

    _run_task(task_id, run)
    return {"task_id": task_id}


# ---- 候选清单 ----

CAND_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([\d:]+)\s*\|\s*([\d:]+)\s*\|"
    r"\s*[\d:]+\s*\|\s*(\S+)\s*\|\s*([\d.]+)")


@app.get("/api/candidates")
def api_candidates(video: str = Query(default="")):
    path = _current_candidates_path(video)
    if not path.exists():
        return {"exists": False, "candidates": []}
    cands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CAND_RE.match(line)
        if m:
            s, e = parse_time(m.group(2)), parse_time(m.group(3))
            cands.append({
                "index": int(m.group(1)),
                "start": s, "start_str": format_time(s),
                "end": e, "end_str": format_time(e),
                "duration": e - s,
                "confidence": m.group(4),
                "score": float(m.group(5)),
            })
    return {"exists": True, "candidates": cands}


@app.post("/api/candidates/add")
def api_candidates_add(payload: dict):
    """把候选段加入 songs.txt（歌名可留空）。"""
    songs_path = _current_songs_path()
    if songs_path.exists():
        text = songs_path.read_text(encoding="utf-8").rstrip("\n")
        blocks = text.split("\n\n") if text else []
    else:
        blocks = []
    for item in payload.get("candidates", []):
        name = (item.get("name") or "").strip() or f"歌曲{len(blocks) + 1}"
        blocks.append(f"{name}\n{item['start_str']}\n{item['end_str']}")
    songs_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return {"count": len(payload.get("candidates", []))}


# ---- 最终清单 ----

def _load_songs(video=""):
    songs_path = _current_songs_path(video)
    if not songs_path.exists():
        return []
    return [{"name": n, "start": s, "start_str": format_time(s),
             "end": e, "end_str": format_time(e), "duration": e - s}
            for n, s, e in parse_songlist(songs_path)]


@app.get("/api/songs")
def api_songs():
    return {"songs": _load_songs()}


@app.post("/api/songs")
def api_songs_save(payload: dict):
    """保存最终清单（前端表格编辑后的完整列表）。"""
    blocks = []
    for i, item in enumerate(payload.get("songs", [])):
        name = (item.get("name") or "").strip()
        if not name:
            raise HTTPException(400, f"第 {i + 1} 条缺少歌曲名")
        s = parse_time(item.get("start_str"))
        e = parse_time(item.get("end_str"))
        if e <= s:
            raise HTTPException(400, f"{name!r} 结束时间早于开始时间")
        blocks.append(f"{name}\n{format_time(s)}\n{format_time(e)}")
    _current_songs_path().write_text("\n\n".join(blocks) + ("\n" if blocks else ""),
                                     encoding="utf-8")
    return {"count": len(blocks)}


# ---- 歌名识别 ----

@app.post("/api/recognize")
def api_recognize(payload: dict):
    """识别一个时间段 [start, end) 的歌名候选（后台任务）。"""
    video = payload.get("video", "")
    try:
        start = float(payload["start"])
        end = float(payload["end"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "start/end 必须是秒数")
    if not video or not Path(video).is_file():
        raise HTTPException(400, "视频路径无效")
    if not 0 <= start < end:
        raise HTTPException(400, "时间段非法")
    task_id = _new_task()
    log = _task_progress(task_id)
    lang = payload.get("lang", "zh")

    def run():
        log(f"正在识别 {payload.get('name', '')} ({format_time(start)} ~ {format_time(end)})...")
        return recognize_segment(video, start, end, lang=lang)

    _run_task(task_id, run)
    return {"task_id": task_id}


@app.post("/api/recognize-all")
def api_recognize_all(payload: dict):
    """依次识别所有候选段，逐段返回结果。

    一次任务跑完所有段，避免每个段都触发模型加载；进度按段数推进。
    """
    video = payload.get("video", "")
    if not video or not Path(video).is_file():
        raise HTTPException(400, "视频路径无效")
    global CURRENT_VIDEO_REQ
    CURRENT_VIDEO_REQ = str(Path(video))
    cands = _load_candidates_payload(video)
    if not cands:
        raise HTTPException(400, "候选清单为空，请先执行分析")
    task_id = _new_task()
    log = _task_progress(task_id)
    lang = payload.get("lang", "zh")

    def run():
        results = []
        for i, c in enumerate(cands, 1):
            log(f"[{i}/{len(cands)}] 识别 {c['start_str']} ~ {c['end_str']}...")
            start, end = c["start"], c["end"]
            res = recognize_segment(video, start, end, lang=lang)
            results.append({"index": c["index"], "start": start,
                            "start_str": c["start_str"], "end": end,
                            "end_str": c["end_str"], **res})
        return {"results": results}

    _run_task(task_id, run)
    return {"task_id": task_id}


def _load_candidates_payload(video_arg=""):
    """从 candidates.txt 读取候选段（供识别复用）。"""
    path = _current_candidates_path(video_arg)
    if not path.exists():
        return []
    cands = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CAND_RE.match(line)
        if m:
            s, e = parse_time(m.group(2)), parse_time(m.group(3))
            cands.append({"index": int(m.group(1)),
                          "start": s, "end": e,
                          "start_str": format_time(s), "end_str": format_time(e)})
    return cands


# ---- 剪切 ----

@app.post("/api/cut")
def api_cut(payload: dict):
    video = payload.get("video", "")
    if not video or not Path(video).is_file():
        raise HTTPException(400, "视频路径无效")
    global CURRENT_VIDEO_REQ
    CURRENT_VIDEO_REQ = str(Path(video))
    sess = _session_paths(video)
    if not sess["songs"].exists():
        raise HTTPException(400, "songs.txt 不存在，请先保存最终清单")
    task_id = _new_task()
    log = _task_progress(task_id)

    def run():
        out_dir = payload.get("out_dir") or str(sess["result"])
        pad = payload.get("pad", 1.5)
        min_duration = payload.get("min_duration", 0.0)
        report = do_cut(video=video, songlist_path=sess["songs"],
                        out_dir=out_dir, pad=pad, min_duration=min_duration)
        return {
            "out_dir": out_dir,
            "items": [{
                "name": r["name"],
                "file": r["out"].name,
                "path": str(r["out"]),
                "actual": r["actual"],
                "actual_str": format_time(r["actual"]),
            } for r in report],
        }

    _run_task(task_id, run)
    return {"task_id": task_id}


@app.post("/api/select-video")
def api_select_video(payload: dict):
    """前端切换/选定视频时调用，设定当前会话（可立即建 temp/result 目录）。"""
    video = payload.get("video", "")
    if not video or not Path(video).is_file():
        raise HTTPException(400, "视频路径无效")
    global CURRENT_VIDEO_REQ
    CURRENT_VIDEO_REQ = str(Path(video))
    sess = _session_paths(video)
    return {"dir": str(_session_dir(video)),
            "temp": str(sess["temp"]), "result": str(sess["result"])}


# ---- 拖入定位：按文件名在本地磁盘找视频路径 ----

@app.post("/api/locate-video")
def api_locate_video(payload: dict):
    """浏览器拖入文件时只能拿到文件名，此接口在常用目录里定位绝对路径。

    搜索顺序：用户提示目录（hint）→ 上次视频所在目录 → 项目根目录。
    同名文件可能有多处，取第一个命中；找不到返回 404。
    """
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少文件名")
    if not name.lower().endswith(tuple(VIDEO_EXTS)):  # 不是视频文件
        raise HTTPException(400, f"不是视频文件: {name}")

    candidates = []
    hint = (payload.get("hint_dir") or "").strip()
    if hint:
        candidates.append(Path(hint))
    last_video = ROOT / "work" / "last_video.txt"
    if last_video.exists():
        prev = Path(last_video.read_text(encoding="utf-8").strip())
        if prev.parent.is_dir():
            candidates.append(prev.parent)
    candidates.append(ROOT)

    for base in candidates:
        try:
            hit = next(p for p in base.iterdir() if p.is_file() and p.name == name)
        except (StopIteration, PermissionError, FileNotFoundError):
            continue
        hit = hit.resolve()  # 一律返回绝对路径，避免 cwd 变化导致前端路径失效
        (ROOT / "work").mkdir(parents=True, exist_ok=True)
        last_video.write_text(str(hit), encoding="utf-8")
        return {"path": str(hit)}
    raise HTTPException(404, f"在常用目录中未找到 {name!r}，请在浏览中手动选择")


# ---- 视频流与本地打开 ----

@app.get("/api/stream")
def api_stream(path: str = Query(...)):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/open")
def api_open(path: str = Query(...)):
    """在资源管理器中打开路径（Windows）。"""
    if os.name == "nt":
        os.startfile(str(Path(path)))  # noqa: S606
        return {"ok": True}
    return {"ok": False, "message": "仅 Windows 支持"}


@app.get("/")
def index():
    # no-cache 确保每次打开都拉到最新版 HTML（否则浏览器缓存旧页，
    # 导致新功能按钮（如识别、全部识别）看不到）
    return FileResponse(ROOT / "static" / "index.html",
                        headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="AutoMusicCut Web 服务")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    import uvicorn
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(
            f"http://localhost:{args.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
