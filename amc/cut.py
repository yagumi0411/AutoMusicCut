"""按最终清单批量无损剪切唱歌片段。"""

import re
import shutil
import subprocess
from pathlib import Path

from .timefmt import format_time, parse_time

_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|\s]+')

# Windows 保留设备名，作为文件名会创建失败（追加下划线规避）
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
MAX_NAME_LEN = 80


def _run_cmd(cmd):
    """执行 ffmpeg/ffprobe；失败时把 stderr 末尾拼进异常，便于定位原因。"""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(
            f"命令失败（退出码 {proc.returncode}）: {cmd[0]}\n"
            f"--- stderr 末尾 ---\n{tail or '(无输出)'}"
        )
    return proc


def ensure_ffmpeg():
    """预检 ffmpeg/ffprobe，缺失时给出安装指引。"""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"未找到 {'/'.join(missing)}，请安装 ffmpeg 并加入 PATH。\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def sanitize_filename(name):
    """歌名 → 安全的文件名片段（保留中文）。

    额外处理：Windows 保留设备名（CON/PRN/AUX/NUL/COM1-9/LPT1-9）加下划线前缀，
    并截断超长名称（避免路径超 260 字符导致 ffmpeg 写出失败）。
    """
    s = _INVALID_FS_CHARS.sub("_", str(name).strip()).strip("_")
    if not s:
        return s
    if s.lower() in _RESERVED_NAMES or s.lower().split(".")[0] in _RESERVED_NAMES:
        s = f"_{s}"
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN].rstrip("_")
    return s


def parse_songlist(path):
    """解析最终清单 songs.txt。

    条目之间以空行分隔，每个条目 3 行：歌曲名、开始、结束。
    # 开头的行视为注释忽略。返回 [(歌名, 开始秒, 结束秒)]。
    """
    text = Path(path).read_text(encoding="utf-8")
    entries, block = [], []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if block:
                entries.append((lineno - len(block), block))
                block = []
            continue
        block.append((lineno, line))
    if block:
        entries.append((lineno - len(block) + 1, block))

    songs = []
    for start_lineno, block in entries:
        if len(block) != 3:
            raise ValueError(
                f"清单条目行数错误（第 {start_lineno} 行起，应有 3 行：歌曲名/开始/结束，实际 {len(block)} 行）"
            )
        (_, name), (_, t_start), (_, t_end) = block
        s = parse_time(t_start)
        e = parse_time(t_end)
        if e <= s:
            raise ValueError(f"结束时间早于开始时间: {name!r} ({t_start} ~ {t_end})")
        songs.append((name, s, e))
    if not songs:
        raise ValueError("清单为空，未找到任何条目")
    return songs


def video_duration(video):
    """用 ffprobe 获取视频时长（秒）。"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]
    out = _run_cmd(cmd).stdout.strip()
    return float(out)


def cut_segment(video, out_path, start, end, pad, duration):
    """ffmpeg 无损剪切一段（-c copy，关键帧对齐，配合 pad 保证不丢内容）。"""
    s = max(0.0, start - pad)
    e = min(duration, end + pad)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video),
        "-map", "0", "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    _run_cmd(cmd)
    # 验证实际切出的时长
    actual = video_duration(out_path)
    return s, e, actual


def cut(video, songlist_path, out_dir="output", pad=1.5, min_duration=0.0,
        overwrite=False, progress_cb=None):
    """按清单批量剪切，返回结果报告列表。

    min_duration: 最小时长（秒），短于该值的条目跳过并警告；
    用于"只剪完整曲目"（如 180 表示至少 3 分钟）。
    overwrite: 输出文件已存在时是否覆盖；False 则跳过并在报告中标记。
    progress_cb: 可选进度回调（日志字符串），供 Web 端逐段显示进度。
    """
    def log(msg):
        print(msg, flush=True)
        if progress_cb:
            progress_cb(msg)

    ensure_ffmpeg()
    video = Path(video)
    songs = parse_songlist(songlist_path)
    duration = video_duration(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    if min_duration > 0:
        kept = []
        for name, s, e in songs:
            if e - s >= min_duration:
                kept.append((name, s, e))
            else:
                skipped.append((name, s, e, e - s))
        songs = kept

    log(f"视频时长: {format_time(duration)}，共 {len(songs)} 段，边界外扩 {pad}s"
        + (f"，最小时长过滤 {min_duration}s" if min_duration > 0 else ""))
    for name, s, e, d in skipped:
        log(f"  跳过: {name!r} 时长 {format_time(d)} < {min_duration}s")

    report = []
    total = len(songs)
    for i, (name, s, e) in enumerate(songs, 1):
        if e > duration:
            log(f"[{i:02d}] 警告: {name!r} 结束时间 {format_time(e)} 超出视频时长，已截断")
        out_path = out_dir / f"{i:02d}_{sanitize_filename(name)}.mp4"
        if out_path.exists() and not overwrite:
            log(f"[{i}/{total}] 已存在，跳过: {out_path.name}（加 --overwrite 可覆盖）")
            report.append({"name": name, "out": out_path, "requested": (s, e),
                           "cut": (s, e), "actual": None, "skipped": True})
            continue
        log(f"[{i}/{total}] 剪切 {name} {format_time(s)} ~ {format_time(e)} ...")
        cut_s, cut_e, actual = cut_segment(video, out_path, s, e, pad, duration)
        log(f"[{i}/{total}] {name} → {out_path.name}  实际时长 {format_time(actual)}")
        report.append({
            "name": name, "out": out_path,
            "requested": (s, e), "cut": (cut_s, cut_e), "actual": actual,
            "skipped": False,
        })
    done = sum(1 for r in report if not r.get("skipped"))
    log(f"完成，共 {done} 段，输出目录: {out_dir}")
    return report
