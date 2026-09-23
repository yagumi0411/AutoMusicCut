"""歌声片段歌名识别：SenseVoiceSmall ASR 转写歌词 → lrclib 歌词搜索 → 相似度确认。

定位：**人工确认的辅助**，不自动填写歌名。
- 本地 ASR 把唱歌段转写成歌词文本（翻唱也能识别，绕开音频指纹"只能匹配原唱"的限制）；
- 从歌词里挑若干短片段作搜索词查 lrclib.net（免费、无需 key、按歌词搜索）；
- 用文本相似度过滤候选曲目，返回 Top-N + 置信度，由用户在 Web 上点选确认。

搜索策略（实证校准）：
长句的歌词检索召回差，且 ASR 转写有同音/词形偏差。因此从转写中抽取多个
短窗口分别查询；中日韩文本没有天然空格，按字符窗口切分，避免把整段歌词当成
一个查询词。候选曲目的评分也按“转写片段能否在完整歌词中匹配”计算，而非拿
片段与整首歌词作全局相似度比较。

模型（SenseVoiceSmall）懒加载单例，一次加载、多段复用。
搜索结果可能为空/不准：冷门歌、纯音乐、BGM 太吵、主播明显改词时给不出答案，
此时返回空由人工处理即可。
"""

import difflib
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import httpx

ASR_MODEL = "iic/SenseVoiceSmall"
LRCLIB_API = "https://lrclib.net/api/search"
MIN_SIM = 0.20            # 片段匹配下限；仅用于人工确认候选，优先保证召回
N_QUERIES = 5             # 短查询个数（每段歌词拆出多少个搜索词）
QUERY_WINDOW = 2          # 拉丁文字每个搜索窗口的词数
CJK_QUERY_WINDOW = 6      # 中日韩文字没有分词边界，使用字符窗口
SEARCH_TOP_K = 5          # 每次查询取回的候选上限
RECOGNITION_WINDOW_SEC = 35
RECOGNITION_SAMPLE_POSITIONS = (0.15, 0.5, 0.85)


# ---- ASR：懒加载单例 ----

_SENSEVOICE = None
_SENSEVOICE_LOCK = threading.Lock()


def ensure_asr():
    """懒加载 SenseVoiceSmall 模型（单例，多段复用）。

    双重检查加锁：/api/recognize 与 /api/recognize-all 可能并发进入，
    无锁时两个线程都会各自 AutoModel(...) 一次，白白多占约 230MB 内存。
    """
    global _SENSEVOICE
    if _SENSEVOICE is None:
        with _SENSEVOICE_LOCK:
            if _SENSEVOICE is None:
                from funasr import AutoModel  # 延迟导入，避免拖慢启动
                _SENSEVOICE = AutoModel(
                    model=ASR_MODEL,
                    model_revision="master",
                    trust_remote_code=False,
                    disable_update=True,
                )
    return _SENSEVOICE


def transcribe(wav_path, lang="auto"):
    """ASR 转写给定 wav 的歌词，返回纯歌词文本（带事件标签剥离后）。

    wav 需为按时间段截取好的窗口（见 recognize_segment）。
    SenseVoice 结果可能是 dict 或 str，统一取文本；
    输出里可能混入 <|zh|>、<|BGM|>、<|withitn|> 等标签，全部剥离，
    只保留歌词本身（否则标签会污染搜索词与相似度比对）。
    """
    model = ensure_asr()
    res = model.generate(
        input=str(wav_path),
        language=lang,      # auto 支持中/英/日/韩/粤自动识别
        use_itn=True,
        batch_size_s=60,
    )
    text = ""
    for r in res:
        s = r.get("text") if isinstance(r, dict) else str(r)
        if s:
            text += s + " "
    text = re.sub(r"<\|[^>]*\|>", "", text)   # 剥离 SenseVoice 标签
    return " ".join(text.split())


_QUERY_UNIT_RE = re.compile(
    r"[一-鿿぀-ヿ가-힣]|[a-zA-Z0-9]+")


def build_queries(text, n=N_QUERIES, window=QUERY_WINDOW):
    """从 ASR 歌词里挑多个短片段作搜索词。

    拉丁文字按词切分；中日韩文字通常没有空格，按字符切分并使用 6 字窗口。
    原实现只按空格切分，中文转写会被视为一个超长查询，几乎无法命中。
    """
    units = _QUERY_UNIT_RE.findall(text)
    nw = len(units)
    if not nw:
        return []
    cjk = any(len(unit) == 1 and not unit.isascii() for unit in units)
    window = CJK_QUERY_WINDOW if cjk else window
    joiner = "" if cjk else " "
    if nw <= window + 2:
        return [joiner.join(units)]
    start = max(1, nw // 5)            # 略过片段开头的切换杂音
    span = nw - start
    queries = []
    for k in range(n):
        i = min(nw - window, start + (span * (k + 1)) // (n + 1))
        query = joiner.join(units[i:i + window])
        if query not in queries:
            queries.append(query)
    return queries


def search_lyrics(query, top_k=SEARCH_TOP_K):
    """按歌词片段查 lrclib，返回候选 [{title, artist, plainLyrics, id}]。

    网络/解析失败返回空列表，不抛异常（识别仅辅助，不该打断流程）。
    """
    params = {"q": query}
    headers = {"User-Agent": "AutoMusicCut/1.0 (local tool)"}
    try:
        r = httpx.get(LRCLIB_API, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001 网络/解析失败视为无结果
        return []
    out = []
    for s in data or []:
        if s.get("plainLyrics"):
            out.append(s)
        if len(out) >= top_k:
            break
    return out


_KEEP_CHARS_RE = re.compile(r"[^A-Za-z一-鿿぀-ヿ가-힣0-9]+")


def _text_sim(a, b):
    """转写片段在完整歌词中的匹配度 0~1（忽略大小写与标点）。

    不能使用 SequenceMatcher 的全局 ratio：即使转写完全正确，它也只是整首
    歌词的一小段，分数仍会低得不合理。这里统计转写文本被歌词中连续匹配块
    覆盖的比例，以适配片段识别场景。
    """
    def compact(s):
        return _KEEP_CHARS_RE.sub("", s).lower()
    a, b = compact(a), compact(b)
    if not a or not b:
        return 0.0
    blocks = difflib.SequenceMatcher(None, a, b).get_matching_blocks()
    matched = sum(block.size for block in blocks if block.size >= 2)
    return matched / len(a)


def confidence_of_sim(sim):
    if sim >= 0.7:
        return "高"
    if sim >= 0.4:
        return "中"
    return "低"


def segment_wav(video, start, end, out_wav):
    """截取 [start, end) 秒为 16k 单声道临时 wav（识别用）。"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
        "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    return out_wav


def recognition_windows(start, end):
    """为长候选段挑选多个转写窗口，避开首尾切换噪声。

    候选段常含前奏、间奏或主播说话。对超过一个窗口的片段，在其前、中、后部
    各抽取约 35 秒；窗口彼此独立评分，只要任一窗口转写到歌词即可召回曲目。
    """
    duration = end - start
    if duration <= RECOGNITION_WINDOW_SEC:
        return [(start, end)]
    travel = duration - RECOGNITION_WINDOW_SEC
    return [
        (start + travel * pos, start + travel * pos + RECOGNITION_WINDOW_SEC)
        for pos in RECOGNITION_SAMPLE_POSITIONS
    ]


def _transcribe_windows(video, windows, lang="auto", work_dir=None):
    """转写给定的音频时间窗口，返回非空文本列表。"""
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".amc_seg_", suffix=".wav",
        dir=str(work_dir) if work_dir else None)
    os.close(fd)
    tmp_wav = Path(tmp_name)
    try:
        texts = []
        for window_start, window_end in windows:
            segment_wav(video, window_start, window_end, tmp_wav)
            text = transcribe(tmp_wav, lang=lang)
            if text:
                texts.append(text)
        return texts
    finally:
        tmp_wav.unlink(missing_ok=True)


def transcribe_segment(video, start, end, lang="auto", work_dir=None):
    """完整转写候选段，返回非空文本列表。"""
    return _transcribe_windows(video, [(start, end)], lang=lang, work_dir=work_dir)


def recognize_segment(video, start, end, lang="auto", work_dir=None):
    """识别单个 [start, end) 秒段的歌名。

    返回 {"transcript", "query", "candidates": [{title, artist, sim, confidence}]}
    candidates 按相似度降序，可能为空。

    work_dir: 临时 wav 的存放目录，默认用系统临时目录。此前写在**源视频
    所在目录**，只读目录/网络盘会失败，且同一时间段的并发识别会撞名。
    """
    texts = _transcribe_windows(
        video, recognition_windows(start, end), lang=lang, work_dir=work_dir)
    queries = []
    for text in texts:
        for query in build_queries(text):
            if query not in queries:
                queries.append(query)

    if not queries:
        return {"transcript": " / ".join(texts), "query": "", "candidates": []}

    # 多查询合并候选，按 trackName+artistName 去重
    merged = {}
    for q in queries:
        for s in search_lyrics(q):
            key = (s.get("trackName", ""), s.get("artistName", ""))
            merged.setdefault(key, s)

    cands = []
    for s in merged.values():
        lyrics = s.get("plainLyrics") or ""
        sim = max(_text_sim(text, lyrics) for text in texts)
        if sim >= MIN_SIM:
            cands.append({
                "title": s.get("trackName", ""),
                "artist": s.get("artistName", ""),
                "sim": round(sim, 3),
                "confidence": confidence_of_sim(sim),
            })
    cands.sort(key=lambda c: c["sim"], reverse=True)
    return {"transcript": " / ".join(texts), "query": " | ".join(queries),
            "candidates": cands}
