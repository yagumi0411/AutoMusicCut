"""歌声片段歌名识别：SenseVoiceSmall ASR 转写歌词 → lrclib 歌词搜索 → 相似度确认。

定位：**人工确认的辅助**，不自动填写歌名。
- 本地 ASR 把唱歌段转写成歌词文本（翻唱也能识别，绕开音频指纹"只能匹配原唱"的限制）；
- 从歌词里挑若干短片段作搜索词查 lrclib.net（免费、无需 key、按歌词搜索）；
- 用文本相似度过滤候选曲目，返回 Top-N + 置信度，由用户在 Web 上点选确认。

搜索策略（实证校准）：
lrclib 的 /api/search 是精确子串匹配——实测 3 词短片段能命中、
整句（8 词）返回 0，且 ASR 转写有同音/词形偏差，整句必漏。
因此用多个 2~3 词短窗口分别查询，合并去重后统一按歌词相似度确认。

模型（SenseVoiceSmall）懒加载单例，一次加载、多段复用。
搜索结果可能为空/不准：冷门歌、纯音乐、BGM 太吵、主播明显改词时给不出答案，
此时返回空由人工处理即可。
"""

import difflib
import re
import subprocess
from pathlib import Path

import httpx

ASR_MODEL = "iic/SenseVoiceSmall"
LRCLIB_API = "https://lrclib.net/api/search"
MIN_SIM = 0.35            # 候选相似度下限（低于该值的候选直接丢弃）
N_QUERIES = 3             # 短查询个数（每段歌词拆出多少个搜索词）
QUERY_WINDOW = 2          # 每个搜索窗口的词数
SEARCH_TOP_K = 5          # 每次查询取回的候选上限


# ---- ASR：懒加载单例 ----

_SENSEVOICE = None


def ensure_asr():
    """懒加载 SenseVoiceSmall 模型（单例，多段复用）。"""
    global _SENSEVOICE
    if _SENSEVOICE is None:
        from funasr import AutoModel  # 延迟导入，避免拖慢启动
        _SENSEVOICE = AutoModel(
            model=ASR_MODEL,
            model_revision="master",
            trust_remote_code=False,
            disable_update=True,
        )
    return _SENSEVOICE


def transcribe(wav_path, lang="zh"):
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


def build_queries(text, n=3, window=2):
    """从 ASR 歌词里挑 n 个 2~3 词短片段作搜索词。

    lrclib 的搜索是精确子串匹配：整句/长句召回差（实测 3 词中、8 词空），
    且 ASR 带同音/词形偏差时整句必漏。策略：均匀分布在歌词各处的短窗口
    分别查询，合并后用歌词相似度统一确认，兼顾召回与准确。
    避开开头（转写最容易歪的部分）。
    """
    t = re.sub(r"[^一-鿿぀-ヿ가-힣a-zA-Z0-9 ]", " ", text).strip()
    words = t.split()
    nw = len(words)
    if not nw:
        return []
    if nw <= window + 2:
        return [" ".join(words)]
    start = max(1, nw // 3)            # 从 1/3 处开始，避开开头
    span = nw - start
    queries = []
    for k in range(n):
        i = min(nw - window, start + (span * (k + 1)) // (n + 1))
        queries.append(" ".join(words[i:i + window]))
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


def _text_sim(a, b):
    """文本相似度 0~1（SequenceMatcher，忽略大小写与标点）。"""
    def compact(s):
        return re.sub(r"[^A-Za-z一-鿿0-9]+", "", s).lower()
    a, b = compact(a), compact(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def confidence_of_sim(sim):
    if sim >= 0.5:
        return "高"
    if sim >= 0.35:
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


def recognize_segment(video, start, end, lang="zh"):
    """识别单个 [start, end) 秒段的歌名。

    返回 {"transcript", "query", "candidates": [{title, artist, sim, confidence}]}
    candidates 按相似度降序，可能为空。
    """
    tmp_wav = Path(video).parent / f".amc_seg_{int(start)}-{int(end)}.wav"
    try:
        segment_wav(video, start, end, tmp_wav)
        text = transcribe(tmp_wav, lang=lang)

        queries = build_queries(text)
        if not queries:
            return {"transcript": text, "query": "", "candidates": []}

        # 多查询合并候选，按 trackName+artistName 去重
        merged = {}
        for q in queries:
            for s in search_lyrics(q):
                key = (s.get("trackName", ""), s.get("artistName", ""))
                merged.setdefault(key, s)

        cands = []
        for s in merged.values():
            sim = _text_sim(text, s.get("plainLyrics") or "")
            if sim >= MIN_SIM:
                cands.append({
                    "title": s.get("trackName", ""),
                    "artist": s.get("artistName", ""),
                    "sim": round(sim, 3),
                    "confidence": confidence_of_sim(sim),
                })
        cands.sort(key=lambda c: c["sim"], reverse=True)
        return {"transcript": text, "query": " | ".join(queries),
                "candidates": cands}
    finally:
        tmp_wav.unlink(missing_ok=True)
