"""歌声区间检测：FireRedVAD AED 帧级概率 → 滑窗聚合 → 候选清单。

核心思路：
- FireRedVAD 的 AED（Audio Event Detection）模型逐帧输出
  speech / singing / music 三类概率，直接区分唱歌、杂谈、音乐；
- 对 singing 概率做滑窗聚合（默认 20s 窗 / 10s 步 / 0.5 阈值），
  相邻唱歌窗间隔 30s 以内合并，得到完整歌曲的候选区间；
- 参数用真实录播 + 人工时间戳标定（13 首真值全部命中、覆盖 88%）。

旧版 demucs+pyin 特征管线已由 FireRedVAD 取代（见 git 历史与
docs/model-plan.md 中的标定分析）。
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from .timefmt import format_time

# ---- 检测参数（用真实数据标定，可用 CLI 覆盖）----
WIN_SEC = 20.0            # 滑窗长度（秒）
WIN_HOP_SEC = 10.0        # 滑窗步长（秒）
WIN_THRESH = 0.5          # 窗内 singing 平均概率阈值
MERGE_GAP = 30.0          # 相邻唱歌窗间隔小于该值（秒）时合并
DEFAULT_MODEL_DIR = "pretrained_models/FireRedVAD/AED"
HF_REPO = "FireRedTeam/FireRedVAD"


def ensure_model(model_dir=DEFAULT_MODEL_DIR):
    """检查 AED 模型文件，缺失时自动从 Hugging Face 下载。"""
    model_dir = Path(model_dir)
    if not (model_dir / "model.pth.tar").exists() or \
       not (model_dir / "cmvn.ark").exists():
        print(f"首次使用：从 Hugging Face 下载 FireRedVAD AED 模型到 {model_dir}")
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, "AED/model.pth.tar",
                        local_dir=str(model_dir.parent))
        hf_hub_download(HF_REPO, "AED/cmvn.ark",
                        local_dir=str(model_dir.parent))
    return model_dir


def audio_cache_path(work_dir, video):
    """16k 音轨缓存路径：按视频名+文件大小哈希命名。

    换视频不互相覆盖，同一视频可跨分析/识别复用。
    """
    video = Path(video)
    key = f"{video.name}|{video.stat().st_size}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return Path(work_dir) / f"audio_{digest}.wav"


def ffmpeg_extract_audio(video, out_wav, sr=16000, progress_cb=None):
    """用 ffmpeg 抽取 16kHz 单声道音轨（AED 输入）。"""
    def log(msg):
        print(msg, flush=True)
        if progress_cb:
            progress_cb(msg)

    log(f"[1/2] 抽取音轨: {video}")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-map", "0:a:0",
        "-ac", "1", "-ar", str(sr),
        "-f", "wav", str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    log(f"      完成: {out_wav}")


def singing_probs(wav_path, model_dir, use_gpu, progress_cb=None):
    """跑 AED，返回 (singing 帧概率, 时长秒)。帧移 10ms。"""
    from fireredvad import FireRedAed, FireRedAedConfig

    def log(msg):
        print(msg, flush=True)
        if progress_cb:
            progress_cb(msg)

    log(f"[2/2] FireRedVAD 歌声检测 ({'cuda' if use_gpu else 'cpu'})...")
    aed = FireRedAed.from_pretrained(
        str(ensure_model(model_dir)), FireRedAedConfig(use_gpu=use_gpu))
    result, probs = aed.detect(str(wav_path))
    probs = probs.numpy()
    log(f"      完成，音频 {result['dur']:.0f}s")
    return probs[:, 1], result["dur"]


def window_candidates(singing_p, win_sec=WIN_SEC, hop_sec=WIN_HOP_SEC,
                      thresh=WIN_THRESH, merge_gap=MERGE_GAP, min_duration=15.0):
    """滑窗聚合 singing 概率 → 候选段列表 [{"start","end","score"}]。

    singing_p 为帧级概率（帧移 10ms），返回秒单位的候选区间。
    """
    frame_hop = 0.01
    n = len(singing_p)
    win_f = int(win_sec / frame_hop)
    hop_f = max(1, int(hop_sec / frame_hop))

    clusters = []  # [start_sec, end_sec, [窗得分]]
    for start in range(0, n - win_f + 1, hop_f):
        score = float(singing_p[start:start + win_f].mean())
        if score < thresh:
            continue
        t = start * frame_hop
        if clusters and t - clusters[-1][1] <= merge_gap:
            clusters[-1][1] = t + win_sec
            clusters[-1][2].append(score)
        else:
            clusters.append([t, t + win_sec, [score]])

    candidates = []
    for t0, t1, scores in clusters:
        if t1 - t0 < min_duration:
            continue
        candidates.append({
            "start": t0,
            "end": min(t1, n * frame_hop),
            "score": float(np.mean(scores)),
        })
    return candidates


def confidence_of(score):
    """窗平均概率 → 置信度档位。"""
    if score >= 0.6:
        return "高"
    if score >= 0.45:
        return "中"
    return "低"


def detect(video, work_dir="work", candidates_path="candidates.txt",
           model_dir=DEFAULT_MODEL_DIR, device="cpu", min_duration=15.0,
           win_sec=WIN_SEC, win_thresh=WIN_THRESH, merge_gap=MERGE_GAP,
           progress_cb=None):
    """检测主流程：产出候选清单文件，返回候选列表。

    progress_cb: 可选的进度回调，参数为日志字符串（供 Web 轮询展示）。
    """
    def log(msg):
        print(msg, flush=True)
        if progress_cb:
            progress_cb(msg)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    video = Path(video)

    audio_wav = audio_cache_path(work_dir, video)
    if not audio_wav.exists():
        ffmpeg_extract_audio(video, audio_wav, progress_cb=progress_cb)

    use_gpu = (device == "cuda")
    if use_gpu:
        import torch
        if not torch.cuda.is_available():
            log("警告: CUDA 不可用，回退到 CPU")
            use_gpu = False
    singing_p, dur = singing_probs(audio_wav, model_dir, use_gpu,
                                   progress_cb=progress_cb)
    candidates = window_candidates(
        singing_p, win_sec=win_sec, thresh=win_thresh,
        merge_gap=merge_gap, min_duration=min_duration)
    for c in candidates:
        c["confidence"] = confidence_of(c["score"])

    lines = [
        "# AutoMusicCut 候选清单（仅供参考，最终以人工清单 songs.txt 为准）",
        f"# 视频: {video.name}（时长 {format_time(dur)}）",
        "# 格式: 序号 | 开始 | 结束 | 时长 | 置信度 | 概率",
    ]
    for i, c in enumerate(candidates, 1):
        d = c["end"] - c["start"]
        lines.append(
            f"{i:3d} | {format_time(c['start'])} | {format_time(c['end'])} "
            f"| {format_time(d)[3:]} | {c['confidence']} | {c['score']:.2f}"
        )
    Path(candidates_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(c["end"] - c["start"] for c in candidates)
    log(f"      候选段 {len(candidates)} 个，合计 {format_time(total)}")
    log(f"      清单已写入: {candidates_path}")
    return candidates
