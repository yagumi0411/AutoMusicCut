"""使用单个 HTDemucs 模型提取人声，保留分离前的立体声信息。"""

import hashlib
import threading
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_SIGNATURE = "955717e8"
MODEL_FILE = "955717e8-8726e21a.th"
MODEL_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + MODEL_FILE
MODEL_DIR = Path(__file__).resolve().parent.parent / "pretrained_models" / "Demucs"
SEPARATION_BLOCK_SEC = 60
CONTEXT_SEC = 1

_MODEL = None
_LOCK = threading.Lock()


def ensure_separator(progress_cb=None):
    """调用者持有 _LOCK；权重在项目内下载、校验，模型只加载一次。"""
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = MODEL_DIR / MODEL_FILE
        if not checkpoint.exists():
            if progress_cb:
                progress_cb("首次使用：下载人声分离模型（约 80 MiB）...")
            partial = checkpoint.with_suffix(".download")
            try:
                digest = hashlib.sha256()
                with httpx.stream("GET", MODEL_URL, timeout=60, follow_redirects=True) as r:
                    r.raise_for_status()
                    with partial.open("wb") as f:
                        for chunk in r.iter_bytes():
                            digest.update(chunk)
                            f.write(chunk)
                if not digest.hexdigest().startswith("8726e21a"):
                    raise RuntimeError("人声分离模型校验失败，请重新下载")
                partial.replace(checkpoint)
            finally:
                partial.unlink(missing_ok=True)
        _MODEL = get_model(MODEL_SIGNATURE, repo=MODEL_DIR).eval()
    return _MODEL


def extract_vocals(mix_wav, out_wav, progress_cb=None):
    """分块分离 44.1k 立体声，输出完整的 16k 单声道人声 WAV。

    每块附带首尾上下文，写入时裁掉上下文，避免重复音频并限制长录音的内存占用。
    分离失败直接报告错误，不悄悄使用原始混音。
    """
    import torch
    from demucs.apply import apply_model

    with _LOCK:
        model = ensure_separator(progress_cb)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with sf.SoundFile(mix_wav) as source, sf.SoundFile(
                out_wav, "w", samplerate=16000, channels=1, subtype="FLOAT") as target:
            if source.samplerate != 44100 or source.channels != 2:
                raise ValueError("人声分离输入必须是 44.1 kHz 立体声")
            block = SEPARATION_BLOCK_SEC * source.samplerate
            context = CONTEXT_SEC * source.samplerate
            for offset in range(0, len(source), block):
                end = min(offset + block, len(source))
                left, right = max(0, offset - context), min(len(source), end + context)
                source.seek(left)
                mix = source.read(right - left, dtype="float32", always_2d=True)
                wav = torch.from_numpy(mix.T.copy())
                ref = wav.mean(0)
                mean, std = ref.mean(), ref.std(unbiased=False) + 1e-8
                if progress_cb:
                    progress_cb(f"提取人声 {offset / 44100:.0f} ~ {end / 44100:.0f} 秒...")
                with torch.inference_mode():
                    stems = apply_model(
                        model, ((wav - mean) / std)[None], device=device,
                        shifts=0, split=True, overlap=0.25, progress=False)
                vocals = (stems[0, model.sources.index("vocals")] * std + mean)
                mono = vocals.mean(0).cpu().numpy()
                # 先重采样含上下文的块，再裁剪，以保留重采样滤波器的边界上下文。
                audio = resample_poly(mono, 160, 441)
                start_sample = round((offset - left) * 16000 / 44100)
                count = round(end * 16000 / 44100) - round(offset * 16000 / 44100)
                target.write(audio[start_sample:start_sample + count])
                del stems, vocals, wav
    return Path(out_wav)
