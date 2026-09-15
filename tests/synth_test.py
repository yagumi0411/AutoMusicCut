"""端到端测试：清单解析、时间格式与批量剪切。

生成 70s 合成视频（音频布局：10~33s 旋律段，36~60s 语调段），
用 songs.txt 清单跑 cut，校验剪切时长与最小时长过滤。

用法: .venv/Scripts/python.exe tests/synth_test.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from amc.cut import cut
from amc.timefmt import format_time, parse_time

SR = 44100
WORK = ROOT / "tests" / "synth_work"
WORK.mkdir(parents=True, exist_ok=True)


def note(freq, dur, amp=0.8):
    """带 3 次谐波的正弦音符。"""
    t = np.arange(int(dur * SR)) / SR
    x = sum(amp / (k + 1) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(3))
    n = len(x)
    fade = min(int(0.02 * SR), n // 4)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return x * env


def make_audio():
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00,
             493.88, 523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66, 261.63]
    segs = [note(f, 1.5) for f in freqs]
    gap = np.zeros(int(0.05 * SR))
    song = np.concatenate([s for pair in [(s, gap) for s in segs] for s in pair])

    y = np.zeros(int(70 * SR), dtype=np.float32)
    y[int(10 * SR):int(10 * SR) + len(song)] = song
    import soundfile as sf
    wav = WORK / "synth.wav"
    sf.write(str(wav), y, SR, subtype="PCM_16")
    return wav


def make_video(wav):
    mp4 = WORK / "synth.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=70",
        "-i", str(wav),
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        "-shortest", str(mp4),
    ], check=True, capture_output=True, encoding="utf-8", errors="replace")
    return mp4


def check(cond, msg):
    if not cond:
        print(f"测试失败: {msg}")
        sys.exit(1)
    print(f"通过: {msg}")


def main():
    # 时间格式
    check(parse_time("005841") == 3521, "HHMMSS 解析 (005841 → 3521s)")
    check(parse_time("01:03:09") == 3789, "HH:MM:SS 解析 (01:03:09 → 3789s)")
    check(format_time(3521) == "00:58:41", "秒数格式化 (3521s → 00:58:41)")

    # 合成视频 + 剪切
    wav = make_audio()
    mp4 = make_video(wav)
    songlist = WORK / "songs.txt"
    songlist.write_text(
        "# 合成测试清单\n合成歌曲\n000010\n000035\n\n短片段\n000050\n000055\n",
        encoding="utf-8")

    report = cut(video=mp4, songlist_path=songlist,
                 out_dir=WORK / "output", pad=0.0, min_duration=10.0)
    check(len(report) == 1, f"最小时长过滤（2 条中保留 1 条）")
    check(abs(report[0]["actual"] - 25) < 3,
          f"剪切时长（实际 {report[0]['actual']:.1f}s，预期约 25s）")

    print("\n全部测试通过")


if __name__ == "__main__":
    main()
