"""FireRedVAD AED 模型在真实数据上的评估。

1. 4 个样品（唱歌1/2、杂谈1/2）：看 speech/singing/music 事件分布
2. 完整视频 + 2026.9.9.txt 真值：帧级阈值扫描 + 段级命中评估

用法: .venv/Scripts/python.exe tests/eval_firered.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from fireredvad import FireRedAed, FireRedAedConfig

from amc.cut import parse_songlist

MODEL_DIR = ROOT / "pretrained_models" / "FireRedVAD" / "AED"
WORK = ROOT / "tests" / "firered_work"
WORK.mkdir(parents=True, exist_ok=True)

FRAME_HOP = 0.01  # AED 帧移 10ms


def load_aed():
    cfg = FireRedAedConfig(use_gpu=False)
    return FireRedAed.from_pretrained(str(MODEL_DIR), cfg)


def extract_16k(video, out_wav):
    """从视频抽 16kHz 单声道 wav（AED 输入）。"""
    if not Path(out_wav).exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video), "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav),
        ], check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
    return Path(out_wav)


def merge_segments(segs, gap=5.0):
    """合并相邻（间隔 < gap 秒）的事件段。"""
    if not segs:
        return []
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def eval_samples(aed):
    print("=== 1. 样品文件事件分布 ===")
    for name in ["唱歌1", "唱歌2", "杂谈1", "杂谈2"]:
        wav = extract_16k(ROOT / "data" / f"{name}.mp4", WORK / f"{name}.wav")
        result, probs = aed.detect(str(wav))
        ev = result["event2timestamps"]
        print(f"\n{name} (时长 {result['dur']:.0f}s):")
        print(f"  事件占比 speech={result['event2ratio']['speech']} "
              f"singing={result['event2ratio']['singing']} "
              f"music={result['event2ratio']['music']}")
        for event in ["speech", "singing", "music"]:
            segs = merge_segments(ev[event])
            total = sum(e - s for s, e in segs)
            print(f"  {event:8s}: {len(segs)} 段, 合计 {total:.0f}s")


def eval_full(aed):
    print("\n=== 2. 完整视频真值评估 ===")
    wav = extract_16k(ROOT / "data" / "完整视频.mp4", WORK / "完整视频.wav")
    truth = parse_songlist(ROOT / "data" / "2026.9.9.txt")
    truth_segs = [(s, e) for _, s, e in truth]
    print(f"真值 {len(truth_segs)} 个唱歌段")

    result, probs = aed.detect(str(wav))
    probs = probs.numpy()  # (T, 3)
    t_axis = np.arange(len(probs)) * FRAME_HOP
    singing_p = probs[:, 1]

    # 真值帧标记
    truth_label = np.zeros(len(probs), dtype=bool)
    for s, e in truth_segs:
        truth_label[(t_axis >= s) & (t_axis < e)] = True
    print(f"帧级真值唱歌占比: {truth_label.mean():.2%}")

    # 帧级阈值扫描
    print("\n帧级阈值扫描（singing 通道）:")
    print(f"  {'阈值':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    best = None
    for th in np.arange(0.1, 0.91, 0.1):
        pred = singing_p >= th
        tp = (pred & truth_label).sum()
        fp = (pred & ~truth_label).sum()
        fn = (~pred & truth_label).sum()
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        print(f"  {th:5.1f} {p:6.3f} {r:6.3f} {f1:6.3f}")
        if best is None or f1 > best[2]:
            best = (th, p, r, f1)
    print(f"最优帧级: 阈值 {best[0]:.1f} → F1 {best[3]:.3f} (P {best[1]:.3f}, R {best[2]:.3f})")

    # 段级命中（默认后处理参数）
    ev = result["event2timestamps"]
    det_segs = merge_segments(ev["singing"])
    print(f"\n段级评估（singing 事件 {len(det_segs)} 段，间隔<5s 已合并）:")
    hit = 0
    for s, e in truth_segs:
        # 与任一检测段 IoU > 0.5 视为命中
        matched = False
        for ds, de in det_segs:
            inter = max(0, min(e, de) - max(s, ds))
            union = max(e, de) - min(s, ds)
            if inter / max(1e-9, union) > 0.5:
                matched = True
                break
        if matched:
            hit += 1
    print(f"  命中 {hit}/{len(truth_segs)} 首 (召回 {hit / len(truth_segs):.2%})")
    # 检测段里有多少是真唱歌（精度）
    tp_sec = 0
    for ds, de in det_segs:
        for s, e in truth_segs:
            tp_sec += max(0, min(e, de) - max(s, ds))
    det_sec = sum(e - s for s, e in det_segs)
    print(f"  检测总时长 {det_sec:.0f}s，其中真唱歌 {tp_sec:.0f}s (精度 {tp_sec / max(1, det_sec):.2%})")

    # 逐首报告
    print("\n逐首命中情况:")
    for name, s, e in truth:
        best_iou, best_seg = 0, None
        for ds, de in det_segs:
            inter = max(0, min(e, de) - max(s, ds))
            union = max(e, de) - min(s, ds)
            iou = inter / max(1e-9, union)
            if iou > best_iou:
                best_iou, best_seg = iou, (ds, de)
        mark = "命中" if best_iou > 0.5 else ("部分" if best_iou > 0.2 else "漏检")
        seg_txt = f"检测 {best_seg[0]:.0f}~{best_seg[1]:.0f}s" if best_seg else "无检测段"
        print(f"  {name:16s} 真值 {s:7.0f}~{e:7.0f}s  IoU {best_iou:.2f}  {mark}  ({seg_txt})")


def main():
    aed = load_aed()
    eval_samples(aed)
    eval_full(aed)


if __name__ == "__main__":
    main()
