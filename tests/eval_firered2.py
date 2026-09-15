"""FireRedVAD 帧概率的滑窗聚合与参数扫描（段级评估）。

detect 只跑一次，缓存帧级概率；随后离线扫描滑窗聚合参数，
找"候选清单"质量最优的组合。

用法: .venv/Scripts/python.exe tests/eval_firered2.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from fireredvad import FireRedAed, FireRedAedConfig

from amc.cut import parse_songlist

MODEL_DIR = ROOT / "pretrained_models" / "FireRedVAD" / "AED"
WORK = ROOT / "tests" / "firered_work"
FRAME_HOP = 0.01  # AED 帧移 10ms


def load_probs():
    cache = WORK / "完整视频_probs.npz"
    if cache.exists():
        d = np.load(cache)
        return d["singing_p"], d["speech_p"], d["music_p"], float(d["dur"])
    aed = FireRedAed.from_pretrained(
        str(MODEL_DIR), FireRedAedConfig(use_gpu=False))
    result, probs = aed.detect(str(WORK / "完整视频.wav"))
    probs = probs.numpy()
    np.savez(cache, singing_p=probs[:, 1], speech_p=probs[:, 0],
             music_p=probs[:, 2], dur=result["dur"])
    return probs[:, 1], probs[:, 0], probs[:, 2], result["dur"]


def window_candidates(prob, win_sec, hop_sec, th, merge_gap):
    """滑窗聚合概率 → 候选段列表。"""
    n = len(prob)
    win_f = int(win_sec / FRAME_HOP)
    hop_f = max(1, int(hop_sec / FRAME_HOP))
    segs = []
    for start in range(0, n - win_f + 1, hop_f):
        if prob[start:start + win_f].mean() >= th:
            t = start * FRAME_HOP
            if segs and t - segs[-1][1] <= merge_gap:
                segs[-1][1] = t + win_sec
            else:
                segs.append([t, t + win_sec])
    return [(s, e) for s, e in segs]


def eval_segments(det_segs, truth_segs):
    hit = 0
    tp_sec = 0.0
    for s, e in truth_segs:
        best_iou = 0.0
        for ds, de in det_segs:
            inter = max(0, min(e, de) - max(s, ds))
            union = max(e, de) - min(s, ds)
            iou = inter / max(1e-9, union)
            best_iou = max(best_iou, iou)
            tp_sec += inter if iou == best_iou else 0  # 用最佳匹配段计 TP
        if best_iou > 0.5:
            hit += 1
    det_sec = sum(e - s for s, e in det_segs)
    truth_sec = sum(e - s for s, e in truth_segs)
    recall = hit / len(truth_segs)
    # 覆盖召回：检测段覆盖的真值时长比例
    cover = 0.0
    for s, e in truth_segs:
        for ds, de in det_segs:
            cover += max(0, min(e, de) - max(s, ds))
    cover_r = cover / truth_sec
    precision = cover / max(1, det_sec)
    return recall, cover_r, precision, det_sec


def main():
    truth_segs = [(s, e) for _, s, e in parse_songlist(ROOT / "data" / "2026.9.9.txt")]
    singing_p, _, _, dur = load_probs()
    print(f"视频 {dur:.0f}s，真值 {len(truth_segs)} 段 "
          f"({sum(e - s for s, e in truth_segs):.0f}s)\n")

    print(f"{'窗长':>4s} {'阈值':>4s} {'合并':>4s} "
          f"{'段命中':>6s} {'时长覆盖':>8s} {'精度':>6s} {'候选总长':>8s}")
    rows = []
    for win_sec in [5, 10, 20]:
        for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for gap in [5, 15, 30]:
                det = window_candidates(singing_p, win_sec, win_sec / 2, th, gap)
                recall, cover_r, precision, det_sec = eval_segments(det, truth_segs)
                rows.append((win_sec, th, gap, recall, cover_r, precision, det_sec))
    # 按覆盖召回排序展示前 12
    rows.sort(key=lambda r: (-r[4], -r[5]))
    for win_sec, th, gap, recall, cover_r, precision, det_sec in rows[:12]:
        print(f"{win_sec:4.0f} {th:4.1f} {gap:4.0f} "
              f"{recall:6.0%} {cover_r:8.1%} {precision:6.1%} {det_sec:8.0f}s")

    print("\n对照：默认 AED 后处理段")
    from fireredvad import FireRedAed, FireRedAedConfig
    aed = FireRedAed.from_pretrained(
        str(MODEL_DIR), FireRedAedConfig(use_gpu=False))
    result, _ = aed.detect(str(WORK / "完整视频.wav"))
    ev = result["event2timestamps"]
    # 与上面同样合并
    merged = []
    for s, e in ev["singing"]:
        if merged and s - merged[-1][1] <= 5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    recall, cover_r, precision, det_sec = eval_segments(
        [(s, e) for s, e in merged], truth_segs)
    print(f"  段命中 {recall:.0%}  时长覆盖 {cover_r:.1%}  "
          f"精度 {precision:.1%}  候选总长 {det_sec:.0f}s")


if __name__ == "__main__":
    main()
