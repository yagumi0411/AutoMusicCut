"""用法: python analyze.py <视频路径> [选项]

自动检测视频中的候选唱歌区间，输出 candidates.txt 供人工筛选。
检测基于 FireRedVAD AED 模型（首次使用自动下载模型）。
"""

import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from amc.detect import (  # noqa: E402
    DEFAULT_MODEL_DIR, MERGE_GAP, WIN_FRAC_MIN, WIN_SEC, WIN_THRESH, detect,
)


def main():
    p = argparse.ArgumentParser(description="检测视频中的候选唱歌区间")
    p.add_argument("video", help="视频文件路径")
    p.add_argument("--work-dir", default="work", help="中间产物目录（默认 work/）")
    p.add_argument("--candidates", default="candidates.txt", help="候选清单输出路径")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="FireRedVAD AED 模型目录，缺失时自动下载")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"],
                   help="AED 推理设备（默认 cpu，模型很小、500 倍实时）")
    p.add_argument("--min-duration", type=float, default=15.0,
                   help="候选段最小长度，秒（默认 15）")
    p.add_argument("--win-sec", type=float, default=WIN_SEC,
                   help=f"滑窗长度，秒（默认 {WIN_SEC}）")
    p.add_argument("--win-thresh", type=float, default=WIN_THRESH,
                   help=f"窗内 singing 平均概率阈值（默认 {WIN_THRESH}）")
    p.add_argument("--merge-gap", type=float, default=MERGE_GAP,
                   help=f"相邻唱歌窗合并间隔，秒（默认 {MERGE_GAP}）")
    p.add_argument("--win-frac-min", type=float, default=WIN_FRAC_MIN,
                   help=f"窗内 singing 压过 speech 与 music 的帧占比下限，"
                        f"用于剔除带 BGM 杂谈的误报（默认 {WIN_FRAC_MIN}，"
                        f"填 0 关闭）")
    args = p.parse_args()

    detect(
        video=args.video,
        work_dir=args.work_dir,
        candidates_path=args.candidates,
        model_dir=args.model_dir,
        device=args.device,
        min_duration=args.min_duration,
        win_sec=args.win_sec,
        win_thresh=args.win_thresh,
        merge_gap=args.merge_gap,
        win_frac_min=args.win_frac_min,
    )


if __name__ == "__main__":
    main()
