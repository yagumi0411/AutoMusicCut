"""用法: python cut.py <视频路径> <清单.txt> [选项]

按最终清单批量无损剪切唱歌片段。

清单格式（条目间空行分隔，每条目 3 行：歌曲名、开始、结束；
时间支持 005841 或 00:58:41，行首 # 为注释）:

    冬眠
    005841
    010309

    旅行的意义
    01:09:26
    01:13:40
"""

import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from amc.cut import cut  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="按清单批量无损剪切唱歌片段")
    p.add_argument("video", help="视频文件路径")
    p.add_argument("songlist", help="最终清单 txt 路径")
    p.add_argument("--out-dir", default="output", help="输出目录（默认 output/）")
    p.add_argument("--pad", type=float, default=1.5,
                   help="起止边界外扩秒数（默认 1.5，配合无损剪切关键帧对齐）")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="最小时长秒数，短于此值的条目跳过（如 180 只剪 3 分钟以上的）")
    args = p.parse_args()

    cut(video=args.video, songlist_path=args.songlist,
        out_dir=args.out_dir, pad=args.pad, min_duration=args.min_duration)


if __name__ == "__main__":
    main()
