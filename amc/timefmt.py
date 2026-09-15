"""时间格式解析与格式化。

支持两种写法（可混用）：
- HHMMSS 纯数字，如 005841 → 00:58:41
- HH:MM:SS 冒号分隔，如 01:09:26
- MM:SS 冒号分隔，如 58:41 → 00:58:41
"""

import re

_PURE_DIGITS_RE = re.compile(r"^\d{6}$")


def parse_time(text):
    """解析时间字符串，返回秒数（float）。"""
    s = str(text).strip()
    if _PURE_DIGITS_RE.match(s):
        h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
    else:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = map(int, parts)
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"无法解析时间: {s!r}（支持 005841 或 00:58:41 或 58:41）")
    if not (0 <= m < 60 and 0 <= sec < 60):
        raise ValueError(f"时间非法: {s!r}")
    return h * 3600 + m * 60 + sec


def format_time(seconds):
    """秒数 → HH:MM:SS 字符串。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
