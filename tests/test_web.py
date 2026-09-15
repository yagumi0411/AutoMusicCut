"""Web API 端到端测试：分析 → 候选 → 加入清单 → 保存 → 剪切。

用法: .venv/Scripts/python.exe tests/test_web.py
（需先启动 app.py）
"""

import sys
import time
from pathlib import Path

import requests

BASE = "http://localhost:8765"
VIDEO = str((Path(__file__).resolve().parent.parent / "data" / "唱歌1.mp4").resolve())


def wait_task(task_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{BASE}/api/task/{task_id}").json()
        if r["status"] == "done":
            return r["result"]
        if r["status"] == "error":
            raise RuntimeError(r["error"])
        time.sleep(1)
    raise TimeoutError("任务超时")


def check(cond, msg):
    if not cond:
        print(f"失败: {msg}")
        sys.exit(1)
    print(f"通过: {msg}")


def main():
    # 1. 分析
    r = requests.post(f"{BASE}/api/analyze", json={"video": VIDEO}).json()
    result = wait_task(r["task_id"])
    check(result and result.get("candidates_path"), "analyze 任务完成")

    # 2. 候选清单
    r = requests.get(f"{BASE}/api/candidates").json()
    check(r["exists"] and len(r["candidates"]) > 0,
          f"候选清单读取（{len(r['candidates'])} 条）")
    first = r["candidates"][0]
    print(f"  首个候选: {first['start_str']} ~ {first['end_str']} "
          f"置信度 {first['confidence']}")

    # 3. 加入最终清单
    r = requests.post(f"{BASE}/api/candidates/add",
                      json={"candidates": [first]}).json()
    check(r["count"] == 1, "候选加入最终清单")

    # 4. 读取最终清单
    r = requests.get(f"{BASE}/api/songs").json()
    check(len(r["songs"]) == 1, f"最终清单读取（{len(r['songs'])} 条）")

    # 5. 修改并保存最终清单
    songs = r["songs"]
    songs[0]["name"] = "测试歌曲"
    r = requests.post(f"{BASE}/api/songs", json={"songs": songs})
    check(r.status_code == 200 and r.json()["count"] == 1, "保存修改后的清单")
    r = requests.get(f"{BASE}/api/songs").json()
    check(r["songs"][0]["name"] == "测试歌曲", "保存内容生效")

    # 6. 剪切
    r = requests.post(f"{BASE}/api/cut", json={"video": VIDEO}).json()
    result = wait_task(r["task_id"])
    check(len(result["items"]) == 1, f"剪切完成（{len(result['items'])} 段）")
    out = Path(result["items"][0]["path"])
    check(out.exists() and out.stat().st_size > 0,
          f"输出文件存在（{out.name}, {out.stat().st_size / 1e6:.1f}MB）")

    # 7. 视频流（Range 支持）
    r = requests.get(f"{BASE}/api/stream", params={"path": VIDEO},
                     headers={"Range": "bytes=0-1023"})
    check(r.status_code == 206, f"视频流 Range 支持（HTTP {r.status_code}）")

    print("\nWeb API 全部测试通过")


if __name__ == "__main__":
    main()
