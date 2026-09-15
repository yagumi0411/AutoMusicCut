# AutoMusicCut

自动识别直播录播中的**唱歌片段**并批量无损剪切。

检测基于 [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) 的 AED
（Audio Event Detection）模型，逐帧区分 speech / singing / music，直接
解决"带 BGM 杂谈 vs 唱歌"的判别问题。对 singing 概率滑窗聚合得到候选区间，
人工确认歌名与起止时间后批量剪切，可选调用 ASR + 歌词库推荐歌名。

## 工作流程

```
① 分析    自动检测候选唱歌区间 → work/<视频名>/temp/candidates.txt
② 人工    对照原视频试听，整理最终清单 → work/<视频名>/temp/songs.txt
③ 剪切    按清单批量无损剪切 → work/<视频名>/result/01_歌曲名.mp4
```

- **候选清单只用于辅助定位**，最终以 `songs.txt` 为准；检测漏掉的唱歌段
  照样可以写进 `songs.txt` 剪切。
- 剪切用 `ffmpeg -c copy` 无损流复制，不重编码；关键帧对齐的 ±1~2 秒误差
  由边界外扩（默认 1.5s）补偿。

## 环境要求

- Windows / Linux / macOS，Python 3.10+
- [ffmpeg](https://ffmpeg.org/)（`ffmpeg` 与 `ffprobe` 需在 PATH 中）
- 首次运行自动下载 AED 模型（约 9MB）；歌名识别另需下载 SenseVoiceSmall
  转写模型（约 230MB）并联网

## 安装

```bash
git clone https://github.com/yagumi0411/AutoMusicCut.git
cd AutoMusicCut
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# 有 NVIDIA GPU 可选（AED 在 CPU 上已约 500 倍实时）：
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 无 GPU：pip install torch

pip install -r requirements.txt
```

依赖：`fireredvad`、`funasr`、`fastapi` + `uvicorn`、`httpx`、`numpy`、
`soundfile`、`huggingface_hub`。

## 用法

### Web 界面（推荐）

```bash
# 双击 start.bat，或手动启动（--port 指定端口，--no-browser 不开浏览器）：
.venv\Scripts\python.exe app.py
```

浏览器打开 http://localhost:8765 ，在页面里完成全部流程：

1. **选视频**：点「浏览」选文件、直接拖文件进页面，或手动填路径（记在浏览器
   本地，刷新后自动恢复）；
2. **分析**：后台执行，右下角可拖动日志面板实时显示进度；候选段以表格展示
   （置信度着色 + 窗平均概率）。检测固定用 CPU，需要 GPU 时走命令行；
3. **试听**：勾选「内嵌播放器」后，点候选行的**开始/结束单元格**分别跳到该
   时间点；播放器带 ±1/5/10 秒微调与 0.5×~2× 倍速，便于对齐歌曲起止；
4. **整理清单**：点「＋清单」加入 songs.txt（歌名默认 `歌曲N`，可直接编辑），
   也可手动增删行、改时间；删除与追加会立即落盘；
5. **识别歌名**（可选）：点候选行「识别」或「全部识别」，自动转写歌词并在
   lrclib 歌词库检索，弹窗给出候选（歌手 + 相似度），点选即填入清单；
6. **剪切**：按清单批量无损剪切到 `work/<视频名>/result/`，完成后可一键打开
   输出目录。

每个视频一个独立会话目录，切换视频互不覆盖：

```
work/<视频名>/
├── temp/    candidates.txt、songs.txt、audio_<哈希>.wav（16k 音轨缓存，可复用）
└── result/  01_歌曲名.mp4、02_歌曲名.mp4 …
```

> 内嵌播放器只支持浏览器可解码的容器（mp4/m4v 等）；flv/mkv/ts 请用外部
> 播放器对照清单试听。「打开输出目录」为 Windows 专属。

### 命令行

#### 1. 自动检测候选唱歌区间

```bash
python analyze.py data/录播.mp4
#   --work-dir work/       中间产物目录（16k 音轨缓存，可复用）
#   --candidates candidates.txt
#   --model-dir pretrained_models/FireRedVAD/AED   模型目录，缺失时自动下载
#   --device cuda|cpu      推理设备（默认 cpu，模型很小、约 500 倍实时）
#   --min-duration 15      候选段最小长度（秒）
#   --win-sec 20           滑窗长度（秒）
#   --win-thresh 0.5       窗内 singing 平均概率阈值，漏检多就调低
#   --win-frac-min 0.08    窗内 singing 压过 speech 与 music 的帧占比下限，
#                          误报多就调大，漏检多就调小或填 0 关闭
#   --merge-gap 30         相邻唱歌窗合并间隔（秒），候选太碎就调大
```

输出 `candidates.txt` 示例：

```
# AutoMusicCut 候选清单（仅供参考，最终以人工清单 songs.txt 为准）
# 视频: 录播.mp4（时长 03:41:20）
# 格式: 序号 | 开始 | 结束 | 时长 | 置信度 | 概率
  1 | 00:03:10 | 00:06:45 | 03:35 | 高 | 0.82
  2 | 00:58:41 | 01:03:09 | 04:28 | 高 | 0.78
  3 | 01:09:26 | 01:13:40 | 04:14 | 中 | 0.61
```

#### 2. 人工筛选

对照原视频试听候选段，整理最终清单 `songs.txt`（条目间空行分隔，每条目 3 行：
歌曲名、开始、结束；`#` 开头为注释行）：

```
冬眠
005841
010309

旅行的意义
01:09:26
01:13:40
```

时间支持 `HHMMSS`（`005841`）、`HH:MM:SS`（`00:58:41`）、`MM:SS`（`58:41`）。

#### 3. 批量剪切

```bash
python cut.py data/录播.mp4 songs.txt
#   --out-dir output/     输出目录
#   --pad 1.5             起止边界外扩秒数
#   --min-duration 180    最小时长过滤（秒），短于该值的条目跳过，
#                         例如 180 表示只剪 3 分钟以上的完整曲目
```

输出 `output/01_冬眠.mp4`、`output/02_旅行的意义.mp4`……并打印每段实际切出时长。

## 检测原理

1. **抽音轨**：ffmpeg 抽取 16kHz 单声道音轨（按视频名+文件大小哈希缓存，重复
   分析直接复用）；
2. **事件检测**：FireRedVAD AED 模型（DFSMN，约 588K 参数，支持 100+ 语言）
   逐帧（10ms）输出 speech / singing / music 三类概率；
3. **滑窗聚合**：对 singing 概率滑窗（默认 20s 窗 / 10s 步）取均值，超过阈值
   （0.5）的窗视为唱歌窗，间隔 30s 以内的相邻窗合并成候选区间；
4. **剔除带 BGM 杂谈的误报**：杂谈配 BGM 时 BGM 旋律会让 singing 概率同样偏高，
   但此时模型始终认为 music 才是主类；真唱时人声突出，singing 会压过 speech 与
   music。因此要求窗内"singing 同时不小于 speech 与 music"的帧占比不低于
   `--win-frac-min`（默认 0.08）。

默认参数用真实录播与人工时间戳标定：13 首真值歌曲全部命中、覆盖唱歌时长的
84.6%，候选区间精度 83.1%；在 2 小时杂谈录播上，用户标注的误报中最长的一处
被完全剔除。不同主播/混音风格下可微调 `--win-frac-min` 与 `--win-thresh`。

## 歌名识别

定位是**人工确认的辅助**，不自动填写歌名：截取该时间段音轨用 SenseVoiceSmall
本地转写歌词（翻唱也能识别，绕开音频指纹"只能匹配原唱"的限制）→ 从转写文本
取 3 个 2 词短窗口查 lrclib.net（其搜索为精确子串匹配，整句必漏）→ 合并去重后
按歌词相似度排序，返回相似度 ≥ 0.35 的 Top-N 供点选。冷门歌、纯音乐、BGM 太吵
或主播改词时可能给不出候选，手动填写即可。

## 项目结构

```
AutoMusicCut/
├── app.py              Web 服务（FastAPI，仅监听 127.0.0.1；文件头有 API 一览）
├── analyze.py          命令行：检测候选唱歌区间
├── cut.py              命令行：按清单批量无损剪切
├── amc/
│   ├── detect.py       FireRedVAD AED 检测 + 滑窗聚合
│   ├── recognize.py    SenseVoiceSmall 转写 + lrclib 歌词检索
│   ├── cut.py          清单解析与 ffmpeg 无损剪切
│   └── timefmt.py      时间解析/格式化
├── static/index.html   Web 前端（单文件，无构建步骤）
└── start.bat           Windows 一键启动
```

## 免责声明

本工具仅供个人学习、整理备份自存录播使用。请尊重内容创作者的权利，
不要将剪切结果用于未经授权的传播或商业用途。

## License

[MIT](LICENSE)
