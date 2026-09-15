# AutoMusicCut

自动识别直播录播中的**唱歌片段**并批量无损剪切。

检测基于 [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) 的 AED
（Audio Event Detection）模型，逐帧区分 speech / singing / music，直接
解决"带 BGM 杂谈 vs 唱歌"的判别问题。对 singing 概率做滑窗聚合得到候选
唱歌区间，人工确认歌名与起止时间后批量剪切。

## 工作流程

```
① analyze    自动检测候选唱歌区间 → candidates.txt（时间+置信度）
② 人工       对照原视频试听，整理最终清单 → songs.txt（歌曲名、开始、结束）
③ cut        按清单批量无损剪切 → output/01_歌曲名.mp4 ...
```

- **候选清单只用于辅助定位**，最终以 `songs.txt` 为准；检测漏掉的唱歌段
  照样可以写进 `songs.txt` 剪切。
- 剪切使用 `ffmpeg -c copy` 无损流复制，不重编码、速度快；关键帧对齐带来
  的 ±1~2 秒误差由边界外扩（默认 1.5s）补偿，保证不丢内容。

## 环境要求

- Windows / Linux / macOS
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/)（需在 PATH 中）
- 首次运行自动从 Hugging Face 下载 AED 模型（约 9MB）

## 安装

```bash
git clone https://github.com/yagumi0411/AutoMusicCut.git
cd AutoMusicCut
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 有 NVIDIA GPU（CUDA 12.8，可选，AED 在 CPU 上已约 500 倍实时）：
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 无 GPU：
pip install torch

pip install -r requirements.txt
```

## 用法

### Web 界面（推荐）

```bash
# 双击 start.bat，或手动启动：
.venv\Scripts\python.exe app.py
```

浏览器自动打开 http://localhost:8765 ，在页面里完成全部流程：

- **选择视频**：点击"浏览"在目录浏览器中选中录播文件（或直接填路径）；
- **分析**：点击后自动检测，候选唱歌段以表格展示（置信度着色）；
- **试听**：勾选"内嵌播放器"，点击候选行视频直接跳到该时间点；
- **整理清单**：点"＋加入清单"把候选行加入 songs.txt，也可手动增删行、
  修改歌名与起止时间；
- **识别歌名**（可选）：点候选行"识别"或"全部识别"，自动转写歌词并在
  本地搜索歌名，弹窗给出 Top-N 候选（含歌手与置信度），点选即填入
  songs.txt；首次使用会自动下载 SenseVoiceSmall 转写模型（约 230MB），
  需要联网查询歌词数据库；
- **剪切**：点击后按清单批量无损剪切，完成后可一键打开输出目录。

> 内嵌播放器依赖浏览器对视频编码的支持：mp4(H.264) 可直接播放；
> flv/mkv 等格式浏览器不支持，仍可用外部播放器对照清单试听。

### 命令行

#### 1. 自动检测候选唱歌区间

```bash
python analyze.py data/录播.mp4
# 可选参数：
#   --work-dir work/       中间产物目录（16k 音轨缓存，可复用）
#   --candidates candidates.txt
#   --device cuda|cpu      推理设备（默认 cpu，模型很小、约 500 倍实时）
#   --min-duration 15      候选段最小长度（秒）
#   --win-sec 20           滑窗长度（秒）
#   --win-thresh 0.5       窗内 singing 平均概率阈值，漏检多就调低
#   --merge-gap 30         相邻唱歌窗合并间隔（秒），候选太碎就调大
```

输出 `candidates.txt` 示例：

```
# AutoMusicCut 候选清单（仅供参考，最终以人工清单 songs.txt 为准）
# 格式: 序号 | 开始 | 结束 | 时长 | 置信度 | 得分
  1 | 00:03:10 | 00:06:45 | 03:35 | 高 | 0.82
  2 | 00:58:41 | 01:03:09 | 04:28 | 高 | 0.78
  3 | 01:09:26 | 01:13:40 | 04:14 | 中 | 0.61
```

#### 2. 人工筛选

对照原视频试听候选段，整理最终清单 `songs.txt`（条目间空行分隔，
每条目 3 行：歌曲名、开始、结束；`#` 开头为注释行）：

```
冬眠
005841
010309

旅行的意义
01:09:26
01:13:40
```

时间支持 `HHMMSS`（如 `005841`）或 `HH:MM:SS`（如 `00:58:41`）两种写法。

#### 3. 批量剪切

```bash
python cut.py data/录播.mp4 songs.txt
# 可选参数：
#   --out-dir output/     输出目录
#   --pad 1.5             起止边界外扩秒数
#   --min-duration 180    最小时长过滤（秒），短于该值的条目跳过
#                         例如 180 表示只剪 3 分钟以上的完整曲目
```

输出 `output/01_冬眠.mp4`、`output/02_旅行的意义.mp4`……并打印每段的
实际切出时长。

## 检测原理

1. **抽音轨**：ffmpeg 抽取 16kHz 单声道音轨；
2. **事件检测**：FireRedVAD 的 AED 模型（DFSMN 架构，约 588K 参数，
   支持 100+ 语言）逐帧（10ms）输出 speech / singing / music 三类概率，
   直接区分唱歌、杂谈与音乐；
3. **滑窗聚合**：对 singing 概率做滑窗（默认 20s 窗 / 10s 步）取均值，
   超过阈值（0.5）的窗视为唱歌窗，间隔 30s 以内的相邻窗合并成候选区间，
   输出候选清单。

默认参数用真实录播与人工时间戳标定：13 首真值歌曲全部命中、
覆盖唱歌时长的 88%。不同主播/混音风格下可微调 `--win-thresh` 与
`--merge-gap`。

旧版 demucs+pyin 特征管线在中文声调语言下区分度不足（标定分析见
[docs/model-plan.md](docs/model-plan.md)），已被 FireRedVAD 取代；
如需进一步提升检测质量，可参考该文档中的模型训练路线。

## 免责声明

本工具仅供个人学习、整理备份自存录播使用。请尊重内容创作者的权利，
不要将剪切结果用于未经授权的传播或商业用途。

## License

[MIT](LICENSE)
