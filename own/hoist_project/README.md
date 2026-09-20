# `own/hoist_project` —— 吊装单场景自有资源子树

本目录是**自有资源**子树（与仓库根下 `open/` 开源资源子树**硬隔离**）。
凡「真实素材派生数据」「私有端点配置」「本场景专用提示词」都只落在这里。

## 目录结构

```
own/hoist_project/
├── config/
│   └── project.hoist.json      # 项目配置（12 类 labels / private_vlm / endpoint allowlist）
├── assets/prompts/
│   └── hoist_zh.md             # 吊装中文标注提示词（含 KB S1 注入与受控词表要求）
├── input/                      # 素材派生数据（关键帧）；不入库，见 .gitignore
└── workspace/                  # 运行产物；不入库，见 .gitignore
```

## 素材来源（不随仓库分发）

- **珠海吊装视频 4 段**：约 31.8 分钟，源分辨率 960×540、30fps → 走「视频抽帧」路径。
- **宿州吊装隐患照片 31 张**：多为 4096×3072 → 走「图片直入」路径。

> 原始素材统一存放在 `D:\WorkPlace\industrial-data\`（本机）或 `lab-server`
> 的内网路径下，**绝不复制进本仓库**；本仓库只保留配置与提示词。

## 关键约定（与架构 §7 / §10 一致）

1. **类别体系**：12 类，`class_id` 顺序即 `labels` 数组顺序，**不可重排**。
   已排除梯具/脚手架/护栏/安全带等高处作业类别。
2. **场景**：本期 `scene` 恒为 `hoist`（`height` 留位不实现）。
3. **端点护栏**：VLM 端点 `119.23.145.227:38091` 是**公网 IP**，
   `is_local_endpoint_host()` 判定为 `False`，因此**必须**用 `allowlist` 模式显式白名单，
   否则护栏会把合法端点拦死：

   ```json
   "inference": {
     "endpoint_policy": {
       "mode": "allowlist",
       "allowlist": ["119.23.145.227:38091"]
     }
   }
   ```

4. **VLM 调用从 `lab-server` 发起**：本机到端点不可达（实测 TCP 超时），
   端点调用必须落在服务器侧。
5. **原始素材目录约定（`media_root`）**：配置里没有「原始素材目录」字段，
   编目（`inventory`）默认扫描 **`input_frames_dir` 同级的 `media/`**，
   即 `own/hoist_project/input/media/`。之所以这样定：让「原始素材」与
   「抽帧产物（`input/frames/`）」分离，互不污染，且不必改动配置结构。
   如需指向别处（如挂载盘），用 `run_inventory(config, media_root=...)`
   或 CLI 的 `--config` 指向的工程里调整；**切勿把真实素材复制进仓库**。

## 抽帧产物：PNG 是交付物，PGM 是中间产物

`extract-video` 阶段对每段视频按 `video.sample_fps` 抽帧，每帧产出两个文件：

- `frame_<i>_<ts>.png` —— **关键帧交付物**（按 `video.scale_width` 缩放），
  写入 `input/frames/<source_id>/`，供母项目 `run_extraction` 消费；
- `frame_<i>_<ts>.pgm` —— **灰度分析中间产物**（按 `video.analysis_width` 缩放），
  留在 `workspace/extracted/video_frames/<source_id>/`。

> 母项目 `imageio` 只能解码 Netpbm，故去重分析依赖 PGM；PNG 仅供交付/可视化。
> **PGM 可安全删除**（保留它是为了调整去重阈值时不必重新解码视频）。
> 请勿把 PGM 当作必须交付的产物。

## 运行顺序

```powershell
# 本机（Windows）小样本冒烟：只探活
pwsh -File scripts/run_hoist_cycle.ps1
```

```bash
# lab-server（Linux）全量：完整链路（当前仅 T05 `evaluate` 未实现）
bash scripts/run_hoist_cycle.sh
```

阶段顺序：`inventory → extract-video → extract → annotate → prioritize →
curate → freeze-holdout → train → evaluate`（`doctor` 可随时单独跑）。

> **已实现（可跑）**：`doctor`（T01）、`inventory / extract-video / extract`（T02）、
> `annotate / prioritize`（T03）、`curate / freeze-holdout / train`（T04）。
> **仍未实现**：`evaluate`（T05）—— CLI 会明确报「尚未实现」并返回退出码 `2`，
> **绝不静默成功**。

### 退出码约定

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 / `doctor` 一切就绪 |
| `1` | `doctor` 读到配置但发现问题（环境问题） |
| `2` | 子命令尚未实现（或参数错误） |
| `3` | 配置**无法读取**（路径不存在 / 非法 JSON / 校验失败） |
| `4` | **已实现**阶段的执行失败（如 ffmpeg 缺失、素材缺失） |

`1` 与 `3` 刻意区分（「配置读不到」vs「读到了但环境有问题」）；`2` 与 `4` 区分
「压根还没实现」与「实现好了但跑失败」。

### T04 新增产物

- `workspace/datasets/<version>/` —— 版本化数据集：`data.yaml` / `manifest.json` /
  `class_histogram.json` / `quality_report.json` / `summary.json`；
- `workspace/holdout/` —— 固定 holdout：`index.json`（冻结清单 + 每帧 sha256）
  + `frames/` + `labels/`（冻结副本）。**跨版本共享**（与 `<version>` 无关）；
- `workspace/runs/<version>/` —— 训练产物：`report.json` / `metrics.json` /
  `command.txt`（`UltralyticsTrainer` 在 `execute=False` 时只准备命令）。

### holdout 铁律（跨轮可比的前提）

`workspace/holdout/index.json` **一旦存在即视为「已冻结」**，后续轮次一律**读回**
该清单、**绝不重新切分**——即便传入不同的 `--version` / `--holdout-ratio` / 随机种子，
即便首轮冻结的是空集合。新增记录只能进 `train` / `val`。`verify_holdout`
重算每帧 sha256，可检出**内容被篡改**与**文件被删**两种情况；`doctor` 也会给出
holdout 的真实完整性结论（`ok` / `tampered` / `missing` / `not_frozen`）。

### 数据集版本号规则

版本号由**调用方指定**（`--version`），**不是时间戳**——因为「跨轮可比」是硬要求
（时间戳版本号每轮都不同，无法跨轮对比指标）。版本号**不得**含 `/`、`\`、`..`
或为绝对路径：它会被拼进 `workspace/datasets/<version>/`，并在同名重建时 `rmtree`，
故 CLI 对其 **fail-loud** 校验，绝不静默越界删除。
