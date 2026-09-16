# 智能交通路口 Qwen3VL -> YOLO 闭环训练项目

> 一个把 `视频抽帧`、`YOLO 初检`、`Qwen3VL 二次复核`、`YOLO 数据集生成`、`增量训练` 和 `困难样本回流` 串成闭环的交通视觉工程项目。

## 项目亮点

- 闭环训练：不是只做一次检测，而是完成“抽帧 -> 复标 -> 训练 -> 回流”的完整链路。
- 模型协同：用 YOLO 负责初检和训练，用 Qwen3VL 负责复核和纠偏。
- 双模式支持：既支持 `mock` 演示，也支持真实 `production` 运行。
- Skill 化设计：将三个核心阶段拆成独立 Skill，适合 Agent 调用和课程展示。
- 工程交付完整：包含主程序、CLI、配置、测试、文档、晨读笔记和 GitHub 展示材料。

## 快速入口

- 主程序入口：[1.py](/D:/ruxian/1.py:1)
- 项目说明书：[README.md](/D:/ruxian/README.md:1)
- 生产接入指南：[docs/production-integration.md](/D:/ruxian/docs/production-integration.md)
- 施工隐患链路说明：[docs/hazard-annotation-chain.md](docs/hazard-annotation-chain.md)
- Skills 接入说明：[docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md)
- 任务要求映射：[docs/requirements-mapping.md](/D:/ruxian/docs/requirements-mapping.md)
- 晨读手写笔记：[docs/晨读手写笔记.md](/D:/ruxian/docs/晨读手写笔记.md)

## 仓库适合谁

- 想做智能交通 / 路口检测课程项目的人
- 想把 Qwen3VL 和 YOLO 串成闭环的人
- 想做 Agent Skills / 可复用技能化项目的人
- 想把“能跑”升级成“可交付、可展示、可维护”的人

## 1. 项目简介

本项目是一个面向智能交通路口场景的视觉数据闭环训练系统，用来把“视频抽帧、目标初检、视觉大模型复核、数据集生成、YOLO 训练、困难样本回流”这一整条链路工程化。

它的核心目标不是只完成一次检测或一次训练，而是建立一个可重复执行、可持续优化的闭环流程：

1. 使用 YOLO 对图片或视频帧做初始检测。
2. 使用 Qwen3VL 对初检结果进行二次复核、补标和纠偏。
3. 自动生成标准 YOLO 格式数据集。
4. 使用 Ultralytics YOLO 执行训练或继续训练。
5. 输出困难样本、复核队列和下一轮数据生产任务。

这个项目特别适合以下场景：

- 智能交通路口目标检测训练
- 视频数据自动抽帧与筛样
- 用视觉大模型辅助标注
- YOLO 数据集自动生成
- 小批量增量训练与闭环迭代

---

## 2. 项目解决了什么问题

在交通路口检测项目里，常见问题通常有：

- 视频很多，但人工逐帧筛选效率低。
- YOLO 初检有一定能力，但对遮挡、密集目标、夜间、远距离、小目标容易漏检或误检。
- 人工标注成本高，且不同人标注标准不一致。
- 数据、训练、评估、回流往往是分开的，难形成稳定闭环。

本项目的思路是：

- 先让 YOLO 做“第一轮自动筛查”。
- 再让 Qwen3VL 做“第二轮更细致的复核”。
- 把结果自动转成训练数据。
- 再训练 YOLO，并从训练结果中继续发现困难样本。

这样就形成了一个“模型帮助模型、数据反哺模型”的闭环。

---

## 3. 项目整体架构

从功能上看，本项目可以分成三层：

### 3.1 技能层 `skills/`

这里定义了三个核心 Skill，用于描述每一步做什么、输入输出是什么、如何在 Agent 环境中调用：

- `video-frame-extraction`
- `images-qwen3vl-annotation`
- `yolo-closed-loop-training`

每个 Skill 都带有：

- `SKILL.md`：技能说明书
- `scripts/run.ps1`：对应阶段的快捷启动脚本

### 3.2 业务实现层 `traffic_cv_closed_loop/`

这是项目真正执行逻辑的核心代码，负责：

- 读取配置
- 执行抽帧或候选样本筛选
- 调用 Qwen3VL
- 生成 YOLO 标签
- 组织数据集
- 调用 Ultralytics YOLO 训练
- 输出指标、报告、困难样本

### 3.3 工程集成层 `docs/`、`examples/`、`scripts/`

这一层负责：

- 给出 mock 和 production 的配置示例
- 提供一键运行脚本
- 提供文档、验收映射、接入说明

---

## 4. 闭环流程说明

项目的标准闭环流程如下：

### 第一步：抽帧 / 候选样本提取

输入可以是图片目录，也可以是从视频中提取出的帧。

这一阶段会：

- 遍历输入图像
- 调用 YOLO 做初始检测
- 根据检测结果和阈值筛出候选样本
- 输出候选清单

### 第二步：Qwen3VL 二次标注

这一阶段会：

- 读取候选图像和 YOLO 初检结果
- 调用 Qwen3VL 做复核
- 判断哪些目标需要保留、删除、补充或修正
- 生成标准 YOLO 标签
- 产出差异说明和人工复核队列

### 第三步：训练与回流

这一阶段会：

- 根据有效样本生成版本化数据集
- 按训练集、验证集、测试集切分
- 调用 Ultralytics YOLO 训练
- 输出指标、权重、日志和摘要
- 生成困难样本与下一轮任务建议

---

## 5. 目录结构详解

项目根目录结构如下：

```text
docs/
examples/
scripts/
skills/
tests/
traffic_cv_closed_loop/
1.py
README.md
pyproject.toml
```

下面是每个目录的详细解释。

### 5.1 `docs/`

文档目录，主要用于说明项目设计、接入方式和验收映射。

关键文件：

- [docs/production-integration.md](/D:/ruxian/docs/production-integration.md)
  说明如何接入真实 Qwen 服务和真实 YOLO 环境。
- [docs/qwenvl-yolo-skills-work-order.md](/D:/ruxian/docs/qwenvl-yolo-skills-work-order.md)
  面向项目立项、汇报、课程作业的工单式说明。
- [docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md)
  说明如何把本项目作为 Cursor / Agent Skills 项目来接入。
- [docs/requirements-mapping.md](/D:/ruxian/docs/requirements-mapping.md)
  将图片里的任务要求逐条映射到仓库中的实际实现。

### 5.2 `examples/`

示例项目目录，用来演示两种使用模式。

#### `examples/mock_project/`

这是演示模式项目。

作用：

- 使用 mock 数据跑通流程
- 不依赖真实 Qwen 服务
- 不依赖真实 YOLO 训练
- 适合快速验证代码逻辑

#### `examples/production_project/`

这是生产模式模板。

作用：

- 连接真实 Qwen3VL 服务
- 使用真实 `yolo` 命令做检测与训练
- 放置权重、配置、输入样本和真实输出结果

#### `examples/hazard_mock_project/` 与 `examples/hazard_project/`

这是**施工安全隐患双通道标注**链路的示例项目。

- `examples/hazard_mock_project/`：mock 演示（provider=mock / detector=sidecar / trainer=mock），
  可用一条命令跑通全链路，无需视频、模型和 ffmpeg。
- `examples/hazard_project/`：真实模式模板（provider=`local_openai`，本地 Qwen3-VL 服务）。

详见 [docs/hazard-annotation-chain.md](docs/hazard-annotation-chain.md)。

### 5.3 `scripts/`

项目快捷脚本目录。

关键文件：

- [scripts/run_mock_pipeline.ps1](/D:/ruxian/scripts/run_mock_pipeline.ps1)
  用于快速运行 mock 闭环。
- [scripts/run_production_pipeline.ps1](/D:/ruxian/scripts/run_production_pipeline.ps1)
  用于快速运行 production 闭环。
- [scripts/curate_video_batch.py](/D:/ruxian/scripts/curate_video_batch.py)
  用于对视频帧标注结果进行筛选和清洗，生成更稳定的训练数据子集。

### 5.4 `skills/`

这是按 Agent Skills 思路组织的三个核心能力目录。

#### `skills/video-frame-extraction/`

功能：

- 负责抽帧和候选样本提取

包含：

- [SKILL.md](/D:/ruxian/skills/video-frame-extraction/SKILL.md)
- [scripts/run.ps1](/D:/ruxian/skills/video-frame-extraction/scripts/run.ps1)

#### `skills/images-qwen3vl-annotation/`

功能：

- 负责调用 Qwen3VL 做二次标注和纠偏

包含：

- [SKILL.md](/D:/ruxian/skills/images-qwen3vl-annotation/SKILL.md)
- [scripts/run.ps1](/D:/ruxian/skills/images-qwen3vl-annotation/scripts/run.ps1)
- [prompts/qwen3vl_traffic_zh.md](/D:/ruxian/skills/images-qwen3vl-annotation/prompts/qwen3vl_traffic_zh.md)

#### `skills/yolo-closed-loop-training/`

功能：

- 负责生成数据集、训练模型、输出指标与回流建议

包含：

- [SKILL.md](/D:/ruxian/skills/yolo-closed-loop-training/SKILL.md)
- [scripts/run.ps1](/D:/ruxian/skills/yolo-closed-loop-training/scripts/run.ps1)
- [templates/next_cycle_tasks_template.md](/D:/ruxian/skills/yolo-closed-loop-training/templates/next_cycle_tasks_template.md)

### 5.5 `traffic_cv_closed_loop/`

这是项目的主代码目录。

关键文件说明：

- [cli.py](/D:/ruxian/traffic_cv_closed_loop/cli.py)
  项目命令行入口，支持 `extract`、`annotate`、`train`、`run-loop` 等命令。
- [config.py](/D:/ruxian/traffic_cv_closed_loop/config.py)
  负责读取并校验项目配置。
- [models.py](/D:/ruxian/traffic_cv_closed_loop/models.py)
  定义候选帧、检测框、标注记录等核心数据结构。
- [io_utils.py](/D:/ruxian/traffic_cv_closed_loop/io_utils.py)
  提供文件读写、目录创建、JSON/JSONL 处理等工具。
- [diagnostics.py](/D:/ruxian/traffic_cv_closed_loop/diagnostics.py)
  提供 `doctor`、`probe-qwen`、`probe-detector` 等诊断能力。

子目录：

- `pipeline/`
  包含抽帧、标注、训练三个业务阶段。
- `adapters/`
  用来适配外部系统，例如 Qwen 和 YOLO。

### 5.6 `tests/`

自动化测试目录。

关键文件：

- [tests/test_pipeline.py](/D:/ruxian/tests/test_pipeline.py)
  验证 mock 模式下完整闭环是否能跑通。
- [tests/test_production_integration.py](/D:/ruxian/tests/test_production_integration.py)
  验证 production 配置、探测逻辑和训练命令构造是否正确。
- [tests/test_skills_delivery.py](/D:/ruxian/tests/test_skills_delivery.py)
  验证三个 Skill、说明文档和交付结构是否齐全。

### 5.7 `1.py`

这是为了方便直接使用而提供的主程序入口。

作用：

- 不需要记住很多 CLI 子命令
- 直接通过 `--stage` 调用三个阶段
- 适合课程演示、项目答辩和本地手动操作

### 5.8 `hazard_annotation/`

**工业施工安全隐患自动标注**链路的独立包（入口 `python -m hazard_annotation`）。

关键模块：

- `__main__.py`：CLI 入口，支持 `extract / annotate / aggregate / train / visualize / all /
  status / doctor / probe-qwen / demo` 与 `--mode quick|full`、`--mock`。
- `knowledge.py`：隐患知识库归一化、压缩注入与场景路由。
- `keyframes.py`：六步关键帧提取（粗采样 → 运动 → 去重 → 场景切换 → 目标强制保留 → 数量/间隔）。
- `imageio.py`：纯 Python Netpbm 解析与 aHash/直方图/差分、PNG 编码。
- `pipeline/`：抽帧、双通道标注、聚合、训练闭环、静态 HTML 可视化。
- `diagnostics.py`：`doctor` / `probe-qwen` / `probe-detector`。

详细说明见 [docs/hazard-annotation-chain.md](docs/hazard-annotation-chain.md)。

---

## 6. 运行模式说明

本项目支持两种运行模式。

### 6.1 演示模式 `mock`

适用场景：

- 首次验证项目结构是否正确
- 不具备真实 Qwen 服务账号
- 不想立刻跑真实 YOLO 训练

特点：

- 运行快
- 依赖少
- 可以完整走通“抽帧 -> 标注 -> 训练”的逻辑链

配置文件：

- [examples/mock_project/config/project.json](/D:/ruxian/examples/mock_project/config/project.json)

### 6.2 生产模式 `production`

适用场景：

- 已经有真实 Qwen 服务地址和 API Key
- 已经有真实 YOLO 权重
- 想对真实图片或视频抽帧结果进行训练

特点：

- 走真实模型调用
- 输出真实训练结果和权重
- 适合项目实际演示与模型迭代

配置文件：

- [examples/production_project/config/project.production.json](/D:/ruxian/examples/production_project/config/project.production.json)

---

## 7. 快速开始

### 7.1 环境要求

建议环境：

- Python 3.11 及以上
- Windows PowerShell
- 如果要跑真实训练，需要本地可执行 `yolo`

### 7.2 安装依赖

基础模式：

```powershell
pip install -e .
```

如果要使用真实 Ultralytics 训练：

```powershell
pip install -e .[production]
```

### 7.3 先跑 mock 演示

```powershell
python -m traffic_cv_closed_loop run-loop --config examples/mock_project/config/project.json
```

这条命令的作用是：

- 跑抽帧阶段
- 跑标注阶段
- 跑训练准备和 mock 训练
- 输出完整 JSON 结果

### 7.4 再跑 production 诊断

```powershell
python -m traffic_cv_closed_loop doctor --config examples/production_project/config/project.production.json
```

这一步的作用是检查：

- 配置文件是否能正常读取
- Qwen 地址是否完整
- API Key 环境变量是否存在
- `yolo` 命令是否能找到
- 权重路径是否存在

### 7.5 生产环境探活

探测 Qwen：

```powershell
python -m traffic_cv_closed_loop probe-qwen --config examples/production_project/config/project.production.json
```

探测 YOLO 检测器：

```powershell
python -m traffic_cv_closed_loop probe-detector --config examples/production_project/config/project.production.json
```

### 7.6 运行真实闭环

```powershell
python -m traffic_cv_closed_loop run-loop --config examples/production_project/config/project.production.json --execute-trainer
```

---

## 8. 主程序 `1.py` 的使用方法

如果你不想记很多命令，可以直接使用 [1.py](/D:/ruxian/1.py:1)。

### 8.1 一键全跑

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage all --execute-trainer
```

这条命令会顺序执行：

1. `extract`
2. `annotate`
3. `train`

### 8.2 单独执行某个阶段

只抽帧：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage extract
```

只标注：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage annotate
```

只训练：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage train --execute-trainer
```

### 8.3 查看状态或做诊断

查看状态：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage status
```

做诊断：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage doctor
```

### 8.4 探测服务

探测 Qwen：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage probe-qwen --image tupian\a85164a58d4396339d86fd77419e632e.png
```

探测 YOLO：

```powershell
python 1.py --config examples/production_project/config/project.production.json --stage probe-detector --image tupian\a85164a58d4396339d86fd77419e632e.png
```

---

## 9. 核心命令详细解释

### 9.1 `extract`

命令：

```powershell
python -m traffic_cv_closed_loop extract --config <your-config.json>
```

作用：

- 读取输入图像
- 做候选样本筛选
- 生成候选记录 `candidates.jsonl`

适用时机：

- 你刚放入一批图片或视频帧，想先筛样本

### 9.2 `annotate`

命令：

```powershell
python -m traffic_cv_closed_loop annotate --config <your-config.json>
```

作用：

- 读取候选样本
- 调用 Qwen3VL 复核
- 输出标签、差异信息和人工复核清单

适用时机：

- 你已经完成初检，想生成更可靠的标注结果

### 9.3 `train`

命令：

```powershell
python -m traffic_cv_closed_loop train --config <your-config.json> --execute-trainer
```

作用：

- 组织数据集版本
- 生成 `data.yaml`
- 调用 YOLO 训练
- 输出训练结果、指标和权重

注意：

- 不加 `--execute-trainer` 时，有些模式下只会准备训练而不真正执行

### 9.4 `run-loop`

命令：

```powershell
python -m traffic_cv_closed_loop run-loop --config <your-config.json> --execute-trainer
```

作用：

- 一次性跑完抽帧、标注、训练全流程

适用时机：

- 想完整验证闭环
- 想做正式演示

### 9.5 `status`

命令：

```powershell
python -m traffic_cv_closed_loop status --config <your-config.json>
```

作用：

- 查看最近一次 `extract`、`annotate`、`train` 的状态摘要

### 9.6 `doctor`

命令：

```powershell
python -m traffic_cv_closed_loop doctor --config <your-config.json>
```

作用：

- 用于生产环境前检查配置和依赖是否齐全

### 9.7 `probe-qwen`

命令：

```powershell
python -m traffic_cv_closed_loop probe-qwen --config <your-config.json> --image <image-path>
```

作用：

- 向 Qwen 服务发送一次探测请求
- 验证接口地址、模型名、API Key、返回格式是否正常

### 9.8 `probe-detector`

命令：

```powershell
python -m traffic_cv_closed_loop probe-detector --config <your-config.json> --image <image-path>
```

作用：

- 用一张图测试 YOLO 检测器是否能正常运行

---

## 10. 配置文件详细解释

项目配置采用 JSON 格式。

推荐参考：

- [examples/mock_project/config/project.json](/D:/ruxian/examples/mock_project/config/project.json)
- [examples/production_project/config/project.production.json](/D:/ruxian/examples/production_project/config/project.production.json)

下面对主要字段做详细说明。

### 10.1 基础信息

- `project_name`
  项目名称，用于报告和输出结果中的标识。

- `scene`
  场景描述，用于 Prompt 和任务说明。

- `labels`
  类别列表，决定最终 YOLO 标签的类别空间。

### 10.2 `input`

- `input.frames_dir`
  输入图片目录。可以是用户手动放入的图片，也可以是从视频提取出来的帧目录。

### 10.3 `output`

- `output.workspace_dir`
  工作目录。所有中间产物和结果都会输出到这里。

### 10.4 `pipeline`

- `frame_step`
  抽帧或采样步长。

- `max_frames_per_source`
  每个输入源最多处理多少张图，防止一次跑太多。

- `min_detection_confidence`
  初检时最低置信度阈值。

- `review_confidence_threshold`
  低于这个阈值的结果通常更容易进入复核队列。

- `include_empty_frames`
  是否允许无目标帧参与后续流程。

- `include_review_queue_in_training`
  是否把人工复核队列中的样本也纳入训练。

- `train_split`、`val_split`、`test_split`
  数据集切分比例。

- `seed`
  随机种子，用于保证切分稳定。

### 10.5 `qwen`

- `provider`
  Qwen 服务提供方式。真实生产通常设为 `openai_compatible`，演示模式可设为 `mock`。

- `base_url`
  Qwen 服务基地址。

- `endpoint`
  如果服务方要求直接填完整接口地址，也可以用这个字段。

- `request_path`
  请求路径，常见为 `/chat/completions`。

- `prompt_template`
  Prompt 模板文件路径。

- `model`
  模型名，例如 `qwen-plus`。

- `api_key_env`
  API Key 所在的环境变量名称，例如 `QWEN_API_KEY`。

- `require_api_key`
  是否强制要求 API Key。

- `timeout_seconds`
  请求超时时间。

- `max_retries`
  失败重试次数。

- `retry_backoff_seconds`
  重试退避时间。

- `temperature`
  采样温度。

- `top_p`
  概率采样阈值。

- `max_tokens`
  最大输出 token 数。

- `response_format`
  返回格式约束，建议为 `json_object`。

- `auth_scheme`
  默认是 `Bearer`。

- `headers`
  固定自定义请求头。

- `header_env`
  从环境变量注入额外请求头。

- `extra_body`
  请求体扩展字段。

### 10.6 `yolo`

- `binary`
  YOLO 命令名称，通常是 `yolo`。

- `task`
  任务类型，通常是 `detect`。

- `detector`
  检测器实现方式，例如 `ultralytics` 或 `sidecar`。

- `trainer`
  训练器实现方式，例如 `ultralytics` 或 `mock`。

- `detector_weights`
  初检权重路径。

- `weights`
  训练或继续训练所用权重路径。

- `detector_confidence`
  检测阶段置信度阈值。

- `detector_iou`
  检测阶段 IOU 阈值。

- `detector_image_size`
  检测阶段输入尺寸。

- `image_size`
  训练阶段输入尺寸。

- `epochs`
  训练轮数。

- `device`
  使用的设备，例如 `cpu`、`0`。

- `batch`
  batch size。

- `workers`
  dataloader worker 数量。

- `patience`
  早停耐心值。

- `cache`
  是否缓存数据。

- `resume`
  是否断点续训。

- `close_mosaic`
  训练后期关闭 mosaic 的轮数。

- `save_period`
  权重周期保存频率。

---

## 11. 输出结果详细解释

默认情况下，所有输出都在 `workspace/` 下面。

### 11.1 `workspace/extracted/`

用于保存抽帧和候选样本阶段的结果。

常见内容：

- `candidates.jsonl`
- `summary.json`

### 11.2 `workspace/review/`

用于保存 Qwen3VL 标注复核阶段的结果。

常见内容：

- `annotations.jsonl`
- `review_queue.jsonl`
- `labels/`
- `raw_responses/`
- `prompts/`
- `summary.json`

### 11.3 `workspace/datasets/`

用于保存版本化 YOLO 数据集。

每个版本通常包含：

- `images/train`
- `images/val`
- `images/test`
- `labels/train`
- `labels/val`
- `labels/test`
- `data.yaml`
- `manifest.json`

### 11.4 `workspace/runs/`

用于保存训练结果。

常见内容：

- 训练日志
- `best.pt`
- `last.pt`
- `metrics.json` 或等价指标输出
- `report.json`
- 图表和混淆矩阵

### 11.5 `workspace/analysis/`

用于保存分析和回流建议。

常见内容：

- `hard_examples.jsonl`
- `next_cycle_tasks.md`

### 11.6 `workspace/probes/`

用于保存 Qwen 或 YOLO 探测结果。

### 11.7 `workspace/status/`

用于保存每个阶段的最新状态摘要。

---

## 12. Skills 与项目的对应关系

图片要求里的三个核心 Skill 和本项目的对应关系如下：

- `video_frame_extraction_skill`
  对应 [skills/video-frame-extraction/SKILL.md](/D:/ruxian/skills/video-frame-extraction/SKILL.md)

- `images_qwenvl_annotation_skill`
  对应 [skills/images-qwen3vl-annotation/SKILL.md](/D:/ruxian/skills/images-qwen3vl-annotation/SKILL.md)

- `yolo6_training_skill`
  对应 [skills/yolo-closed-loop-training/SKILL.md](/D:/ruxian/skills/yolo-closed-loop-training/SKILL.md)

说明：

- 图片中第三个名字虽然写成 `yolo6_training_skill`，但本项目已经以更通用的 YOLO 闭环训练 Skill 完成了同类目标。

---

## 13. 测试与验证

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖内容包括：

- mock 闭环能否跑通
- production 配置与探测逻辑是否合理
- Skill 目录和交付文档是否完整
- 施工隐患链路：图像/直方图工具、知识库归一化与路由、thinking JSON 解析、mock 全链路产物、doctor 与 probe-qwen

---

## 14. 常见问题

### 14.1 `doctor` 报 API Key 缺失

说明：

- 你的 `qwen.api_key_env` 所指定的环境变量没有设置

解决方式：

```powershell
$env:QWEN_API_KEY="你的真实 Key"
```

### 14.2 `doctor` 报 `yolo_binary_missing`

说明：

- 系统里找不到 `yolo` 命令

解决方式：

```powershell
pip install -e .[production]
yolo help
```

### 14.3 权重文件不存在

说明：

- `yolo.detector_weights` 或 `yolo.weights` 路径不对

解决方式：

- 把权重放到配置对应路径
- 或把配置改成正确的绝对路径 / 相对路径

### 14.4 Qwen 返回格式不稳定

说明：

- 大模型可能返回非标准 JSON 或返回内容过长

建议：

- 保持 `response_format = json_object`
- 保持 Prompt 中明确要求“只输出 JSON”
- 控制 `max_tokens`

---

## 15. 推荐阅读顺序

如果你第一次接触这个仓库，建议这样看：

1. 先看本 README，理解整体结构。
2. 再看 [docs/requirements-mapping.md](/D:/ruxian/docs/requirements-mapping.md)，确认它如何对应任务要求。
3. 再看 [docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md)，理解 Skills 接入方法。
4. 然后看 [examples/mock_project/config/project.json](/D:/ruxian/examples/mock_project/config/project.json)，先跑 mock。
5. 最后再看 [examples/production_project/config/project.production.json](/D:/ruxian/examples/production_project/config/project.production.json)，接入真实服务。

---

## 16. 结论

本项目已经具备一套较完整的 Qwen3VL -> YOLO 闭环训练工程能力，特点是：

- 有明确的三阶段职责划分
- 有独立的 Skill 目录与说明
- 有 mock 和 production 两种模式
- 有统一的配置和输出目录
- 有主程序、CLI、脚本、文档和测试

如果你把它用于课程作业、毕业设计、项目答辩或团队内部原型验证，这个仓库已经具备比较规范的交付基础。

---

## 17. 工业施工安全隐患自动标注链路（新增）

在原有交通闭环之外，仓库新增了**工业施工安全隐患自动标注**链路，入口为
`python -m hazard_annotation`，与交通链路完全隔离、互不影响。

核心特点：

- **双通道标注**：通道 A 输出 YOLO 目标框（`labels/*.txt`），通道 B 输出隐患事件
  （`hazards/*.json`），共同驱动数据集与回流。
- **主模型**：`qwen3-vl:30b-a3b-thinking-q8_0`，OpenAI 兼容协议（`local_openai`，
  `require_api_key=false`，`base_url` 完全可配置）。
- **零第三方依赖**：仅标准库；视频解码仅依赖可选的外部 `ffmpeg`。
- **quick / full 双档**：通过 `--mode` 或配置内 `run_mode` 切换。
- **一键 mock**：`python -m hazard_annotation demo --mock`，无需视频 / 模型 / ffmpeg。

### 17.1 快速开始

```powershell
# 一键 mock 演示（自动生成 mock 资产并跑通全链路）
python -m hazard_annotation demo --mock

# 等价地：all --mock 会在缺少输入帧时自动生成 mock 资产
python -m hazard_annotation all --mock --config examples/hazard_mock_project/config/project.json

# 诊断与探活
python -m hazard_annotation doctor --config examples/hazard_project/config/project.json
python -m hazard_annotation probe-qwen --config examples/hazard_project/config/project.json
```

### 17.2 产物

`workspace/` 下产出 `review/labels/*.txt`、`hazards/*.json`、`hazards/report.json`、
`datasets/<ver>/data.yaml`、`analysis/hard_examples.jsonl`、`analysis/next_cycle_tasks.md`，
以及**自包含静态 HTML 报告** `visualization/report.html`（base64 缩略图 + SVG 图表 + 场景 Tab +
mock/真实模式标识）。

### 17.3 4090D（24GB）显存注意事项

链路不在进程内加载模型，只通过 HTTP 调用本地服务：

- 模型服务侧负责量化与显存/内存卸载；`30b-a3b` 的 q8 权重约 30GB 量级，24GB 卡上需部分卸载，
  显存吃紧时优先改用 q4 量化。
- 标注链路侧用 `quick` 档 + `combined_call=true` + `concurrency=1` + `max_keyframes` 限制并发与关键帧数量，
  避免多请求叠加显存峰值。

完整说明见 [docs/hazard-annotation-chain.md](docs/hazard-annotation-chain.md)。

---

## 18. 许可协议

本仓库当前采用 [MIT License](/D:/ruxian/LICENSE)。

这意味着：

- 你可以学习、复制、修改和分发本项目
- 你可以把它作为课程项目、原型项目或个人作品集的一部分
- 使用时需要保留原始许可声明
