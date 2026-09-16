# Cursor / Agent Skills 接入说明

本文档说明如何把当前仓库作为一个可发现、可调用、可验证的 Agent Skills 项目来使用，并让三个核心技能形成完整闭环。

## 1. 目标

接入完成后，Agent 应该能够：

- 发现 `skills/` 目录下的三个核心 Skill
- 根据任务意图自动选择合适的 Skill
- 按 `视频抽帧 -> Qwen3VL 二次标注 -> YOLO 训练` 的顺序串联执行
- 使用统一的项目配置文件和输出目录
- 产出可追踪的数据集版本、训练结果和回流任务

## 2. 仓库内的 Skills 目录

当前项目已经提供以下 Skill：

- `skills/video-frame-extraction`
- `skills/images-qwen3vl-annotation`
- `skills/yolo-closed-loop-training`

每个 Skill 都包含：

- `SKILL.md`：技能定义、适用场景、输入输出、执行步骤
- `scripts/run.ps1`：本地 PowerShell 启动入口

其中：

- `images-qwen3vl-annotation` 额外包含 `prompts/qwen3vl_traffic_zh.md`
- `yolo-closed-loop-training` 额外包含 `templates/next_cycle_tasks_template.md`

## 3. 在 Cursor / Agent 环境中的推荐放置方式

推荐直接把整个仓库作为工作区打开，并保持以下结构不变：

```text
skills/
  video-frame-extraction/
    SKILL.md
    scripts/run.ps1
  images-qwen3vl-annotation/
    SKILL.md
    prompts/qwen3vl_traffic_zh.md
    scripts/run.ps1
  yolo-closed-loop-training/
    SKILL.md
    templates/next_cycle_tasks_template.md
    scripts/run.ps1
```

推荐做法：

1. 在 Cursor 中直接打开仓库根目录 `D:\ruxian`
2. 保持 `skills/` 位于工作区根目录，不要再套一层子目录
3. 让 Agent 在当前工作区内运行，这样它能同时读取 `skills/`、`docs/`、`examples/` 和源码目录
4. 让 Agent 默认优先参考各个 `SKILL.md`，再调用对应脚本或 Python CLI

如果你的 Agent Skills 环境支持自动发现 `SKILL.md`，那么当前目录结构已经满足发现条件。

如果你的环境需要手动登记 Skill，建议登记以下三个目录：

- `skills/video-frame-extraction`
- `skills/images-qwen3vl-annotation`
- `skills/yolo-closed-loop-training`

## 4. 三个 Skill 的职责边界

### `video-frame-extraction`

职责：

- 接收视频或帧目录
- 调用 YOLO 初检
- 抽取高价值关键帧
- 生成候选样本清单，供二次标注使用

对应命令：

```powershell
python -m traffic_cv_closed_loop extract --config <config.json>
```

### `images-qwen3vl-annotation`

职责：

- 接收关键帧和 YOLO 初检结果
- 调用 Qwen3VL 做二次复核和补标
- 输出 YOLO 标签、差异报告和人工复核队列

对应命令：

```powershell
python -m traffic_cv_closed_loop annotate --config <config.json>
```

### `yolo-closed-loop-training`

职责：

- 接收已修正的数据集
- 生成版本化训练数据
- 调用 YOLO 训练与评估
- 输出新权重、指标、困难样本和下一轮任务建议

对应命令：

```powershell
python -m traffic_cv_closed_loop train --config <config.json> --execute-trainer
```

## 5. 推荐的 Agent 调用策略

当用户目标是“根据视频生成训练数据并继续训练模型”时，推荐 Agent 按下面的顺序调用：

1. 先读取项目配置文件
2. 调用 `video-frame-extraction`
3. 调用 `images-qwen3vl-annotation`
4. 调用 `yolo-closed-loop-training`
5. 最后汇总 `workspace/` 下的产物路径和指标

当用户只说“帮我抽帧”时，只调用 `video-frame-extraction`。

当用户只说“帮我标注图片”时，只调用 `images-qwen3vl-annotation`。

当用户已经有数据集，只想“继续训练”时，只调用 `yolo-closed-loop-training`。

## 6. 推荐配置文件

开发验证推荐：

- `examples/mock_project/config/project.json`

生产闭环推荐：

- `examples/production_project/config/project.production.json`
- `examples/production_project/config/project.video_batch_20260713_30572.json`

如果要跑真实 Qwen + 真实 YOLO，请同时准备：

- Qwen 服务地址
- Qwen API Key 环境变量
- YOLO 权重文件
- `yolo` 命令可执行环境

## 7. 本地运行方式

### 方式一：直接调用 Python CLI

```powershell
python -m traffic_cv_closed_loop run-loop --config examples/mock_project/config/project.json
```

### 方式二：按 Skill 分步运行

```powershell
powershell -ExecutionPolicy Bypass -File skills/video-frame-extraction/scripts/run.ps1 -Config examples/mock_project/config/project.json
powershell -ExecutionPolicy Bypass -File skills/images-qwen3vl-annotation/scripts/run.ps1 -Config examples/mock_project/config/project.json
powershell -ExecutionPolicy Bypass -File skills/yolo-closed-loop-training/scripts/run.ps1 -Config examples/mock_project/config/project.json -ExecuteTrainer
```

## 8. 输出目录约定

当前项目统一将结果写入配置文件指定的 `workspace/` 目录，主要包括：

- `workspace/extracted/`
- `workspace/review/`
- `workspace/datasets/`
- `workspace/runs/`
- `workspace/analysis/`
- `workspace/probes/`
- `workspace/status/`

这保证 Agent 在多轮对话里可以持续复用已有中间结果，而不是每次都从零开始。

## 9. 验收建议

完成接入后，建议按下面顺序验收：

1. 运行 `python -m unittest discover -s tests -v`
2. 用 mock 配置执行一次完整闭环
3. 用生产配置执行 `doctor`
4. 用真实视频执行一次 `extract -> annotate -> train`
5. 检查是否成功生成数据集版本、训练权重和困难样本回流清单

## 10. 注意事项

- `skills/` 目录名称和 `SKILL.md` 文件名称不要改动
- 三个 Skill 的职责不要混写到同一个 `SKILL.md`
- 生产运行前先跑 `doctor`，避免 Qwen 地址、API Key 或权重路径错误
- 回流训练时优先保留数据集版本和指标对比，避免模型变好还是变差无法追踪
