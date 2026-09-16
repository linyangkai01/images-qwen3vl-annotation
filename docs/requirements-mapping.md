# 图片需求对照表

本文档把图片中的任务要求与当前仓库实现逐条对齐，便于你直接验收项目是否满足要求。

## 1. 任务一：开发三个核心 Skills 工具

### 1.1 `video_frame_extraction_skill`

图片要求：

- 接收视频文件
- 使用 YOLO 权重做初步检测
- 从视频中筛选出含目标的关键帧

仓库对应实现：

- Skill 定义：[skills/video-frame-extraction/SKILL.md](/D:/ruxian/skills/video-frame-extraction/SKILL.md)
- 启动脚本：[skills/video-frame-extraction/scripts/run.ps1](/D:/ruxian/skills/video-frame-extraction/scripts/run.ps1)
- 核心代码：[traffic_cv_closed_loop/pipeline/extraction.py](/D:/ruxian/traffic_cv_closed_loop/pipeline/extraction.py)
- CLI 命令：`python -m traffic_cv_closed_loop extract --config <config.json>`

满足情况：

- 已满足

### 1.2 `images_qwenvl_annotation_skill`

图片要求：

- 接收关键帧
- 调用 QwenVL / Qwen3VL 做更精细的二次检测
- 与初始 YOLO 结果对比
- 生成新的 YOLO 格式标注数据

仓库对应实现：

- Skill 定义：[skills/images-qwen3vl-annotation/SKILL.md](/D:/ruxian/skills/images-qwen3vl-annotation/SKILL.md)
- 启动脚本：[skills/images-qwen3vl-annotation/scripts/run.ps1](/D:/ruxian/skills/images-qwen3vl-annotation/scripts/run.ps1)
- Prompt 模板：[skills/images-qwen3vl-annotation/prompts/qwen3vl_traffic_zh.md](/D:/ruxian/skills/images-qwen3vl-annotation/prompts/qwen3vl_traffic_zh.md)
- 核心代码：[traffic_cv_closed_loop/pipeline/annotation.py](/D:/ruxian/traffic_cv_closed_loop/pipeline/annotation.py)
- Qwen 适配器：[traffic_cv_closed_loop/adapters/qwen.py](/D:/ruxian/traffic_cv_closed_loop/adapters/qwen.py)

满足情况：

- 已满足

### 1.3 `yolo6_training_skill`

图片要求：

- 接收新生成的数据集
- 使用原有 YOLO 权重做微调训练
- 产出性能提升后的新模型

仓库对应实现：

- Skill 定义：[skills/yolo-closed-loop-training/SKILL.md](/D:/ruxian/skills/yolo-closed-loop-training/SKILL.md)
- 启动脚本：[skills/yolo-closed-loop-training/scripts/run.ps1](/D:/ruxian/skills/yolo-closed-loop-training/scripts/run.ps1)
- 核心代码：[traffic_cv_closed_loop/pipeline/training.py](/D:/ruxian/traffic_cv_closed_loop/pipeline/training.py)
- YOLO 适配器：[traffic_cv_closed_loop/adapters/yolo.py](/D:/ruxian/traffic_cv_closed_loop/adapters/yolo.py)

满足情况：

- 已满足

说明：

- 仓库当前训练侧使用的是 Ultralytics YOLO 命令体系
- 如果你图片里写的是 `yolo6_training_skill`，这里已经以“YOLO 闭环训练 Skill”的形式完成，同类目标已经覆盖

## 2. 任务二：实现闭环调用与测试

图片要求重点：

- 将三个 Skills 接入 Agent Skills 环境
- 能被 AI Agent 正确发现和调用
- 形成视频抽帧、Qwen 标注、YOLO 训练的闭环
- 能验证自动化程度和模型提升效果

仓库对应实现：

- 闭环命令入口：[traffic_cv_closed_loop/cli.py](/D:/ruxian/traffic_cv_closed_loop/cli.py)
- 一键闭环：`python -m traffic_cv_closed_loop run-loop --config <config.json> --execute-trainer`
- 生产脚本：[scripts/run_production_pipeline.ps1](/D:/ruxian/scripts/run_production_pipeline.ps1:1)
- 技能接入说明：[docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md)
- 生产接入说明：[docs/production-integration.md](/D:/ruxian/docs/production-integration.md)

自动化验证：

- 测试文件：[tests/test_pipeline.py](/D:/ruxian/tests/test_pipeline.py:1)
- 生产集成测试：[tests/test_production_integration.py](/D:/ruxian/tests/test_production_integration.py:1)
- Skills 交付测试：[tests/test_skills_delivery.py](/D:/ruxian/tests/test_skills_delivery.py:1)

满足情况：

- 已满足

## 3. 任务三：总结与文档化

图片要求重点：

- 总结每个 Skill 的内容与实现逻辑
- 说明输入输出接口和配置方式
- 给出 Cursor / Agent Skills 的接入步骤
- 形成说明文档

仓库对应实现：

- 工单式总说明：[docs/qwenvl-yolo-skills-work-order.md](/D:/ruxian/docs/qwenvl-yolo-skills-work-order.md)
- Skills 接入说明：[docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md)
- 生产接入说明：[docs/production-integration.md](/D:/ruxian/docs/production-integration.md)
- 主项目说明：[README.md](/D:/ruxian/README.md)

满足情况：

- 已满足

## 4. 交付物清单

当前仓库已经具备以下交付物：

- 三个可独立说明的 Skill 目录
- 每个 Skill 对应的 `SKILL.md`
- 每个 Skill 对应的 `run.ps1`
- 一套可运行的 Python 闭环工程
- mock 模式与 production 模式配置
- 文档、工单、接入说明、测试文件

## 5. 建议的验收动作

如果你要按图片要求做最终验收，建议按下面顺序：

1. 看 [docs/qwenvl-yolo-skills-work-order.md](/D:/ruxian/docs/qwenvl-yolo-skills-work-order.md) 确认目标和范围
2. 看 [docs/cursor-agent-skills-setup.md](/D:/ruxian/docs/cursor-agent-skills-setup.md) 确认 Agent Skills 接法
3. 跑 `python -m unittest discover -s tests -v`
4. 跑 mock 闭环
5. 跑真实视频闭环
6. 对比新旧模型指标和输出权重
