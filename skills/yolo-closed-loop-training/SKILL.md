<metadata>
name: yolo_closed_loop_training
title: YOLO闭环训练与回流技能
description: 读取修正后的YOLO标签和图像数据，自动构建版本化数据集，调用YOLO进行训练或继续训练，输出新权重、评估指标、困难样本及下一轮回流任务建议。
inputs:
  image_folder: str # 图像目录
  labels_folder: str # YOLO标签目录
  train_config: str # 项目配置文件路径
  base_weight_path: str # 初始或上轮最佳权重路径
  execute_trainer: bool # 是否真正执行训练
outputs:
  dataset_yaml: str
  run_report_json: str
  best_weight_path: str
  hard_examples_jsonl: str
  next_cycle_tasks_md: str
</metadata>

[适用场景]

- 已具备YOLO训练数据集，准备继续训练
- 需要比较训练前后模型表现
- 需要提取困难样本继续回流

[执行说明]

调用 `traffic_cv_closed_loop.pipeline.training` 阶段逻辑，或直接执行命令：

```powershell
python -m traffic_cv_closed_loop train --config examples/mock_project/config/project.json --execute-trainer
```

功能说明：

1. 读取已确认的标注记录。
2. 生成带版本号的YOLO数据集目录和 `data.yaml`。
3. 按训练集、验证集、测试集切分数据。
4. 调用YOLO训练器执行训练或准备训练命令。
5. 输出指标、最佳权重、困难样本和下一轮任务建议。

[输入解释]

- `image_folder`
  训练图像目录。
- `labels_folder`
  YOLO标签目录。
- `train_config`
  项目级配置文件。
- `base_weight_path`
  用于微调的初始权重。
- `execute_trainer`
  为 `true` 时真正运行YOLO训练。

[输出解释]

- `dataset_yaml`
  训练数据集配置文件。
- `run_report_json`
  本轮训练报告。
- `best_weight_path`
  最佳权重文件路径。
- `hard_examples_jsonl`
  困难样本列表。
- `next_cycle_tasks_md`
  下一轮数据回流建议。

[下一步衔接]

将困难样本重新回流给 `video_frame_extraction` 和 `images_qwen3vl_annotation`，进入下一轮闭环优化。
