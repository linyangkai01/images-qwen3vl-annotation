<metadata>
name: images_qwen3vl_annotation
title: Qwen3VL交通图像二次标注技能
description: 读取关键帧图片及YOLO初始检测结果，调用Qwen3VL进行复核、补标和纠偏，输出标准YOLO标签、差异报告和人工复核队列。
inputs:
  image_folder: str # 关键帧图像目录
  detect_cache_json: str # YOLO初始检测缓存或候选检测结果
  prompt_template: str # Qwen3VL提示词模板路径
  label_list: list[str] # 允许输出的类别集合
  review_threshold: float # 进入人工复核队列的阈值
outputs:
  labels_folder: str
  annotations_jsonl: str
  review_queue_jsonl: str
  annotation_summary_json: str
</metadata>

[适用场景]

- 关键帧已经准备好，需要自动做二次标注
- 希望修正YOLO的漏检、误检和类别偏差
- 需要输出可直接训练的YOLO标签

[执行说明]

调用 `traffic_cv_closed_loop.pipeline.annotation` 阶段逻辑，或直接执行命令：

```powershell
python -m traffic_cv_closed_loop annotate --config examples/mock_project/config/project.json
```

功能说明：

1. 读取关键帧和对应的YOLO初始检测框。
2. 根据Prompt模板调用Qwen3VL做图像级复核。
3. 对比YOLO与Qwen3VL结果，识别新增、删除和修正目标。
4. 生成标准YOLO标签文件。
5. 对低置信度或冲突样本写入人工复核队列。

[输入解释]

- `image_folder`
  抽帧阶段输出的关键帧目录。
- `detect_cache_json`
  YOLO初始检测结果缓存。
- `prompt_template`
  Qwen3VL专用提示词模板。
- `label_list`
  最终允许输出的标签类别。
- `review_threshold`
  低于该阈值或冲突较大的样本进入复核。

[输出解释]

- `labels_folder`
  生成后的YOLO标签目录。
- `annotations_jsonl`
  每张图的标注记录与差异结果。
- `review_queue_jsonl`
  人工复核队列。
- `annotation_summary_json`
  标注阶段摘要。

[下一步衔接]

将标注结果交给 `yolo_closed_loop_training` 技能，执行数据集构建、训练和评估。
