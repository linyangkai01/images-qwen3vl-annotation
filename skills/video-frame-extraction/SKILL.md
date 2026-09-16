<metadata>
name: video_frame_extraction
title: YOLO视频智能抽帧筛选技能
description: 读取交通路口监控视频或视频抽帧目录，使用YOLO进行初始目标检测，按采样步长和置信度筛选包含目标的关键帧，输出关键帧图片及对应的初始检测缓存结果。
inputs:
  video_path: str # 输入视频文件路径，或已抽帧图片目录
  weight_path: str # YOLO权重文件路径，例如 weights/yolo11l.pt
  frame_step: int # 采样间隔，默认由配置文件控制
  conf_threshold: float # 初检置信度阈值，默认0.25
  output_img_dir: str # 关键帧输出目录，对应 workspace/extracted
  device: str # 运行设备，例如 cpu 或 0
outputs:
  image_folder: str
  detect_cache_json: str
  candidates_jsonl: str
  extraction_summary_json: str
</metadata>

[适用场景]

- 需要从长视频中自动筛选训练样本
- 已经有YOLO权重，希望先做自动初检
- 需要为后续Qwen3VL复标提供候选关键帧

[执行说明]

调用 `traffic_cv_closed_loop.pipeline.extraction` 阶段逻辑，或直接执行命令：

```powershell
python -m traffic_cv_closed_loop extract --config examples/mock_project/config/project.json
```

功能说明：

1. 读取输入视频或图片目录。
2. 调用YOLO进行初始检测。
3. 按 `frame_step`、检测结果变化和置信度筛选关键帧。
4. 将关键帧复制到工作目录并生成候选样本清单。
5. 输出初始检测缓存，供后续标注技能对比漏检和误检。

[输入解释]

- `video_path`
  原始视频路径，或预先抽好的帧目录。
- `weight_path`
  YOLO检测模型权重路径。
- `frame_step`
  抽样频率，值越大，抽帧越稀疏。
- `conf_threshold`
  低于该值的检测结果通常不会进入候选集。
- `output_img_dir`
  关键帧输出位置。
- `device`
  使用CPU还是GPU执行检测。

[输出解释]

- `image_folder`
  关键帧图像目录。
- `detect_cache_json`
  YOLO原始检测缓存。
- `candidates_jsonl`
  候选关键帧清单。
- `extraction_summary_json`
  本阶段运行摘要。

[下一步衔接]

将输出交给 `images_qwen3vl_annotation` 技能，做二次标注和标签修正。
