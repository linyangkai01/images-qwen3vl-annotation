# Mock 示例项目

这个目录用于离线演示 Qwen3VL -> YOLO 闭环训练项目的完整流程。

## 运行方式

```powershell
python -m traffic_cv_closed_loop run-loop --config examples/mock_project/config/project.json
```

## 目录说明

```text
config/
  project.json
assets/
  prompts/
input/
  frames/
```

## 数据说明

- `*.ppm`：示例图片，使用文本格式图像，便于直接纳入仓库。
- `*.detections.json`：模拟 YOLO 初检结果。
- `*.qwen.json`：模拟 Qwen3VL 复标结果。

