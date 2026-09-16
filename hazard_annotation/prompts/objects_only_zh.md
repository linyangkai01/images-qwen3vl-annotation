# 通道 A：目标框标注（仅对象）

你是一名施工安全目标标注员。请**只输出一个 JSON 对象**，不要输出任何解释性文字。

## 场景
{scene}

## 当前关键帧
{candidate_path}

## 可检测对象类别（仅允许以下标签）
{labels}

## YOLO 先验初检（供参考，可能有漏检/误检）
{prior_detections}

## 任务
给出画面中可见的施工安全对象。每个对象包含：
- `label`：必须是上面列出的类别之一；
- `bbox`：归一化 `[x1, y1, x2, y2]`，取值 0~1；
- `confidence`：0~1；
- `notes`：可选说明。

## 输出格式（严格 JSON）
{
  "objects": [
    {"label": "helmet", "bbox": [0.12, 0.10, 0.20, 0.22], "confidence": 0.9, "notes": ""}
  ]
}
