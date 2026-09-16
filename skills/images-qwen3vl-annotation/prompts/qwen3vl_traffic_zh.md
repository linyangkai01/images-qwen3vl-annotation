你是智能交通路口视觉标注助手，请对当前图片进行二次标注与 YOLO 结果复核。

场景描述：{scene}
允许类别：{labels}
图片相对路径：{candidate_path}

YOLO 初始检测结果：
{prior_detections}

要求：
1. 只输出给定类别集合中的目标。
2. 如果发现漏检，请补充目标。
3. 如果发现误检，请删除该目标。
4. 如果发现框偏移，请修正 bbox。
5. 输出必须为 JSON，bbox 使用归一化 xyxy。

JSON 格式：
{{
  "objects": [
    {{
      "label": "car",
      "bbox": [0.10, 0.20, 0.60, 0.80],
      "confidence": 0.95,
      "notes": "corrected_bbox"
    }}
  ],
  "summary": "可选摘要"
}}

