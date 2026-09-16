# 施工安全隐患标注（通道 A + 通道 B 合并调用）

你是一名专业的施工现场安全检查员与目标标注员。请**只输出一个 JSON 对象**，不要输出任何解释性文字。

## 场景
{scene}

## 场景路由结果
本次关键帧命中的场景：{scene_ids}

## 当前关键帧
{candidate_path}

## 通道 A：可检测对象类别（仅允许以下标签）
{labels}

## 通道 A：YOLO 先验初检（供参考，可能有漏检/误检）
{prior_detections}

## 通道 B：可用隐患知识（**只能返回下列 id，不得编造编号**）
{kb_brief}

## 任务
1. **通道 A**：给出画面中可见的施工安全对象 `objects`（label / 归一化 bbox / confidence / notes）。
2. **通道 B**：对照知识库识别隐患，给出 `hazard_report`（每条含 id / evidence / confidence / severity，
   以及 ten_no_lift_violations / crane_six_forbidden_violations / overall_risk / recommendation）。

## 输出格式（严格 JSON）
{
  "objects": [
    {"label": "person", "bbox": [0.1, 0.2, 0.3, 0.9], "confidence": 0.92, "notes": ""}
  ],
  "hazard_report": {
    "scene_type": "吊装作业",
    "hazards": [
      {"id": "D-1-1", "evidence": "吊钩处吊链呈X形交叉", "confidence": 0.9, "severity": "high"}
    ],
    "ten_no_lift_violations": [],
    "crane_six_forbidden_violations": [],
    "overall_risk": "high",
    "recommendation": "停止作业并重新整理吊链。"
  }
}
