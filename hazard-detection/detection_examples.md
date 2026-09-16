# Few-shot 示例

## 示例1：吊装作业 - 多隐患场景

### 输入
图片描述：一张施工现场照片，画面中有一台龙门吊正在吊运散件钢材，吊链呈交叉状态，吊物下方有2名工人未佩戴安全帽，区域照明偏暗。

### 模型输出
```json
{
  "scene_type": "吊装作业",
  "hazards": [
    {
      "id": "D-1-1",
      "category": "吊索具缺陷与使用不当",
      "description": "绳索交叉缠绕",
      "confidence": 0.95,
      "evidence": "龙门吊吊钩处可见吊链明显交叉呈X形",
      "severity": "high",
      "norm_reference": "GB 6067.1-2010",
      "action": "停止作业，重新整理绳索"
    },
    {
      "id": "D-3-1",
      "category": "被吊物品不安全状态",
      "description": "散件未用料斗吊运/未可靠捆绑固定",
      "confidence": 0.88,
      "evidence": "吊运的钢材为散件状态，未见料斗或可靠捆绑",
      "severity": "high",
      "norm_reference": "安全操作规程",
      "action": "必须用料斗或其他防脱落措施"
    },
    {
      "id": "D-5-1",
      "category": "人员不安全行为",
      "description": "人员未佩戴安全帽",
      "confidence": 0.92,
      "evidence": "吊物下方2名工人头部无安全帽",
      "severity": "high",
      "action": "要求佩戴安全帽"
    },
    {
      "id": "D-5-3",
      "category": "人员不安全行为",
      "description": "人员站在吊物下方",
      "confidence": 0.90,
      "evidence": "2名工人位于吊物正下方区域，未保持安全距离",
      "severity": "critical",
      "action": "立即撤离吊物下方区域"
    },
    {
      "id": "D-4-1",
      "category": "作业环境与照明",
      "description": "吊装作业照明不足",
      "confidence": 0.75,
      "evidence": "画面整体偏暗，阴影区域较大",
      "severity": "medium",
      "action": "增设照明设备"
    }
  ],
  "overall_risk": "critical",
  "ten_no_lift_violations": ["TNR-7: 吊下站人、物件带浮件不吊"],
  "crane_six_forbidden_violations": [],
  "recommendation": "立即停止吊装作业，撤离吊物下方人员，要求全员佩戴安全帽，整理吊链并使用料斗吊运散件，增设照明后方可恢复作业。"
}
```

---

## 示例2：高处作业 - 梯具使用不规范

### 输入
图片描述：一名工人独自站在A字梯最高处进行天花板作业，梯子放置在瓷砖地面上，梯脚无橡胶垫，工人未系安全带。

### 模型输出
```json
{
  "scene_type": "高处作业",
  "hazards": [
    {
      "id": "G-4-6",
      "category": "梯具使用不规范",
      "description": "踩于最高梯面作业",
      "confidence": 0.93,
      "evidence": "工人双脚位于A字梯最高一级踏面",
      "severity": "high",
      "action": "禁止踩最高梯面"
    },
    {
      "id": "G-4-5",
      "category": "梯具使用不规范",
      "description": "禁止单人作业（下方无人稳固梯具）",
      "confidence": 0.91,
      "evidence": "梯子周围未见其他人员扶持",
      "severity": "high",
      "action": "安排专人在下方扶持梯具"
    },
    {
      "id": "G-4-4",
      "category": "梯具使用不规范",
      "description": "梯脚无防滑措施",
      "confidence": 0.85,
      "evidence": "梯脚直接接触瓷砖地面，未见防滑橡胶垫",
      "severity": "medium",
      "action": "安装防滑脚垫"
    },
    {
      "id": "G-2-2",
      "category": "个人防护用品（PPE）",
      "description": "人员未系安全带",
      "confidence": 0.88,
      "evidence": "工人腰间未见安全带",
      "severity": "critical",
      "action": "必须系安全带后方可作业"
    }
  ],
  "overall_risk": "critical",
  "ten_no_lift_violations": [],
  "crane_six_forbidden_violations": [],
  "recommendation": "立即停止作业。安排专人在下方扶持梯具，工人不得站在最高梯面，应系安全带，梯脚需安装防滑垫。"
}
```

---

## 示例3：无隐患场景

### 输入
图片描述：整洁的吊装作业现场，行车吊运中，吊链垂直受力无缠绕，吊物使用专用吊具，下方无人员，操作员佩戴安全帽和反光背心，照明充足。

### 模型输出
```json
{
  "scene_type": "吊装作业",
  "hazards": [],
  "overall_risk": "none",
  "ten_no_lift_violations": [],
  "crane_six_forbidden_violations": [],
  "observations": [
    "吊链垂直受力，无缠绕",
    "使用专用吊具",
    "吊物下方无人员",
    "操作员佩戴安全帽和反光背心",
    "照明充足"
  ],
  "recommendation": "当前作业状态符合安全规范，继续保持。"
}
```

---

## 示例4：整改前后对比

### 输入
图片A（整改前）：脚手架外侧护栏缺失，工人在无防护的2m高平台上作业。
图片B（整改后）：脚手架外侧已安装护栏，工人佩戴安全帽。

### 模型输出
```json
{
  "before": {
    "hazards": [
      {"id": "G-1-1", "description": "平台无防护栏/脚手架护栏缺失"},
      {"id": "G-2-1", "description": "人员未佩戴安全帽"}
    ]
  },
  "after": {
    "resolved": ["G-1-1", "G-2-1"],
    "unresolved": [],
    "new_hazards": []
  },
  "rectification_rate": "100%",
  "recommendation": "隐患已全部整改，脚手架护栏和安全帽均已到位，符合安全要求。"
}
```
