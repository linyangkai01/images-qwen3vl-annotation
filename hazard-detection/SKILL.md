# 施工现场隐患识别 Skill

## 概述

本 Skill 为施工现场规范化隐患识别提供结构化知识库，配合本地视觉大模型（如 Qwen-VL、InternVL、LLaVA 等）使用，用于从图片/视频中自动识别吊装作业和高处作业的安全隐患。包含吊装"十不吊"红线和行车六严禁规则检查。

## 文件结构

```
hazard-detection/
├── SKILL.md                    # 本文件 - 使用说明
├── knowledge_base.json         # 结构化隐患知识库（模型上下文注入用）
├── prompt_templates.md         # 场景化 Prompt 模板
└── detection_examples.md       # Few-shot 示例
```

## 使用方式

### 1. 作为 System Prompt 注入知识库

将 `knowledge_base.json` 的内容加载到模型的 system prompt 或上下文中，使模型在分析图片前"预装"所有隐患知识。

```
角色：你是一名施工现场安全检查员。
以下是隐患识别知识库：
{knowledge_base.json 内容}

请分析以下图片，识别其中存在的安全隐患...
```

### 2. 使用 Prompt 模板

根据实际场景选择 `prompt_templates.md` 中的模板：

- **场景A**：已知场景类型（吊装/高处），定向识别
- **场景B**：未知场景，先分类再识别
- **场景C**：视频帧批量分析
- **场景D**：对比检查（有无隐患）

### 3. 本地模型部署建议

| 模型 | 推荐场景 | 最低显存 |
|------|---------|---------|
| Qwen2.5-VL-7B | 通用图片识别 | 16GB |
| InternVL2-8B | 高精度场景理解 | 24GB |
| LLaVA-1.6-34B | 最高精度 | 48GB |

建议将 `knowledge_base.json` 作为 few-shot 或 retrieval 的知识源，而非直接全量注入 system prompt（token 限制）。

## 输出格式

模型识别结果统一输出为 JSON 格式，便于下游系统对接：

```json
{
  "scene_type": "吊装作业",
  "hazards": [
    {
      "id": "D-1-1",
      "category": "吊索具缺陷与使用不当",
      "description": "绳索交叉缠绕",
      "confidence": 0.92,
      "evidence": "图片左侧可见吊链明显交叉",
      "severity": "high",
      "norm_reference": "GB 6067.1-2010"
    }
  ],
  "ten_no_lift_violations": ["TNR-7: 吊下站人、物件带浮件不吊"],
  "crane_six_forbidden_violations": [],
  "overall_risk": "high",
  "recommendation": "立即停止作业，整理吊链后重新吊装"
}
```

## 扩展

- 添加新隐患：编辑 `knowledge_base.json`，按现有格式追加
- 添加新场景：在 `prompt_templates.md` 中新增模板
- 添加示例：在 `detection_examples.md` 中补充 few-shot 示例
