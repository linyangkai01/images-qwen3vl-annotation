# hazard-detection/ 目录说明

> 本目录是"工业施工安全隐患"项目的**领域知识资产层**，不是代码。
> 它定义隐患编号体系、规范依据、提示词设计与示例，供 `hazard_annotation/` 代码消费。
> 最后核实：2026-09-19（全仓 Grep 引用链 + git 跟踪状态 + 运行时调用路径）

---

## 一、目录定位

本目录解决的问题是：**让视觉大模型说"人话"而不是"编规范号"**。

做法是把领域知识从代码里剥离出来，固化成本目录下的资产：

- 隐患有哪些、怎么编号、严重度几级、依据哪条国标 → 由 `knowledge_base.json` 定义
- 模型怎么被提问、看到什么示例 → 由 `prompt_templates.md` / `detection_examples.md` 定义

代码只负责"读知识、问模型、回填字段"，**规范依据永远只能来自本目录**。

---

## 二、六个文件逐一说明

### 1. `knowledge_base.json` — 运行时核心资产（唯一被代码读取的文件）

**主要目的**
隐患编号、名称、视觉线索、严重度、风险类型、规范依据、整改动作的**唯一权威来源**。

**内容概述**
- 顶层键：`meta` / `scenes` / `ten_no_lift` / `crane_six_forbidden` / `severity_levels`
- `meta.version = 1.0.0`，`meta.total_hazards = 58`
- 三层结构 `scenes → hazard_categories → hazards`
- 实际隐患 **58 条**：
  - `S1` 吊装作业 27 条（吊索具 7 / 吊架托架 5 / 被吊物 7 / 作业环境 3 / 人员行为 5）
  - `S2` 高处作业 30 条（防护设施 5 / PPE 4 / 安全带绳 3 / 梯具 11 / 移动脚手架 4 / 作业行为 3）
  - `S3` 通用 1 条（场地积水）
- 每条隐患字段：`id` `name` `description` `visual_cues` `severity` `risk_type` `norm_ref` `action`
- 另有红线规则：`ten_no_lift` 10 条、`crane_six_forbidden` 6 条
- `norm_ref` 覆盖率 **58/58（100%）**

**依赖关系（被谁读）**

运行时读取链：
- `traffic_cv_closed_loop/config.py:155-172` — `_resolve_kb_path()`。当配置 `knowledge.kb_path` 为空时，先查 `project_dir/hazard-detection/knowledge_base.json`，再**逐级向上遍历祖先目录**。三个 hazard 示例项目的 `kb_path` 均为空字符串，完全依赖此自动发现机制——**"知识库找不到"类故障的唯一根因就在这里**
- `hazard_annotation/knowledge.py:53` — `KnowledgeBase.load(path)`，唯一解析入口
- `hazard_annotation/pipeline/annotation.py:216` — `kb = KnowledgeBase.load(config.kb_path)`，标注主链路
- `hazard_annotation/diagnostics.py:117-135` 与 `:270` — `doctor` 自检与告警
- `hazard_annotation/pipeline/visualization.py:497-498` — 读 `severity_levels` 渲染配色

工具与测试：
- `scripts/fill_norm_ref.py:40` — `DEFAULT_KB` 常量
- 5 个测试文件引用：`test_hazard_knowledge.py`、`test_hazard_pipeline.py`、`test_hazard_adversarial_qa.py`、`test_hazard_diagnostics.py`、`test_dataset_image_formats.py`

**在流程中的角色 — 契约层**
模型只被允许输出 `id` + `evidence`（+ confidence/severity），
`category` / `description` / `norm_reference` / `action` 全部由本文件**本地回填**。
这是"杜绝模型编造规范号"这一设计承诺的物理载体。

**风险提示**
`diagnostics.py` 会上浮 `knowledge_base_meta_count_stale` 告警。
**禁止读取 `meta.total_hazards` 做断言**，一律以解析出的 58 条为准。

---

### 2. `SKILL.md` — 上游原型使用说明（零代码引用，内容已与实现脱节）

**主要目的**
上游原型阶段写的"怎么用这套知识库"的说明书。

**内容概述**
2,603 字节，Jul 13 上游原始文件，git 已跟踪。含：文件结构图、System Prompt 注入方式、4 个场景模板索引、本地模型选型建议（Qwen2.5-VL-7B / InternVL2-8B / LLaVA-1.6-34B）、输出 JSON 格式、扩展指引。

**依赖关系**
无任何 Python 代码引用。仅 `deliverables/software-company/ROADMAP-工业安全检测训练闭环.md` 把它列为"新增文件"。

**在流程中的角色 — 历史存档层，且需警惕误导**

`SKILL.md` 正文描述的是"把 `knowledge_base.json` **全量**注入 system prompt"。
但实际实现走的是第三条路：`hazard_annotation/knowledge.py:122-137` 的
`KnowledgeCompressor.build_brief(scene_ids, max_chars)` —— 按场景过滤 + 按严重度排序 + 按字符预算截断，产出**压缩编号清单**。

> 文件自己在第 48 行也写了"建议不要全量注入"——它描述的是上游原型的设想，本项目已用 `KnowledgeCompressor` 取代。

**真正面向使用者的新版技能说明在 `skills/hazard-detection/SKILL.md`**（带 `metadata` 头，已同步 58 条 / 双通道 / `KnowledgeCompressor` / 六步法抽帧）。两者是**同一主题的新旧两版**，不是同一个文件的两个副本。

**易混淆点**：`tests/test_skills_delivery.py` 测的三个 `SKILL.md` 是上游技能包
（`skills/video-frame-extraction/`、`skills/images-qwen3vl-annotation/`、`skills/yolo-closed-loop-training/`），
**与本文件无关**，本文件无任何测试覆盖。

---

### 3. `prompt_templates.md` — 提示词设计稿（零代码引用，功能已被取代）

**主要目的**
给视觉大模型设计好的四套提问模板，覆盖不同识别情境。

**内容概述**
9,631 字节，Jul 13 上游原始文件。四套模板：
- **场景A** 已知场景类型，定向识别 —— 内含吊装/高处两份完整检查清单，逐条列出 `D-1-1`~`D-5-5`、`G-1-1`~`G-6-3`，并附"十不吊"与"行车六严禁"原文
- **场景B** 未知场景，先分类再识别（三步法）
- **场景C** 视频帧批量分析（逐帧 + 持续性隐患判定）
- **场景D** 整改前后对比（输出 `rectification_rate`）
- 末尾"使用建议"：temperature 0.1~0.3、多轮对话先分类再识别

**依赖关系**
无任何 Python 代码引用。被设计文档引用：
- `ARCH-hazard-annotation.md:7`（列为"设计基线"）、`:204`（称场景路由调用场景 B 第一步——**代码里没有这个调用**）
- `PRD-hazard-annotation.md:154`（P1-1 时间聚合"对齐场景 C"——**P1-1 至今未实现**）

**实际运行时提示词来源（与本文件无关）**
- 兜底：`hazard_annotation/prompts/combined_zh.md`
- 项目级覆盖：`examples/<project>/assets/prompts/combined_zh.md`（**优先级更高，实测胜出**）
- 解析逻辑：`hazard_annotation/pipeline/annotation.py` 的 `_resolve_prompt()`

**在流程中的角色 — 设计参考层**
是"提示词该长什么样"的设计稿与需求来源，不参与运行。
P1 需求（时间聚合、整改对比）实现时，本文件的场景 C / 场景 D 是直接参照物。

---

### 4. `detection_examples.md` — Few-shot 示例集（零代码引用）

**主要目的**
提供"输入画面 → 期望 JSON 输出"的标注范例，供 few-shot 注入或人工校对格式。

**内容概述**
5,413 字节，Jul 13 上游原始文件。四个完整示例：
1. 吊装作业多隐患（5 条隐患，`overall_risk: critical`）
2. 高处作业梯具违规（4 条隐患）
3. **无隐患场景**（`hazards: []`，用 `observations` 描述合规状态）
4. 整改前后对比（`rectification_rate: 100%`）

每个隐患项展示 `id / category / description / confidence / evidence / severity / norm_reference / action` 的写法，
`evidence` 一律遵循"**在哪 + 长什么样**"的写法要求。

**依赖关系**
无任何 Python 代码引用。被 `ARCH:7`、`ARCH:245`、`PRD:135`、`PRD:155`、`ROADMAP:69` 引用。

**⚠️ 关键问题**
`PRD:135` 要求 demo 资产"对齐 `detection_examples.md` 示例"，
但实际 demo 由 `hazard_annotation/demo.py` 纯代码合成几何图形（`MOCK_FRAME_SIZE = 64`），
**与本文件记载的场景没有任何映射关系，也没有对应的 few-shot 注入逻辑**。

**在流程中的角色 — 素材层（尚未接线）**
本项目"few-shot 示例"这一环**完全缺失**：素材备好了，没有代码消费它。
示例 3（无隐患场景）尤其重要——它是抑制"为了填满 JSON 而编造隐患"的范式。

---

### 5. `规范化施工隐患点分类整理（优化文档）.md` — 知识库的人工溯源稿

**主要目的**
隐患分类体系的**原始人工整理稿**，是 `knowledge_base.json` 的源头。

**内容概述**
5,525 字节，Jul 13 上游原始文件。按"隐患编号 / 隐患点 / 备注"三列整理的完整清单，
编号体系与知识库**一一对应**：`D-1-1`~`D-5-5`、`G-1-1`~`G-6-3`、`T-1`，
附录含"十不吊"十条与"行车六严禁"六条原文。

**独特价值：备注列里的"可识别性判断"**
这些信息**知识库里没有**：
- `D-1-7` 吊装无牵引绳 → "可能存在视频角度问题，可能实际有"
- `D-2-2` 吊架下的吊链斜拉 → "难以识别"
- `D-2-3` 使用卡板吊重物 → "有规范"
- `D-3-1` 散件未用料斗 → "禁止，有规范，**必须用料斗或其他防脱落措施**"

**依赖关系**
无任何 Python 代码引用。仅 `ROADMAP:69` 列为"新增文件"。

**在流程中的角色 — 溯源层，不可删除**
它直接回答"**哪些隐患在监控视角下根本不可能标出来**"——
是评估标注可行性、解释模型漏检、撰写论文"局限性"章节的第一手依据。**建议保留并显式引用。**

---

### 6. `knowledge_base.json.bak-20260916-171616` — 备份文件

**主要目的**
`scripts/fill_norm_ref.py` 批量补 `norm_ref` 之前的原文件快照（当时覆盖率 11/58）。

**内容概述**
30,129 字节。结构与现版本一致，差异在 `norm_ref` 字段与 `meta.total_hazards`。

**依赖关系与状态**
- git **未跟踪**（`git ls-files hazard-detection/` 只返回 5 个文件）
- 已被 `.gitignore:72` 的 `hazard-detection/*.bak-*` 规则正确排除
- **注意**：文件系统 mtime 显示 `Jul 13 11:54`，但文件名时间戳是 `20260916-171616`。
  这是 `shutil.copy2` 保留原文件 mtime 的正常行为，**不是文件损坏**

**在流程中的角色 — 安全网（已完成使命）**
`norm_ref` 已 100% 补齐且测试全绿，此备份可删。
若保留，注意它会让本目录在人工浏览时产生"有两个知识库"的困惑。

---

## 三、关联与依赖关系总览

**知识消费是单向的：代码 → 知识库**

```
规范化施工隐患点分类整理.md  ──(人工整理, 非代码)──▶  knowledge_base.json
                                                          │
                                        KnowledgeBase.load ─┼─▶ 标注 / doctor 自检 / 可视化配色
                                                          │
                                    KnowledgeCompressor ──┴─▶ 提示词注入（按场景过滤 + 字符预算截断）
```

**与本目录之外的关联**

- `skills/hazard-detection/SKILL.md` — 新版技能说明（现行版），取代本目录的 `SKILL.md`
- `hazard_annotation/prompts/combined_zh.md` — **实际运行时提示词**，取代 `prompt_templates.md`
  - `hazard_only_zh.md` / `objects_only_zh.md` — 合法待用分支（`combined_call=false` 时生效，当前三项目均为 true）
  - `scene_route_zh.md` — **孤儿文件**，文档声称在用但代码从未加载
- `docs/hazard-annotation-chain.md` — 链路说明
- `deliverables/software-company/` — 设计、需求、审计文档（PRD / ARCH / AUDIT 等）
- `scripts/fill_norm_ref.py` — 规范依据回填工具（默认干跑，`--apply` 才写入并自动备份）

---

## 四、在开发、协作、维护流程中分别扮演的角色

**契约层（1 个，必须严格维护）**
- `knowledge_base.json` —— 代码与领域知识之间唯一的接口。改它等于改系统行为，任何字段变更都要跑测试。

**设计参考层（3 个，只读不改）**
- `prompt_templates.md` —— 提示词设计稿，需求来源
- `detection_examples.md` —— 输出格式范式与 few-shot 素材
- `规范化施工隐患点分类整理（优化文档）.md` —— 编号体系溯源 + 可识别性备注

这一层的共同角色：**给"该做成什么样"提供依据**，同时是文档与代码不一致时的对照基准。

**历史存档层（2 个，可清理）**
- `SKILL.md` —— 上游原型说明，内容已过时
- `knowledge_base.json.bak-*` —— 修复前快照

---

## 五、维护指引

- **新增隐患** → 编辑 `knowledge_base.json` 按现有格式追加，并补齐 `norm_ref`
- **补规范依据** → `python scripts/fill_norm_ref.py`（默认干跑，`--apply` 才写入）
- **新增场景** → 在 `hazard_annotation/prompts/` 下新增提示词模板（**不是**改 `prompt_templates.md`）
- **改提示词** → 优先改项目级 `examples/<project>/assets/prompts/combined_zh.md`（它优先于包内兜底）
- **改动后必跑** → `tests/test_hazard_knowledge.py`、`tests/test_hazard_pipeline.py`

## 六、待办与已知不一致

1. `SKILL.md` 描述"全量注入"，实际是 `KnowledgeCompressor` 压缩注入 —— 建议加一行"本文件为上游原型说明"
2. `detection_examples.md` 的 few-shot 从未被代码消费，`PRD:135` 声称"对齐"但无映射
3. `prompt_templates.md` 的 `PRD:154/155` 引用指向未实现的 P1 需求，属悬空引用
4. 两份 `combined_zh.md` 措辞分叉，包内兜底版对三个项目是死文件 —— 建议统一
5. `hazard_annotation/prompts/scene_route_zh.md` 是孤儿文件，且 `ARCH:327` 声称它在用
