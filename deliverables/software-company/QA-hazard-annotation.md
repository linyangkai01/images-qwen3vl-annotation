# 工业施工安全隐患自动标注系统 — 独立 QA 验证报告

- **文档类型**：独立对抗性测试报告（QA Engineer）
- **验证对象**：`hazard_annotation/` 新增包 + `traffic_cv_closed_loop/` 改造（T1–T5）
- **上游依据**：`PRD-hazard-annotation.md`（P0 验收标准）、`ARCH-hazard-annotation.md`（9 条关键决策）
- **验证环境**：Windows；Python 3.13.14（`C:\Users\81945\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe`）；**无 ffmpeg、无 yolo/ultralytics、无真实 qwen3-vl 服务**；零第三方依赖
- **验证结论**：**不建议直接进 4090D 全量验证**——需先处理 1 条 HIGH（阻断 P0-4 验收）与 1 条 MEDIUM（P0-1 关键帧正确性），另有 4 条 LOW 可列为已知问题

---

## 1. 测试产物

### 1.1 新增独立测试文件

- `tests/test_hazard_adversarial_qa.py` —— **71 个用例**，7 个测试类，覆盖 6 大高危区 + PRD 验收 + CLI 回归：
  - `ThinkingJsonParserAdversarialTest`（16）
  - `KnowledgeBaseAdversarialTest`（9）
  - `SceneRouterAdversarialTest`（3）
  - `ImageIoAdversarialTest`（13）
  - `ConfigAdversarialTest`（8）
  - `KeyframeAdversarialTest`（9）
  - `GracefulDegradationTest`（7）
  - `PrdAcceptanceFindingsTest`（2）
  - `CliConfigRegressionTest`（4）

  命名约定：`test_finding_*` 的用例断言的是**应有的正确行为**，在缺陷未修复前**预期失败**——它们就是本报告每条发现的、可复现的机器证据。

### 1.2 修改的既有测试（测试代码 Bug，QA 自行修复）

- `tests/test_hazard_pipeline.py::HazardPipelineTest._prepare`：新增 `shutil.rmtree(project / "workspace", ignore_errors=True)`。
  - 原因见 **F8**：该测试复制示例工程时连带复制了 `.gitignore` 掉的 `examples/hazard_mock_project/workspace/`，其中残留多份带时间戳的 `datasets/v*/data.yaml`，导致 `test_data_yaml_uses_hazard_labels` 断言 `len(data_yamls) == 1` 失败（实测 `3 != 1`）。**在交付工作树的当前状态下，工程师自述的“55 用例全绿”不可复现。**

---

## 2. 测试运行结果

### 2.1 全量回归（命令 + 结果）

```
python -m unittest discover -s tests
Ran 126 tests in 160.293s
FAILED (failures=8)
```

- 126 = 工程师既有 55（交通 9 + 施工 46）+ QA 新增 71
- **8 个失败全部是 QA 的 `test_finding_*`（即本报告的 8 条发现）**
- 其余 **118 用例全部通过**，包括交通 9 个测试**零回归**

### 2.2 端到端 demo

```
python -m hazard_annotation demo --mock
EXIT=0
```

产物齐全：`extracted/candidates.jsonl`、`extracted/summary.json`、`hazards/report.json`、`review/labels/*.txt`、`review/raw_responses/*.json`、`review/review_queue.jsonl`、`datasets/v*/data.yaml`、`analysis/hard_examples.jsonl`、`analysis/next_cycle_tasks.md`、`visualization/report.html`。
关键指标：`total_frames=5 / final_keyframes=5 / scene_cuts=2 / force_kept=5`；`overall_risk=critical`；`invalid_id_count=0`；**`required_field_completeness=0.2727`（见 F1）**。

---

## 3. 发现的问题清单（按严重级别）

### F1 — 【HIGH · 阻断 P0-4 验收】必填字段完整率 27% ≪ 95%

- **文件:行号**：`hazard_annotation/pipeline/annotation.py:198-210`（`_completeness`，`required = ("id","category","severity","norm_reference","action")`）；`hazard_annotation/knowledge.py:194-211`（`backfill`）；数据源 `hazard-detection/knowledge_base.json`
- **复现**：`python -m hazard_annotation demo --mock` → `workspace/review/summary.json` 的 `required_field_completeness = 0.2727`（3/11）。逐条看：`D-1-1`/`D-3-1` 通过，其余 9 条（`D-5-1`/`D-5-3`/`D-4-1`/`G-4-6`/`G-4-5`/`G-4-4`/`G-2-2`）的 `norm_reference` 全为空。
- **期望 vs 实际**：PRD P0-4「通道 B JSON 必填字段完整率 ≥ 95%」→ 实际 **27.27%**。
- **根因**：`knowledge_base.json` 共 58 条隐患，**仅 11 条（18.97%）带 `norm_ref`**（`D-1-1..D-1-7`、`D-2-3`、`D-3-1`、`D-4-3`、`G-1-1`），其余 47 条为空。`backfill` 只在 `definition.norm_ref` 非空时回填，因此 `norm_reference` 无法补齐，被 `_completeness` 计为缺失。**即使模型输出 100% 正确，该验收指标在当前知识库下也永远无法达标。**
- **归属判定**：**数据/需求缺口**（非代码逻辑 Bug）——需 PM/架构与用户决策：①补全知识库 47 条 `norm_ref`；或 ②修订验收口径（`norm_reference` 允许为空/可选）；或 ③在 KB 缺失时回退到一个默认规范号并显式标注。**这是本次交付对答辩最直接的硬伤。**

### F2 — 【MEDIUM · 源码】场景切换帧 / 目标保底帧会被去重（或运动）剔除后永久丢失

- **文件:行号**：`hazard_annotation/keyframes.py:97-109`（视频路径）与 `:208-219`（图片路径）。顺序为 `_step_motion` → `_step_dedup` → `_step_scene_cut_*` → `_step_force_keep_*` → `_step_cap_and_gap`；`_step_cap_and_gap`（:365）只遍历**幸存帧**。
- **复现**（纯 Python，无需 ffmpeg）：图片目录放两张 16×16 纯色帧（黑 `0` → 白 `255`），运行 `KeyframeExtractor.extract`。结果：`scene_cuts=0`、`final_keyframes=1`（白帧被 aHash 去重剔除，场景检测再也看不到它）。同样，把一张带 `person` 目标、但与上一帧近似重复的帧喂进去，该帧也会在 `force_keep` 之前被去重删除。
- **期望 vs 实际**：ARCH 决策 2 第 4 步「场景切换强制保留……**防止去重把‘切换瞬间’吃掉**」；PRD P0-1.4/1.5 明确要求「场景切换帧强制保留」「目标命中帧强制保留，**保证隐患帧不被去重误删**」。实际：去重先于场景检测/目标保底执行，被去重的帧无法被“强制保留”，上述两条 P0 要求在最需要它们时失效。
- **归属判定**：**源码 Bug**（实现顺序与设计意图/PRD 不符）→ 建议转 Engineer。修复方向：先算场景/目标索引，令 `_step_dedup`/`_step_motion` 跳过这些索引；或让 `_step_cap_and_gap` 能从未幸存的全量帧中补回被标记为 scene/force 的帧。

### F3 — 【MEDIUM · 源码/设计】思维链中的 ```json 草稿会覆盖最终答案

- **文件:行号**：`traffic_cv_closed_loop/adapters/qwen.py:44-49`（`parse` 中 fenced 短路）、`:80-86`（`_extract_fenced`）
- **复现**：`ThinkingJsonParser().parse('```json\n{"draft": 1}\n```\n最终答案：{"final": 2}')` → 返回 `{"draft": 1}`（错误），应为 `{"final": 2}`。
- **期望 vs 实际**：解析器目标为“取最后一个平衡 JSON（最终答案在后）”。但 `parse` 一旦发现 ```json 围栏就把它当作唯一候选，未再与全文的 last-balanced 结果比较；当**思维链带围栏草稿、最终答案不带围栏**时返回草稿。ARCH 决策 4 第 3 步（优先围栏）与第 4 步（取最后一个平衡 JSON）本身存在冲突，实现选了前者。
- **归属判定**：**源码/设计**（建议转 Engineer，改法：取“围栏候选”与“全文 last-balanced 候选”中结束位置更靠后者）。注：工程内常见情形（答案也带围栏）可正常工作，故定级 MEDIUM。

### F4 — 【LOW · 源码】`//` 注释剥离会破坏字符串内的 URL，且静默返回错误内容

- **文件:行号**：`traffic_cv_closed_loop/adapters/qwen.py:128`（`re.sub(r"//[^\n\r]*", "", text)`，在 `_repair` 中先于字符串感知处理）
- **复现**：`ThinkingJsonParser().parse("{'url': 'http://example.com/x', 'objects': []}")` → `{'url': 'http:'}`（静默数据损坏，不抛异常）。
- **期望 vs 实际**：修复应保持字符串内容不变；实际把 `//` 之后内容整体删除。
- **归属判定**：**源码 Bug**（低概率，但违反“不得静默返回错误结果”）→ 可转 Engineer；建议改为字符串感知的注释剥离，或仅在 `json.loads` 失败后、且更保守地剥离。

### F5 — 【LOW · 源码】P5/P6 的 CRLF 头部导致栅格偏移一个字节

- **文件:行号**：`hazard_annotation/imageio.py:56-59`（`if magic in ("P5","P6"): index += 1`，只跳过一个空白字节）
- **复现**：`PgmImage.load(b"P5\r\n1 1\r\n255\r\n\x07")` → `pixels == b"\n"`（应为 `b"\x07"`）。
- **期望 vs 实际**：Windows 生成的 Netpbm 常用 `\r\n`；当前实现把 `\n` 当成第一个像素。ffmpeg 输出为 `\n`，故实际影响有限。
- **归属判定**：**源码 Bug**（LOW）→ 可转 Engineer（跳过 maxval 后的整段空白）。

### F6 — 【LOW · 源码】像素数与 `width×height` 不符时静默补零/截断

- **文件:行号**：`hazard_annotation/imageio.py:94-96`（P5 补零/截断）、`:116-119`（P6 补零/截断）
- **复现**：`PgmImage.load(b"P5\n2 2\n255\n" + bytes([1,2]))` → 返回 `[1,2,0,0]`（补零，无异常）；`b"P5\n1 1\n255\n" + bytes([1,2,3,4])` → 返回 `[1]`（静默截断）。
- **期望 vs 实际**：畸形栅格应清晰报错；实际静默产生“看起来正常”的数据，补零还会制造虚假运动/直方图。
- **归属判定**：**源码 Bug**（LOW，存在“优雅容错 vs 明确报错”的设计歧义，建议转 Engineer 明确策略）。

### F7 — 【LOW · 算法局限】aHash 无法区分纯色/低纹理帧

- **文件:行号**：`hazard_annotation/imageio.py:170-181`（`ahash` 以 `value > mean` 置位）
- **复现**：`hamming(ahash(全黑 16×16), ahash(全白 16×16)) == 0`（应大于 0）。
- **期望 vs 实际**：纯色黑与纯色白差异极大，但均值哈希均置 0，被判为重复。这是 aHash 固有限制，也是 F2 的放大器（低纹理监控画面下会误删场景切换）。
- **归属判定**：**算法/设计局限**（非实现错误）——建议在去重时辅以直方图距离或直接采用 F2 的“先保底后去重”修复即可缓解。

### F8 — 【MEDIUM · 测试代码，已由 QA 修复】工程师自测非封闭，交付树现状下并非全绿

- **文件:行号**：`tests/test_hazard_pipeline.py:112-113`（`assertEqual(len(data_yamls), 1)`）
- **复现**：在含 `examples/hazard_mock_project/workspace/`（任何一次 demo 后即存在，已被 `.gitignore`）的工作树上运行 `python -m unittest tests.test_hazard_pipeline` → `AssertionError: 3 != 1`。
- **期望 vs 实际**：测试应与环境无关、可重复；实际依赖“示例工程的 workspace 不存在”这一隐式前提。工程师跑测试时（12:17）workspace 尚未生成，随后跑 demo（12:20）才生成，故其“55 全绿”结论在当前工作树上不可复现。
- **归属判定**：**测试代码 Bug → QA 已自行修复**（`_prepare` 中清理 `workspace/`）。修复后 `tests.test_hazard_pipeline` 6 用例全绿。

### 非阻断观察（不单独计为 Bug）

- **O1** `parse("{")` 返回 `{}`（把半截 `{` 修复为空对象），属“静默空 dict”的边界，可接受但建议收紧。
- **O2** P2/P5 的 `maxval != 255` 时**不缩放**（存原始值），影响阈值语义；ffmpeg 固定输出 255，实际无碍。
- **O3** 思维链中**未闭合**花括号（未用 ` thinking` 包裹）会导致整体抛 `ValueError`——**安全行为**（不返回错误 JSON、不静默空 dict），但属解析器能力边界，已如实记录。
- **O4** `examples/hazard_mock_project/workspace/`、`examples/mock_project/workspace/probes/` 等残留产物（均 gitignore），是 F8 的诱因，建议交付前清理。

---

## 4. PRD 第 5 节验收标准逐条对照

- **P0-1 关键帧提取** —— **部分满足**。六步法齐全、`summary.json` 五项计数齐全、`max_keyframes`/`min_gap_seconds` 生效（已用合成帧序列验证）；但**场景切换/目标保底被去重吞掉（F2）**，与“强制保留”不符。300~500 数量与漏帧率 ≤5% **无法验证**（无 ffmpeg、无真实视频）。
- **P0-2 类别体系** —— **满足（附条件）**。配置 `labels` 为 12 类施工类别、无交通残留；`data.yaml` 的 `names` 与配置一致（`overhead_crane`/`guardrail` 已断言）；PPE 缺失由通道 B 关联推断（`_PPE_CONFLICTS`）。注意：架构把 PRD 建议的 17 类收敛为 12 类（合并 `rigging`、删 `spreader_beam`/`hopper`），**需导师确认**。
- **P0-3 知识库注入** —— **满足**。`build_brief` 输出长度在 200/300/1000/4000/8000 预算下均 ≤ 上限；超限按 severity 从高到低截断并告警（tiny 预算只保留 critical）；S1 附加十不吊/行车六严禁、S2 不附加；模型返回 id 100% 命中（`invalid_id` 对编造编号生效）；`category/norm_reference/action` 本地回填且与知识库一致。**但 `norm_reference` 多数为空 → 见 F1。**
- **P0-4 双通道** —— **部分满足**。每关键帧同时有 `labels/*.txt` 与 `hazards/*.json`；`combined_call=true` 时单帧仅 1 次调用（mock 分支）；**但必填字段完整率 27% ≪ 95%（F1）**。
- **P0-5 本地模型接入** —— **部分满足**。`ThinkingJsonParser` 对 14 类对抗输入全部正确（取最后平衡 JSON、字符串感知、三种包裹、有限修复、垃圾输入抛异常）；`probe-qwen` 在 mock 下返回连通性与耗时。**“对含 thinking 响应解析成功率 ≥ 95%”“断网可跑”“单帧耗时”无法验证**（无真实 30B 服务）。
- **P0-6 训练闭环** —— **部分满足**。`data.yaml`、困难样本（含 critical/high 回流）、`next_cycle_tasks.md` 均生成。**真实 YOLO 训练、mAP50/P/R、混淆矩阵无法验证**（无 yolo/ultralytics）；mock trainer 指标可被可视化读取。
- **P0-7 mock 模式** —— **满足**。干净环境一条命令 `python -m hazard_annotation demo --mock` 退出码 0；三类场景（吊装多隐患/高处违规/无隐患）结构正确；HTML 顶部“演示数据（mock）”标识正确。
- **P0-8 配置扩展** —— **满足**。新字段全部有默认值；旧 `examples/mock_project/config/project.json` 与 `examples/production_project/config/project.production.json` 均可加载；`max_keyframes=0`/`sample_fps<=0`/`phash_size<2`/`kb_max_chars` 过小均触发 `validate()` 报错；`apply_profile("quick"/"full")` 覆盖帧预算/超时/并发/预算/tokens 均已断言。
- **可视化（第 4 节）** —— **部分满足**。自包含 HTML 生成、缩略图内联为 `data:image/png;base64`、SVG 柱/饼/折线、风险配色、mock 横幅均就绪；**浏览器实际渲染未做人工确认**（本环境无 GUI）。

---

## 5. 智能路由判定

- **源码有 Bug → 转 Engineer**：**F2**（关键帧去重顺序）、**F3**（围栏草稿覆盖最终答案）、**F4**（`//` 破坏字符串）、**F5**（CRLF 偏移）、**F6**（栅格长度不校验）。
- **测试代码有 Bug → QA 自行修复**：**F8**（已修，`tests/test_hazard_pipeline.py::_prepare`）。
- **数据/需求缺口 → 需 PM/架构/用户决策**：**F1**（知识库 `norm_ref` 覆盖不足导致 P0-4 不可达）。
- **算法局限（记录在案）**：**F7**（aHash 纯色不可分，可随 F2 一并缓解）。
- **其余 118 用例全部通过**，含交通 9 用例零回归。

---

## 6. 因环境缺失而无法验证的项（如实列出）

1. **无 ffmpeg**：视频→关键帧真实链路、`fps`/`scale`/`format=gray` 抽帧、`select='gt(scene,N)'+showinfo` 场景分数解析、`extract_indices` 采样索引↔文件映射、1 小时视频压缩比与漏帧率。
2. **无 yolo/ultralytics**：真实目标检测、`yolo train`、mAP50/P/R、混淆矩阵；仅验证了“缺二进制时抛明确错误”（`FileNotFoundError: YOLO binary not found`）。
3. **无真实 qwen3-vl（Ollama/vLLM）服务**：端到端 thinking 解析成功率、单帧耗时、断网可跑、`extra_body(think:false)` 是否被服务端真正支持。
4. **4090D 显存与 q8 模型冲突**：架构第 9 节第 1 条的待用户拍板事项，未验证。
5. **浏览器渲染**：HTML 未做人工目视确认。

---

## 7. 已尝试但“未攻破”的攻击手段（证明这些部分确实稳）

- **ThinkingJsonParser**：思维链含成对花括号（取到最终答案）、字符串内含 `{}` 与转义引号、`\`\`\`json` 围栏、` thinking`、`<|thinking|>`、```thinking``` 围栏、尾逗号、单引号、`True/False/None`、`//` 与 `/* */` 注释、截断补齐、顶层数组、纯垃圾/空串抛异常、`repair=False` 抛异常、`_extract_json_from_text` 向后兼容——**均正确**。
- **知识库**：58 条（27+30+1）、`id_index` 全覆盖、S3 直挂 `T-1` 正确解析、`meta.total_hazards=53` 陈旧值未被代码依赖、`build_brief` 预算/截断/场景隔离、非法 id 标记、重复 id 拒绝、模型编造规范号被覆盖——**均正确**。
- **图像链路**：P2/P3/P5/P6 四格式、头部注释、多空白分隔、坏 magic 报错、截断头部报错、aHash 自距离 0、结构化图可区分、`motion_ratio`/`histogram16`/`histogram_delta` 手算值、尺寸不一致不崩溃——**均正确**。
- **配置**：旧配置加载、新段默认值、`kb_path` 自动向上查找、`to_dict` 新段、quick/full 双档、用户 profile 叠加、非法值校验——**均正确**。
- **关键帧**：运动剔除、去重、目标保底（在幸存帧上）、`min_gap`、`max_keyframes`、图片目录端到端——**均正确**（缺陷仅在“去重吞掉保底帧”的顺序上）。
- **优雅降级**：无 ffmpeg 视频路径抛 `FfmpegUnavailableError`（含 doctor 指引）、无 yolo 抛明确错误、未知 detector 抛 `ValueError`、`doctor` 在无 ffmpeg 下给出 video/keyframe/knowledge 三段且标出 `meta_count_stale`、**失败帧 100% 进入 `review_queue.jsonl` 且原始响应写入 `review/raw_responses/`（“严禁静默丢帧”成立）**——**均正确**。
- **CLI**：用 argparse 实证还原了原写法——子解析器 `--config` 的默认值**确实会覆盖**父级 `--config`（`['--config', X, 'run-loop']` 得到 `examples/mock_project/config/project.json`）。**工程师把默认值改为 `argparse.SUPPRESS` 的说法成立、改动必要且无副作用**（三种调用顺序均正确）。

---

## 8. 总体判断（能否直接进 4090D）

**结论：不建议直接进 4090D 全量验证。**

- **必须先修**：**F1**（否则 P0-4「完整率 ≥95%」在答辩时无法展示，且是“规范依据”这一卖点的直接证据缺失）；**F2**（否则 P0-1 的“场景/目标强制保留”名不副实，真实视频的关键帧质量存疑）。
- **建议一并修**：**F3**（30B thinking 模型下真实概率不低）。
- **可列为已知问题**：F4、F5、F6、F7。
- **已修**：F8。
- **可以放心进 4090D 的部分**：配置体系、知识库归一化/压缩/回填、纯 Python 图像算法、双通道落盘与复核队列、mock 全链路、CLI、交通零回归——这些在对抗性测试下表现稳健。

> 一句话：**功能骨架与工程健壮性扎实（118/126 通过，且失败全部是有意设计的对抗性断言），但两条 P0 验收（P0-4 完整率、P0-1 强制保留）当前不达标，需先修再上 4090D。**
