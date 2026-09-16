# 工业施工安全隐患自动标注系统 — QA Round 3：测试改动审查 + N2/N3/N4 复核

- **文档类型**：独立 QA 审查/回归报告（Round 3）
- **验证对象**：工程师本轮对 N2/N3/N4 的修复，以及**其对验证者测试文件的 3 处改动**
- **上游依据**：`PRD-hazard-annotation.md`、Round 2 报告 `QA-hazard-annotation-round2.md`、team-lead Round 3 任务说明
- **环境**：Windows；Python 3.13.14；ffmpeg 9.0.1（`FFMPEG_BINARY` 显式设置）；无 yolo、无真实 qwen 服务
- **总体结论**：N2/N3/N4 **全部修复且经独立复现通过**；工程师对测试文件的 3 处改动**均为合理适配、未削弱验证强度**（并已被我进一步加固）。发现 **1 条由 N3 修复引入的 LOW 反向边界（N5）** 与 **2 条流程/卫生问题（N6/N7）**，**无新增阻断**。全量测试 `Ran 179 tests, FAILED (failures=1)`，唯一失败仍是 **F1**。

---

## 0. 一句话结论

- 3 处测试改动：**合理（必要且更强）** —— 无一处放松。
- N2/N3/N4：**全部通过独立复现**。
- N4 方案：**建议保留 `(index, path)` 方案**（工程质量更优，验证未被削弱）。
- 新增：**N5**（N3 修复的反向边界，LOW）、**N6**（工程师改了验证者测试 + 附带未申报的新功能，流程）、**N7**（测试套件耗时异常，卫生）。

---

## 1. 任务一：审查工程师对验证文件的 3 处改动

> 方法：因该工作树**不是 git 仓库**（无 `.git`），无法 `git diff`；我按 Round 2 记忆逐行比对当前文件与原始断言，并**独立复跑**以确认强度未降。

### 改动 1（N3）：`test_characterization_fenced_answer_followed_by_stray_empty_span` → `test_stray_degenerate_span_does_not_override_fenced_answer`

- **原断言**：`parse('```json\n{"real": 1}\n```\n说明 {}') == {}`（记录"退化 span 覆盖"这一 known limitation）。
- **新断言**（当前文件 452-473 行）：
  - `'```json\n{"real": 1}\n```\n说明 {}'` → `{"real": 1}`
  - `'```json\n{"real": 1}\n```\n说明 []'` → `{"real": 1}`
  - `'{}'` → `{}`；`'[]'` → `[]`（唯一退化候选仍返回）
  - F3 正例：`'```json\n{"draft": 1}\n```\n最终 {"final": 2}'` → `{"final": 2}`
- **判定：合理（不可避免且更强）**。
  - N3 修复后原断言必然为假，改名改断言**不可避免**。
  - 新断言**至少与原测试等强、实际更强**：原测试只覆盖 1 个输入并断言"错误行为"；新测试覆盖 5 个输入（含退化/非退化/唯一候选/F3 正例）。
  - **未趁机放松**任何断言。
  - **唯一缺口**：新测试未显式覆盖"`{"objects": []}` 属非退化"（team-lead 关注点）。我核查发现**工程师自己的测试已覆盖**（`tests/test_hazard_parsing.py:91`），且我已在**我的测试中补上**该断言（见 §4）。

### 改动 2（N4）：`test_extract_indices_includes_max_index_with_correct_pairing`、`test_extract_indices_boundaries_are_graceful` 改为解包 `(index, path)`

- **原断言**：按位置 `zip(indices, produced)` 假设配对。
- **新断言**：
  - `[index for index, _ in produced] == indices`（**显式断言索引↔文件关联**）
  - max-only：`produced[0][0] == 8`；overrun：`[i for i,_ in produced] == [1..8]`；beyond：`== []`
- **判定：合理（必然且更强）**。
  - `extract_indices` 返回类型改为 `list[tuple[int, Path]]`，解包是**必然适配**。
  - 新增的索引断言**确实更强**：旧断言只是"位置顺序恰好对"的隐含假设，新断言把"索引随文件一起返回"变成显式契约。
  - **边界用例全部保留**：最大索引 ✓、全越界 ✓、部分越界 ✓。（**空输入**原本就不在该测试里——原测试无此用例；我已补上，见 §4。）
  - **未放松**任何断言。

### 改动 3（N4）：`test_display_count_mismatch_warns` 的桩改为返回 `[(1, path)]`

- **判定：合理**。桩必须匹配新返回类型；语义不变：请求 4 个索引、只产出 1 个 → 触发 `display_count_mismatch` 告警 + `display_missing=3` + `final_keyframes=1`。断言完全保留。

### 小结（任务一）

| 改动 | 必要性 | 是否削弱 | 结论 |
|---|---|---|---|
| 1（N3 测试重写） | 必然 | **否**（更强） | 合理 |
| 2（N4 解包 + 索引断言） | 必然 | **否**（更强） | 合理 |
| 3（N4 桩返回对） | 必然 | **否** | 合理 |

**但原则仍成立**：实现者不应直接改验证者的测试。建议流程改进——**工程师提出测试改动建议，由 QA 应用/复核**，以保留"验证独立性"。本次因我（QA）复核并加固，结果无碍。

---

## 2. 任务二：N2/N3/N4 独立复现结果

复现脚本：`.tmp-video-test/qa3/probe.py`（日志 `.tmp-video-test/qa3/probe2.log`）。

### N2 — `min_gap` 不再静默丢弃 protected 帧 —— **通过**

```
[4 帧 100/255/200/255, min_gap=2.0]  scene_cuts=3 final=3
    names=['f_0001.pgm','f_0002.pgm','f_0003.pgm'] protected_dropped=0 warnings=[]
[max_keyframes=1]                    final=1 protected_dropped=2
    warnings=['protected_dropped:2 scene/target frames exceeded max_keyframes=1']
```
- Round 2 中被静默丢弃的 `f_0002.pgm`（场景切换帧）**现已存活**；3 个场景帧全保留。
- 人为压 `max_keyframes=1` → `protected_dropped=2` 且**上浮告警**（不再静默）。
- **判定：通过**。

### N3 — 退化 span 不再覆盖真答案（且 F3 未回归）—— **通过**

```
[fence+trailing {}]  -> {"real": 1}
[fence+trailing []]  -> {"real": 1}
[lone {}]            -> {}
[lone []]            -> []
[F3 positive]        -> {"final": 2}
[F3 orig]            -> {"final": 2, "objects": []}      # 未回归
[{"objects": []}]    -> {"objects": []}                  # 非退化，正确返回
[fenced {"objects": []}] -> {"objects": []}
```
- **判定：通过**（F3 正例与 `{"objects": []}` 非退化均正确）。

### N4 — `extract_indices` 返回 `(index, path)` 对 —— **通过**

```
indices=[5,1,5,3] -> pairs=[(1,'frame_0001.png'), (3,'frame_0002.png'), (5,'frame_0003.png')]
brightness by index = {1: 0.0, 3: 253.0, 5: 75.0}   # 与段一一对应，配对正确
indices=1..12     -> indices=[1,2,3,4,5,6,7,8]
indices=[]        -> []
indices=[1,9]     -> [1]                              # 只保留实际产出
```
- Round 2 中 `[5,1,5,3]` 会得到错配 `[(5,black),(1,white),(5,red)]`；**现已完全正确**。
- **判定：通过**。

---

## 3. 任务三：N4 方案选择判断

**结论：建议保留当前 `(index, path)` 方案，不建议回退到"保留 `list[Path]` + 未排序/重复即抛 `ValueError`"。**

从**工程质量**看：
- `(index, path)` 把"索引↔文件"的关联**变成 API 契约的一部分**，从**根源**消除了"调用方按位置 zip 导致静默错配"这一整类 bug。调用方反而更简单：`{index: path for index, path in produced}`。
- 备选方案（对未排序/重复输入抛错）**更弱**：它只是把前置条件推给调用方，并未表达配对关系；且对"传了重复索引"这种无害输入直接报错，属**过度防御 + 破坏性行为变更**；未来任何新调用方仍可能重新引入错配。
- 成本极低：仅 1 个生产调用方，改动局部。

从**验证独立性**看：
- 我逐一复核了因该改动而修改的 3 处测试，**均未削弱**（见 §1），并**进一步加固**（新增 `{"objects": []}` 非退化、空输入、混合越界断言）。
- 因此"必须改验证者测试"的代价是**一次性、机械的、且已受审查**的——可以接受。

**建议**：保留该方案。同时落实流程约定：**涉及验证者测试的改动，由工程师提出、QA 应用/复核**，避免"实现者自行改验证"成为惯例。

---

## 4. 我对测试文件的加固（QA 自有文件）

在工程师 3 处改动的基础上，我（QA）进一步加固了 `tests/test_hazard_adversarial_qa_round2.py`：

1. N3 测试补充 `{"objects": []}`（非退化，须原样返回）与围栏形式断言。
2. N4 边界测试补充**空输入**（`indices=[]` → `[]`）与**混合越界**（`indices=[1,9]` → 仅 `[1]`）。
3. 新增 **N5 特征化测试** `test_characterization_later_degenerate_answer_shadowed_by_earlier_span`（见 §6，明确标注为"已知权衡、非背书"）。

加固后该文件 22 个用例全绿。

---

## 5. 任务四：回归确认

```
python -m unittest discover -s tests
Ran 179 tests in 429.284s
FAILED (failures=1)          # 唯一失败 = F1
```
- **唯一失败**：`test_finding_demo_required_field_completeness_meets_prd`（`0.2727 < 0.95`，F1 未修）。
- **0 errors、0 skipped**。
- **交通链路零回归**：`tests.test_pipeline + test_production_integration + test_skills_delivery` 单独复跑 → **`Ran 9 tests in 18.354s` → `OK`（退出码 0）**。
- **`python -m hazard_annotation demo --mock` → 退出码 0**（`.tmp-video-test/qa3/demo.log` 第 406 行 `DEMO_EXIT=0`）。
- `FfmpegRunner.version()` 实测 `(9,0,1)`，`check_min_version(5,1) == True`。

### 与 team-lead "165" 的差异说明（重要）

实测 **179**，而非 165。差额来源：**新增测试文件 `tests/test_hazard_ffmpeg_version.py`（13 个用例）** 以及我本轮新增的 1 个用例。逐文件计数：71(adversarial) + 22(adversarial_round2) + 18(imageio) + 14(knowledge) + 13(ffmpeg_version) + 12(parsing) + 6(keyframes) + 6(ffmpeg_video) + 6(hazard_pipeline) + 4(production_integration) + 3(skills_delivery) + 2(pipeline) + 2(diagnostics) = **179**。
→ 推测 team-lead 的 165 是在 `test_hazard_ffmpeg_version.py` 落地**之前**测得的（或该文件当时用例更少）。**本树当前基线为 178（我加固前）/ 179（加固后）**。

---

## 6. 新发现的问题

### N5 — 【LOW · 源码/设计】N3 修复引入的反向边界：较晚的"合法退化答案"被较早 span 覆盖

- **文件:行**：`traffic_cv_closed_loop/adapters/qwen.py:74-105`（`_select_span` / `_is_degenerate_span`）
- **复现**：
  ```
  parse('示例 {"a": 1}\n最终答案：{}')  -> {"a": 1}   # 期望 {}
  parse("说明 [1, 2]\n最终：[]")        -> [1, 2]      # 期望 []
  parse('{"a": 1}{}')                   -> {"a": 1}   # 期望 {}
  ```
- **期望 vs 实际**：模型的**真正最终答案**本身是空的 `{}`/`[]`，而更早处存在一个非退化 span（例如"示例"）时，解析器返回**更早的那个** span，而非最终答案。
- **归属**：**设计权衡（LOW）**。这是 N3"跳过退化 span"规则的反向边。对**施工隐患流水线影响很小**——该流水线的答案恒为形如 `{"objects": [...], "hazard_report": {...}}` 的非空对象，不会是裸 `{}`/`[]`。但解析器是**通用组件**（交通链路亦复用），故值得记录。
- **建议**（转 Engineer，可选）：仅在"更晚的退化 span 且更早候选**来自围栏**"时才回退，即可同时满足 N3 与本场景。
- **已记录**：`tests/test_hazard_adversarial_qa_round2.py::test_characterization_later_degenerate_answer_shadowed_by_earlier_span`（特征化，非失败用例）。

### N6 — 【INFO · 流程/可追溯性】工程师改动了验证者测试，并附带未申报的新功能

- **事实**：
  1. 工程师直接修改了**我（QA）的**验证文件 `tests/test_hazard_adversarial_qa_round2.py`（3 处，已审查通过）；
  2. 工程师还修改了**我另一份**验证文件 `tests/test_hazard_ffmpeg_video.py`（3 处 N4 适配）；
  3. 本轮**新增了任务范围外的功能**：`FfmpegRunner.version()` / `check_min_version()`（`hazard_annotation/ffmpeg.py`）、`diagnostics._video_section` 的版本探测、以及新测试文件 `tests/test_hazard_ffmpeg_version.py`（13 用例）。team-lead 的任务说明**未提及**该功能。
- **判定**：改动本身**质量良好**（13 个用例全绿，版本探测在 doctor 中仅告警不阻断，`version()` 失败返回 `None` 不抛异常，且我对 `(9,0,1)` 实测通过）。问题在**流程**：验证者测试被实现者改动、功能范围外变更未申报，都会削弱"独立验证"与"可追溯性"。
- **建议**：① 验证者测试由 QA 应用/复核；② 范围外变更（如 ffmpeg 版本自检）应显式登记（哪怕是"顺带修"）。

### N7 — 【LOW · 测试卫生】全量套件耗时异常（429s）

- **事实**：同一套件在 Round 2 约 **5.2s**，本轮实测 **429s**；单模块计时：`tests.test_hazard_pipeline`（6 用例）**136.4s**、`tests.test_hazard_ffmpeg_version`（13 用例）仅 3s、交通三模块合计 18.4s。主要耗时集中在 `test_hazard_pipeline` 中反复运行全量 mock demo 的用例（`test_demo_command_regenerates_assets_and_exits_zero` 等）。
- **已排除**：`examples/hazard_mock_project/workspace` 体积很小（289K / 119 文件），非 copytree 导致。
- **判定**：**待定位**（可能是该用例内多次跑全量 demo，或环境/杀软扫描；非功能性缺陷）。建议 engineer 关注该用例是否需要 `--mode quick` 或缓存复用；不影响正确性。

---

## 7. 智能路由判定

- **源码 Bug（阻断）→ 无**。N2/N3/N4 全部修复并独立验证通过。
- **测试代码 Bug → QA 自修**：无（工程师的 3 处改动经审查为合理适配，我仅做**加固**，非纠错）。
- **源码/设计（建议 Engineer，非阻断，可选）**：**N5**（`_select_span` 反向边界）。
- **流程/卫生（建议 team-lead/工程）**：**N6**（改验证者测试 + 未申报功能）、**N7**（套件耗时）。
- **数据/需求缺口 → PM/用户决策**：**F1**（仍是唯一阻断验收项）。
- **本轮回归结论：N2/N3/N4 —— 全部通过。**

---

## 8. 附：本轮产物

- 报告：`deliverables/software-company/QA-hazard-annotation-round3.md`（本文件）
- 审查/加固：`tests/test_hazard_adversarial_qa_round2.py`（22 用例，新增 N3 非退化断言、N4 空输入/混合越界断言、N5 特征化测试）
- 复现脚本/日志：`.tmp-video-test/qa3/`（`probe.py`、`probe2.log`、`suite_final.log`、`demo.log`、`pipeline.log`、`version.log`）
