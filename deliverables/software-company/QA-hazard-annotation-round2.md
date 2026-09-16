# 工业施工安全隐患自动标注系统 — 独立 QA 回归验证报告（Round 2）

- **文档类型**：独立对抗性回归验证报告（QA Engineer）
- **验证对象**：工程师本轮对 F2–F12 的修复（`keyframes.py` / `ffmpeg.py` / `adapters/qwen.py` / `imageio.py` 四个文件）
- **上游依据**：`PRD-hazard-annotation.md`（P0 验收标准）、`ARCH-hazard-annotation.md`、Round 1 报告 `QA-hazard-annotation.md`
- **验证环境**：Windows；Python 3.13.14（`C:\Users\81945\.workbuddy-ai\binaries\python\versions\3.13.12\python.exe`）；**ffmpeg 9.0.1 可用**（`...\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe`，并显式置于 PATH）；无 yolo/ultralytics、无真实 qwen3-vl 服务；零第三方依赖
- **验证结论**：**F2–F12 全部修复且经真实 ffmpeg 独立复现通过**；工程阻断项已清零。**唯一剩余阻断 = F1**（P0-4 完整率 27.27%），属数据/需求缺口而非代码缺陷。详见第 8 节。

---

## 0. 一句话结论

Round 1 的 8 条 `test_finding_*` 中 **7 条转绿**（F2/F3/F4/F5/F6/F7），F9–F12 首次在真实 ffmpeg 下实测通过；**仅 F1 仍红**。主理人自述的验证数据（场景索引 `[4,6]`/`[7,11]`、亮度指纹）**已被我独立复现**。新发现 4 条 LOW（1 条测试代码已自修、3 条源码/设计观察），**无新增阻断**。

---

## 1. 测试产物

### 1.1 新增测试文件

- **`tests/test_hazard_adversarial_qa_round2.py` —— 21 个用例，5 个测试类**：
  - `FfmpegRound2Test`（5，真实 ffmpeg）：F9 `-vsync` 已移除 / F10 整数与**小数** fps 对齐 / F11 最大索引与越界边界 / 端到端计数。
  - `KeyframeProtectedRound2Test`（3）：F2 最坏情况（aHash **完全相同**的场景切换帧、与前一帧完全相同的目标帧）、图片目录路径回归。
  - `DisplayWarningRound2Test`（2）：F12 抽取失败告警、产出数量不符告警。
  - `ImageIoRound2Test`（4）：F5（P6 CRLF）、F6（P6 栅格不符）、单尾随空白容忍、F7。
  - `QwenParserRound2Test`（7）：F3/F4 + 原有能力回归 + 1 条对 F3 修复副作用的**特征化记录**测试。

  关键设计：**自建测试视频**（4 段纯色，见第 4 节），亮度指纹可把每个采样索引唯一钉到某一段；覆盖工程师用例**未覆盖**的小数 fps（0.5/1.5）、最大索引、越界、抽取失败。

### 1.2 修改的既有测试（测试代码 Bug，QA 自修）

- **`tests/test_hazard_adversarial_qa.py:655` `GracefulDegradationTest.test_video_without_ffmpeg_raises_clear_error`**
  - **问题**：该用例依赖"环境无 ffmpeg"这一隐式前提（`FfmpegRunner(config.video.ffmpeg_binary).available()` 走 `shutil.which`）。ffmpeg 一旦进入 PATH，用例由 pass 变 **ERROR**（真实 ffmpeg 去解假 mp4 → `RuntimeError` 而非 `FfmpegUnavailableError`）。
  - **修复**：显式 `config.video.ffmpeg_binary = "definitely-not-a-real-ffmpeg-binary"`，使其封闭。
  - **旁证**：主理人"`Ran 138 tests, FAILED (failures=1)`"的基线是在**无 ffmpeg 的 PATH** 下得到的；同一份代码在**有 ffmpeg 的 PATH** 下会多出 1 个 ERROR。修复后两种环境均稳定。

---

## 2. 完整测试运行结果

```
# 命令（含 ffmpeg 于 PATH）
python -m unittest discover -s tests

# 结果
Ran 159 tests
FAILED (failures=1)
```

- **159 = 既有 138 + 本轮新增 21**；**0 errors、0 skipped**。
- **唯一失败 = F1**：`test_finding_demo_required_field_completeness_meets_prd` → `AssertionError: 0.2727 not greater than or equal to 0.95`。
- 重的端到端/ffmpeg 用例确实执行：`test_mock_demo_runs_clean_end_to_end ... ok`、`test_full_video_extraction_counts_match_records ... ok`、`FfmpegRound2Test.* ... ok`。
- 分组补充实测：交通链路 `tests.test_pipeline + test_production_integration + test_skills_delivery` → **Ran 9 tests, OK（零回归）**；`test_hazard_parsing/imageio/keyframes/diagnostics` → **Ran 34 tests, OK**。

> 注：本报告运行时 shell 输出流一度异常，故所有结论均以**落盘日志 + 独立脚本实测**为准（日志见 `.tmp-video-test/qa2/`）。

---

## 3. F2–F12 逐条回归结论

判定口径：**通过**＝独立复现出修复后的正确行为；**未通过**＝复现失败；**无法验证**＝环境/数据缺失。

### F2 — 场景/目标保底帧不被 motion/dedup 删除 —— **通过**

- **复现 A（图片，aHash 相同的最坏情况）**：`python _qa2_f2_f12.py`（脚本已挪至项目根，属临时产物）
  - 帧 `100 → 120`：`hamming(ahash(100), ahash(120)) == 0`；**未加保护**时 `_step_motion` 只留 `[0]`（删 1）、`_step_dedup` 只留 `[0]`（删 1）——证明**两个过滤器都会把它吃掉**；**加 `protected` 后** `scene_cuts=1`、`records=[f_0000.pgm, f_0001.pgm]`。
- **复现 B（视频，真实阈值）**：`.tmp-video-test/qa2/extra.log`
  - 我的 8 s 视频，`motion_min_ratio=0.002, dedup_hamming_threshold=4, min_gap=0` → `sampled_frames=8, motion_removed=4, dedup_removed=0, scene_cuts=3, final_keyframes=4, display_missing=0`，**存活场景索引 = [3,5,7]**（白/红/lime 三段切换帧）。
  - 说明：4 个段内重复帧被 motion 正确删除，而 3 个**场景切换帧全部存活**（若无 `protected`，它们会被 dedup 以 hamming=0~1 < 4 删除）。
- **复现 C（目标帧与前一帧完全相同）**：`force_kept=1`，`f_0001.pgm`（含 `person`）存活。
- **判定：通过**。`protected` 机制在**图片路径与视频路径**均真实生效，修复方向与 ARCH 决策 2/PRD P0-1.4/1.5 一致。

### F3 — 思维链中的 ```json 草稿不再覆盖最终答案 —— **通过**

- 复现：`ThinkingJsonParser().parse('```json\n{"draft": 1, "objects": []}\n```\n最终答案：{"final": 2, "objects": []}')` → **`{"final": 2, "objects": []}`**（`_qa2_qwen.log`）。
- 两个围栏场景 `...\n```json\n{"final": 2}\n``` ` → `{"final": 2}`。
- 判定：**通过**。副作用见 **N3**（新发现，LOW）。

### F4 — `//` 注释剥离不再破坏字符串 —— **通过**

- 复现：`parse("{'url': 'http://example.com/x', 'objects': []}")` → `url == "http://example.com/x"`；`parse('{"note": "it\'s fine // ok", "objects": []}')` → 原样保留；真实注释 `// c` / `/* c */` 仍被剥离。
- 直接验证 `_strip_comments`：`{'u': 'http://a/b'}` 与 `{"u": "http://a/b"}` 均不变；`{"a":1} // c` → `{"a":1} `。
- 判定：**通过**（字符串感知的注释剥离实现正确）。

### F5 — P5/P6 CRLF 头不再偏移栅格 —— **通过**

- 复现：`b"P5\r\n1 1\r\n255\r\n\x07"` → `pixels == b"\x07"`；`b"P6\r\n1 1\r\n255\r\n" + bytes([10,20,30])` → `pixels == bytes([10,20,30])`。
- 判定：**通过**（P5/P6 均正确；`\r\n` 视为单一分隔符）。

### F6 — 栅格长度不符时抛错而非静默补零/截断 —— **通过**

- 复现：P5 短栅格（期望 4 得 2）与超长（期望 1 得 4）均抛 `ValueError`；P6 同理。
- 边界：单尾随空白（`bytes([7]) + b"\n"`）**被容忍**（符合 Netpbm 规范），其余多余字节抛错。
- 判定：**通过**。

### F7 — aHash 能区分纯色黑/白 —— **通过**

- 复现：`hamming(ahash(bytes([0]*256),16,16,8), ahash(bytes([255]*256),16,16,8)) > 0`（新增全图亮度高位，恰好差 1 bit）。
- 判定：**通过**。注：仅差 1 bit，`dedup_hamming_threshold=4` 时仍会被判重——**真正保住切换帧的是 F2 的 `protected`**，F7 是必要但不充分的缓解。

### F9 — `-fps_mode passthrough` 替换被移除的 `-vsync 0` —— **通过**

- 复现（`.tmp-video-test/qa2/`）：
  - 旧写法：`ffmpeg -i mine.mp4 -vf fps=1 -vsync 0 ...` → `Unrecognized option 'vsync'. Error splitting the argument list: Option not found`（**证实 ffmpeg 9.x 已移除 `-vsync`**）。
  - 新写法：`-fps_mode passthrough` → `frame= 8`，产出 8 帧。
- 判定：**通过**（`_FPS_MODE_ARGS = ("-fps_mode","passthrough")` 已生效且必要）。

### F10 — 场景索引映射与吸附 —— **通过（含小数 fps）**

- 我的视频（真实切换 t=2/4/6 s，25 fps 源）：

  | `sample_fps` | 场景索引 | 映射时间戳 | 切换后帧亮度指纹 |
  |---|---|---|---|
  | 0.5 | `[2,3,4]` | `[2.0,4.0,6.0]` | 255 / 75 / 150 |
  | 1.0 | `[3,5,7]` | `[2.0,4.0,6.0]` | 255 / 75 / 150 |
  | 1.5 | `[4,7,10]` | `[2.0,4.0,6.0]` | 255 / 75 / 150 |
  | 2.0 | `[5,9,13]` | `[2.0,4.0,6.0]` | 255 / 75 / 150 |

  每个索引都**恰好落在切换后的第一帧**，且 `set(indices) ⊆ available_indices`（吸附有效）。
- 主理人数据独立复现：`synthetic.mp4` `fps=1.0 → [4,6]`、`fps=2.0 → [7,11]`（**完全一致**）。
- 判定：**通过**（`int(round(seconds*fps))+1` + `_nearest_index` 对整数与小数 fps 均正确）。

### F11 — 1-based→0-based 选择器（最大索引盲点） —— **通过**

- 我的视频 `fps=1.0`，`coarse_n=8`：

  | 请求 indices | 产出 | 说明 |
  |---|---|---|
  | `[1..8]` | **8** | 亮度 `[0,0,253,253,75,75,149,149]`，与段一一对应；**最大索引 8 已能产出**（原 bug 盲点） |
  | `[8]` | 1 | 亮度 149（lime） |
  | `[1..12]` | 8 | 越界部分被忽略，不报错 |
  | `[9,10]` | 0 | 全越界返回空 |
  | `[]` | 0 | 空请求返回空 |

- 判定：**通过**。生产调用方 `_extract_display_frames` 传 `sorted({frame.index ...})`，配对正确。
- 遗留：**N4**（LOW，非阻断）——`extract_indices` 对**未排序/含重复**的 indices 会静默错配，生产调用方当前不受影响。

### F12 — `final_keyframes` 反映实际记录 + `display_missing` 告警 —— **通过**

- 复现（`.tmp-video-test/qa2/`，脚本 `_qa2_f2_f12.py`）：
  - **抽取失败**（`extract_indices` 抛 `RuntimeError`）：`final_keyframes=0`、`display_missing=4`、`records=0`、`warnings=['display_extract_failed:boom: no decoder', 'display_missing:4 selected frames had no extracted image']` → **告警而非静默跳过**。
  - **产出数量不符**（请求 4 得 1）：`warnings=['display_count_mismatch:requested=4,produced=1', 'display_missing:3 ...']`、`final_keyframes=1`。
- 真实端到端：`extracted/summary.json` 的 `totals.final_keyframes == len(records)`，`warnings=[]`（无异常时不上浮噪音告警）；`per_source[*].display_missing=0`。
- 判定：**通过**。

### 小结

| 发现 | 级别 | 判定 |
|---|---|---|
| F2 | MEDIUM | **通过** |
| F3 | MEDIUM | **通过** |
| F4 | LOW | **通过** |
| F5 | LOW | **通过** |
| F6 | LOW | **通过** |
| F7 | LOW | **通过** |
| F9 | HIGH（视频阻断） | **通过** |
| F10 | HIGH（视频阻断） | **通过** |
| F11 | HIGH（视频阻断） | **通过** |
| F12 | MEDIUM | **通过** |
| **F1** | **HIGH（P0-4 阻断）** | **未通过（未修）** |

---

## 4. 我构造的测试视频与亮度指纹

**参数**（`.tmp-video-test/qa2/mine.mp4`）：

- 时长 8 s，`320x240`，25 fps，`h264 crf18`，`yuv420p`
- 4 段纯色（硬切）：`black` 0–2 s → `white` 2–4 s → `0xFF0000` 4–6 s → `lime` 6–8 s
- 真实切换时刻：**t = 2.0 / 4.0 / 6.0 s**（对应 1 fps 分析索引 3 / 5 / 7）

**亮度指纹（`fps=1.0`，逐帧灰阶均值）**：

```
idx:   1     2     3     4     5     6     7     8
luma:  0.0   0.0   253.0 253.0 75.0  75.0  149.0 149.0
段:    black black white white red   red   lime  lime
```

> 说明：ffmpeg 的 `green` 实为 `(0,128,0)`（luma≈75，与 red 的 76 几乎相同），故第 4 段改用 `lime`（luma≈150）以保证指纹可区分——这是构造测试视频时的一个**实际坑点**，一并记录。

**主理人 `synthetic.mp4` 独立复现**（`.tmp-video-test/qa2/synth.log`）：

| 项 | 主理人数据 | 我的复现 | 判定 |
|---|---|---|---|
| `scene_cut_indices(fps=1.0)` | `[4,6]` | `[4,6]` | ✅ 一致 |
| `scene_cut_indices(fps=2.0)` | `[7,11]` | `[7,11]` | ✅ 一致 |
| frame_0001 (ts=0.0) 亮度 | 127.5 | 127.5 | ✅ |
| frame_0003 (ts=2.0) 亮度 | 126.6 | 126.6 | ✅ |
| frame_0004 (ts=3.0) 亮度 | 0.0 | 0.0 | ✅ |
| frame_0006 (ts=5.0) 亮度 | 92.8 | 92.5 | ✅（差 0.3，`scale_width` 取值差异） |

**结论：主理人的验证数据全部复现成功，无推翻项。**

---

## 5. 新发现的问题

### N1 — 【LOW · 测试代码 · **已由 QA 修复**】`test_video_without_ffmpeg_raises_clear_error` 非封闭

- **文件:行**：`tests/test_hazard_adversarial_qa.py:655`
- **复现**：将 ffmpeg 加入 PATH 后运行 `python -m unittest tests.test_hazard_adversarial_qa.GracefulDegradationTest` → 该用例 **ERROR**（`RuntimeError: ffmpeg coarse sampling failed`，而非 `FfmpegUnavailableError`）。
- **根因**：用例依赖"环境无 ffmpeg"的隐式前提，未显式指定不存在的二进制。
- **归属**：**测试代码 Bug → QA 自修（已完成）**。修复后 7/7 OK。
- **附带影响**：主理人的"138 tests / 1 failure"基线是**无 ffmpeg PATH** 下的结果；有 ffmpeg 时会多 1 ERROR。已在第 2 节说明。

### N2 — 【LOW · 源码/设计】`min_gap` 可静默丢弃"场景切换帧"（protected 帧）

- **文件:行**：`hazard_annotation/keyframes.py:437-466`（`_step_cap_and_gap`）
- **复现**（`.tmp-video-test/qa2/extra.log`）：图片目录 4 帧 `[100,255,200,255]`，`min_gap_seconds=2.0` → `scene_cuts=3` 但 `final_keyframes=2`，`f_0002.pgm`（场景切换帧）被丢弃，且 **`warnings=[]`**。
- **期望 vs 实际**：PRD P0-1.4 要求"场景切换帧强制保留"；实际 `min_gap` 可在两个 protected 帧间距 < `min_gap` 时丢弃后者，**且不告警**。
- **归属**：**设计张力**（step 6 的 `min_gap` 是既定设计，Round 1 亦未涉及）。建议二选一：① 对 scene/force 帧豁免 `min_gap`；② 丢弃 protected 帧时上浮告警。**非阻断，建议转 Engineer 可选修**。

### N3 — 【LOW · 源码/设计】F3 修复引入的 `_later_span` 退化覆盖风险

- **文件:行**：`traffic_cv_closed_loop/adapters/qwen.py:49-80`（`_later_span`）
- **复现**：`ThinkingJsonParser().parse('```json\n{"real": 1}\n```\n说明 {}')` → 返回 **`{}`**（F3 修复前返回 `{"real": 1}`）。`[]` 同理。
- **期望 vs 实际**：围栏内是真正的答案，其后一个"退化 span"（空对象/空数组）却因"位置更靠后"而胜出，且**不抛异常**（静默降级为空结果）。
- **归属**：**设计张力**——ARCH 决策 4 字面要求"取最后一个平衡 JSON"，故当前行为**符合文字设计**，但与"解析成功率"意图相悖。建议：当后出现的 span 为退化（`{}` / `[]`）时回退到前一个候选。**定级 LOW**（真实模型在围栏答案后紧跟裸 `{}` 的概率低）。
- **本报告已用 `test_characterization_fenced_answer_followed_by_stray_empty_span` 明确记录当前行为**（不作为失败用例，避免与既定设计冲突）。

### N4 — 【LOW · 源码健壮性】`extract_indices` 对未排序/重复 indices 静默错配

- **文件:行**：`hazard_annotation/ffmpeg.py:113-148`
- **复现**：`extract_indices(video, fps=1.0, indices=[5,1,5,3])` → 产出 3 帧，若调用方按原 `indices` 做 `zip`，得 `[(5,black),(1,white),(5,red)]`（**错配**）。
- **根因**：方法内部用 `sorted(set(indices))` 生成选择器，但返回值是 ffmpeg 的输出顺序；方法**未返回实际索引**，配对依赖调用方传入有序去重序列。
- **现状**：唯一生产调用方 `KeyframeExtractor._extract_display_frames` 传入 `sorted({...})`，**当前不受影响**。
- **归属**：**潜在 API 契约问题（LOW）**。建议返回 `list[tuple[int, Path]]` 或在 docstring 中明确"indices 必须有序去重"。**记录为已知问题，非阻断**。

### 无新增阻断问题

Round 1 的 F1 仍为唯一阻断；本轮**未发现新的源码阻断缺陷**。

---

## 6. PRD 验收标准更新后的逐条判定

| 条目 | Round 1 | **Round 2** | 变化说明 |
|---|---|---|---|
| **P0-1 关键帧提取** | 部分满足（F2 吞帧） | **满足（附条件）** | F2 已修：场景/目标保底在 motion/dedup **之前**，`protected` 机制经图片+视频双路径实测生效；六步法/五项计数/`max_keyframes`/`min_gap` 均验证。**仍未验证**：1 小时真实视频的漏帧率 ≤5%、压缩比（无长视频）。附 N2 观察。 |
| **P0-2 类别体系** | 满足（附条件） | **满足（附条件）** | 无变化；12 类收敛仍需导师确认。 |
| **P0-3 知识库注入** | 满足 | **满足** | 无变化；`norm_reference` 覆盖不足的后果归入 F1。 |
| **P0-4 双通道** | 部分满足（F1） | **仍部分满足（F1）** | **必填字段完整率 = 0.2727 ≪ 0.95**，本轮**独立重测**确认未变（`_qa2_f1.log`）。**唯一未达标验收项。** |
| **P0-5 本地模型接入** | 部分满足 | **部分满足（前进）** | `ThinkingJsonParser` 对抗输入全部正确（含 F3/F4 修复）；"真实 30B 服务解析成功率/单帧耗时/断网"仍**无法验证**。 |
| **P0-6 训练闭环** | 部分满足 | **部分满足** | 无变化；真实 YOLO 训练/mAP 仍无法验证。 |
| **P0-7 mock 模式** | 满足 | **满足** | 一条命令 `demo --mock` 退出码 0，产物齐全（本轮再次实测）。 |
| **P0-8 配置扩展** | 满足 | **满足** | 无变化。 |
| 可视化 | 部分满足 | 部分满足 | 无 GUI，未做目视确认。 |

**净变化**：**P0-1 由"部分满足"升为"满足（附条件）"**；P0-5 在解析器维度前进；**F1 仍是唯一缺口**。

---

## 7. 智能路由判定

- **源码有 Bug（需转 Engineer，阻断）**：**无**。F2–F12 全部修复且独立验证通过。
- **源码/设计（建议转 Engineer，非阻断，可选）**：
  - **N2** `min_gap` 静默丢弃 protected 帧（`keyframes.py:437-466`）
  - **N3** `_later_span` 退化 span 覆盖（`qwen.py:49-80`）
  - **N4** `extract_indices` 未排序/重复输入的静默错配（`ffmpeg.py:113-148`）
- **测试代码有 Bug → QA 自修**：**N1**（已完成）。
- **数据/需求缺口 → PM/架构/用户决策**：**F1**（知识库 47/58 条 `norm_ref` 为空 → P0-4 不可达）。
- **全部通过（本轮回归）**：**F2–F12 —— 是**。

---

## 8. 能否进 4090D 的最终判断

**结论：视频/关键帧/解析/图像/配置/交通链路已具备进 4090D 的条件（工程阻断项清零）；但 P0-4 验收仍不达标，取决于 F1 的处置。**

对比 Round 1：

- Round 1 结论为"**不建议直接进**"，须先修 **F1 + F2**，建议修 **F3**。
- Round 2：**F2、F3、F4、F5、F6、F7、F9、F10、F11、F12 全部修复并通过独立回归**；视频链路**首次可在真实 ffmpeg 下跑通**（F9–F11 的 ffmpeg 9.x 兼容性是真·阻断级 bug，现已解决）。**工程侧已无阻断。**

**剩余唯一阻断 = F1**，性质为**数据/需求缺口**（非代码缺陷）：

- 实测：KB `norm_ref` 覆盖 **11/58 = 18.97%**；mock demo 必填字段完整率 **27.27%**。**即使模型 100% 正确，该指标在当前知识库下也不可能达到 95%。**
- **需 PM/导师决策三选一**：① 补全 47 条 `norm_ref`；② 修订验收口径（`norm_reference` 允许为空）；③ KB 缺失时回退默认规范号并显式标注。

**行动建议**：

1. **可以进 4090D 的部分**：关键帧六步法（含真实视频）、thinking 稳健解析、图像链路、配置体系、双通道落盘与复核队列、mock 全链路、CLI、交通零回归——这些在对抗性测试下表现稳健。
2. **进 4090D 前必须决策 F1**：若答辩需展示 P0-4「≥95%」，则必须先补数据或改口径，否则该验收项仍红。
3. **N2/N3/N4 可列为已知问题**（均 LOW、非阻断），建议随下一轮一并处理。

---

## 9. 已尝试的攻击手段（含被推翻/无法验证项）

**成功攻破（发现的问题）**：非封闭测试（N1，已修）、`min_gap` 静默丢 protected 帧（N2）、`_later_span` 退化覆盖（N3）、`extract_indices` 错配（N4）。

**已尝试但"未攻破"（证明这些修复确实稳）**：

- **F2**：aHash **完全相同**的场景切换帧（`100→120`）、与前一帧**完全相同**的目标帧、真实视频默认阈值——`protected` 均正确保住；未加保护时两过滤器确实会删（对照实验）。
- **F3/F4**：围栏草稿+裸最终答案、双围栏、单引号 URL、双引号内 `//`、含撇号的双引号串、真实 `//` 与 `/* */` 注释、嵌套转义引号——全部正确。
- **F5/F6**：P5/P6 CRLF、CR-only、短/超长栅格、单尾随空白——全部符合预期。
- **F10**：小数 fps（0.5/1.5）与整数 fps（1.0/2.0）——映射与吸附均正确。
- **F11**：最大索引、全越界、部分越界、空请求、乱序重复——除 N4 外均正确。
- **F12**：抽取抛异常、产出数量不符——均**告警**而非静默。
- **回归面**：图片目录路径（签名变更后）正常；交通链路 9/9 零回归；`parse()` 原有能力（think/pipe/thinking 围栏/纯 JSON/数组/截断修复/尾逗号/单引号/字面量/last-balanced）全部保持。

**无法验证（环境/数据缺失，如实列出）**：

1. 真实 **qwen3-vl 30B** 服务的解析成功率、单帧耗时、断网可跑、`extra_body(think:false)` 是否被服务端支持。
2. 真实 **YOLO/ultralytics** 训练、mAP50/P/R、混淆矩阵。
3. **1 小时长视频**的漏帧率 ≤5% 与压缩比（仅有 7–8 s 合成视频）。
4. **4090D 显存**与 q8 模型的冲突（架构第 9 节待拍板项）。
5. 浏览器**目视渲染**（无 GUI）。

---

## 10. 附：本轮产物路径

- 报告：`deliverables/software-company/QA-hazard-annotation-round2.md`（本文件）
- 新增测试：`tests/test_hazard_adversarial_qa_round2.py`（21 用例）
- 修改测试：`tests/test_hazard_adversarial_qa.py:655`（N1 封闭化）
- 实测日志/中间产物：`.tmp-video-test/qa2/`（`extra.log`、`synth.log`、`final_suite*.log` 等）
