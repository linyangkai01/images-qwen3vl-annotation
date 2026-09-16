# 工业施工安全隐患自动标注链路（hazard_annotation）

> 在原有「智能交通路口 Qwen3VL → YOLO 闭环」之上新增的**双通道**标注链路：
> 通道 A 输出 YOLO 目标框（`labels/*.txt`），通道 B 输出隐患事件（`hazards/*.json`），
> 二者共同驱动数据集构建、训练与困难样本回流。

本链路**只使用 Python 标准库**（无 numpy / OpenCV / Pillow），视频解码仅依赖外部
`ffmpeg` CLI（可选，缺失时自动降级为图片目录模式）。

---

## 1. 双通道设计

| 通道 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| A 目标检测 | 关键帧 + YOLO 初检先验 | `review/labels/<source>/<frame>.txt` | YOLO 标准标注（12 类） |
| B 隐患事件 | 关键帧 + 压缩后的隐患知识 | `hazards/<source>/<frame>.json` | 隐患编号 / 等级 / 依据 / 整改建议 |

最终 12 个类别：

```text
person helmet vest harness hook rigging overhead_crane load ladder scaffold guardrail pallet
```

主模型：`qwen3-vl:30b-a3b-thinking-q8_0`，通过 OpenAI 兼容协议
（`qwen.provider = "local_openai"`，`require_api_key = false`，`base_url` 完全可配置）。

---

## 2. 包结构

```text
hazard_annotation/
├── __main__.py          # CLI 入口：python -m hazard_annotation <stage>
├── config.py            # 载入配置 + quick/full 档位覆盖
├── models.py            # HazardDef / HazardEvent / HazardReport / KeyframeRecord / DualChannelRecord
├── imageio.py           # 纯 Python Netpbm(P2/P3/P5/P6) 解析 + aHash/直方图/差分 + PNG 编码
├── ffmpeg.py            # ffmpeg CLI 薄封装（缺失时抛 FfmpegUnavailableError）
├── knowledge.py         # 知识库归一化 + 压缩注入 + 场景路由
├── keyframes.py         # 六步关键帧提取
├── diagnostics.py       # doctor / probe-qwen / probe-detector
├── demo.py              # mock 资产生成 + 一键 demo
├── prompts/             # combined / objects_only / hazard_only / scene_route 提示词
└── pipeline/
    ├── extraction.py    # 抽帧 → candidates.jsonl
    ├── annotation.py    # 双通道标注 → labels/ + hazards/ + annotations.jsonl
    ├── aggregate.py     # 跨帧聚合 → hazards/report.json
    ├── training.py      # 数据集 + 训练 + 困难样本回流
    └── visualization.py # 自包含静态 HTML 报告
```

---

## 3. 六步关键帧提取

1. **粗采样**：视频按 `sample_fps` 解码为灰度分析帧（`ffmpeg`），图片目录则直接读取。
2. **运动过滤**：帧间差分运动占比低于 `motion_min_ratio` 的帧丢弃。
3. **去重**：aHash + 汉明距离 ≤ `dedup_hamming_threshold` 视为重复。
4. **场景切换**：直方图差 / `ffmpeg scene` 检测切换点并保留。
5. **目标强制保留**：命中 `force_keep_labels`（如 hook / overhead_crane / ladder）的帧强制保留。
6. **数量与间隔约束**：`max_keyframes` 上限 + `min_gap_seconds` 最小间隔。

---

## 4. 知识库、压缩注入与场景路由

- 知识库文件：`hazard-detection/knowledge_base.json`（定义 **58** 条隐患；`meta.total_hazards` 已由 53 修正为 58，
  `norm_ref` 覆盖率 **100%**，58 条全部具备规范依据）。
- 归一化兼容两种结构：`scenes[].hazard_categories[].hazards[]`（S1/S2）与
  `scenes[].hazards[]`（S3），统一为扁平 `HazardDef` 列表 + `id_index`。
- `KnowledgeCompressor.build_brief()` 按严重度排序，生成「编号 | 名称 | 视觉线索 | 等级」
  的压缩清单（受 `kb_max_chars` 限制），S1 场景额外附带「十不吊 / 行车六严禁」红线。
- `KnowledgeCompressor.backfill()` 用 `id_index` 回填 `category / severity / norm_reference /
  action`，并标记 `invalid_id`。
- `SceneRouter` 规则优先：命中 `hook/rigging/overhead_crane/load` → S1；
  命中 `ladder/scaffold/guardrail/harness` → S2；两者都有 → S1+S2。
  多图源逐帧路由，单视频源按前 5 帧投票做视频级路由。

---

## 5. thinking 稳健 JSON 解析

`ThinkingJsonParser`：

1. 去除 ` thinking` / `<|thinking|>` / ```thinking``` 围栏；
2. 字符串感知状态机扫描，取**最后一个**平衡 JSON 片段（最终答案在推理之后）；
3. 有界修复：注释、Python 字面量（True/False/None）、尾逗号、单引号转双引号、截断闭合。

> 这修复了旧实现「取第一个 JSON 片段」导致 thinking 模型返回解析错误的缺陷。

---

## 6. quick / full 双档模式

| 项 | quick | full |
| --- | --- | --- |
| 关键帧上限 | 较少 | 较多 |
| Qwen 并发 | 1 | 提高 |
| 采样 fps | 较低 | 较高 |
| 适用 | 快速验证 / 显存紧张 | 正式生产 |

档位通过 `ProjectConfig.apply_profile(mode)` 以点路径覆盖实现，`--mode` 优先于配置内的 `run_mode`。

---

## 7. 一键 mock 演示（无需视频 / 模型 / ffmpeg）

```powershell
python -m hazard_annotation demo --mock
```

`demo` 会：生成三类场景的 mock 帧与 sidecar → 跑完
`extract → annotate → aggregate → train → visualize`。

`all --mock` 在项目缺少输入帧时也会**自动生成 mock 资产**，等价于 demo。

---

## 8. 常用命令

```powershell
# 全链路（mock）
python -m hazard_annotation all --mock --config examples/hazard_mock_project/config/project.json

# 单阶段
python -m hazard_annotation extract   --config <cfg>
python -m hazard_annotation annotate  --config <cfg>
python -m hazard_annotation aggregate --config <cfg>
python -m hazard_annotation train     --config <cfg> [--execute-trainer]
python -m hazard_annotation visualize --config <cfg>

# 诊断
python -m hazard_annotation doctor    --config <cfg>
python -m hazard_annotation probe-qwen --config <cfg> [--image <path>]
python -m hazard_annotation status    --config <cfg>
```

`doctor` 报告包含 `video`（路径 / ffmpeg 可用性）、`keyframe`（六步参数 / 场景检测后端）、
`knowledge`（知识库存在 / 隐患条数 / 场景 id）三个施工专属检查项。
`probe-qwen` 返回连通性、模型名与单帧耗时。

---

## 9. 输出产物

```text
workspace/
├── extracted/          candidates.jsonl, frames/, summary.json
├── review/             annotations.jsonl, review_queue.jsonl, labels/, raw_responses/, summary.json
├── hazards/            <source>/<frame>.json, report.json
├── datasets/<ver>/     images/, labels/, data.yaml, manifest.json
├── runs/<ver>/         metrics.json, report.json, trainer_summary.md, latest.json
├── analysis/           hard_examples.jsonl, next_cycle_tasks.md
├── visualization/      report.html   （自包含，base64 缩略图 + SVG 图表 + 场景 Tab）
└── status/             extract.json / annotate.json / aggregate.json / train.json / visualize.json
```

HTML 报告为**自包含静态文件**，浏览器直接打开即可，缩略图内联为 base64 PNG，
并带有 mock / 真实模式标识。

---

## 10. 4090D（24GB）显存注意事项

本链路**不在进程内加载模型**，只通过 HTTP 调用本地服务，因此显存占用由模型服务决定。
责任分工如下：

- **模型服务侧（Ollama / vLLM）**：负责量化与显存/内存卸载策略。
  `30b-a3b` 的 q8 权重约 30GB 量级，24GB 卡上需要部分卸载到内存；如显存吃紧，
  优先改用 q4 量化，或确保 `a3b`（激活 3B）专家卸载生效。
- **标注链路侧**：通过 `quick` 档 + `qwen_combined_call=true` + `qwen_concurrency=1`
  限制并发与单帧开销，避免多请求叠加显存峰值；关键帧数量也受 `max_keyframes` 约束。

建议流程：先 `doctor` 看配置 → 再 `probe-qwen` 看连通性与单帧耗时 → 最后跑 `all`。

---

## 11. 测试

```powershell
python -m unittest discover -s tests -v
```

施工链路的测试文件：

- `tests/test_hazard_imageio.py`：Netpbm 解析 / aHash / 直方图 / 差分。
- `tests/test_hazard_knowledge.py`：知识库归一化 / 压缩预算 / 回填 / 场景路由。
- `tests/test_hazard_parsing.py`：thinking JSON 稳健解析。
- `tests/test_hazard_pipeline.py`：mock 全链路产物、报告结构、困难样本、场景区分。
- `tests/test_hazard_diagnostics.py`：doctor 检查项与 probe-qwen 连通性/耗时。
