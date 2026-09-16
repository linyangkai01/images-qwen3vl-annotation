# 工业安全检测训练闭环 — 改造现状核对与路线图

> 生成时间：2026-09-16
> 核对对象：上游仓库 `huazi32/images-qwen3vl-annotation`、51CTO 博客（代码天地，2026-01-29）、本地工作区 `D:\WorkPlace\images-qwen3vl-annotation`

---

## 一、三份资料的关系（先把基线说清楚）

### 1.1 上游仓库 `huazi32/images-qwen3vl-annotation`

- **定位**：智能交通路口的 Qwen3VL → YOLO 闭环训练工程
- **链路**：视频抽帧 → YOLO 初检 → Qwen3VL 二次复核 → YOLO 数据集生成 → 增量训练 → 困难样本回流
- **三层架构**：
  - 技能层 `skills/`：`video-frame-extraction`、`images-qwen3vl-annotation`、`yolo-closed-loop-training`
  - 业务层 `traffic_cv_closed_loop/`：`cli.py`、`config.py`、`models.py`、`io_utils.py`、`diagnostics.py` + `pipeline/` + `adapters/`
  - 集成层 `docs/`、`examples/`、`scripts/`
- **场景覆盖**：**只有交通**（person / car / bus / truck / motorcycle / traffic_light），**零工业安全内容**
- **提交历史**：仅 2 次（2026-07-16 初始化 + 文档与许可），未做后续功能迭代

### 1.2 51CTO 博客（代码天地，2026-01-29）

- **讲的是方法论**：用 Agent Skills 把 Qwen3VL → YOLO 闭环串起来
- **三个 Skill 的分工**：
  1. `video-frame-extraction`：YOLO 检测 + 抽帧
  2. `images-qwen3vl-annotation`：Qwen3VL 对帧做二次检测；**当 Qwen 检出数 > YOLO 检出数时，把差异帧导出成 YOLO dataset**
  3. `yolo-closed-loop-training`：用差异集微调原模型
- **核心观点**：Skill 是"个人/团队知识上下文"的载体，避免重复写提示词与重复调试
- **重要提醒**：博客作者明确说**闭环训练效果还没跑完**（"再等我跑完当前的历史视频…训练时间比较长"）。所以这份资料只能当**方法参考**，不能当已验证的效果背书

### 1.3 本地工作区（当前实际状态）

**不是从零开始。** 本地已经在上游基础上做了大量改造，改造进度约 **85%**。

---

## 二、本地已经完成了什么

### 2.1 交通链路：保留且零回归

- `traffic_cv_closed_loop/` 未做破坏性改动
- 交通三模块测试单独复跑：`Ran 9 tests` → `OK`（退出码 0）

### 2.2 新增工业施工安全隐患链路 `hazard_annotation/`（入口 `python -m hazard_annotation`）

- `__main__.py`：CLI，支持 `extract / annotate / aggregate / train / visualize / all / status / doctor / probe-qwen / demo`，含 `--mode quick|full`、`--mock`
- `keyframes.py`：**六步关键帧提取**（粗采样 → 运动过滤 → 去重 → 场景切换 → 目标强制保留 → 数量/间隔约束）
- `imageio.py`：纯 Python Netpbm 解析 + aHash/直方图/差分 + PNG 编码（**无 numpy / OpenCV / Pillow**）
- `ffmpeg.py`：ffmpeg CLI 薄封装 + 版本自检（最低 5.1）
- `knowledge.py`：知识库归一化 + 压缩注入 + 场景路由
- `pipeline/`：`extraction.py` / `annotation.py`（双通道）/ `aggregate.py` / `training.py` / `visualization.py`
- `prompts/`：combined / objects_only / hazard_only / scene_route 四套提示词

### 2.3 新增知识库 `hazard-detection/`

- `knowledge_base.json`：**58 条隐患**（S1 吊装 27 + S2 高处 30 + S3 通用 1）
- `SKILL.md`、`detection_examples.md`、`prompt_templates.md`、`规范化施工隐患点分类整理（优化文档）.md`

### 2.4 双通道设计（这是相对上游最核心的改造）

- **通道 A（目标框）**：12 类施工对象 → `review/labels/*.txt`
  - 类别：`person helmet vest harness hook rigging overhead_crane load ladder scaffold guardrail pallet`
  - 关键设计：**不设 `no_helmet` 这类"缺失态"负类**，PPE 缺失由"person 框与 helmet/harness 框的空间关联 + 知识库规则"推断
- **通道 B（隐患事件）**：`hazards/*.json`，字段含 `id / category / severity / norm_reference / action / evidence`
  - 关键设计：模型只输出 `id` + `evidence`，`category / norm_reference / action` 由系统从知识库**本地回填**，杜绝模型编造规范号

### 2.5 工程配套

- `examples/hazard_mock_project/`（mock，一键跑通）+ `examples/hazard_project/`（真实模式模板）
- `docs/hazard-annotation-chain.md`（链路说明）、`docs/class-diagram.mermaid`、`docs/sequence-diagram.mermaid`
- `deliverables/software-company/`：PRD + ARCH + QA×3 + norm_ref 待补清单 + 临时文件清理清单
- 测试：**179 个用例，1 个失败**（唯一失败即 F1），0 errors / 0 skipped
- 可视化：自包含静态 HTML（base64 缩略图 + SVG 图表 + 场景 Tab + mock/真实标识）
- 一键 mock：`python -m hazard_annotation demo --mock` → 退出码 0

---

## 三、距离"工业安全检测训练闭环"还差什么

### P0 — 阻断级（不做完，闭环不成立）

**P0-A　F1：知识库规范依据缺口 —— ✅ 已解决（2026-09-16）**

- 原状：58 条隐患中仅 **11 条**带 `norm_ref`，覆盖率 **18.97%**；通道 B 必填字段完整率仅 **27.27%**，PRD 要求 ≥ 95%
- 处置：生成候选依据清单（`norm_ref-待补清单-候选建议.md`，47 条带置信度标注）+ 合并工具（`scripts/fill_norm_ref.py`），
  一键写入后覆盖率 **11/58 → 58/58（100%）**，`meta.total_hazards` 同步由 53 修正为 58
- 验证：`PrdAcceptanceFindingsTest` 全部通过；`test_finding_demo_required_field_completeness_meets_prd` 转绿
- 回滚：原文件已备份为 `hazard-detection/knowledge_base.json.bak-<时间戳>`
- **遗留提醒**：候选标准号中仅 JGJ 276-2012、JGJ 80-2016、GB/T 17889.1-2021 经联网核验；
  其余按标准体系推断，**建议逐条复核版本年号**（`norm_ref` 会用于下整改单，需准确）

**P0-B　ffmpeg 当前不可用（真实视频抽帧链路跑不了）**

- 实测：`PATH` 无 ffmpeg；conda `hazard` 环境**已不存在**（`D:\working\anaconda3\envs\` 下只剩 `vsa-agent`）；base 环境无 ffmpeg；无 winget 安装痕迹
- 影响：六步关键帧提取的**真实视频路径**（`coarse_sample` / `extract_indices` / `scene_cut_indices`）目前无法执行；只能跑图片目录模式与 mock
- 注：链路本身已实现且单测通过，缺的只是**运行时依赖**

**P0-C　无版本控制**

- 实测：工作区**没有 `.git`**
- 影响：改造过程不可回溯、无法 `git diff`（QA Round 3 已因此被迫"凭记忆逐行比对"）；也影响你后续往 GitHub 推改造版

### P1 — 重要（影响闭环成色）

**P1-A　真实模型服务未接入**

- PRD 第 6 节 10 个待确认问题**基本未答**，其中最关键的 3 个：
  - `qwen3-vl:30b-a3b-thinking-q8_0` 自部署服务的 `base_url` / 端口 / 路径是什么？
  - 是否需要凭据（token / 内网白名单）？
  - 服务端 GPU 显存多大，q8 的 30B MoE 能否整机驻留，单帧耗时多少？
- 现状：所有真实标注能力都**未经真实服务验证**

**P1-B　真实 YOLO 训练未验证**

- `ultralytics` / `torch` **未安装**（pyproject 里 `production` extra 声明了 `ultralytics>=8.3.0`）
- PRD 目标：mAP50 ≥ 0.60 + P/R + 混淆矩阵，**尚未产出任何真实指标**

**P1-C　闭环"不闭"——回流靠人工触发**

- 现状：`training.py` 产出 `analysis/hard_examples.jsonl` + `analysis/next_cycle_tasks.md`（含 4 条建议动作）
- 缺的是：**"回流清单 → 下一轮数据集 → 新权重 → 新旧指标对比"没有串成一条可执行命令**
- 也就是说，目前是"半自动闭环"，不是"自动闭环"

**P1-D　通道一致性校验缺失（PRD P1-3）**

- 设想：模型报"未戴安全帽"但通道 A 检出 helmet 框时，标记可疑并进复核队列，降低幻觉
- 现状：未实现

**P1-E　指标导出缺失（PRD P1-5）**

- mAP50/P/R、隐患等级分布、压缩比尚未汇总成一份可直接进论文的指标文件

### P2 — 增强

- 隐患时间聚合与持续性判定（PRD P1-1）
- 整改前后对比视图（PRD P1-2）
- 批量并发与断点续跑（PRD P1-4）
- 多机位 / 多视频批处理（PRD P2-1）
- 主动学习采样（PRD P2-4）
- **工业环境鲁棒性**：光照波动 ±40%、LED 衰减、粉尘、遮挡、小目标——目前系统未体现针对性设计
- **hazard 链路 Skill 化收口**：`skills/` 下仍只有 3 个交通 Skill，隐患链路未做成独立 Skill
- 测试卫生：全量套件耗时 **429s**（Round 2 仅 5.2s），主因 `test_hazard_pipeline` 反复跑全量 mock demo

### 已知遗留（低优先级，非阻断）

- **N5**：`_select_span` 反向边界——较晚的"合法退化答案"（裸 `{}` / `[]`）被较早 span 覆盖。对隐患流水线影响极小（答案恒为非空对象），但解析器是通用组件
- **N6**：流程问题——工程师改动了验证者测试文件，且有范围外变更未申报
- 知识库 `meta.hazard_count` 仍写 53（陈旧字段，实际 58），`doctor` 会告警

---

## 四、路线图（四闸门制）

### 闸门一：打通"真实视频 → 关键帧"

- **目标**：让六步关键帧提取的真实视频路径可用
- **动作**：
  1. 重建 ffmpeg（conda 用户态，绕开管理员权限；或其它可用方式）
  2. 合成/投喂一段测试视频，跑 `python -m hazard_annotation extract --config <cfg>`
  3. 校验 `summary.json` 的 总帧数 / 采样帧数 / 运动剔除 / 去重剔除 / 最终关键帧数
- **验收**：1 小时视频 → 300~500 关键帧；人工抽检漏帧率 ≤ 5%；所有阈值改配置即生效
- **产物**：一份真实视频抽帧实测报告

### 闸门二：接真实模型，跑通真实标注

- **前置**：闸门一 + **F1 补全**
- **需要你提供**：自部署服务的 base_url / 端口 / 路径 / 模型名 / 凭据要求 / 服务端显存
- **动作**：`doctor` → `probe-qwen` → `annotate`
- **验收**：单帧耗时可见；thinking JSON 解析成功率 ≥ 95%；`combined_call=true` 时单帧只发 1 次请求；全链路无第三方 API 调用
- **产物**：真实标注结果 + 隐患报告 + HTML 可视化

### 闸门三：跑真训练 + 闭环自动化

- **动作**：
  1. 装 `ultralytics`（约 2GB+）
  2. 跑 `train --execute-trainer`，产出 mAP50/P/R + 混淆矩阵
  3. 把"回流清单 → 下一轮数据集 → 训练 → 新旧指标对比"做成**一条命令**
- **验收**：mAP50 ≥ 0.60（或你另给的线）；critical/high 隐患帧进入困难样本清单；闭环可一键复跑
- **产物**：训练指标 + 下一轮数据集 + 对比报告

### 闸门四：工业环境鲁棒性 + 范围扩展

- **动作**：
  1. 补光照波动 / 粉尘 / 遮挡 / 小目标的样本分层与增强策略
  2. 通道一致性校验（P1-3）+ 指标导出（P1-5）
  3. 范围扩展（若需要，见第五节决策 1）
- **产物**：鲁棒性专项报告 + 论文素材

---

## 五、需要你拍板的三件事

**决策 1 — 范围**

- A：沿用现有"施工安全"（吊装 S1 + 高处 S2 + 通用 S3，58 条隐患，12 类），做深做透
- B：扩展到"工厂工业安全"（人员 PPE 违规、危险区域入侵、消防通道堵塞、机械防护、有限空间等，需新增场景与类别）
- C：分层都要（施工 + 工厂），分阶段做

**决策 2 — 闭环"闭"到什么程度**

- A：半自动（现状：产出回流清单 + 下一轮建议，人工触发下一轮）—— 毕设够用
- B：全自动（一条命令从视频跑到新权重 + 新旧指标对比 + 自动生成下一轮数据集）

**决策 3 — 下一步先解锁哪个阻断**

- A：先补 ffmpeg + 真实视频跑通抽帧（闸门一，解锁视频链路）
- B：先等你补完 47 条 norm_ref（解锁 F1 验收）
- C：先接真实 qwen3-vl 服务（闸门二，解锁真实标注）
- D：先装 ultralytics 跑真训练（闸门三，解锁 mAP 指标）

**另建议（不需拍板，我可以直接做）**：把工作区纳入 git 版本控制，让后续改造可 diff、可回溯。
