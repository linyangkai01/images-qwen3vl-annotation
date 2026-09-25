# YOLO 与 VLM 方向深度调研报告

> 生成时间：2026-09-25
> 方法：3 路并行调研（YOLO 训练侧 / VLM 与开放词汇 / 推理加速与部署）+ 实读项目代码与需求文档
> 证据标注：**【实】**=论文或官方文档已验证；**【推】**=我的推理；**【算】**=按公式推算
> 面向：本项目（工业吊装安全隐患检测冷启动）

---

## 一、项目画像（调研的约束基线）

### 1.1 任务目标

一句话：用「YOLO 初检筛候选 + 私有 VLM 复核」的两段式半自动标注，**以最低人工成本**为吊装场景从零产出可训练的 YOLO 数据集，跑通冷启动训练与评估闭环，**交付吊装场景的 YOLO 检测权重**。

**本期只做吊装单场景**（高空素材缺失，已决策不做）。
**不交付隐患结论报告** —— 隐患语义标注降级为内部手段，用于约束模型输出空间、提升框标注质量。

### 1.2 数据特点（决定一切方法的上限）

```
源素材：960×540 / 30fps，珠海 4 段视频 31.8 分钟；宿州照片 31 张（4096×3072）
抽帧：415 帧（非均匀关键帧提取，平均间隔约 4.6 秒）
标注：323 帧 / 1553 个框（Qwen3-VL 自动标注，含噪声）
数据集：292 条（train 173 / val 49 / holdout 9）
类别：12 类，极度长尾
      person 281 / helmet 223 / vest 194 / overhead_crane 81 / load 54
      balance_beam 42 / rigging 39 / hook 35 / spreader 15 / pallet 2
      另 2 类（hopper / ebike）零样本
小目标：helmet 短边中位 10.8px（96% 的框 <32px），vest 25.6px
```

**三个致命约束**：① **292 条样本**；② **1:140 长尾**；③ **helmet 10.8px 小目标**。

### 1.3 现有方案与当前指标

```
链路：抽帧 → yolo11n 初检筛候选 → Qwen3-VL-30B-A3B-Thinking 复核标注 → 过滤 → 训练 → 评估
初检：yolo11n（COCO 闭集），415 → 323 候选（657 框）
VLM 复核：1553 框（VLM 是 YOLO 的 2.4 倍），15.9 小时
训练：yolo11n，50 epoch，10.6 分钟
指标（31 帧分层评估集）：
  balance_beam mAP50 0.764 / person 0.667 / spreader 0.083 / 其余 7 类 ≈ 0
```

**已知失败点**：
- 初检筛不掉任何帧（**322/323 = 99.7% 进复核队列**）→ 两段式降本假设失败
- VLM 标注占全链路耗时 **97.5%**
- hook/rigging 四项排查全否定（不是混淆 / 阈值 / 损失权重 / 尺寸）→ **样本量不足 + 类内方差大**

### 1.4 算力与延迟要求（关键：这两条决定了方法取舍）

```
训练/离线：lab-server，单张 RTX 4090D 24GB，Ubuntu 24.04，工作区 /data/users/lyk
延迟要求：【实·需求文档 1.3.7】不做模型部署、推理服务化、实时告警
          → 本期无延迟约束，纯离线批处理
约束：不接公有云 API；真实素材不出网；主链路零新增第三方依赖
```

**⚠️ 这条直接决定了：所有推理加速/量化/蒸馏方法本期价值极低。** 详见第五节。

### 1.5 验收口径

三档：A 档（短边≥100px）mAP50≥0.70 且 R≥0.80；B 档（32-100px）≥0.50/≥0.65；C 档（<32px）≥0.35/≥0.55。**必须逐类报告，不允许只给平均值。**

---

## 二、结论摘要

### 2.1 三句话结论

1. **本项目的瓶颈不在推理速度，而在数据质量与规模。** 推理环节占全链路耗时 **<0.1%**（415 帧 × yolo11n 约 3~6 秒，而 VLM 标注 15.9 小时）。所有加速/量化/蒸馏手段本期投入产出比近乎为零。

2. **YOLO 侧最高性价比的三件事**：**P2 检测头 + Soft-NMS + 稀有类离线过采样**。P2 与 Soft-NMS 都是零/低计算开销，Soft-NMS 甚至不增加 FLOPs（+0.050 mAP50【实】）。

3. **VLM 侧最高价值的一件事**：用 **YOLOE（Ultralytics 内置，视觉提示模式）** 替换 yolo11n 做初检——它能用 2~3 张示例框定义"吊钩/平衡梁"这种**语言难以描述**的工业类别，这是文本提示类模型做不到的。

### 2.2 我自己判掉的三个矛盾（调研结果互相冲突）

**矛盾 1：SAHI 切片 —— 一处建议 P0，但我要降级**

调研给出 VisDrone 上 AP50 **25.8→38.5** 的亮眼数字。但：

- VisDrone 原图是 **2000×1500 级航拍图**，切成 640 patch 后目标**相对被放大**
- 我们是 **960×540**，本身就不大。切片到 480 再上采样到 640，放大倍数约 1.33×
- 而**直接把 imgsz 提到 1280**（原图 960×540 上采样）→ helmet 从 7.2px 变成 14.4px，**同样是 2×，且实现成本低一个数量级**

**我的判断：imgsz 提升优先，SAHI 切片降为"二级选项，需消融"。** 切片的真实额外收益（背景解耦、样本增广）对 292 条有价值，但会切断 person/overhead_crane 等大目标，**必须做消融才能定**。

**矛盾 2：co-teaching —— YOLO 侧说"不适用"，VLM 侧说"最贴合"**

- YOLO 侧理由：292 张不支撑双网络分歧筛选
- VLM 侧理由：Per-Object Co-teaching 在 KITTI 上 **31.12→46.61（+15.5）**【实】

**我的判断：给"中"可行性，理由是规模差 24 倍。** KITTI 实验用了 7000+ 张，我们 292 张。但"两个模型互选干净样本"本质是**过滤噪声**而非学新东西，在 292 张上可能仍能工作。**必须先做 30 分钟的最小验证**（见 4.5）。注意它的关键设计：**刻意排除 objectness，只用 ℓ_box + ℓ_cls** —— 这是它能抗伪标签噪声的原因。

**矛盾 3：换 Instruct —— 官方数据支持，但社区实测有反向证据**

- Qwen3-VL 技术报告【实】：grounding/OCR 类任务 Instruct 与 Thinking 相当，Instruct 局部更优
- 社区实测【实】：30B-A3B 的 Instruct 版在**空间逻辑/小目标 grounding 上明显弱于 Thinking**

**我的判断：不能直接换，必须先跑 20 帧 A/B。** 我们的任务正是 grounding，恰好落在社区实测的短板区。提速是确定的（输出 token 降一个量级），但**质量是否下降未知**。而且当前端点 403，**这件事现在做不了**。

---

## 三、方法对比表

### 3.1 YOLO 侧

| 方法 | 核心思路 | 适配本项目场景 | 预期收益 | 实现成本 | 依赖与风险 | 可行性 |
|---|---|---|---|---|---|---|
| **P2 检测头** | 增加 stride=4 的高分辨率特征层，专治小目标 | helmet 10.8px、vest 25.6px 正对口 | **+0.036 mAP50**【实·SOD-YOLO VisDrone】 | 低：官方 `yolov8-p2.yaml` 或抄 yaml | 算力+30%、显存↑；4090D 24G 无压力 | **高** |
| **Soft-NMS** | 用分数衰减代替硬抑制，缓解密集小目标互相压制 | 吊装场景人/吊物/索具密集 | **+0.050 mAP50，零 FLOPs**【实】 | 极低：后处理参数 | 需确认 Ultralytics 接口支持方式 | **高** |
| **稀有类离线过采样 2~3×** | 复制含稀有类的图片，原图保留 | spreader 15 / hook 35 / pallet 2 | **AP_small +6%**【实·Kisantal】 | 低：脚本复制 + 改 data.yaml | 4× 反而掉点，且大目标会降 | **高** |
| **BACL 式两阶段**（冻结 backbone，只重训 head） | 解耦表征与分类器：不平衡不影响表征，只重平衡分类头 | 1:140 长尾，8 类为 0 | LVIS rare AP **1.1→16.0（+14.9）**，frequent **零损失**【实】 | 中：`freeze=` 参数即可，无需改代码 | 原论文是两阶段架构，迁移到 YOLO 单阶段绝对值不可直接套 | **中高** |
| **imgsz 960→1280** | 提高推理分辨率，直接放大目标 | helmet 从 7.2px → 14.4px【算】 | 小目标最直接杠杆 | 极低：一个参数 | 173 张下过拟合风险↑，需配早停 | **高** |
| **小数据集 recipe** | lr0=0.001、freeze=10、warmup 3~5、patience 30、epochs 150~300 | 292 条 | 工程经验，无对照数字 | 极低 | 来源为工程博客，非同行评审 | **高**（低成本试） |
| **TTA** | 测试时多尺度/翻转增强后融合 | 离线无延迟约束，正好可用 | COCO mAP **+1.2**，AP_small **+1.0**【实】 | 低 | 推理成本 2~3×，但我们没延迟约束 | **中高** |
| **YOLO11s/m 或 YOLO26s** | 换模型规格 | 33 模型基准：小数据集上 **x 系列反而过拟合** | 见左 | 极低 | YOLO26 NMS-free 与现有 eval 假设不同 | **中** |
| **卸载 albumentations** | 官方隐藏坑：只要装了就强制叠加模糊 | 对 10.8px 的 helmet 是负向的 | 避免掉点 | 极低 | 无 | **高**（必做） |
| **SAHI 切片微调+推理** | 切 patch 训练与推理 | ⚠️ 我们是小图，收益可能远小于 VisDrone | VisDrone AP50 **25.8→38.5**【实】 | 中：需切图+改数据 | **会切断大目标**；必须消融 | **中** |
| **mosaic 0.5 + close_mosaic 15** | mosaic 把 4 图拼进 640，等于每张子图仅 320² | 与"放大目标"直接冲突【推】 | 推断正向 | 极低 | 无实测 | **中**（试） |

### 3.2 VLM 侧

| 方法 | 核心思路 | 适配本项目场景 | 预期收益 | 实现成本 | 依赖与风险 | 可行性 |
|---|---|---|---|---|---|---|
| **YOLOE（视觉提示）替换初检** | 用 2~3 张示例框定义类别，不需语言描述 | **吊钩/平衡梁/索具这类工业件语言难描述，视觉提示正对口** | LVIS 26m **35.4 mAP**；初检框数 657→1200~1500【推】 | 低：Ultralytics 内置 `YOLOE` | ⚠️ 首次文本提示会**联网拉 254MB CLIP** → 必须用视觉提示/prompt-free 回避 | **高** |
| **换 Qwen3-VL Instruct** | 去掉 thinking，输出 token 降一个量级 | 15.9h → 1.5~3h【推】 | **15.9h → 1.5~3h** | 低（改 model 名） | ⚠️ 社区实测 Instruct 在 **grounding 上弱于 Thinking**；且当前端点 403 | **中**（需 A/B） |
| **OVD 框作提议喂 VLM** | 不让 VLM 全图找，只验证候选框 | 搜索空间与输出 token 同时下降【推 −50% token】 | 提速 + 提召回 | 中：改 prompt 流程 | 会继承 OVD 漏检 | **中高** |
| **Per-Object Co-teaching** | 两个 YOLO 用 anchor 级选择损失互选干净样本，**刻意排除 objectness** | 伪标签噪声正对口 | KITTI **31.12→46.61（+15.5）**；掺 10% 真标 → 57.97【实】 | 中：~100 行 loss mask | **无官方代码**；KITTI 7000+ 张 vs 我们 292 张 | **中**（先验证） |
| **双源 IoU 一致性过滤** | YOLOE 框 vs VLM 框，一致的入库，冲突的进人工队列 | 无需人工即可给伪标签打质量分 | 提供噪声分 | 低 | 需两个来源都检出 | **高** |
| **主动学习选帧送人工** | 只人工精标"两源不一致 + 低置信"的帧 | 极小数据集 | 10~25% 精标带来 **+11~+31 mAP**【实】 | 低 | 需人工预算 | **高** |
| **cleanlab.object_detection** | 置信学习找标签问题 | 1553 个伪标签框 | 找错标/漏标 | 低 | 需要模型预测，非纯无监督 | **中** |
| DINO-X / Grounding DINO 1.6 | 最强开放词汇 | — | COCO 56.0 AP【实】 | — | **⛔ 只有云 API，违反不出网约束** | **不可行** |
| OWLv2 当主力 | 开放词汇 | — | 零样本 KITTI 仅 32.34 mAP【实】 | — | 零样本弱 | **低** |
| VLM 直出框当最终 GT | — | — | ODinW-13 SOTA 仅 **48.6 mAP**【实】 | — | 不够当 GT | **不可行** |

### 3.3 推理加速 / 量化 / 压缩

| 方法 | 核心思路 | 本期价值 | 触发条件 | 可行性 |
|---|---|---|---|---|
| **批处理参数写对**（batch/stream/imgsz） | `predict(source=dir, imgsz=960, batch=32, stream=True)` | 3~5 秒/轮 | 现在 | **高**（10 分钟） |
| **TensorRT FP16 导出** | FP16 推理 | 2~4× | 确定部署到 Jetson/工控机 | **中**（未来） |
| **INT8 PTQ** | 训练后量化 | 1.5~3.3×，mAP 掉 3~7%【实】 | 边缘实时 <25FPS | **中**（未来，且**必须逐类校验 helmet**） |
| **QAT** | 量化感知训练 | 挽回 0.030~0.044 mAP【实】 | 仅当 PTQ 掉点>2 且模型≥s | **低**（nano 上 QAT≈PTQ） |
| **剪枝 / 结构重参数化** | 压缩参数量 | 1.3~1.7×，AP50 −2.7【实】 | — | **不做**（长尾尾部风险） |
| **知识蒸馏** | 教师软标签 | +0.5~2 mAP | — | **不做**（教师自己 8 类为 0，会稀释尾部） |
| **多卡并行** | — | 415 帧用不上 | — | **不做** |

---

## 四、按优先级排序的落地建议（含最小验证方案）

### P0-1 · P2 检测头 + Soft-NMS（收益最高、成本最低）

**核心思路**：P2 是 stride=4 的高分辨率特征层，直接提升小目标定位能力；Soft-NMS 用分数衰减代替硬抑制，缓解密集小目标互相压制。

**适配场景**：helmet 10.8px、vest 25.6px 是本项目最难的点，且吊装画面里人/吊物/索具密集。

**预期收益**：P2 **+0.036 mAP50**；Soft-NMS **+0.050 mAP50 且零 FLOPs**；合计（SOD-YOLO 全套）mAP50 0.436→0.526【实·VisDrone】。

**实现成本**：低。Ultralytics 官方已有 `yolov8-p2.yaml`；YOLO11 官方无 p2，社区有 Yolov11-P2-Head（只抄 yaml）。Soft-NMS 是后处理参数。

**依赖与风险**：P2 使算力 +30%、显存上升（4090D 24G 无压力）；anchor 尺寸分布偏移需确认 TAL 自适应。

**可行性：高**。收益有论文实测、成本是改 yaml + 参数、无外部依赖。

**最小验证方案（约 40 分钟）**：
```
基线：yolo11n，imgsz=640，50 epoch → 记录逐类 mAP50
对照：yolo11n + P2 yaml，同参数 → 记录逐类 mAP50
重点看：helmet / vest 两类的 mAP50 变化（这是目标类）
再叠加 Soft-NMS，看是否有额外增益
判据：helmet 或 vest 的 mAP50 有任何提升即采纳；若下降则放弃 P2
```

### P0-2 · 稀有类离线过采样 2~3×（含原图）

**核心思路**：复制含稀有类（spreader / hook / pallet / balance_beam）的图片 2~3 倍，**原图必须保留**。

**适配场景**：spreader 15 框、pallet 2 框，当前为 0。

**预期收益**：AP_small **+6%（2×）→ +7.1%（原图+增强图混合 p=0.5 最优）**【实·Kisantal COCO】。**注意 4× 反而掉点，且大目标会持续下降。**

**实现成本**：低。脚本复制图片 + 重建 data.yaml，零新增依赖。

**依赖与风险**：会轻微损伤头部类；pallet 只有 2 框，过采样后仍预期 ≈0，**建议单列"不可学"并如实披露**。

**可行性：高**。

**最小验证方案（约 30 分钟）**：
```
构造：含稀有类的图复制 2×（原图保留）→ 新 data.yaml
训练同参数，对比：spreader / hook / rigging 的 mAP50，
同时确认 person / balance_beam 没有明显下降
判据：稀有类任一类 mAP50 从 0 变为 >0.1 即采纳
```

### P0-3 · imgsz 提升 + mosaic 下调 + 卸载 albumentations

**核心思路**：目标太小，最直接的办法是**放大**。

**适配场景**：960×540 原图，helmet 在 imgsz=640 下只有 7.2px，提到 1280 变成 14.4px【算】。

**预期收益**：推断为小目标最强杠杆（比 SAHI 切片更直接）。

**实现成本**：极低，全是参数。

**关键风险（必须做）**：**卸载 albumentations** —— Ultralytics 官方行为：只要环境装了它，即使所有参数置 0，仍会强制叠加 blur / median blur / grayscale / CLAHE（各 p=0.01），**且无任何参数可关**。对 10.8px 的 helmet，模糊是纯负向的。

**可行性：高**。

**最小验证方案（约 1 小时）**：
```
先确认 pip list 里有没 albumentations，有就卸载
跑 3 组：imgsz=640 / 960 / 1280，各 50 epoch
对比：逐类 mAP50，重点 helmet / vest，同时看 person 是否因过拟合下降
判据：取 helmet+vest 综合最优的 imgsz
```

### P1-1 · YOLOE 视觉提示替换 yolo11n 做初检

**核心思路**：YOLOE 支持**文本提示 / 视觉提示 / prompt-free** 三种模式。视觉提示可以用 2~3 张示例框定义类别。

**适配场景（这是它最大的价值）**：吊钩、平衡梁、索具这类工业件**用语言很难准确描述**（"hook" 在 COCO 语义里是衣钩/挂钩），但给 2~3 张示例图就明确了。这是文本提示类模型（YOLO-World / Grounding DINO）做不到的。

**预期收益**：LVIS minival 26m **35.4 mAP**【实】；推理 4~8GB VRAM，4090D 无压力。初检框数 657 → 1200~1500【推】。

**实现成本**：低，Ultralytics 内置 `YOLOE`。

**依赖与风险（关键）**：
- ⚠️ **首次文本提示会联网拉取 254MB CLIP** —— 违反"不出网"约束。**必须用视觉提示或 prompt-free 路径回避**，或预置 CLIP 权重
- ⚠️ prompt-free 模式使延迟 **+89%**，且 4585 词表与我们的类别不匹配 → **不用**
- ⚠️ **必须清醒**：初检"框数够了"≠"能筛帧"。99.7% 进复核的根因是筛选用了"有没有目标"这个几乎恒真的判据，**换 OVD 也一样会全量通过**。要真正降本，必须改筛选判据本身（见 P1-3）

**可行性：高**。

**最小验证方案（约 2 小时）**：
```
对全部 415 帧跑 YOLOE-26m（视觉提示，每类给 2~3 张示例框）
统计：12 类各自的检出框数，与 VLM 的 1553 框对比召回
重点看：hook / rigging / spreader / balance_beam 这 4 类（当前 YOLO 检不出）
判据：若这 4 类能被稳定检出 → 采纳为初检器
注意：全程断网验证，确认没有联网拉 CLIP
```

### P1-2 · BACL 式两阶段（冻结 backbone 只重训 head）

**核心思路**：长尾问题的本质不是表征学不好，而是分类器被头部类主导。**先正常训全模型，再冻结 backbone+neck，只重训检测头**，同时对稀有类做类平衡采样。

**适配场景**：1:140 长尾，8 类为 0，而 person/balance_beam 已经能学到 0.667/0.764 —— 说明**表征是好的，只是头偏置了**。

**预期收益**：LVIS 上 rare AP **1.1→16.0（+14.9）**，frequent **30.9→30.9 零损失**【实·BACL CVPR】。

**实现成本**：中，但**纯 API 参数**（Ultralytics `freeze=`），无需改代码。

**依赖与风险**：原论文是 Faster R-CNN 两阶段架构，**迁移到 YOLO 单阶段的绝对值不可直接套用**，只能作为方向性证据。原论文的 FCBL 损失 + FHM 模块需要改损失，本项目不可直接套用。

**可行性：中高**。

**最小验证方案（约 1.5 小时）**：
```
Stage1：正常训 50 epoch（已有 best.pt 可复用）
Stage2：freeze=backbone+neck 层数，lr 降 10×，
        训练集换成「稀有类 3× 过采样」版本，训 20~30 epoch
对比：spreader/hook/rigging 的 mAP50 是否从 0 起来，
      person/balance_beam 是否保持
判据：稀有类起来且头部类掉点 <0.05 → 采纳
```

### P1-3 · 改初检"筛选判据"（真正解决 99.7% 进复核）

**这是我自己加的一条，调研里没直接给，但它才是降本的根因。**

**问题**：当前判据是「这帧有没有检测到目标」。吊装作业画面里几乎每帧都有人和设备 → **判据恒真 → 筛不掉任何帧**。换任何更准的检测器都救不了，因为**判据错了，不是检测器错了**。

**改法**：筛选判据应该是「这帧**有没有不确定性**」，而不是「有没有目标」：
- 帧间差异大（新目标出现 / 状态变化）
- 该帧的类别组合少见（首次出现的组合）
- VLM 与初检器**不一致**的帧

**可行性：中**（需要先有 P1-1 的双源才能做一致性判据）。

### P2-1 · Per-Object Co-teaching（先验证，别直接上）

**核心思路**：两个随机初始化的 YOLO，用**同伴计算的 anchor 级选择损失（ℓ_box + ℓ_cls，刻意排除 objectness）**互相筛选干净样本。这个"排除 objectness"的设计是它能抗伪标签噪声的关键。

**适配场景**：我们的标签是 VLM 伪标签，含噪声 —— 正对口。

**预期收益**：KITTI **31.12→46.61（+15.5）**；掺入 10% 真标 → **57.97**，25% → 77.80【实】。

**实现成本**：中，~100 行 loss mask + 课程学习 warmup。**无官方代码**。

**依赖与风险（我判的）**：KITTI 实验用 7000+ 张，我们 **292 张，差 24 倍**。这是最大的不确定性。

**可行性：中** —— 收益诱人但规模不匹配，必须先验证。

**最小验证方案（约 3 小时）**：
```
用当前 292 条训两个 yolo11n（不同种子）
实现 anchor 级选择损失（排除 objectness），互选 Top-K 干净样本
在**评估结果**上对比：是否比单模型基线好
先只看 hook/rigging/spreader（噪声最可疑的类）
判据：若两个模型的一致性筛选能把这 3 类的 mAP50 拉起来 → 投入；否则放弃
```

### P2-2 · 换 Qwen3-VL Instruct（需 A/B，且当前阻塞）

**预期收益**：15.9h → 1.5~3h【推】。

**风险（我判的）**：官方说 grounding 上 Instruct 与 Thinking 相当甚至更优，但**社区实测 30B-A3B 的 Instruct 在空间逻辑/小目标 grounding 上明显弱于 Thinking**。我们的任务正是 grounding，恰好落在短板区。

**⛔ 当前阻塞**：端点 403，做不了。

**最小验证方案（端点恢复后，约 1 小时）**：
```
抽 20 帧（覆盖各类，含小目标密集帧）
分别用 Thinking 和 Instruct 标注
对比：框数量、框 IoU 一致性、类别准确率、小目标（helmet/vest）检出数
判据：若 Instruct 的小目标检出数不低于 Thinking → 换；否则不换
```

### P3 · 推理加速（现在只做一件事）

**现在该做（10 分钟）**：推理脚本用 `predict(source=dir, imgsz=960, batch=32, stream=True)`，并在评估脚本里记一行 `time.perf_counter()` 留 baseline。同时**保留固定 val 集作为未来 INT8 校准集**。

**未来该做（触发条件驱动）**：确定部署到 Jetson/工控机 → TRT FP16（2~4×）；边缘实时 <25FPS → INT8 PTQ。

**永远不做**：剪枝、蒸馏、NCNN/OpenVINO/TFLite、多卡并行、nano 上的 QAT。

**⚠️ 一个比硬件更重要的成本项**：**Ultralytics 是 AGPL-3.0**，工厂商用部署需要 Enterprise 商用授权。这比硬件更可能是真实支出，建议提前确认。

---

## 五、明确「不做」清单（含理由）

| 方法 | 不做的理由 |
|---|---|
| **推理加速 / 量化 / 蒸馏 / 剪枝** | 本期不做部署（需求 1.3.7），推理占全链路 **<0.1%**。触发条件未满足 |
| **Ultralytics `copy_paste`** | 官方要求 polygon 标签，**纯 bbox 的 detect 不生效** |
| **置信度校准（Temperature/Platt）** | mAP 按置信度排序计算，**单调变换不改变排序 → 对 mAP 数学上零增益** |
| **OHEM / 难例挖掘** | ①Ultralytics 无实现；②标签是伪标签，挖掘会**系统性放大噪声** |
| **知识蒸馏** | 教师模型自己 8 类为 0，软标签在长尾类上全是噪声；COCO 预训练权重已完成"迁移"职能 |
| **DINO-X / Grounding DINO 1.6** | **只有云 API**，违反"不出网/不接公有云"硬约束 |
| **多视角 3D 重建（VGGT / DUSt3R）** | 输出**尺度歧义（非米制）**；VGGT 明确败于大面积运动物体（吊物正是）；无 3D 真值可验证 |
| **单目深度估计** | 信噪比约 1:1，误差与 2D 投影歧义同量级 → **净负担** |
| **改损失函数（Seesaw/EQLv2）** | 无任何 Ultralytics 适配实现，需改 `v8DetectionLoss`；对 292 条收益 < 数据侧 |
| **VLM 直出框当最终 GT** | ODinW-13 SOTA 仅 48.6 mAP，不够 |
| **hook/rigging 继续排查** | 四项已全否定（混淆/阈值/损失权重/尺寸）→ **样本量不足 + 类内方差大**，要么补数据要么合并类 |

---

## 六、风险与依赖汇总

**阻塞项**
- **VLM 端点 403**（出口 IP `211.142.244.134`）→ 所有需要重新标注的工作都做不了。需老总协调确认白名单/token

**合规风险**
- **Ultralytics AGPL-3.0**：商用部署需 Enterprise 授权
- **YOLOE 首次文本提示联网拉 CLIP**：必须用视觉提示路径回避
- **开源/自有素材隔离**：`own/` 与 `open/` 硬隔离

**方法风险**
- 292 条样本下，任何"学习类"方法（co-teaching、蒸馏）都有过拟合与规模不匹配风险
- helmet 10.8px 可能**物理上不可分** —— 若 P2 + imgsz1280 + 过采样后仍 ≈0，应诚实判定"该分辨率下不可学"，转两阶段 ROI 方案（先检 person，再在人框内判 helmet）
- pallet 仅 2 框，**任何方法都救不了**，建议单列"不可学"

**评估风险（重要）**
- 49 张 val / 9 张 holdout 下，spreader(15 框)、pallet(2 框) 的 per-class AP **方差极大**
- "8 类 ≈0" 中有相当部分是**评估样本量不足的噪声**，而非纯模型问题
- **建议至少用 173+49 做 k-fold（如 3 折）取均值再判断**，否则 A/B 实验无法区分

---

## 七、附 · 调研来源

**YOLO 模型与训练**

- [SOD-YOLO: Enhancing YOLO-Based Detection of Small Objects (arXiv 2507.12727)](https://arxiv.org/html/2507.12727v1)
- [YOLO Evolution: A Comprehensive Benchmark and Architectural Review (arXiv 2411.00201)](https://arxiv.org/html/2411.00201v2)
- [YOLO26: Key Architectural Enhancements (arXiv 2509.25164)](https://arxiv.org/html/2509.25164v5)
- [YOLOv13 (arXiv 2506.17733)](https://arxiv.org/html/2506.17733v1)
- [Augmentation for small object detection (Kisantal et al., arXiv 1902.07296)](https://ar5iv.labs.arxiv.org/html/1902.07296)
- [BACL: Balanced Classification for Long-Tailed Object Detection (arXiv 2308.02213)](https://ar5iv.labs.arxiv.org/html/2308.02213)
- [SAHI (arXiv 2202.06934)](https://ar5iv.labs.arxiv.org/html/2202.06934)
- [Ultralytics 数据增强指南](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [Ultralytics TTA 教程](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)
- [Ultralytics Issue #6201: 小数据集训练最佳实践](https://github.com/ultralytics/ultralytics/issues/6201)
- [ultralytics yolov8-p2.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)
- [Yolov11-P2-Head](https://github.com/YingXiong47/Yolov11-P2-Head)

**VLM 与开放词汇检测**

- [YOLOE 文档 (Ultralytics)](https://docs.ultralytics.com/models/yoloe/)
- [YOLO-World 文档 (Ultralytics)](https://docs.ultralytics.com/models/yolo-world/)
- [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [DINO-X (arXiv 2411.14347)](https://arxiv.org/html/2411.14347v1)
- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/html/2511.21631)
- [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Per-Object Co-teaching for VLM Pseudo Labels (arXiv 2511.09955)](https://arxiv.org/html/2511.09955)
- [autodistill](https://github.com/autodistill/autodistill)
- [obss/SAHI](https://github.com/obss/sahi)
- [cleanlab.object_detection API](https://docs.cleanlab.ai/stable/cleanlab/object_detection/index.html)
- [3F-OVD / NEU-171K 细粒度工业 OVD (arXiv 2503.14862)](https://arxiv.org/abs/2503.14862)

**推理加速与部署**

- [Ultralytics 导出模式文档（22 种格式 / quantize / QAT vs PTQ mAP 表）](https://docs.ultralytics.com/modes/export/)
- [Ultralytics 预测模式文档（batch / stream / quantize）](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics NVIDIA Jetson 指南（Orin 实测 ms/im 与 mAP 表）](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [Quantization Robustness for Object Detection (arXiv 2508.19600)](https://arxiv.org/abs/2508.19600)
- [Mask-Guided Local Feature Quantization for Small Object Detection (IET)](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.70476)
- [YOLOs-CPP-TensorRT Benchmarks（RTX 2000 Ada 实测）](https://github.com/Geekgineer/YOLOs-CPP-TensorRT/blob/main/benchmarks/README.md)
- [NVIDIA Jetson Orin Nano Super 官方页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
