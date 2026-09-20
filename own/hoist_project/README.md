# `own/hoist_project` —— 吊装单场景自有资源子树

本目录是**自有资源**子树（与仓库根下 `open/` 开源资源子树**硬隔离**）。
凡「真实素材派生数据」「私有端点配置」「本场景专用提示词」都只落在这里。

## 目录结构

```
own/hoist_project/
├── config/
│   └── project.hoist.json      # 项目配置（12 类 labels / private_vlm / endpoint allowlist）
├── assets/prompts/
│   └── hoist_zh.md             # 吊装中文标注提示词（含 KB S1 注入与受控词表要求）
├── input/                      # 素材派生数据（关键帧）；不入库，见 .gitignore
└── workspace/                  # 运行产物；不入库，见 .gitignore
```

## 素材来源（不随仓库分发）

- **珠海吊装视频 4 段**：约 31.8 分钟，源分辨率 960×540、30fps → 走「视频抽帧」路径。
- **宿州吊装隐患照片 31 张**：多为 4096×3072 → 走「图片直入」路径。

> 原始素材统一存放在 `D:\WorkPlace\industrial-data\`（本机）或 `lab-server`
> 的内网路径下，**绝不复制进本仓库**；本仓库只保留配置与提示词。

## 关键约定（与架构 §7 / §10 一致）

1. **类别体系**：12 类，`class_id` 顺序即 `labels` 数组顺序，**不可重排**。
   已排除梯具/脚手架/护栏/安全带等高处作业类别。
2. **场景**：本期 `scene` 恒为 `hoist`（`height` 留位不实现）。
3. **端点护栏**：VLM 端点 `119.23.145.227:38091` 是**公网 IP**，
   `is_local_endpoint_host()` 判定为 `False`，因此**必须**用 `allowlist` 模式显式白名单，
   否则护栏会把合法端点拦死：

   ```json
   "inference": {
     "endpoint_policy": {
       "mode": "allowlist",
       "allowlist": ["119.23.145.227:38091"]
     }
   }
   ```

4. **VLM 调用从 `lab-server` 发起**：本机到端点不可达（实测 TCP 超时），
   端点调用必须落在服务器侧。

## 运行顺序

```powershell
# 本机（Windows）小样本冒烟：只探活
pwsh -File scripts/run_hoist_cycle.ps1
```

```bash
# lab-server（Linux）全量：完整链路（当前 T02–T05 未实现，会明确报「尚未实现」）
bash scripts/run_hoist_cycle.sh
```

阶段顺序：`inventory → extract-video → extract → annotate → prioritize →
curate → freeze-holdout → train → evaluate`（`doctor` 可随时单独跑）。
