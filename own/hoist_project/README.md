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
5. **原始素材目录约定（`media_root`）**：配置里没有「原始素材目录」字段，
   编目（`inventory`）默认扫描 **`input_frames_dir` 同级的 `media/`**，
   即 `own/hoist_project/input/media/`。之所以这样定：让「原始素材」与
   「抽帧产物（`input/frames/`）」分离，互不污染，且不必改动配置结构。
   如需指向别处（如挂载盘），用 `run_inventory(config, media_root=...)`
   或 CLI 的 `--config` 指向的工程里调整；**切勿把真实素材复制进仓库**。

## 抽帧产物：PNG 是交付物，PGM 是中间产物

`extract-video` 阶段对每段视频按 `video.sample_fps` 抽帧，每帧产出两个文件：

- `frame_<i>_<ts>.png` —— **关键帧交付物**（按 `video.scale_width` 缩放），
  写入 `input/frames/<source_id>/`，供母项目 `run_extraction` 消费；
- `frame_<i>_<ts>.pgm` —— **灰度分析中间产物**（按 `video.analysis_width` 缩放），
  留在 `workspace/extracted/video_frames/<source_id>/`。

> 母项目 `imageio` 只能解码 Netpbm，故去重分析依赖 PGM；PNG 仅供交付/可视化。
> **PGM 可安全删除**（保留它是为了调整去重阈值时不必重新解码视频）。
> 请勿把 PGM 当作必须交付的产物。

## 运行顺序

```powershell
# 本机（Windows）小样本冒烟：只探活
pwsh -File scripts/run_hoist_cycle.ps1
```

```bash
# lab-server（Linux）全量：完整链路（当前 T03–T05 未实现，会明确报「尚未实现」）
bash scripts/run_hoist_cycle.sh
```

阶段顺序：`inventory → extract-video → extract → annotate → prioritize →
curate → freeze-holdout → train → evaluate`（`doctor` 可随时单独跑）。

> 已实现（可跑）：`doctor`、`inventory`、`extract-video`、`extract`（T01–T02）。
> 仍未实现：`annotate / prioritize / curate / freeze-holdout / train / evaluate`
> —— CLI 会明确报「尚未实现」并返回退出码 `2`，**绝不静默成功**。
> 已实现阶段的执行失败（如 ffmpeg 缺失）返回退出码 `4`。
