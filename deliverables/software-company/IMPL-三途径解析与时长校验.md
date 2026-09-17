# 三条解析途径 + 视频时长校验：实现说明

> 实现时间：2026-09-16
> 对应需求：支持三种解析途径并可切换；明确抽帧触发方式与频率；base64 编码与逗号拼接；预留 Bearer Token；输出结构化结果；视频时长校验与提示
> 新增测试：`tests/test_hazard_inference.py`（46 个用例，全绿）

---

## 一、三条解析途径

### 1.1 途径清单与能力边界

由 `inference.backend` 选择，可选四个值：

**`private_vlm`** —— 实验室私有多模态模型 API（Ollama-native `/api/generate`）
- 能力：通道 A（目标框）+ 通道 B（隐患事件）**都能做**
- 实现：`PrivateVlmAnnotator`（`traffic_cv_closed_loop/adapters/qwen.py`）
- 认证：`Authorization: Bearer <token>`

**`local_yolo`** —— 本机 Ultralytics YOLO
- 能力：**只能做通道 A**（目标框）。隐患事件需要多模态模型
- 实现：`UltralyticsDetector`（复用现有）

**`server_yolo`** —— lab-server 上的 YOLO（经 ssh 委派执行）
- 能力：**只能做通道 A**
- 实现：`RemoteYoloDetector`（`hazard_annotation/inference.py`）
- 流程：`scp` 上传图片 → `ssh` 执行 `yolo predict` → `scp` 下载标签

**`mock`** —— 离线 sidecar，不联网
- 能力：两者都有（替身），用于链路验证

### 1.2 关键约束（已实现为明确报错）

用 `local_yolo` / `server_yolo` 去取 `annotator`（即想要通道 B）会**直接报错**，而不是静默返回空结果：

```
inference.backend=local_yolo 不提供多模态标注能力；
隐患事件（通道 B）需要 private_vlm 或 mock。
若只想跑通道 A，请直接使用 detector。
```

### 1.3 切换方式

改配置一个字段即可：

```json
"inference": { "backend": "private_vlm" }
```

或命令行覆盖（如果走 CLI 的话）。三条途径共用同一套下游逻辑，切换不改代码。

### 1.4 自检

`InferenceRouter.describe()` 输出途径说明与配置检查结果，可直接给 `doctor` 用：

```json
{
  "backend": "private_vlm",
  "available_backends": ["mock", "local_yolo", "server_yolo", "private_vlm"],
  "endpoint": "http://119.23.145.227:38091/api/generate",
  "model": "qwen3-vl:30b-a3b-thinking-q8_0",
  "images_mode": "array",
  "auth_env": "LAB_VLM_TOKEN",
  "auth_token_present": false,
  "max_video_seconds": 60.0,
  "on_over_limit": "warn",
  "issues": ["auth_token_missing:LAB_VLM_TOKEN"],
  "notes": ["请设置环境变量 LAB_VLM_TOKEN=<Bearer Token>"]
}
```

---

## 二、视频抽帧的触发方式与频率

### 2.1 触发方式

**入口命令**：`python -m hazard_annotation extract --config <cfg>`

三种触发场景：

**手动单阶段** —— 只跑抽帧，用于调试与验证
```
python -m hazard_annotation extract --config <cfg>
```

**全链路触发** —— 抽帧作为第一步自动执行
```
python -m hazard_annotation all --config <cfg>
```

**定时/事件触发（待接入）** —— 目前无调度器，属于 S3 阶段的补齐项

### 2.2 抽帧频率：两层频率，不是一个固定值

**第一层：粗采样频率（固定）**
- 配置：`video.sample_fps`，默认 `1.0`
- 含义：先把视频按 1 帧/秒解码成灰度分析帧
- 作用：把 1 小时视频从 9 万帧降到 3600 帧，避免逐帧处理
- 图片目录模式：直接读取，不做采样

**第二层：关键帧输出频率（内容自适应）**
最终关键帧数量**不是固定值**，由六步法逐步筛选决定：

1. **粗采样** —— 按 `sample_fps` 解码
2. **运动过滤** —— 帧间差分运动占比低于 `motion_min_ratio`（默认 0.002）的帧丢弃
3. **感知去重** —— aHash 汉明距离 ≤ `dedup_hamming_threshold`（默认 4）视为重复
4. **场景切换强制保留** —— 直方图差 / ffmpeg scene 检测到的切换点保留
5. **目标强制保留** —— 命中 `force_keep_labels`（8 类）的帧强制保留
6. **数量与间隔约束** —— 上限 `max_keyframes`（默认 400）+ 最小间隔 `min_gap_seconds`（默认 2.0 秒）

**两个兜底频率约束**：
- `max_keyframes`：上限 400，防止长视频产出过多
- `min_gap_seconds`：相邻关键帧至少间隔 2 秒，避免同一动作刷屏

**所以"频率"的正确理解是**：粗采样固定 1fps，关键帧输出密度由内容决定，最终受 400 张 / 2 秒间隔两个约束封顶。

### 2.3 时长校验的接入位置

抽帧**开始前**先做视频时长校验（见第六节），校验不通过会打印提示或直接拒绝。

---

## 三、base64 编码与逗号拼接

### 3.1 实现位置

`PrivateVlmAnnotator.encode_images()`（`traffic_cv_closed_loop/adapters/qwen.py`）

### 3.2 两种模式，由 `qwen.images_mode` 选择

**`array`（默认）** —— 返回 JSON 数组，每个元素是一张图的纯 base64
```json
"images": ["iVBORw0KGgo...", "iVBORw0KGgo..."]
```

**`comma_joined`** —— 返回单个字符串，多张图的 base64 之间用**英文逗号** `,` 拼接
```json
"images": "iVBORw0KGgo...,iVBORw0KGgo..."
```

### 3.3 两个实现细节

**纯 base64，不带 `data:` 前缀**
OpenAI 兼容格式需要 `data:image/png;base64,xxx`，但你给的接口示例是纯 base64 字符串。本实现按你的示例走，**不加前缀**。

**多图单请求**
`annotate_multi(image_paths, prompt)` 支持一次请求送多张图，对应你说的"多张图片之间用英文逗号隔开"。

### 3.4 ⚠️ 需要你确认的一点

你给的请求体示例是 `"images": ["base64编码1", "base64编码2"]`（JSON **数组**），但文字描述是"多张图片之间用英文逗号隔开"（像是**单个字符串**）。

**两种我都实现了，默认用 `array`（对齐你的示例）。** 如果服务端实际要求逗号拼接的字符串，把配置改成：

```json
"qwen": { "images_mode": "comma_joined" }
```

不需要改代码。

---

## 四、Bearer Token 配置（已预留）

### 4.1 配置项

```json
"qwen": {
  "api_key_env": "LAB_VLM_TOKEN",
  "auth_scheme": "Bearer",
  "require_api_key": true
}
```

- `api_key_env`：**存 token 的环境变量名**。走 `private_vlm` 途径时默认就是 `LAB_VLM_TOKEN`
- `auth_scheme`：默认 `Bearer`，拼成 `Authorization: Bearer <token>`
- `require_api_key`：`true` 时缺 token 直接报错；`false` 时放行

### 4.2 填入方式

```bash
# Linux / macOS
export LAB_VLM_TOKEN="你的 token"

# Windows Git Bash
export LAB_VLM_TOKEN="你的 token"
```

### 4.3 安全约定

- **token 不写进配置文件**，只通过环境变量注入
- **token 不写进日志**：`request_preview` 里只记录环境变量**名字**（`auth_env`），不记录值
- 缺失时 `describe()` 会给出明确指引，不会静默失败

---

## 五、结构化解析结果

### 5.1 输出结构

```
workspace/
├── extracted/
│   ├── frames/<source>/<frame>.png      关键帧
│   ├── candidates.jsonl                 候选记录（含 YOLO 初检先验）
│   └── summary.json                     抽帧摘要
│       ├── inference_backend            新增：本次使用的解析途径
│       ├── video_duration               新增：时长校验报告
│       └── totals                       9 项计数
│
├── review/
│   ├── labels/<source>/<frame>.txt      通道 A：YOLO 标准标签
│   ├── hazards/<source>/<frame>.json    通道 B：隐患事件
│   ├── annotations.jsonl                双通道合并记录
│   ├── review_queue.jsonl               人工复核队列
│   ├── raw_responses/                   原始模型响应（可追溯）
│   └── summary.json                     标注摘要
│
└── datasets/<ver>/                      数据集
```

### 5.2 通道 B 隐患事件结构

```json
{
  "scene_type": "吊装作业",
  "hazards": [
    {
      "id": "D-5-3",
      "category": "人员不安全行为",
      "severity": "critical",
      "norm_reference": "GB 6067.1-2010",
      "action": "立即撤离吊物下方",
      "evidence": "人员站在吊物正下方",
      "confidence": 0.85
    }
  ],
  "overall_risk": "critical",
  "recommendation": "..."
}
```

**关键设计**：模型只输出 `id` + `evidence`，`category / norm_reference / action` 由知识库**本地回填**，杜绝编造规范号。

### 5.3 原始响应可追溯

每次调用都把请求预览与原始响应写进 `raw_responses/`，解析失败时帧进复核队列且**不丢原始响应**。

---

## 六、视频时长校验（1 分钟上限）

### 6.1 探测方式

`FfmpegRunner.probe_duration()`：
1. 优先 `ffprobe`（容器精确时长）
2. 回退解析 `ffmpeg -i` stderr 里的 `Duration: 00:01:23.45`
3. 都失败返回 `0.0`，视为**未知**，**不阻断流程**（避免把探测失败误判成超限）

### 6.2 三种处置策略（`inference.on_over_limit`）

**`warn`（默认）** —— 打印告警但放行
```
[时长校验] 告警：视频时长 3600.0s 超过私有多模态模型上限 60s；
已按 on_over_limit=warn 放行，但结果可能不完整，建议改用 segment 模式
```

**`error`** —— 直接拒绝并抛 `VideoDurationError`，附上修改指引

**`segment`** —— 计算需要切成几段并给出具体区间
```
[时长校验] 超限：时长 90.0s 超过上限 60s，建议切成 2 段（每段 ≤ 55s）后再送模型
          建议分段（共 2 段）：0-55s、55-90s
```

### 6.3 配置

```json
"inference": {
  "max_video_seconds": 60,
  "on_over_limit": "segment",
  "segment_seconds": 55
}
```

`segment_seconds` 默认 55 秒，留 5 秒余量。

### 6.4 校验结果落盘

写进 `extracted/summary.json` 的 `video_duration` 字段：

```json
{
  "video_path": "...",
  "duration_seconds": 90.0,
  "limit_seconds": 60.0,
  "status": "over_limit",
  "action": "segment",
  "segment_count": 2,
  "message": "时长 90.0s 超过上限 60s，建议切成 2 段（每段 ≤ 55s）后再送模型",
  "warnings": ["video_over_limit_segmented"]
}
```

同时把告警码写进 summary 的 `warnings` 数组，便于下游聚合。

---

## 七、怎么跑

### 7.1 新增的示例项目

`examples/hazard_lab_vlm_project/` —— 已经配好私有 VLM 途径

```
examples/hazard_lab_vlm_project/
├── assets/prompts/combined_zh.md
└── config/project.json          ← provider=private_vlm，端点已填
```

关键配置：

```json
"inference": {
  "backend": "private_vlm",
  "max_video_seconds": 60,
  "on_over_limit": "warn",
  "segment_seconds": 55
},
"qwen": {
  "provider": "private_vlm",
  "base_url": "http://119.23.145.227:38091",
  "request_path": "/api/generate",
  "model": "qwen3-vl:30b-a3b-thinking-q8_0",
  "api_key_env": "LAB_VLM_TOKEN",
  "require_api_key": true,
  "images_mode": "array",
  "stream": false,
  "options": { "temperature": 0.7, "max_tokens": 8192 }
}
```

### 7.2 运行步骤

```bash
# 1) 填入 token
export LAB_VLM_TOKEN="你的 token"

# 2) 看途径自检（会报 token 是否就位、端点、时长策略）
python -m hazard_annotation doctor --config examples/hazard_lab_vlm_project/config/project.json

# 3) 抽帧（含时长校验）
python -m hazard_annotation extract --config examples/hazard_lab_vlm_project/config/project.json

# 4) 标注（走私有 VLM）
python -m hazard_annotation annotate --config examples/hazard_lab_vlm_project/config/project.json
```

### 7.3 切换到其他途径

只改 `inference.backend`：
- 改 `local_yolo` → 用本机 Ultralytics
- 改 `server_yolo` → 委派 lab-server 执行（需 ssh 免密已配置）
- 改 `mock` → 离线跑通链路

---

## 八、验收点

**途径切换**
- 四个途径都能构造 router 且 `describe()` 正确
- `local_yolo` / `server_yolo` 取 annotator 时**明确报错**而非静默返回空
- 未知途径名报错

**base64 与拼接**
- array 模式返回 list，元素是**纯 base64**（无 `data:` 前缀）
- comma_joined 模式返回 str，**英文逗号**分隔，每段可独立解码
- 请求体含 `model / prompt / options / stream / images`，**不含 `messages`**

**Bearer Token**
- 有 token 时注入 `Authorization: Bearer <token>`
- 无 token 且 `require_api_key=true` 时报错
- token **不落盘、不写日志**

**时长校验**
- 45s → ok 放行
- 60s（恰好）→ ok 放行
- 90s + segment → 切成 2 段
- 3600s + warn → 告警放行
- 3600s + error → 抛错
- 探测失败 → 视为未知，**不误伤**

**回归**
- 交通链路 `test_pipeline` / `test_production_integration` / `test_skills_delivery` 全绿
- 隐患链路既有测试全绿
- `python -m hazard_annotation demo --mock` 退出码 0

---

## 九、连通性探测命令（新增）

### 9.1 为什么需要它

在正式跑真实素材之前，需要先确认「端点通不通、token 对不对、请求体格式合不合」。这一步如果用真实素材试，既浪费又违反隐私约束。

所以做了一个**分层探测命令**，用**本地合成的图**验证全链路，**不涉及任何真实素材**。

### 9.2 三层探测，逐层加深

**第 1 层：TCP 握手** —— 不发送任何数据，只确认端口可达

**第 2 层：无凭据 GET** —— 看服务是否在线、是否强制鉴权。403/401 是**好信号**（说明服务在线且认证层在工作）

**第 3 层：合成图真实请求** —— 默认**不执行**，加 `--send-request` 才发。用的是本地生成的合成 PNG（画面上有几个色块和近似人形/吊钩/吊物的矩形），便于人工判断模型是否真的"看到了"图片

### 9.3 用法

```bash
# 只看前两层（不发任何业务数据）
python -m hazard_annotation probe-inference --config examples/hazard_lab_vlm_project/config/project.json

# 三层都跑（需先设置 token；用合成图，不碰真实素材）
export LAB_VLM_TOKEN="你的 token"
python -m hazard_annotation probe-inference --config examples/hazard_lab_vlm_project/config/project.json --send-request
```

### 9.4 实测结果（2026-09-17）

```
=== 私有 VLM 端点探测 ===
  途径      : private_vlm
  端点      : http://119.23.145.227:38091/api/generate
  模型      : qwen3-vl:30b-a3b-thinking-q8_0
  images    : array
  token 环境 : LAB_VLM_TOKEN（未设置）

  [1/3] TCP 握手      : 可连接（47.95ms）
  [2/3] 无凭据 GET    : HTTP 403（强制鉴权，正常）
  [3/3] 合成图请求    : 未执行（加 --send-request 可执行）

  待处理：
    - auth_token_missing:LAB_VLM_TOKEN

  结论：服务在线且强制鉴权；但 token 未设置，带认证的请求还发不了
```

**三个结论**：
1. **TCP 可达**（47.95ms）—— 网络路径没问题
2. **服务在线且强制鉴权** —— 未带 token 被拒，符合预期
3. **token 尚未配置** —— 这是唯一剩下的阻塞

### 9.5 ⚠️ 一个重要发现：403 来自 nginx，不是应用层

响应体预览是：

```html
<html><head><title>403 Forbidden</title></head>
<body><center><h1>403 Forbidden</h1></center>
<hr><center>nginx/1.</center></body></html>
```

**这是 nginx 的默认错误页**，说明端点前面挂了 nginx 反向代理。

**影响**：鉴权可能在 **nginx 层**做（而非应用层）。这意味着：

- token 的传递方式（`Authorization: Bearer`）需要与 nginx 的配置匹配
- 如果 nginx 用的是别的鉴权头（如 `X-Auth-Token`），需要改 `qwen.auth_scheme` 或 `qwen.headers`
- 配置里已经预留了 `qwen.headers` 和 `qwen.header_env`，可以灵活适配

**建议**：拿到 token 后，先用 `--send-request` 试一次。如果返回 403，说明鉴权头不对，此时需要向部署方确认 nginx 的鉴权配置。

---

## 十、待你补齐 / 确认

**P0-1 填入 Bearer Token**
```bash
export LAB_VLM_TOKEN="你的 token"
```
未填时 `doctor` 会提示，不会静默失败。

**P0-2 确认 `images` 传参格式**
你给的示例是 JSON 数组，文字描述是逗号拼接。**我默认用数组，两种都支持。** 请确认服务端实际要哪种，如需切换改 `qwen.images_mode` 即可。

**P1-1 lab-server YOLO 途径的连接参数**
`inference.server.*` 已按现有部署填了默认值（host=`lab-server`、remote_dir=`/data/users/lyk/.tmp/yolo-inference`、activate 脚本路径）。如与实际不符请告知。

**P1-2 真实运行授权**
代码已实现并单测覆盖，但**尚未对任何真实素材实际调用过该接口**。首次真实运行前请明确授权。

---

## 附：本次改动清单

**新增文件**
- `hazard_annotation/inference.py` —— 途径路由 + 时长校验 + 远程 YOLO 委派 + 端点探测
- `tests/test_hazard_inference.py` —— **56 个用例**
- `examples/hazard_lab_vlm_project/config/project.json` —— 私有 VLM 示例配置
- `examples/hazard_lab_vlm_project/assets/prompts/combined_zh.md`

**修改文件**
- `traffic_cv_closed_loop/adapters/qwen.py` —— 新增 `PrivateVlmAnnotator`（含 `annotate_multi`），工厂支持 `private_vlm`
- `traffic_cv_closed_loop/config.py` —— 新增 `inference` 段、`qwen.images_mode` / `options` / `stream`，`api_key_env` 按 provider 智能默认
- `hazard_annotation/ffmpeg.py` —— 新增 `probe_duration()` + `_parse_seconds()`
- `hazard_annotation/pipeline/extraction.py` —— 抽帧前接入时长校验，summary 增加 `inference_backend` / `video_duration`
- `hazard_annotation/__main__.py` —— 新增 `probe-inference` 阶段与 `--send-request` 开关
