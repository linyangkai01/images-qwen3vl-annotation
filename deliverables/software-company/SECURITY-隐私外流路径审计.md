# 隐私安全审计：真实素材外流路径排查

> 审计时间：2026-09-16
> 审计范围：`D:\WorkPlace\images-qwen3vl-annotation` 全仓（含 `hazard_annotation/`、`traffic_cv_closed_loop/`、`examples/`、`docs/`）
> 触发原因：用户明确要求——**规避上传真实的图片和视频到网上，确保隐私安全**
> 保护对象：`D:\WorkPlace\industrial-data\` 下的宿州 31 张隐患图、珠海 4 个视频（3×mp4 + 1×.qt，共约 296MB）、隐患知识库

---

## 一、一句话结论

**全仓只有一个数据出网点**：`traffic_cv_closed_loop/adapters/qwen.py:535` 的 `urllib.request.urlopen`，目标地址完全由配置项 `qwen.base_url` 决定。

**没有**遥测、没有埋点、没有云 SDK（无 boto3 / oss2 / cos）、没有上传逻辑、没有密钥文件。

也就是说：**数据会不会外流，只取决于配置里的 `base_url` 指向哪里。** 而当前**没有任何护栏**阻止它指向公网。

---

## 二、逐项排查结果

### 2.1 出网代码点：仅 1 处

- `traffic_cv_closed_loop/adapters/qwen.py:8-9` 引入 `urllib.error` / `urllib.request`
- `traffic_cv_closed_loop/adapters/qwen.py:535` 执行 `urllib.request.urlopen(request, timeout=...)`
- 全仓无 `requests` / `httpx` / `aiohttp` 引入
- 全仓无 `requests.post` / `requests.put` / `upload` / `boto3` / `oss2` 等外发或云存储调用

### 2.2 各配置的 `base_url` 指向

- `examples/hazard_project/config/project.json` → `http://localhost:11434/v1`（本地 Ollama）**安全**
- `examples/hazard_mock_project/config/project.json` → `""` + `provider=mock` **安全**
- `examples/mock_project/config/project.json` → `endpoint: ""` **安全**
- `examples/production_project/config/project.production.json` → `https://ws-qi1jm9l6h7j1j0hl.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` **指向阿里云百炼（第三方云）**
- `examples/production_project/config/project.video_batch_20260713_30572.json` → 同上 **指向阿里云百炼**

### 2.3 隐患链路对云服务的引用

- 在 `hazard_annotation/`、`examples/hazard_project/`、`examples/hazard_mock_project/`、`docs/hazard-annotation-chain.md` 中检索 `百炼` / `dashscope` / `aliyun` / `maas` → **零匹配**
- 隐患链路是干净的：`provider=local_openai` + `require_api_key=false` + `base_url=http://localhost:11434/v1`

### 2.4 密钥文件

- 项目内未发现 `.env` / `*.env` / 密钥文件（`.gitignore` 已含 `.env` / `.env.*` / `*.env` / `*.local`）

### 2.5 `base_url` 本地性校验

- 在 `traffic_cv_closed_loop/` 与 `hazard_annotation/` 中检索 `localhost` / `127.0.0.1` / `is_local` / `allow_remote` / `require_local` → **零匹配**
- **结论：没有任何代码阻止 `base_url` 指向公网。**

### 2.6 `.gitignore` 覆盖情况

已覆盖：
- `examples/mock_project/workspace/`
- `examples/production_project/workspace/`
- `examples/hazard_mock_project/workspace/`
- `examples/hazard_mock_project/input/frames/`
- `examples/hazard_project/workspace/`
- `examples/hazard_project/input/frames/`
- `examples/production_project/input/frames/video_batch_20260713_30572/`
- `examples/production_project/input/frames/user_batch_20260712/`

**未覆盖（缺口）**：
- `examples/hazard_project/input/video/` —— 而隐患链路的配置里 `video.video_path` 正是 `input/video/site_monitor.mp4`。若把真实视频放这里，一旦初始化 git 就会被纳入版本控制
- 无通用 `*.mp4` / `*.mov` / `*.qt` / `*.jpg` / `*.png` 规则兜底

### 2.7 项目内是否已混入真实素材

- 全仓图片/视频文件仅 **1 个**：`examples/production_project/input/holdout_ood/family_car_ood_001.png`（交通链路的 holdout 图，非本次工业素材）
- **宿州隐患图与珠海视频均未复制进项目**，仍完整保留在 `D:\WorkPlace\industrial-data\`

### 2.8 可视化 HTML

- 产物：`examples/hazard_mock_project/workspace/visualization/report.html`
- 该文件为**自包含**，缩略图以 base64 内联。文件本身留在本地，`workspace/` 已被 `.gitignore` 覆盖
- 但需注意：**分享这个 HTML 文件等于分享其中的图像**。若用真实素材生成，外发前需脱敏

---

## 三、风险清单

**R1【中】交通链路生产配置指向第三方云**
`examples/production_project/` 的两个配置指向阿里云百炼。若误用工业真实素材跑交通链路，帧会上传到公网。

**R2【中】缺 `base_url` 本地性护栏**
改一个配置字段就能把数据发出去，没有代码层面的拦截。

**R3【低】`.gitignore` 未覆盖 `examples/hazard_project/input/video/`**
真实视频若放此处，初始化 git 后会被跟踪。

**R4【低】可视化 HTML 内联 base64**
分享文件即分享图像。

---

## 四、建议措施

**M1【建议优先】加 `base_url` 本地性校验（护栏）**
默认只允许本机与内网地址（`localhost` / `127.0.0.1` / `::1` / `10.x` / `172.16-31.x` / `192.168.x`）；若配置为公网地址，需显式打开一个开关（如 `qwen.allow_remote=true`）才放行，否则启动即报错。这样"误用云 API"从"改一个字段"变成"必须显式声明"。

**M2 补 `.gitignore`**
新增 `examples/hazard_project/input/video/`，并加通用兜底规则 `*.mp4` / `*.mov` / `*.qt` / `*.avi` / `*.jpg` / `*.jpeg` / `*.png` / `*.ppm` / `*.pgm`。

**M3 交通链路生产配置加警示**
在 `examples/production_project/config/*.json` 的 `base_url` 旁显式标注"⚠️ 公网地址，禁止用于涉密素材"，或直接把 `base_url` 置空并让用户自行填写。

**M4 素材存放纪律**
真实素材统一留在 `D:\WorkPlace\industrial-data\`，**不复制进项目目录**；项目内只放 mock 资产。

**M5 可视化脱敏（可选）**
给可视化加一个开关：生成报告时可选"缩略图打码"或"不内联图像，只留文件引用"，便于对外分享。

---

## 五、本次审计的自身合规声明

- 本次审计全程仅在本机执行：目录列举、文件元数据统计、JSON 本地解析、PDF 本地字节扫描
- **未上传、未外发任何真实素材**，也未将任何素材内容用于联网检索
- 联网检索仅涉及公开资料（51CTO 博客、GitHub 仓库），不含用户数据
