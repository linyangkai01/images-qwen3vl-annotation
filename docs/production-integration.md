# 生产接入指南

本文档说明如何把当前项目从离线示例切换到真实 Qwen3VL 服务和真实 YOLO 训练环境。

## 1. 准备条件

### Qwen3VL 服务

需要准备以下信息：

- 服务 `base_url` 或完整 `endpoint`
- 模型名，例如 `Qwen/Qwen2.5-VL-72B-Instruct`
- API Key 对应的环境变量名
- 如果服务要求自定义 Header，还需要对应的 Header 名和环境变量名。大多数 Bearer Token 场景只需要 API Key。

### YOLO 环境

需要准备以下内容：

- 已安装并可执行的 `yolo` 命令
- 初检权重 `detector_weights`
- 训练权重 `weights`
- 训练使用的 GPU 设备编号，例如 `0`

## 2. 复制生产配置

建议从 [project.production.json](/D:/ruxian/examples/production_project/config/project.production.json) 开始修改。

重点替换：

- `input.frames_dir`
- `output.workspace_dir`
- `qwen.base_url` 或 `qwen.endpoint`
- `qwen.model`
- `qwen.api_key_env`
- `qwen.header_env`
- `yolo.detector_weights`
- `yolo.weights`
- `yolo.device`

## 3. 设置环境变量

可参考 [env.production.example](/D:/ruxian/examples/production_project/env.production.example)。

PowerShell 示例：

```powershell
$env:QWEN_API_KEY = "your-api-key"
```

如果你的服务还要求额外请求头，再补充环境变量，并在 `qwen.header_env` 里显式配置。例如：

```json
"header_env": {
  "X-App-Id": "QWEN_APP_ID"
}
```

## 4. 先跑诊断

```powershell
python -m traffic_cv_closed_loop doctor --config examples/production_project/config/project.production.json
```

这个命令会检查：

- 输入目录是否存在
- Prompt 模板是否存在
- Qwen 服务地址是否完整
- API Key 和可选 Header 环境变量是否存在
- `yolo` 命令是否在 PATH 中
- 权重路径是否存在

## 5. 做联调探活

### 探测 Qwen3VL

```powershell
python -m traffic_cv_closed_loop probe-qwen --config examples/production_project/config/project.production.json
```

如果要指定某张图片：

```powershell
python -m traffic_cv_closed_loop probe-qwen --config examples/production_project/config/project.production.json --image path/to/frame.jpg
```

### 探测 YOLO 初检

```powershell
python -m traffic_cv_closed_loop probe-detector --config examples/production_project/config/project.production.json
```

## 6. 运行真实闭环

```powershell
python -m traffic_cv_closed_loop run-loop --config examples/production_project/config/project.production.json --execute-trainer
```

如果只想先看训练命令是否准备正确，可以只执行：

```powershell
python -m traffic_cv_closed_loop train --config examples/production_project/config/project.production.json
```

这会输出 `prepared` 状态，并把实际训练命令写入运行目录。

## 7. 查看结果

关键产物：

- `workspace/review/raw_responses/`：Qwen3VL 原始响应
- `workspace/datasets/<version>/data.yaml`：YOLO 数据集配置
- `workspace/runs/<version>/command.txt`：真实训练命令
- `workspace/runs/<version>/metrics.json`：解析后的训练指标
- `workspace/runs/<version>/trainer_summary.md`：训练摘要
- `workspace/analysis/hard_examples.jsonl`：困难样本

## 8. 常见问题

### Qwen 返回不是 JSON

建议检查：

- `qwen.response_format` 是否为 `json_object`
- Prompt 是否明确要求只输出 JSON
- 服务是否完全兼容 OpenAI `chat/completions`

### `doctor` 报 `yolo_binary_missing`

说明系统 PATH 中找不到 `yolo` 命令。需要先安装 Ultralytics，并确认命令行可直接执行：

```powershell
yolo help
```

### 权重路径不存在

如果使用本地权重，请填写绝对路径或相对项目路径。  
如果使用模型名而不是文件路径，例如 `yolo11n.pt`，项目会按模型名透传给 Ultralytics。

## 9. 生产建议

- 先用 `probe-qwen` 和 `probe-detector` 做小流量联调。
- 再用少量样本跑 `extract -> annotate -> train`。
- 指标和标签质量稳定后，再扩大到批量数据。
- 训练集和验证集要做版本留档，避免回流样本污染评估结果。
