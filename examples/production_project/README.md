# 生产项目样例

这个目录提供真实环境接入模板，不包含真实权重、真实图片和真实凭据。

建议使用顺序：

1. 修改 [project.production.json](/D:/ruxian/examples/production_project/config/project.production.json)
2. 设置环境变量，参考 [env.production.example](/D:/ruxian/examples/production_project/env.production.example)
3. 运行 `doctor`
4. 运行 `probe-qwen` 和 `probe-detector`
5. 运行真实闭环

