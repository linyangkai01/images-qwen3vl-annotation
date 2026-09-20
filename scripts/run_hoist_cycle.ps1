# run_hoist_cycle.ps1 —— 本机（Windows）小样本冒烟入口
#
# 用法：
#   pwsh -File scripts/run_hoist_cycle.ps1                 # 默认只跑 doctor
#   pwsh -File scripts/run_hoist_cycle.ps1 -Stage doctor
#   pwsh -File scripts/run_hoist_cycle.ps1 -Stage inventory # T02–T05 未实现，会明确报错（退出码 2）
#
# 说明：本机只做小规模验证（单测 / 冒烟 / 小样本），**不跑全量、不调 VLM、不联网**。

param(
    [string]$Config = "own/hoist_project/config/project.hoist.json",
    [string]$Stage = "doctor"
)

$ErrorActionPreference = "Stop"

# 定位仓库根（本脚本位于 scripts/ 下）。
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host "[run_hoist_cycle] repo=$RepoRoot"
    Write-Host "[run_hoist_cycle] stage=$Stage"
    Write-Host "[run_hoist_cycle] config=$Config"

    python -m industrial_hazard_coldstart $Stage --config $Config
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "[run_hoist_cycle] stage '$Stage' 退出码=$code（非 0：可能是「尚未实现」或探活发现问题）"
    } else {
        Write-Host "[run_hoist_cycle] stage '$Stage' 完成（退出码 0）"
    }
    exit $code
}
finally {
    Pop-Location
}
