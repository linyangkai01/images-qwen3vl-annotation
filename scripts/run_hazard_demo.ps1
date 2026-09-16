# 一键跑通施工安全隐患标注 mock 全链路（无视频、无模型、无 ffmpeg）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/run_hazard_demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_hazard_demo.ps1 -Mode full
#
# 产物位于 examples/hazard_mock_project/workspace/：
#   extracted/  hazards/  review/  datasets/  runs/  analysis/  visualization/report.html

param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[hazard] working dir: $root"
Write-Host "[hazard] running: python -m hazard_annotation demo --mode $Mode"

python -m hazard_annotation demo --mode $Mode
$code = $LASTEXITCODE

$report = Join-Path $root "examples/hazard_mock_project/workspace/visualization/report.html"
if ($code -eq 0 -and (Test-Path $report)) {
    Write-Host "[hazard] OK. HTML report: $report"
} else {
    Write-Host "[hazard] demo finished with exit code $code"
}

exit $code
