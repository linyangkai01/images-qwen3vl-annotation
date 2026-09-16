# 用 ffmpeg 合成一段测试视频（可选，用于演示"视频 -> 关键帧"真实路径）
#
# 无 ffmpeg 时本脚本会优雅跳过（不报错），因此不影响 mock 全链路。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/make_mock_video.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/make_mock_video.ps1 -Seconds 5
#
# 产物：examples/hazard_project/input/video/site_monitor.mp4

param(
    [int]$Seconds = 5,
    [int]$Fps = 25,
    [string]$Size = "640x480"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "[hazard] ffmpeg not found in PATH - skipping mock video generation."
    exit 0
}

$outDir = Join-Path $root "examples/hazard_project/input/video"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir "site_monitor.mp4"

Write-Host "[hazard] generating $Seconds s test video -> $outFile"
& ffmpeg -hide_banner -y -f lavfi -i "testsrc=size=$Size:rate=$Fps:duration=$Seconds" `
    -pix_fmt yuv420p $outFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "[hazard] OK. mock video: $outFile"
} else {
    Write-Host "[hazard] ffmpeg failed with exit code $LASTEXITCODE"
}
exit $LASTEXITCODE
