param(
    [string]$Config = "examples/production_project/config/project.production.json",
    [switch]$ExecuteTrainer
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $repoRoot
}

function Import-LocalEnvFile {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            return
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($name) {
            Set-Item -Path ("Env:" + $name) -Value $value
        }
    }
}

$localEnvPath = Join-Path (Split-Path -Parent $PSScriptRoot) "examples/production_project/.env.production.local"
Import-LocalEnvFile -Path $localEnvPath

Write-Host "Running doctor checks..."
python -m traffic_cv_closed_loop doctor --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Doctor checks failed. Fix config or environment before continuing."
}

Write-Host "Probing Qwen service..."
python -m traffic_cv_closed_loop probe-qwen --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Qwen probe failed."
}

Write-Host "Probing YOLO detector..."
python -m traffic_cv_closed_loop probe-detector --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "YOLO detector probe failed."
}

$args = @("-m", "traffic_cv_closed_loop", "run-loop", "--config", $Config)
if ($ExecuteTrainer) {
    $args += "--execute-trainer"
}

Write-Host "Running end-to-end closed loop..."
python @args
