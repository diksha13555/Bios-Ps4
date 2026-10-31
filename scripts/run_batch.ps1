param(
    [string[]]$Videos,
    [int]$MaxFrames = 150,
    [int]$VizMaxSide = 640
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$run = Join-Path $root "scripts\run_ps4.py"
$out = Join-Path $root "outputs"
$summary = @()
foreach ($v in $Videos) {
    if (-not (Test-Path $v)) { Write-Warning "skip missing $v"; continue }
    Write-Host "=== $v ===" -ForegroundColor Cyan
    & $py $run --input $v --output-dir $out --max-frames $MaxFrames --viz-max-side $VizMaxSide
    if ($LASTEXITCODE -ne 0) { throw "failed $v" }
}
Write-Host "Done." -ForegroundColor Green
