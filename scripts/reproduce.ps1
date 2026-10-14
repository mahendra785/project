# Regenerate every MG-CRC number, table and paper figure input (Windows).
#   ./scripts/reproduce.ps1          full run (CPU random-forest backbone)
#   ./scripts/reproduce.ps1 -Fast    reuse backend/var/posteriors
param([switch]$Fast, [string]$Backbone = "rf", [int]$Size = 256, [int[]]$Seeds = @(0, 1, 2),
      [string[]]$Datasets = @("fracatlas", "kvasir-seg"))
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "backend\.venv\Scripts\python.exe"
Push-Location (Join-Path $root "backend")
try {
  if (-not $Fast) {
    foreach ($ds in $Datasets) { foreach ($s in $Seeds) {
      & $py scripts/experiments/train_backbone.py --dataset $ds --backbone $Backbone --size $Size --seed $s
      if ($LASTEXITCODE -ne 0) { throw "training failed: $ds seed $s" } } }
  }
  foreach ($ds in $Datasets) { foreach ($s in $Seeds) {
    & $py scripts/experiments/mgcrc.py --dataset $ds --backbone $Backbone --seed $s
    if ($LASTEXITCODE -ne 0) { throw "experiment failed: $ds seed $s" } } }
  & $py scripts/experiments/simulation.py
  & $py scripts/experiments/sim_diagnostic.py
  foreach ($ds in $Datasets) { & $py scripts/experiments/posthoc_multi.py --dataset $ds --backbone $Backbone }
  & $py scripts/experiments/aggregate.py | Out-Null
  & $py scripts/experiments/paper_macros.py | Out-Null
  $dest = Join-Path $root "docs\research\results\backbones"
  New-Item -ItemType Directory -Force $dest | Out-Null
  Get-ChildItem var/posteriors/*/*.json | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $dest "$($_.Directory.Name)_$($_.Name)") }
  Write-Host "done -> docs/research/results/summary.md, paper/data/*.csv"
} finally { Pop-Location }
