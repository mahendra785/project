<#
.SYNOPSIS
    One-time setup for PUMS: Python venv + dependencies, npm install, samples.

.EXAMPLE
    ./scripts/setup.ps1
#>

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '[1/4] Creating the Python virtual environment ...' -ForegroundColor Cyan
Set-Location (Join-Path $root 'backend')
if (-not (Test-Path '.venv')) { python -m venv .venv }

Write-Host '[2/4] Installing backend dependencies ...' -ForegroundColor Cyan
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip --quiet
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt

Write-Host '[3/4] Generating synthetic sample phantoms ...' -ForegroundColor Cyan
& '.\.venv\Scripts\python.exe' scripts\make_sample_images.py

Write-Host '[4/4] Installing frontend dependencies ...' -ForegroundColor Cyan
Set-Location (Join-Path $root 'frontend')
npm install

Set-Location $root
Write-Host ''
Write-Host 'Setup complete. Start both services with ./scripts/dev.ps1' -ForegroundColor Green
