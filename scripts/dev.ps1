<#
.SYNOPSIS
    Start the PUMS backend and frontend together (Windows / PowerShell).

.DESCRIPTION
    Launches uvicorn and next dev in separate windows so their logs stay
    readable. Run scripts/setup.ps1 once first.

.EXAMPLE
    ./scripts/dev.ps1
#>

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$venvPython = Join-Path $root 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error "Backend venv not found. Run ./scripts/setup.ps1 first."
}

Write-Host 'Starting PUMS backend on http://localhost:8000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "Set-Location '$root\backend'; & '$venvPython' -m uvicorn app.main:app --reload --port 8000"
)

Write-Host 'Starting PUMS frontend on http://localhost:3000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host ''
Write-Host 'Backend  : http://localhost:8000/docs' -ForegroundColor Green
Write-Host 'Frontend : http://localhost:3000' -ForegroundColor Green
Write-Host ''
Write-Host 'The first analysis builds the synthetic calibration set (~15s).'
