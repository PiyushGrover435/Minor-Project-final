# setup_env.ps1
# Helper script to create a Python 3.11 virtualenv and install requirements for Sentin-Edge AI.
# Usage: Open PowerShell in this folder and run: .\setup_env.ps1

$ErrorActionPreference = 'Stop'

Write-Host "Sentin-Edge environment setup helper" -ForegroundColor Cyan

# Helper to run a command and return success
function Try-Run($cmd) {
    try {
        & cmd /c $cmd 2>$null
        return $true
    } catch {
        return $false
    }
}

Write-Host "Locating Python 3.11 interpreter..."

$pythonCmd = $null
# Try py launcher first
if (Try-Run 'py -3.11 --version') { $pythonCmd = 'py -3.11' }
# Try common executable names
elseif (Try-Run 'python3.11 --version') { $pythonCmd = 'python3.11' }
elseif (Try-Run 'python --version') {
    $ver = & python --version 2>&1
    if ($ver -match '3\.11') { $pythonCmd = 'python' }
}

if (-not $pythonCmd) {
    Write-Host "Could not find Python 3.11 on PATH. Please install Python 3.11 from https://www.python.org/downloads/ and ensure the interpreter is available as 'py -3.11', 'python3.11' or 'python' (3.11)." -ForegroundColor Yellow
    exit 1
}

Write-Host "Using Python command: $pythonCmd"

# Create venv
& $pythonCmd -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create virtualenv" -ForegroundColor Red; exit 1 }

Write-Host "To activate the venv run: .\.venv\Scripts\Activate.ps1"
# Do not attempt to activate automatically in this script to avoid session scope issues

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtualenv Python executable not found at $venvPython" -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip and installing requirements..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to upgrade pip in virtualenv" -ForegroundColor Red; exit 1 }

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to install requirements in virtualenv" -ForegroundColor Red; exit 1 }

Write-Host "Setup complete. To activate and run the app:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  python sentin_edge.py" -ForegroundColor Green
