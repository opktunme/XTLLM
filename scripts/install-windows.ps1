$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Python = Get-Command py.exe -ErrorAction SilentlyContinue
if (-not $Python) {
    throw 'Python 3.11+ was not found. Install it from python.org or run: winget install Python.Python.3.11'
}

$Version = & py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$Version -lt [version]'3.11') {
    throw "Python 3.11+ is required; found $Version"
}

$Venv = Join-Path $Root '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $Venv 'Scripts\python.exe'))) {
    & py -3 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the XTLLM Python environment.' }
}
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install numpy 'huggingface_hub[cli]'
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }

Write-Host ''
Write-Host 'XTLLM launcher environment is ready.' -ForegroundColor Green
Write-Host 'Next:'
Write-Host '  .\xtllm.cmd doctor'
Write-Host '  .\xtllm.cmd models'
Write-Host '  .\xtllm.cmd setup qwen36 --model-root D:\XTLLM-Models'
