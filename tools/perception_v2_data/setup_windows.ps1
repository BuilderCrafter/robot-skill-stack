$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE). See the error above." }
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Install 64-bit Python 3.11, including the Python launcher, then reopen PowerShell. Example: winget install -e --id Python.Python.3.11"
}
& py -3.11 --version
Check-Exit "Python 3.11 check"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
    Check-Exit "venv creation"
}
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -c "import sys; assert sys.version_info[:2] == (3,11), 'Existing .venv is not Python 3.11'; assert sys.prefix != sys.base_prefix, 'Not a venv'"
Check-Exit "isolated environment check"
& $Python -m pip install --upgrade "pip==25.1.1"
Check-Exit "pip installation"
& $Python -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
Check-Exit "CUDA PyTorch installation"
& $Python -m pip install -r requirements-training.txt
Check-Exit "training dependencies"
& $Python -m pip check
Check-Exit "dependency validation"
& $Python check_environment.py --download-model
Check-Exit "GPU/model check"
& $Python -m pip freeze | Out-File -Encoding utf8 training-environment-lock.txt
Write-Host "Ready. Python: $Python"
