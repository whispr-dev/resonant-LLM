# ----- Fren-Agent PowerShell setup (Windows) -----
# Run from PowerShell. If you hit policy errors:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# cd into project root
Set-Location "C:\code\fren-agent"

# Create and activate venv if needed
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    python -m venv .venv
}

# Activate venv (idempotent)
. ".\.venv\Scripts\Activate.ps1"

# Upgrade pip + install deps
python -m pip install --upgrade pip wheel
pip install -r ".\requirements.txt"

# Ensure assets folder exists
if (-not (Test-Path ".\assets")) { New-Item -ItemType Directory -Path ".\assets" | Out-Null }

# NLTK VADER lexicon warmup (PowerShell-native)
python -c "import nltk; 
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print('VADER already present.')
except Exception:
    nltk.download('vader_lexicon'); 
    print('VADER downloaded.')"

Write-Host "`nSetup complete. To launch:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python .\main.py"

