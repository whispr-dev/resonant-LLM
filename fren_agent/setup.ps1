Set-Location C:\code\fren-agent

# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip + install deps
python -m pip install --upgrade pip wheel
pip install -r .\requirements.txt

# NLTK VADER lexicon warmup (optional)
python - << 'PY'
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    nltk.download('vader_lexicon')
print("VADER ready.")
PY

Write-Host "Done. To run: `python .\main.py`"
