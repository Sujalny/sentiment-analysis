@echo off
cd /d "%~dp0"
set "SENTIMENT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SENTIMENT_PYTHON%" (
  echo Create the virtual environment and install requirements as described in README.md.
  pause
  exit /b 1
)
start "Sentiment Studio Browser" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:5000/'"
"%SENTIMENT_PYTHON%" app.py
pause
