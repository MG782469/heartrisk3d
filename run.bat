@echo off
REM HeartRisk 3D - one-click start (Windows)
cd /d "%~dp0"
if not exist artifacts\models.joblib (
  echo Training models, first run only ...
  python ml\train.py || goto :error
)
start "" http://127.0.0.1:8000
python app\server.py 8000
goto :eof
:error
echo Training failed. Run: pip install -r requirements.txt
pause
