@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Create .venv and install dependencies using README.md first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0compare_app.py" --open-browser %*
if errorlevel 1 pause
