@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please install dependencies first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
pause