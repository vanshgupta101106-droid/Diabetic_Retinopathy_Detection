@echo off
title Diabetic Retinopathy Detection Web Server
echo =========================================================
echo   Starting Diabetic Retinopathy Detection AI Service...
echo =========================================================
if exist "%~dp0Diabetic_retinopathy_detection" (
    cd /d "%~dp0Diabetic_retinopathy_detection"
) else (
    cd /d "%~dp0"
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    pause
    exit /b 1
)
echo.
echo Interactive Web App: http://127.0.0.1:8000/ui
echo Swagger API Docs:    http://127.0.0.1:8000/docs
echo Health Check:        http://127.0.0.1:8000/
echo.
echo Opening browser in 3 seconds...
start "" "http://127.0.0.1:8000/ui"
echo.
.\.venv\Scripts\uvicorn.exe src.app.main:app --host 127.0.0.1 --port 8000 --reload
pause

