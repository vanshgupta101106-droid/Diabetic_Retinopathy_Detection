Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Starting Diabetic Retinopathy Detection AI Service...  " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if (Test-Path "$PSScriptRoot\Diabetic_retinopathy_detection") {
    Set-Location -Path "$PSScriptRoot\Diabetic_retinopathy_detection"
} else {
    Set-Location -Path "$PSScriptRoot"
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Virtual environment not found in .venv."
    exit 1
}

Write-Host ""
Write-Host "Interactive Web App: http://127.0.0.1:8000/ui" -ForegroundColor Green
Write-Host "Swagger API Docs:    http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Health Check:        http://127.0.0.1:8000/" -ForegroundColor Gray
Write-Host ""

Start-Process "http://127.0.0.1:8000/ui"

& ".\.venv\Scripts\uvicorn.exe" src.app.main:app --host 127.0.0.1 --port 8000 --reload
