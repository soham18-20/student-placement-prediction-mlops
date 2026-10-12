# PlacementPulse - Local Execution Helper Script for Windows PowerShell
param (
    [string]$Command = "serve"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🎓 Student Placement Prediction System with MLOps" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

switch ($Command.ToLower()) {
    "test" {
        Write-Host "Running Pytest Unit & Integration Tests..." -ForegroundColor Yellow
        python -m pytest tests -v
    }
    "train" {
        Write-Host "Running Model Training Pipeline & Versioning..." -ForegroundColor Yellow
        python -m src.train
    }
    "lint" {
        Write-Host "Running Code Linting..." -ForegroundColor Yellow
        python -m flake8 src tests --max-line-length=127
    }
    "serve" {
        Write-Host "Starting FastAPI Dev Server at http://127.0.0.1:8000" -ForegroundColor Green
        Write-Host "• Web Dashboard: http://127.0.0.1:8000" -ForegroundColor White
        Write-Host "• Swagger Docs:  http://127.0.0.1:8000/docs" -ForegroundColor White
        Write-Host "• Health Endpoint: http://127.0.0.1:8000/health" -ForegroundColor White
        python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
    }
    default {
        Write-Host "Usage: .\run.ps1 [serve | test | train | lint]" -ForegroundColor Red
    }
}
