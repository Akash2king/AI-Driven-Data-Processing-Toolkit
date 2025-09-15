#!/usr/bin/env pwsh

Write-Host "Starting LLM-Guided Survey Data Cleaning Tool - Streamlit Frontend" -ForegroundColor Green
Write-Host ""

Write-Host "Installing/Updating Python packages..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
Write-Host "Frontend will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Backend should be running at: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
