@echo off
REM ===========================================
REM LangChain Platform - Clean Server Start
REM ===========================================
REM This script ensures a clean server restart

setlocal enabledelayedexpansion

echo.
echo ============================================
echo LangChain Platform - Clean Server Start
echo ============================================
echo.

REM Navigate to deployment directory
cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Step 1: Stop ALL Python processes
echo [1/4] Stopping ALL Python processes...
taskkill /f /im python.exe 2>nul
if %errorlevel% == 0 (
    echo       Python processes terminated
) else (
    echo       No Python processes running
)
timeout /t 2 /nobreak >nul

REM Step 2: Clear Python cache
echo [2/4] Clearing Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" (
    rd /s /q "%%d" 2>nul
)
del /s /q *.pyc 2>nul
echo       Cache cleared

REM Step 3: Verify .env file exists
echo [3/4] Checking configuration...
if exist ".env" (
    echo       .env file found
    REM Show API_KEY_ENABLED setting
    for /f "tokens=1,* delims==" %%a in ('findstr /r "^API_KEY_ENABLED=" .env') do (
        echo       API_KEY_ENABLED = %%b
    )
) else (
    echo       WARNING: .env file not found!
    echo       Please copy .env.example to .env and configure it.
    pause
    exit /b 1
)

REM Step 4: Verify virtual environment
echo [4/4] Checking virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo       Virtual environment found
) else (
    echo       ERROR: Virtual environment not found at .venv
    echo       Please create it with: python -m venv .venv
    pause
    exit /b 1
)

echo.
echo ============================================
echo Starting server...
echo ============================================
echo.
echo  Chat UI:  http://localhost:8000/chat
echo  API Docs: http://localhost:8000/docs
echo  Health:   http://localhost:8000/health
echo.
echo  Press Ctrl+C to stop the server
echo ============================================
echo.

REM Start server with full reload capability
.venv\Scripts\python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload --reload-include "*.html" --reload-include "*.css" --reload-include "*.js"
