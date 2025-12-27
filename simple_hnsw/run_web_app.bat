@echo off
setlocal

echo [1/2] Checking Backend Environment...

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Create/Activate Virtual Environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate venv and install requirements
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r web_app/backend/requirements.txt
if %errorlevel% neq 0 (
    echo Error installing backend dependencies.
    pause
    exit /b 1
)

echo.
echo [1/2] Starting Backend (Port 8000)...
start "HNSW Backend" cmd /k "call .venv\Scripts\activate.bat && python web_app/backend/server.py"

echo.
echo [2/2] Starting Frontend (Port 5173)...
cd web_app/frontend
if not exist "node_modules" (
    echo Installing Frontend dependencies...
    call npm install
)
start "HNSW Frontend" cmd /k "npm run dev"

echo.
echo ---------------------------------------------------
echo Web App is starting!
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Close the opened command windows to stop the services.
echo ---------------------------------------------------
pause
