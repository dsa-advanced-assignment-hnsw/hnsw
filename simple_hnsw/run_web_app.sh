#!/bin/bash

# Ensure we are in project root
cd "$(dirname "$0")"

echo "Starting HNSW Web Visualization..."

# ==========================================
# 1. Backend Setup & Start
# ==========================================
echo "[1/2] Checking Backend Environment..."

# Check/Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate Environment
source .venv/bin/activate

# Install Dependencies (ensures they are always up to date)
echo "Installing/Updating backend dependencies..."
pip install -r web_app/backend/requirements.txt > /dev/null

echo "Starting Backend (FastAPI on port 8000)..."
python web_app/backend/server.py > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID."

# ==========================================
# 2. Frontend Setup & Start
# ==========================================
echo "[2/2] Checking Frontend Environment..."
cd web_app/frontend

# Check/Install Node Modules
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (this may take a moment)..."
    npm install
fi

echo "Starting Frontend (Vite on port 5173)..."
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID."

# ==========================================
# 3. Running
# ==========================================
echo "---------------------------------------------------"
echo "Web App is running!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "---------------------------------------------------"
echo "Press Ctrl+C to stop all services."

# Trap to kill processes on exit
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

wait