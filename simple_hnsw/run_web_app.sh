#!/bin/bash

# Ensure we are in project root
cd "$(dirname "$0")"

echo "Starting HNSW Web Visualization..."

# 1. Start Backend
echo "[1/2] Starting Backend (FastAPI on port 8000)..."
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD web_app/backend/server.py > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID."

# 2. Start Frontend
echo "[2/2] Starting Frontend (Vite on port 5173)..."
cd web_app/frontend
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID."

echo "---------------------------------------------------"
echo "Web App is running!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "---------------------------------------------------"
echo "Press Ctrl+C to stop all services."

# Trap to kill processes on exit
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

wait
