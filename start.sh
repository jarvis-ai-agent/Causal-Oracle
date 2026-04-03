#!/bin/bash
# Start Causal Oracle — backend + frontend

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Causal Oracle..."
echo "Root: $ROOT"

# Start backend
cd "$ROOT/backend"
source .venv/bin/activate
echo "[Backend] Starting FastAPI on http://localhost:8000 ..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd "$ROOT/frontend"
echo "[Frontend] Starting Vite on http://localhost:5173 ..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✓ Backend:  http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo "✓ API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
