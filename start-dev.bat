@echo off
setlocal

echo [LexigenAI] Starting backend and frontend in dev mode...

if not exist "frontend\node_modules" (
  echo [LexigenAI] Installing frontend dependencies...
  cd frontend
  call npm install
  cd ..
)

start "LexigenAI Backend" cmd /k "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
start "LexigenAI Frontend" cmd /k "cd frontend && npm run dev"

echo [LexigenAI] Done.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173

endlocal
