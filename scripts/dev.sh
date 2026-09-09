#!/usr/bin/env bash
# Start the PUMS backend and frontend together (macOS / Linux / Git Bash).
# Run scripts/setup.sh once first.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -x "$root/backend/.venv/bin/python" ]; then
  python_bin="$root/backend/.venv/bin/python"
elif [ -x "$root/backend/.venv/Scripts/python.exe" ]; then
  python_bin="$root/backend/.venv/Scripts/python.exe"
else
  echo "Backend venv not found. Run scripts/setup.sh first." >&2
  exit 1
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting PUMS backend  -> http://localhost:8000/docs"
(cd "$root/backend" && "$python_bin" -m uvicorn app.main:app --reload --port 8000) &

echo "Starting PUMS frontend -> http://localhost:3000"
(cd "$root/frontend" && npm run dev) &

echo
echo "The first analysis builds the synthetic calibration set (~15s)."
wait
