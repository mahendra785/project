#!/usr/bin/env bash
# One-time setup for PUMS: Python venv + dependencies, npm install, samples.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Creating the Python virtual environment ..."
cd "$root/backend"
[ -d .venv ] || python3 -m venv .venv

if [ -x .venv/bin/python ]; then python_bin=.venv/bin/python; else python_bin=.venv/Scripts/python.exe; fi

echo "[2/4] Installing backend dependencies ..."
"$python_bin" -m pip install --upgrade pip --quiet
"$python_bin" -m pip install -r requirements.txt

echo "[3/4] Generating synthetic sample phantoms ..."
"$python_bin" scripts/make_sample_images.py

echo "[4/4] Installing frontend dependencies ..."
cd "$root/frontend"
npm install

echo
echo "Setup complete. Start both services with scripts/dev.sh"
