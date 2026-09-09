#!/usr/bin/env bash
# Run the whole test suite: backend pytest + frontend node --test.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -x "$root/backend/.venv/bin/python" ]; then
  python_bin="$root/backend/.venv/bin/python"
else
  python_bin="$root/backend/.venv/Scripts/python.exe"
fi

echo "=== Backend (pytest) ==="
(cd "$root/backend" && "$python_bin" -m pytest)

echo
echo "=== Frontend (node --test) ==="
(cd "$root/frontend" && npm test)

echo
echo "=== Frontend typecheck ==="
(cd "$root/frontend" && npm run typecheck)
