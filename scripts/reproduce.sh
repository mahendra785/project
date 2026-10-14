#!/usr/bin/env bash
# Regenerate every MG-CRC number, table and paper figure input.
#
#   ./scripts/reproduce.sh            # full: train backbones (CPU RF) + experiments + aggregate
#   ./scripts/reproduce.sh --fast     # skip training; reuse backend/var/posteriors
#   BACKBONE=unet-r34 SIZE=512 ./scripts/reproduce.sh   # GPU backbone (needs requirements-torch.txt)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$root/backend/.venv/bin/python"; [ -x "$py" ] || py="$root/backend/.venv/Scripts/python.exe"
BACKBONE="${BACKBONE:-rf}"; SIZE="${SIZE:-256}"; SEEDS="${SEEDS:-0 1 2}"; DATASETS="${DATASETS:-fracatlas kvasir-seg}"
cd "$root/backend"
if [ "${1:-}" != "--fast" ]; then
  for ds in $DATASETS; do for s in $SEEDS; do
    "$py" scripts/experiments/train_backbone.py --dataset "$ds" --backbone "$BACKBONE" --size "$SIZE" --seed "$s"
  done; done
fi
for ds in $DATASETS; do for s in $SEEDS; do
  "$py" scripts/experiments/mgcrc.py --dataset "$ds" --backbone "$BACKBONE" --seed "$s"
done; done
"$py" scripts/experiments/simulation.py
"$py" scripts/experiments/sim_diagnostic.py
for ds in $DATASETS; do "$py" scripts/experiments/posthoc_multi.py --dataset "$ds" --backbone "$BACKBONE"; done
"$py" scripts/experiments/aggregate.py > /dev/null
"$py" scripts/experiments/paper_macros.py > /dev/null
mkdir -p "$root/docs/research/results/backbones"
for d in var/posteriors/*/; do for f in "$d"*.json; do [ -e "$f" ] && cp "$f" "$root/docs/research/results/backbones/$(basename "$d")_$(basename "$f")"; done; done
echo "done -> docs/research/results/summary.md, paper/data/*.csv"
