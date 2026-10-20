#!/usr/bin/env bash
# Revision runs registered in pre-registration Amendment 3 (frozen-model protocol).
# Usage: ./scripts/run_revision.sh [train|calibrate|ablate|aggregate|all]
set -euo pipefail
cd "$(dirname "$0")/../backend"
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
EXP=scripts/experiments
LOG=../docs/research/results_v2/logs
mkdir -p "$LOG"
STAGE=${1:-all}
DATASETS_2D=(kvasir-seg fracatlas)
ARCHS=(unet-r34 unet-mitb2)
SEEDS=(0 1 2)
have_task08() { [ -d var/datasets/msd-hepatic-vessel ] || ls ../data/datasets/msd-hepatic-vessel/*/imagesTr >/dev/null 2>&1; }

train() {
  for ds in "${DATASETS_2D[@]}"; do for a in "${ARCHS[@]}"; do for s in "${SEEDS[@]}"; do
    [ -f "var/posteriors_frozen/$ds/${a}_seed$s.json" ] && continue
    $PY $EXP/train_frozen.py --dataset "$ds" --arch "$a" --seed "$s" 2>&1 | tee "$LOG/train_${ds}_${a}_$s.log"
  done; done; done
  if have_task08; then for a in "${ARCHS[@]}"; do for s in "${SEEDS[@]}"; do
    [ -f "var/posteriors_frozen/msd-hepatic-vessel/${a}_seed$s.json" ] && continue
    $PY $EXP/train_frozen_msd.py --arch "$a" --seed "$s" 2>&1 | tee "$LOG/train_task08_${a}_$s.log"
  done; done; fi
}

calibrate() {
  for f in var/posteriors_frozen/*/*_seed*.json; do
    ds=$(basename "$(dirname "$f")"); stem=$(basename "$f" .json); a=${stem%_seed*}; s=${stem##*_seed}
    $PY $EXP/mgcrc_v2.py --dataset "$ds" --arch "$a" --seed "$s" --tag main 2>&1 | tee "$LOG/main_${ds}_${a}_$s.log"
  done
}

ablate() {  # secondary analyses, U-Net ResNet-34, every seed (descriptive, no decision rule)
  for f in var/posteriors_frozen/*/unet-r34_seed*.json; do
    ds=$(basename "$(dirname "$f")"); s=$(basename "$f" .json); s=${s##*_seed}
    run() { $PY $EXP/mgcrc_v2.py --dataset "$ds" --arch unet-r34 --seed "$s" --boot 50 --boot-reps 50 "$@" 2>&1 | tail -1; }
    run --tag abltau25 --tau 0.25
    run --tag abltau75 --tau 0.75
    run --tag ablq25 --small-q 25
    run --tag ablq50 --small-q 50
    [ "$ds" != msd-hepatic-vessel ] && run --tag ablconn4 --conn4
    for n in 50 100 200; do run --tag ablncal$n --ncal $n; done
  done
}

aggregate() { $PY $EXP/aggregate_v2.py; }

case $STAGE in
  train) train ;; calibrate) calibrate ;; ablate) ablate ;; aggregate) aggregate ;;
  all) train; calibrate; ablate; aggregate ;;
esac
