#!/bin/bash
# Sequential controller gain sweep (from-scratch grid->neurons->memory, splitting):
#   Run 1: saturation_grow_gain=0.02 (mild damp), pop50/kfold5, quick levers
#   Run 2: saturation_grow_gain=1.0  (≈undamped force-grow), pop50/kfold5, production-ish
# Both at 5° matched IC, lamarckian, from scratch. Run 2 starts after Run 1 exits.
set -u
cd /Users/lacg/wnn
export PYTHONPATH="/Users/lacg/wnn/src/wnn"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=3
PY=/Users/lacg/wnn-venv/bin/python
BASE=logs/controller/sweep_20260609
D1="$BASE/run1_gain0p02"
D2="$BASE/run2_gain1p0"
mkdir -p "$D1" "$D2"

echo "[seq] $(date '+%Y-%m-%d %H:%M:%S') START RUN 1 (gain 0.02, pop50/kfold5)"
$PY -u tests/run_phased_ga.py \
  --pop 50 --num-eval-folds 5 --elitism 0.2 --crossover-rate 0.5 \
  --tilt 5 --body-rate 0.5 --yaw-rate 0.3 \
  --lamarckian --skip-stages bits,connections \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 \
  --steps 800 --eval-episodes 12 --universe-episodes 6 \
  --rg-rounds 4 --rg-episodes-per-round 12 \
  --neurons-gens 40 --neurons-patience 2 \
  --memory-gens 60 --memory-patience 3 --check-interval 5 \
  --saturation-grow-gain 0.02 \
  --train-workers 4 \
  --base-seed 20260609 --report-seed 7777 \
  --save-stage-checkpoints "$D1" --save-winner "$D1/winner.pkl" \
  > "$D1/run.out" 2>&1
RC1=$?
echo "[seq] $(date '+%Y-%m-%d %H:%M:%S') RUN 1 done (exit $RC1) -> START RUN 2 (gain 1.0, production-ish)"

$PY -u tests/run_phased_ga.py \
  --pop 50 --num-eval-folds 5 --elitism 0.2 --crossover-rate 0.5 \
  --tilt 5 --body-rate 0.5 --yaw-rate 0.3 \
  --lamarckian --skip-stages bits,connections \
  --grid-state-neurons 8 12 16 20 --grid-bits 24 30 36 \
  --steps 1000 --eval-episodes 20 --universe-episodes 8 \
  --rg-rounds 6 --rg-episodes-per-round 16 \
  --neurons-gens 100 --neurons-patience 4 \
  --memory-gens 200 --memory-patience 8 --check-interval 5 \
  --saturation-grow-gain 1.0 \
  --train-workers 4 \
  --base-seed 20260609 --report-seed 7777 \
  --save-stage-checkpoints "$D2" --save-winner "$D2/winner.pkl" \
  > "$D2/run.out" 2>&1
RC2=$?
echo "[seq] $(date '+%Y-%m-%d %H:%M:%S') RUN 2 done (exit $RC2). SEQUENCE COMPLETE."
