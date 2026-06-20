#!/bin/bash
# QUICK controller fitness-weight RE-SWEEP on the FIXED scorer (delta-mode fix
# 9d466189 + 1-bit recurrence fix 2d71f709). The original wsweep (20260610, →C10)
# was decided on the BUGGED scorer, so the weight choice is invalid. This re-runs
# the SAME 18 combos, SAME seeds (base 20260609 / report 99990001), SAME default
# (delta) substrate — only the scorer changed + shorter gens (quick) + throttled
# (RAYON=2) so it shares the box with the deadline-priority XDS worker.
# Compare the held-out ranking here vs wsweep_phased_20260610 to re-pick weights.
# v2 (multi-seed): held-out scored at --report-episodes 100 (NOT the search's 8 eps;
# the 8-ep held-out was the cause of W1's noisy/inflated 87.5% → real ~58%) over 4
# fixed report seeds (--report-seeds), reported as mean±std — robust to the documented
# single-seed controller-eval variance.
set -u
cd /Users/lacg/wnn
export PYTHONPATH="/Users/lacg/wnn/src/wnn"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2   # throttled (XDS priority)
PY=/Users/lacg/wnn-venv/bin/python
BASE=logs/controller/wsweep_refixed_ms_20260620
mkdir -p "$BASE"

# combo  err   stable jerk  mono   (identical 18 combos to the original sweep)
COMBOS=(
  "W1 0.50 0.40 0.05 0.05"  "W2 0.40 0.50 0.05 0.05"  "W3 0.60 0.30 0.05 0.05"  "W4 0.45 0.35 0.10 0.10"
  "C1 0.20 0.40 0.20 0.20"  "C2 0.20 0.50 0.10 0.20"  "C3 0.20 0.50 0.20 0.10"  "C4 0.30 0.30 0.20 0.20"
  "C5 0.30 0.40 0.10 0.20"  "C6 0.30 0.40 0.20 0.10"  "C7 0.30 0.50 0.10 0.10"  "C8 0.40 0.20 0.20 0.20"
  "C9 0.40 0.30 0.10 0.20"  "C10 0.40 0.30 0.20 0.10" "C11 0.40 0.40 0.10 0.10" "C12 0.50 0.20 0.10 0.20"
  "C13 0.50 0.20 0.20 0.10" "C14 0.50 0.30 0.10 0.10"
)
WANT=("$@")
in_want() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

for entry in "${COMBOS[@]}"; do
  set -- $entry
  NAME=$1 E=$2 S=$3 J=$4 M=$5
  in_want "$NAME" || continue
  D="$BASE/$NAME"; mkdir -p "$D"
  echo "[resweep] $(date '+%Y-%m-%d %H:%M:%S') START $NAME  err=$E stable=$S jerk=$J mono=$M"
  $PY -u tests/run_phased_ga.py \
    --pop 30 --num-eval-folds 3 --elitism 0.2 --crossover-rate 0.5 \
    --tilt 5 --body-rate 0.5 --yaw-rate 0.3 \
    --lamarckian --skip-stages bits,connections \
    --grid-state-neurons 8 12 16 --grid-bits 24 30 \
    --steps 500 --eval-episodes 8 --universe-episodes 5 \
    --rg-rounds 3 --rg-episodes-per-round 8 \
    --neurons-gens 15 --neurons-patience 3 \
    --memory-gens 25 --memory-patience 4 --check-interval 3 \
    --saturation-grow-gain 1.0 \
    --fit-weight-err-sq "$E" --fit-weight-stable "$S" \
    --fit-weight-jerk "$J" --fit-weight-mono "$M" \
    --train-workers 2 \
    --base-seed 20260609 \
    --report-episodes 100 \
    --report-seeds 99990001 99990101 12345 67890 \
    --save-stage-checkpoints "$D" --save-winner "$D/winner.pkl" \
    > "$D/run.out" 2>&1
  echo "[resweep] $(date '+%Y-%m-%d %H:%M:%S') DONE $NAME (exit $?)"
done
echo "[resweep] $(date '+%Y-%m-%d %H:%M:%S') RESWEEP COMPLETE."
