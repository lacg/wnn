#!/bin/bash
# Controller fitness-weight sweep on the FULL phased_ga pipeline (the SAME
# pipeline as the gain runs — representative, unlike the old neurons-only
# curriculum --mode sweep). Per combo: grid → GA-neurons → GA-memory, with all
# four gated options ON:
#   - splitting           (WNN_STATE_SPLIT=1, conflict-driven state in training)
#   - lamarckian          (--lamarckian: carry+write-back trained cells across gens)
#   - cell persistence    (--save-winner saves the genome WITH its cells)
#   - per-stage held-out  (--report-seed: honest held-out at end of neurons + memory)
# Runs all 18 combos (W1-W4 + C1-C14) sequentially. Detached + checkpointed.
set -u
cd /Users/lacg/wnn
export PYTHONPATH="/Users/lacg/wnn/src/wnn"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=3
PY=/Users/lacg/wnn-venv/bin/python
BASE=logs/controller/wsweep_phased_20260610
mkdir -p "$BASE"

# combo  err   stable jerk  mono   (the 18 SWEEP_COMBOS from run_curriculum_ga.py)
COMBOS=(
  "W1 0.50 0.40 0.05 0.05"  "W2 0.40 0.50 0.05 0.05"  "W3 0.60 0.30 0.05 0.05"  "W4 0.45 0.35 0.10 0.10"
  "C1 0.20 0.40 0.20 0.20"  "C2 0.20 0.50 0.10 0.20"  "C3 0.20 0.50 0.20 0.10"  "C4 0.30 0.30 0.20 0.20"
  "C5 0.30 0.40 0.10 0.20"  "C6 0.30 0.40 0.20 0.10"  "C7 0.30 0.50 0.10 0.10"  "C8 0.40 0.20 0.20 0.20"
  "C9 0.40 0.30 0.10 0.20"  "C10 0.40 0.30 0.20 0.10" "C11 0.40 0.40 0.10 0.10" "C12 0.50 0.20 0.10 0.20"
  "C13 0.50 0.20 0.20 0.10" "C14 0.50 0.30 0.10 0.10"
)
# Allow a subset for cull-down rounds:  run_weight_sweep_phased.sh W2 C2 C11 ...
WANT=("$@")
in_want() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

for entry in "${COMBOS[@]}"; do
  set -- $entry
  NAME=$1 E=$2 S=$3 J=$4 M=$5
  in_want "$NAME" || continue
  D="$BASE/$NAME"; mkdir -p "$D"
  echo "[wsweep] $(date '+%Y-%m-%d %H:%M:%S') START $NAME  err=$E stable=$S jerk=$J mono=$M"
  $PY -u tests/run_phased_ga.py \
    --pop 30 --num-eval-folds 3 --elitism 0.2 --crossover-rate 0.5 \
    --tilt 5 --body-rate 0.5 --yaw-rate 0.3 \
    --lamarckian --skip-stages bits,connections \
    --grid-state-neurons 8 12 16 --grid-bits 24 30 \
    --steps 500 --eval-episodes 8 --universe-episodes 5 \
    --rg-rounds 3 --rg-episodes-per-round 8 \
    --neurons-gens 30 --neurons-patience 4 \
    --memory-gens 50 --memory-patience 6 --check-interval 5 \
    --saturation-grow-gain 1.0 \
    --fit-weight-err-sq "$E" --fit-weight-stable "$S" \
    --fit-weight-jerk "$J" --fit-weight-mono "$M" \
    --train-workers 4 \
    --base-seed 20260609 --report-seed 99990001 \
    --save-stage-checkpoints "$D" --save-winner "$D/winner.pkl" \
    > "$D/run.out" 2>&1
  echo "[wsweep] $(date '+%Y-%m-%d %H:%M:%S') DONE $NAME (exit $?)"
done
echo "[wsweep] $(date '+%Y-%m-%d %H:%M:%S') SWEEP COMPLETE (${#COMBOS[@]} combos or subset)."
