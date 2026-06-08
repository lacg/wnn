#!/bin/bash
# Phase-1 pure-tilt crossover scan DRIVER (runs sequentially, resumable).
# Recipe per tilt: grid -> NEURONS -> MEMORY (skip bits+connections), lamarckian.
# Each run auto-computes its PID baseline -> a WNN-vs-PID pair per tilt.
# Goal: find the tilt where the WNN catches/passes PID (PID is near-optimal at 5°).
#
# Resumable: skips any tilt whose winner.pkl already exists. RAYON=3 (off IDS's back).
# Launched detached by scripts/launch_tilt_sweep_phase1.sh.
set -uo pipefail
cd /Users/lacg/wnn
VENV_PY="$(pwd)/wnn/bin/python"
export PYTHONPATH="$(pwd)/src/wnn:${PYTHONPATH:-}"

TILTS="10 15 20 25 30 35 40 45"   # 5° already done (Arm A); skipped here
BASE="logs/controller/tilt_sweep_phase1"
SUMMARY="$BASE/SUMMARY.md"
mkdir -p "$BASE"
[ -f "$SUMMARY" ] || printf "# Phase-1 pure-tilt crossover scan (WNN grid->neurons->memory vs PID)\n\nbase-seed 5005, report-seed 9009, pop 50, neurons 30/2, memory 60/2, body 0.5 yaw 0.3.\n\n| tilt | WNN held-out (err/stable) | PID held-out (err/stable) | WNN-PID err gap |\n|---|---|---|---|\n" > "$SUMMARY"

for T in $TILTS; do
  DIR="$BASE/tilt${T}"
  if [ -f "$DIR/winner.pkl" ]; then
    echo "[$(date '+%H:%M:%S')] tilt ${T}° already done (winner.pkl present) — skip"
    continue
  fi
  mkdir -p "$DIR"
  LOG="$DIR/tilt${T}.log"
  echo "[$(date '+%H:%M:%S')] === tilt ${T}° START ==="
  RAYON_NUM_THREADS=3 "$VENV_PY" -u tests/run_phased_ga.py \
    --tilt "$T" --body-rate 0.5 --yaw-rate 0.3 --steps 250 \
    --pop 50 --elitism 0.2 --check-interval 5 \
    --neurons-gens 30 --neurons-patience 2 \
    --skip-stages bits,connections \
    --memory-gens 60 --memory-patience 2 \
    --eval-episodes 100 --universe-episodes 8 --num-eval-folds 5 \
    --fit-weight-err-sq 0.40 --fit-weight-stable 0.30 --fit-weight-jerk 0.10 --fit-weight-mono 0.20 \
    --base-seed 5005 --report-seed 9009 --lamarckian \
    --save-stage-checkpoints "$DIR" --save-winner "$DIR/winner.pkl" \
    > "$LOG" 2>&1
  rc=$?
  # Extract the final (memory) held-out WNN result + the PID held-out baseline.
  WNN=$(grep -E "RESULT —" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*stable=([0-9.]+)%[[:space:]]*err=([0-9.]+)°.*/\2°\/\1%/')
  PID=$(grep -E "vs PID  \(held-out\)" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*stable=([0-9.]+)%[[:space:]]*err=([0-9.]+)°.*/\2°\/\1%/')
  echo "[$(date '+%H:%M:%S')] === tilt ${T}° DONE rc=$rc | WNN ${WNN:-?} | PID ${PID:-?} ==="
  printf "| %s° | %s | %s | (see log) |\n" "$T" "${WNN:-?}" "${PID:-?}" >> "$SUMMARY"
done
echo "[$(date '+%H:%M:%S')] ===== PHASE-1 TILT SWEEP COMPLETE ====="
