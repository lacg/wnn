#!/bin/bash
# Frame-misalignment fix VALIDATION sweep (27/06/2026, commit ae3e0214).
# Re-runs the obs-feature controller variants now that arch_shape_from_spec uses
# spec.num_features() — so the forced full-state prefix finally lands on the recurrent
# state (320 at nf=10) instead of inside the sensor region (288). Before the fix EVERY
# obs run was effectively memoryless (grid configs gave identical CE/stable — the tell).
#
# Question: does the fix turn coupled-anchor 70.5% -> ~S16 (87.2%)? And do the other
# obs families (decouple, peraxis, tilt, pwm) — all previously mis-wired — now recover?
#
# All variants share the EXACT S16/absolute base recipe (weights .25/.35/.20/.15/.05,
# grid 8/12/16 x 24/30b, levels 16, bpf 8, lamarckian, folds=3, neurons+memory stages).
# Only the obs flags differ. SEQUENTIAL (one GPU job at a time; never contend the IDS
# worker or itself — the prior seed-11 anchor run was OOM-Killed under contention).
#
# Interleaved per feedback_sweeps_always_interleave: SEED is the OUTER loop, so round 1
# (seed 09) covers every variant before round 2 (seed 10) — enables early read/cull.
set -u
cd /Users/lacg/wnn
ROOT=logs/controller/FrameFixVal_20260627
mkdir -p "$ROOT"
LOG="$ROOT/driver.log"; exec >>"$LOG" 2>&1

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05

# variant name | obs flags (only the delta from the base recipe)
VARIANTS=(
  "s16|"
  "anchor|--obs-yaw-err"
  "decouple|--obs-yaw-err --decouple-outputs"
  "peraxis|--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw"
  "tilt|--obs-tilt-p --obs-tilt-i"
  "pwm|--obs-pwm"
)
SEEDS="20260609 20260610"

run_one() {
  local name="$1" flags="$2" seed="$3"
  local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[ffv] $(date '+%H:%M:%S') SKIP ${name} seed=${seed} (already done)"; return; fi
  echo "[ffv] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} flags=[${flags}]"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
    --no-delta-control $flags --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds 5 \
    --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$seed" --runs 1 \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[ffv] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} rc=$rc (continuing)"
  else
    echo "{\"name\":\"${name}\",\"seed\":${seed},\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > "$dir/done.json"
    echo "[ffv] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"
  fi
}

echo "[ffv] $(date '+%Y-%m-%d %H:%M:%S') START frame-fix validation (${#VARIANTS[@]} variants x 2 seeds, sequential, seed-interleaved)"
for seed in $SEEDS; do
  echo "[ffv] ===== ROUND seed=${seed} ====="
  for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"; flags="${v#*|}"
    run_one "$name" "$flags" "$seed"
  done
done
echo "{\"frame_fix_val_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_frame_fix_val_done.json
echo "[ffv] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
