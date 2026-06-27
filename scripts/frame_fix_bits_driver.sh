#!/bin/bash
# Frame-fix HIGHER-BITS round (27/06/2026). Repeats the EXACT validation sweep
# (same 6 variants, same seeds 09/10, seed-interleaved, sequential) but with:
#   --grid-bits 48 72   (was 24 30) → feature-sampling suffix ~32-64b (was ~8-22b),
#                       i.e. ~3-6 bits/feature instead of ~1.4 (relieves the starvation)
#   --num-eval-folds 5  (was 3 — now the locked rule; damps overfit + variance)
# Everything else identical to frame_fix_validation_driver.sh → apples-to-apples on the
# SAME seeds, so the only deltas are bits + folds. Question: does more feature-wiring lift
# the whole field (including the s16 baseline) past the ~77-81% folds=3 cluster?
#
# NOTE: as neurons grow, total bits/neuron can exceed 64 → address uses a lossy hash
# (same one IDS runs at 100b — separates addresses fine). Uniform across variants ⇒ fair.
#
# SELF-GATING: waits for the validation sweep to finish before starting (chains overnight).
set -u
cd /Users/lacg/wnn
ROOT=logs/controller/FrameFixBits_20260627
mkdir -p "$ROOT"
LOG="$ROOT/driver.log"; exec >>"$LOG" 2>&1

# wait for the prerequisite sweep (poll; this runs detached so sleep is fine here)
echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') WAITING for val sweep (/tmp/wnn_frame_fix_val_done.json)"
while [ ! -f /tmp/wnn_frame_fix_val_done.json ]; do sleep 60; done
echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') val sweep done — starting higher-bits round"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05

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
  if [ -f "$dir/done.json" ]; then echo "[ffb] $(date '+%H:%M:%S') SKIP ${name} seed=${seed} (already done)"; return; fi
  echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} flags=[${flags}]"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 48 72 --levels 16 --bits-per-feature 8 \
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
    echo "[ffb] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} rc=$rc (continuing)"
  else
    echo "{\"name\":\"${name}\",\"seed\":${seed},\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > "$dir/done.json"
    echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"
  fi
}

echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') START higher-bits round (grid-bits 48 72, folds 5, ${#VARIANTS[@]} variants x 2 seeds)"
for seed in $SEEDS; do
  echo "[ffb] ===== ROUND seed=${seed} ====="
  for v in "${VARIANTS[@]}"; do
    name="${v%%|*}"; flags="${v#*|}"
    run_one "$name" "$flags" "$seed"
  done
done
echo "{\"frame_fix_bits_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_frame_fix_bits_done.json
echo "[ffb] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
