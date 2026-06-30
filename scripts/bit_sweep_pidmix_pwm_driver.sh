#!/bin/bash
# Bit-budget sweep for pidmix_pwm (30/06/2026). Tests whether a MIDDLE-GROUND per-neuron
# address width beats both R1(~52b eff, 87.2%) and R2(~115b eff, 85.2%), or whether the
# peak sits AT/BELOW 52 (the prediction from the wish_bits=0 / 2× sparser-memory analysis).
#
# CLEAN DESIGN: folds=5 FIXED at every point (no folds-3 confound — the R1/R2 ambiguity was
# exactly that R1 used folds=3). Bits is the ONLY free variable. 2 seeds (09+10) per bit-width
# because pidmix_pwm's seed-to-seed SD is ±8-12pp (frame-fix), so a single seed would misread
# the curve. seed-outer order: seed09 full 3-pt curve first (scout), then seed10. grid-bits
# grows ~+20 in the neurons-GA, so:
#     grid 24 -> eff ~44  (below 52: does saturation finally bite?)
#     grid 40 -> eff ~60  (the 52-64 region)
#     grid 64 -> eff ~84  (the "middle ground" hypothesis, head-on)
# Readout = held-out (ho-mem) vs bits, PLUS the split-pressure wish_bits/saturation + cells
# (distinct addresses) at each point — the inflection IS the peak, measured not inferred.
#
# Recipe is byte-identical to frame_fix_pidmix_driver.sh's pidmix_pwm cell EXCEPT grid-bits
# (swept) and folds (pinned 5). Self-gates: waits for the frame-fix sweep to finish first so
# it never competes with the planned runs for cores/memory.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/BitSweep_pidmix_pwm_20260630.log
exec >>"$LOG" 2>&1

echo "[bitsweep] $(date '+%Y-%m-%d %H:%M:%S') WAITING for frame-fix sweep (/tmp/wnn_frame_fix_pidmix_done.json)"
while [ ! -f /tmp/wnn_frame_fix_pidmix_done.json ]; do sleep 60; done
echo "[bitsweep] $(date '+%Y-%m-%d %H:%M:%S') frame-fix done — starting folds=5 bit sweep"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
PIDFLAGS="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
FLAGS="$PIDFLAGS --obs-pwm"               # = pidmix_pwm (nf=19)
ROOT=logs/controller/BitSweep_pidmix_pwm_20260630
SEEDS="20260609 20260610"                  # 2-seed avg (pidmix_pwm seed-SD is ±8-12pp — single seed misleads)
FOLDS=5                                     # PINNED — the whole point
GBITS_SWEEP="24 40 64"                      # single value per run -> attributable to that width

run_one() {
  local gbits="$1" SEED="$2"
  local dir="$ROOT/pidmix_pwm_b${gbits}_seed${SEED}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[bitsweep] $(date '+%H:%M:%S') SKIP b=${gbits}"; return; fi
  echo "[bitsweep] $(date '+%Y-%m-%d %H:%M:%S') START b=${gbits} folds=${FOLDS} seed=${SEED}"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits $gbits --levels 16 --bits-per-feature 8 \
    --no-delta-control $FLAGS --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds $FOLDS \
    --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$SEED" --runs 1 \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  if [ $? -ne 0 ]; then echo "[bitsweep] $(date '+%H:%M:%S') FAIL b=${gbits} (continuing)"
  else echo "{\"variant\":\"pidmix_pwm\",\"grid_bits\":${gbits},\"folds\":${FOLDS},\"seed\":${SEED}}" > "$dir/done.json"
       echo "[bitsweep] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE b=${gbits}"; fi
}

# seed-outer: seed09 full curve first (scout), then seed10 completes the 2-seed avgs
for seed in $SEEDS; do for gb in $GBITS_SWEEP; do run_one "$gb" "$seed"; done; done
echo "{\"bit_sweep_pidmix_pwm_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_bit_sweep_pidmix_pwm_done.json
echo "[bitsweep] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
