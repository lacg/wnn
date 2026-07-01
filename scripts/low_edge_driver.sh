#!/bin/bash
# Low-edge lean-the-architecture sweep (01/07/2026). Goal is NOT to break 90% — it's to find how
# LEAN (few bits + few state neurons = small memory, fast, FPGA-friendly) we can go before stable%
# finally cliffs. The bit-sweep {24,40,64} showed the peak is ≤24 (b24 pooled 83.5, best+tightest);
# b24 seed10 grew to only eff-25 and was the fastest/tightest cell → plateau holds ≥eff-25, lower
# edge un-mapped. This probes BELOW that. Plan: .claude/plans/controller_break_90pct.md.
#
# CONSTRAINT (evaluator.py:316): state_bits_per_neuron >= state_neurons (forced 1-bit/neuron prev-state
# prefix). So going leaner = lower BOTH. grid-state-neurons {4,8,12} + grid-bits {12,16,20} (all valid;
# grid auto-filters any sn>bits pair). eff-sb lands below the b24 plateau.
#
# 2 substrates (s16 obs-OFF = cleanest lean baseline; pidmix_pwm = continuity w/ the bit-sweep curve)
# × grid-bits {12,16,20} × 2 seeds = 12 cells. folds=5. seed-outer. Self-gates on the A/B/C finishing.
# NB: magnitude-aware patience is now the controller DEFAULT (01/07) but passed explicitly for parity.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/LowEdge_20260701.log
exec >>"$LOG" 2>&1

echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') WAITING for state-integral A/B/C (/tmp/wnn_state_integral_done.json)"
while [ ! -f /tmp/wnn_state_integral_done.json ]; do sleep 60; done
echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') A/B/C done — starting low-edge lean sweep"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
PIDFLAGS="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
ROOT=logs/controller/LowEdge_20260701
GSN="4 8 12"                                 # leaner state layer (bits must be >= max sn = 12)
GBITS_SWEEP="12 16 20"                        # below the b24 plateau
SEEDS="20260609 20260610"                     # seed-outer
# substrate: name | obs-flags  (s16 = empty)
SUBS=( "s16|" "pidmix_pwm|$PIDFLAGS --obs-pwm" )

run_one() {
  local name="$1" flags="$2" gbits="$3" seed="$4"
  local dir="$ROOT/${name}_b${gbits}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[lowedge] $(date '+%H:%M:%S') SKIP ${name} b=${gbits} seed=${seed}"; return; fi
  echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') START ${name} b=${gbits} seed=${seed}"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons $GSN --grid-bits $gbits --levels 16 --bits-per-feature 8 \
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
  if [ $? -ne 0 ]; then echo "[lowedge] $(date '+%H:%M:%S') FAIL ${name} b=${gbits} seed=${seed} (continuing)"
  else echo "{\"substrate\":\"${name}\",\"grid_bits\":${gbits},\"seed\":${seed}}" > "$dir/done.json"
       echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} b=${gbits} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for s in "${SUBS[@]}"; do
    IFS='|' read -r name flags <<< "$s"
    for gb in $GBITS_SWEEP; do run_one "$name" "$flags" "$gb" "$seed"; done
  done
done
echo "{\"low_edge_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_low_edge_done.json
echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
