#!/bin/bash
# Low-edge INPUT-BITS lean sweep v2 (01/07/2026). v1 was WRONG (varied TOTAL bits). MECHANISM (found
# in recurrent_genome.py): total_state_bits = prefix_factor·state_neurons + state_suffix(=INPUT-bits).
# NEURONS phase grows sn (delta 1); INPUT-bits (suffix) are mutated ONLY in the BITS phase, which we
# SKIP → suffix is FIXED by the grid seed = grid_bits − pf·grid_sn. So "8b" = 8 INPUT bits, sn separate,
# total = 8 + sn. The bit-sweep's REAL axis was input-bits: {24,40,64 total} = suffix {16,28,56} →
# 83.5/77.4/74.8, input-16 best. This maps BELOW 16 to find the lean cliff = min viable memory (FPGA).
#
# Design: FIX --grid-state-neurons 8 (neurons phase grows the real sn; we only pin the suffix). Sweep
# INPUT-bits {4,8,12,16} → grid_bits = input + 8 = {12,16,20,24}. input-16 (grid 24) = the bit-sweep
# anchor for continuity. x-axis reported = input-bits = total_sb − sn. Goal = LEAN (small/fast/FPGA),
# NOT break 90. 2 substrates (s16 obs-OFF cleanest; pidmix_pwm continuity) × 4 inputs × 2 seeds = 16.
# folds=5. seed-outer. Chains after the A/B/C. (magnitude-aware patience is now the controller DEFAULT.)
set -u
cd /Users/lacg/wnn
LOG=logs/controller/LowEdge_20260701.log
exec >>"$LOG" 2>&1

echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') WAITING for state-integral A/B/C (/tmp/wnn_state_integral_done.json)"
while [ ! -f /tmp/wnn_state_integral_done.json ]; do sleep 60; done
echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') A/B/C done — starting input-bits lean sweep"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
PIDFLAGS="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
ROOT=logs/controller/LowEdge_20260701
GSN=8                                          # fixed seed sn; neurons phase grows the REAL sn
INPUTS="4 8 12 16"                              # INPUT-bits (suffix). grid_bits = input + GSN
SEEDS="20260609 20260610"
SUBS=( "s16|" "pidmix_pwm|$PIDFLAGS --obs-pwm" )

run_one() {
  local name="$1" flags="$2" input="$3" seed="$4"
  local gbits=$(( input + GSN ))               # total = input + forced-prefix(sn seed)
  local dir="$ROOT/${name}_in${input}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[lowedge] $(date '+%H:%M:%S') SKIP ${name} in=${input} seed=${seed}"; return; fi
  echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') START ${name} in=${input} (grid_bits=${gbits}) seed=${seed}"
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
  if [ $? -ne 0 ]; then echo "[lowedge] $(date '+%H:%M:%S') FAIL ${name} in=${input} seed=${seed} (continuing)"
  else echo "{\"substrate\":\"${name}\",\"input_bits\":${input},\"grid_bits\":${gbits},\"seed\":${seed}}" > "$dir/done.json"
       echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} in=${input} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for s in "${SUBS[@]}"; do
    IFS='|' read -r name flags <<< "$s"
    for inp in $INPUTS; do run_one "$name" "$flags" "$inp" "$seed"; done
  done
done
echo "{\"low_edge_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_low_edge_done.json
echo "[lowedge] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
