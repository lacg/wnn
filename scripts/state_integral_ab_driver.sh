#!/bin/bash
# Break-90% experiment (01/07/2026). Tests whether the ~90% controller-stability ceiling is
# an INTEGRATOR gap (fixable) vs a capacity gap (not). Substrate = s16 (obs-OFF, cleanest base).
# Plan: .claude/plans/controller_break_90pct.md
#   A_ctrl     : s16, small state (gsn 8 12 16), no integral       — baseline anchor
#   B_integral : A + --state-integral (recurrent state trained to a DIRECT PID-integral target)
#                → the targeted fix (WNN_STATE_INTEGRAL_TARGET=1). "use small --grid-state-neurons"
#   C_grow     : s16, GROWN state (gsn 24 32 40), no integral      — capacity control
# Success = B pooled ho-mem >90% AND B>C (→ integrator not capacity). 3 arms × 2 seeds = 6 runs.
# folds=5 fixed, grid-bits 24 (bits are wasted — keep archs small, lower OOM risk). seed-outer.
# Self-gates: waits for the bit-sweep to finish so it never competes for cores.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/StateIntegral_20260701.log
exec >>"$LOG" 2>&1

echo "[stateint] $(date '+%Y-%m-%d %H:%M:%S') WAITING for bit-sweep (/tmp/wnn_bit_sweep_pidmix_pwm_done.json)"
while [ ! -f /tmp/wnn_bit_sweep_pidmix_pwm_done.json ]; do sleep 60; done
echo "[stateint] $(date '+%Y-%m-%d %H:%M:%S') bit-sweep done — starting s16 state-integral A/B/C"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/StateIntegral_20260701
SEEDS="20260609 20260610"                   # seed-outer (all 3 arms at seed09 first, then seed10)
# arm: name | integral-flag | grid-state-neurons
ARMS=( "A_ctrl||8 12 16" "B_integral|--state-integral|8 12 16" "C_grow||24 32 40" )

run_one() {
  local name="$1" intflag="$2" gsn="$3" seed="$4"
  local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[stateint] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
  echo "[stateint] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} int=[${intflag}] gsn=[${gsn}]"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons $gsn --grid-bits 24 --levels 16 --bits-per-feature 8 \
    --no-delta-control $intflag --integral-leak 0.99 --integral-scale 1.0 \
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
  if [ $? -ne 0 ]; then echo "[stateint] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
  else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"gsn\":\"${gsn}\",\"integral\":\"${intflag}\"}" > "$dir/done.json"
       echo "[stateint] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for a in "${ARMS[@]}"; do
    IFS='|' read -r name intflag gsn <<< "$a"
    run_one "$name" "$intflag" "$gsn" "$seed"
  done
done
echo "{\"state_integral_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_state_integral_done.json
echo "[stateint] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
