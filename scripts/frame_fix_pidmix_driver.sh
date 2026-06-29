#!/bin/bash
# Full-PID-mix test (28/06/2026). Two NEW variants, each under BOTH existing recipes so they
# slot directly into the Round 1 / Round 2 comparison tables (apples-to-apples, same seeds):
#   pidmix      = --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i
#                 → roll/pitch P+I + yaw P+I + gyro D = the full 3-axis PID (nf=15). The one
#                   combination no prior variant had (peraxis lacked yaw; anchor lacked roll/pitch I).
#   pidmix_pwm  = pidmix + --obs-pwm  → full PID + actuator-state accumulator (nf=19). Tests whether
#                   the accumulator adds steady-state value ON TOP of a complete PID.
# Recipes (matching the two rounds exactly so the numbers are comparable):
#   R1 → FrameFixVal_20260627  : --grid-bits 24 30 --num-eval-folds 3   (folds=3 ONLY to match the
#        existing Round-1 table; the folds=5 rule still governs standalone work — see CLAUDE.md)
#   R2 → FrameFixBits_20260627 : --grid-bits 100   --num-eval-folds 5
# 2 variants × 2 recipes × 2 seeds (09/10) = 8 runs, SEQUENTIAL. Self-gates on the current Round-2
# sweep finishing. pidmix_pwm at 100b (nf=19) is the biggest memory test yet — OOM watch applies.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/FrameFixPidmix_20260628.log
exec >>"$LOG" 2>&1

echo "[pid] $(date '+%Y-%m-%d %H:%M:%S') WAITING for current Round-2 sweep (/tmp/wnn_frame_fix_bits_done.json)"
while [ ! -f /tmp/wnn_frame_fix_bits_done.json ]; do sleep 60; done
echo "[pid] $(date '+%Y-%m-%d %H:%M:%S') Round-2 done — starting full-PID-mix runs"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
PIDFLAGS="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

VARIANTS=( "pidmix|$PIDFLAGS" "pidmix_pwm|$PIDFLAGS --obs-pwm" )
# recipe: label | ROOT | grid-bits | folds
RECIPES=( "R1|logs/controller/FrameFixVal_20260627|24 30|3" "R2|logs/controller/FrameFixBits_20260627|100|5" )
SEEDS="20260609 20260610"

run_one() {
  local name="$1" flags="$2" root="$3" gbits="$4" folds="$5" seed="$6"
  local dir="$root/${name}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[pid] $(date '+%H:%M:%S') SKIP ${name} ${root##*/} seed=${seed}"; return; fi
  echo "[pid] $(date '+%Y-%m-%d %H:%M:%S') START ${name} ${root##*/} seed=${seed} bits=[${gbits}] folds=${folds}"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits $gbits --levels 16 --bits-per-feature 8 \
    --no-delta-control $flags --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds $folds \
    --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$seed" --runs 1 \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  if [ $? -ne 0 ]; then echo "[pid] $(date '+%H:%M:%S') FAIL ${name} ${root##*/} seed=${seed} (continuing)"
  else echo "{\"name\":\"${name}\",\"root\":\"${root##*/}\",\"seed\":${seed}}" > "$dir/done.json"
       echo "[pid] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} ${root##*/} seed=${seed}"; fi
}

# seed-interleaved: round per seed, both recipes, both variants
for seed in $SEEDS; do
  for rc in "${RECIPES[@]}"; do
    IFS='|' read -r rlabel root gbits folds <<< "$rc"
    for v in "${VARIANTS[@]}"; do
      IFS='|' read -r name flags <<< "$v"
      run_one "$name" "$flags" "$root" "$gbits" "$folds" "$seed"
    done
  done
done
echo "{\"frame_fix_pidmix_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_frame_fix_pidmix_done.json
echo "[pid] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
