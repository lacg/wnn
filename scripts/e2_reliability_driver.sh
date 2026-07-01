#!/bin/bash
# E2 reliability sweep (01/07/2026) — plan .claude/plans/controller_break_90_v2.md.
# Post-ki=0 re-anchor: the gap is PD-approximation quality + yaw observability + GA
# reliability (NOT a missing integrator). Base recipe = the A_ctrl cell (s16, obs-OFF,
# gsn 8 12 16, grid-bits 24, folds 5, C10 weights) whose no-immigrant anchor is
# 84.3±4.4 pooled (StateIntegral_20260701). ALL arms ride --immigrants 0.15 (E1):
#   IMM   : base + immigrants only            — isolates E1 vs the 84.3 anchor
#   LONG  : + --steps 2000                    — settling precision pressure (lever 3)
#           (its held-out is at 2000 steps — NOT directly comparable; re-score the
#            winner at 500 steps via scripts/e4_best_of_k.py for the cross-arm read)
#   CURR  : + --difficulty-adaptive           — mastery-gated hard-IC curriculum (built, never run)
#   ANCH  : + --obs-yaw-err --neurons-gens 30 — yaw-anchor retry under a healthier search
#           (attacks the committee's common-mode yaw band; extra gens for the extra DOF)
#   GAMMA : + --threshold-gamma 2.0           — E3 hover-dense thermometer decode
# 5 arms × 2 seeds = 10 cells, seed-outer. ONE controller at a time (waits on low-edge).
# Arm R (action-repeat) intentionally ABSENT — needs a careful Rust+Metal parity pass
# (GPU split-trainer kernels too); its cells append later via the resume-skip design.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/E2Reliability_20260702.log
exec >>"$LOG" 2>&1

echo "[e2] $(date '+%Y-%m-%d %H:%M:%S') WAITING for low-edge (/tmp/wnn_low_edge_done.json)"
while [ ! -f /tmp/wnn_low_edge_done.json ]; do sleep 60; done
echo "[e2] $(date '+%Y-%m-%d %H:%M:%S') low-edge done — starting E2 reliability sweep"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/E2Reliability_20260702
SEEDS="20260609 20260610"                   # seed-outer (all arms at seed09 first, then seed10)
# arm: name | extra flags (placed AFTER the base flags → later argparse occurrence wins)
ARMS=(
  "IMM|--immigrants 0.15"
  "LONG|--immigrants 0.15 --steps 2000"
  "CURR|--immigrants 0.15 --difficulty-adaptive"
  "ANCH|--immigrants 0.15 --obs-yaw-err --neurons-gens 30"
  "GAMMA|--immigrants 0.15 --threshold-gamma 2.0"
)

run_one() {
  local name="$1" extra="$2" seed="$3"
  local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[e2] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
  echo "[e2] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} extra=[${extra}]"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 24 --levels 16 --bits-per-feature 8 \
    --no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds 5 \
    --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$seed" --runs 1 \
    $extra \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  if [ $? -ne 0 ]; then echo "[e2] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
  else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"extra\":\"${extra}\"}" > "$dir/done.json"
       echo "[e2] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for a in "${ARMS[@]}"; do
    IFS='|' read -r name extra <<< "$a"
    run_one "$name" "$extra" "$seed"
  done
done
echo "{\"e2_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_e2_done.json
echo "[e2] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE"
