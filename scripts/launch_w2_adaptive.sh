#!/bin/bash
# Adaptive difficulty curriculum (H4-v3: mastery-gated + backtracking) + high bpf —
# the COMBINED test. Keep bpf=24's sub-degree low-d precision (0.96°) AND rescue the
# starved high-d coverage by pouring budget into the failing shells (advance on
# mastery, regress -0.1 to consolidate, re-approach). d_start 0.1 / step 0.1 (10
# levels), generous neurons-gens so the loop can dwell on the hard shells.
# vs bpf=8 anchor 72.2% + bpf=24 fixed-curriculum (the sweep). base-seed 20260609.
set -u
cd /Users/lacg/wnn
BPF="${1:-24}"
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/W2_adaptive_bpf${BPF}_20260621/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-adaptive] $(date '+%Y-%m-%d %H:%M:%S') START adaptive curriculum bpf=$BPF"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --bits-per-feature "$BPF" \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --difficulty-adaptive --difficulty-start 0.1 --difficulty-step 0.1 \
  --mastery-threshold 0.95 --dwell-gens 4 --max-attempts 6 \
  --holdout-per-shell \
  --neurons-gens 90 --neurons-patience 5 --check-interval 5 \
  --memory-gens 40 --memory-patience 8 \
  --pop 30 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.50 --fit-weight-jerk 0.05 --fit-weight-mono 0.05 \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed 20260609 --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[w2-adaptive] $(date '+%Y-%m-%d %H:%M:%S') DONE bpf=$BPF (exit $?)"
