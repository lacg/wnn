#!/bin/bash
# bits_per_feature (input thermometer resolution) sweep on the BEST recipe so far:
# W2 weights + difficulty curriculum (the noh3 config that reached the 72.2% plateau
# at bpf=8). Higher bpf = finer address resolution → can sense/correct smaller
# attitude deviations (attacks the 0.94° hover floor + high-tilt degradation), but
# also more distinct addresses (sparser visits → density tradeoff). Does finer
# encoding break the 72% plateau? Everything else identical to W2_diffcurric_noh3
# (base-seed 20260609 = direct A/B; bpf=8 anchor = 72.2±4.4%). RAYON=2 (XDS priority).
# Arg: bits_per_feature (e.g. 12, 16, 24).
set -u
cd /Users/lacg/wnn
BPF="${1:-12}"
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/W2_bpf${BPF}_20260621/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-bpf] $(date '+%Y-%m-%d %H:%M:%S') START bits_per_feature=$BPF"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --bits-per-feature "$BPF" \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --difficulty-curriculum --difficulty-phases 5 --difficulty-start 0.2 \
  --neurons-gens 50 --neurons-patience 5 --check-interval 5 \
  --memory-gens 40 --memory-patience 8 \
  --pop 30 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.50 --fit-weight-jerk 0.05 --fit-weight-mono 0.05 \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed 20260609 --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[w2-bpf] $(date '+%Y-%m-%d %H:%M:%S') DONE bits_per_feature=$BPF (exit $?)"
