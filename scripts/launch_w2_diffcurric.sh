#!/bin/bash
# DIFFICULTY curriculum (H4-v2, WNN-correct) on the W2 weights: ramp the IC
# magnitude d=0.2->1.0 over the NEURONS stage, full 3-axis throughout, then MEMORY
# at full. Tests the hover-density insight — start where the substrate already
# hovers (100%) and grow outward, addresses overlapping so cells transfer (unlike
# the moot axis curriculum). eval=100. Multi-seed held-out at FULL difficulty.
# Base-seed 20260609 = the re-sweep's W2 seed (A/B vs plain-W2 71.5%). RAYON=2.
# Arg: "noh3" (curriculum alone, isolate its effect) or "h3" (+ --decouple-outputs).
set -u
cd /Users/lacg/wnn
TAG="${1:-noh3}"
H3_FLAG=""
[ "$TAG" = "h3" ] && H3_FLAG="--decouple-outputs"
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/W2_diffcurric_${TAG}_20260621/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-diffcurric] $(date '+%Y-%m-%d %H:%M:%S') START tag=$TAG H3=[$H3_FLAG]"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --difficulty-curriculum --difficulty-phases 5 --difficulty-start 0.2 \
  $H3_FLAG \
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
echo "[w2-diffcurric] $(date '+%Y-%m-%d %H:%M:%S') DONE tag=$TAG (exit $?)"
