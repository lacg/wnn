#!/bin/bash
# H3-ONLY test on the W2 weights — does decouple-outputs (orthogonal action
# space) make the FULL 3-axis problem tractable, beating plain-W2's 71.5%/4.17°?
# NO axis-curriculum (H4 was moot for a WNN: single-axis cells fill addresses
# disjoint from multi-axis → no transfer). Normal NEURONS→MEMORY on full 3-axis.
# eval=100 (fine stability gradient so the GA can steer on stability; GPU-batched
# scoring keeps it cheap — training/DAGGER dominates per-gen, not the eval count).
# Honest multi-seed held-out (100 eps × 4 seeds). Base-seed 20260609 = the
# re-sweep's W2 seed (direct A/B vs plain-W2 71.5%). Throttled RAYON=2 (XDS priority).
# Substrate arg: "delta" (default) or "abs" (--no-delta-control).
set -u
cd /Users/lacg/wnn
SUB="${1:-delta}"
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DELTA_FLAG=""; TAG="delta"
if [ "$SUB" = "abs" ]; then DELTA_FLAG="--no-delta-control"; TAG="abs"; fi
DIR=logs/controller/W2_h3only_${TAG}_20260621/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-h3only] $(date '+%Y-%m-%d %H:%M:%S') START — W2 + H3(decouple), 3-axis, eval=100, substrate=$TAG"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --decouple-outputs $DELTA_FLAG \
  --neurons-gens 30 --neurons-patience 5 --check-interval 5 \
  --memory-gens 40 --memory-patience 8 \
  --pop 30 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.50 --fit-weight-jerk 0.05 --fit-weight-mono 0.05 \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed 20260609 --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[w2-h3only] $(date '+%Y-%m-%d %H:%M:%S') DONE substrate=$TAG (exit $?)"
