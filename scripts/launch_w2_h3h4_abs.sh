#!/bin/bash
# H3+H4 test on the W2 weights (re-sweep winner, fixed scorer): does decouple-
# outputs (H3, orthogonal action space) + axis-curriculum (H4) push PAST plain
# W2's 71.5%/4.17° toward PID (100%/2.46°)? DELTA substrate (same as the re-sweep
# — the "abandon delta" call was a buggy-scorer artifact; delta reaches 71.5% on
# the fixed scorer). Fast search (eval 8) + honest multi-seed held-out (100 eps ×
# 4 seeds). Validated curriculum gens (neurons 70 / pat 2 / check 3 / memory 40,
# from the prior H4 runs). Base-seed 20260609 = the re-sweep's W2 seed (rough A/B).
# Throttled RAYON=2 — XDS keeps priority.
set -u
cd /Users/lacg/wnn
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/W2_h3h4_abs_20260621/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-h3h4] $(date '+%Y-%m-%d %H:%M:%S') START — W2 weights + H3(decouple) + H4(axis-curriculum), ABSOLUTE"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --no-delta-control --decouple-outputs --axis-curriculum \
  --neurons-gens 70 --neurons-patience 2 --check-interval 3 \
  --memory-gens 40 --memory-patience 2 \
  --pop 30 --num-eval-folds 3 \
  --eval-episodes 8 --steps 500 --tilt 5.0 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.50 --fit-weight-jerk 0.05 --fit-weight-mono 0.05 \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed 20260609 --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[w2-h3h4] $(date '+%Y-%m-%d %H:%M:%S') DONE (exit $?)"
