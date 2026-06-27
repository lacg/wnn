#!/bin/bash
# Yaw-anchor + FEATURE-BALANCE CAP test. The connectivity diag showed the coupled anchor
# over-wires the OUTPUT layer to obs_yaw_err (2.14x); S16 wires evenly (~0.6x all features)
# → robust. This forces S16-like even wiring via (no feature >1.5x
# the least-wired). Tests: does balanced wiring let the yaw anchor generalize? S16 recipe +
# --obs-yaw-err. A/B vs coupled anchor (70.5%/67.5%) and S16 (87.2%/88.5%). Args: SEED.
set -u
cd /Users/lacg/wnn
SEED="${1:-20260609}"
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/YawHighCov_20260627/YAWHC_seed${SEED}; mkdir -p "$DIR"
echo "[yaw-hc] $(date '+%Y-%m-%d %H:%M:%S') START obs_yaw_err + high-coverage (grid-bits 72-88, suffix~full frame) seed=$SEED"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 72 88 --levels 16 --bits-per-feature 8 \
  --no-delta-control \
  --obs-yaw-err --integral-leak 0.99 --integral-scale 1.0 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
  --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
  --pop 24 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
  --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed "$SEED" --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" > "$DIR/run.out" 2>&1
echo "{\"yaw_hc_done\":true,\"seed\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_hc_seed${SEED}_done.json
echo "[yaw-hc] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE seed=$SEED"
