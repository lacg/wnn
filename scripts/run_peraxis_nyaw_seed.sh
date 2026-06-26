#!/bin/bash
# Per-axis roll+pitch WITHOUT yaw (drop-yaw clean per-axis). S16/absolute recipe
# (weights .25/.35/.20/.15/.05) + obs-peraxis-p + obs-peraxis-i + --no-obs-peraxis-yaw,
# so the controller sees gravity-observable roll+pitch error & integral (4 features,
# nf=13), dropping the drifting dead-reckoned yaw that cratered A1/ISO_P/ISO_I (~14%).
# Tests whether clean per-axis P+I beats the blind S16 baseline (85.4%).
# GATED: only meaningful if the obs-tilt-i probe recovered (clean-reference family).
# Args: SEED (default 20260609). Writes PERAXIS_NYAW_seed{SEED}/.
set -u
cd /Users/lacg/wnn
SEED="${1:-20260609}"
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/IntegralAB_20260625/PERAXIS_NYAW_seed${SEED}; mkdir -p "$DIR"
echo "[peraxis-nyaw] $(date '+%Y-%m-%d %H:%M:%S') START roll+pitch P+I (no yaw, nf=13) seed=$SEED"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
  --no-delta-control \
  --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --integral-leak 0.99 --integral-scale 1.0 \
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
echo "{\"peraxis_nyaw_done\":true,\"seed\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_peraxis_nyaw_done.json
echo "[peraxis-nyaw] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE"
