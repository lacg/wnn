#!/bin/bash
# Yaw-anchor OVERFIT PROBE: identical to run_yaw_anchor_seed.sh (S16 recipe +
# --obs-yaw-err) EXCEPT --num-eval-folds 5 (was 3). Tests the diagnosis that the
# anchor's in-sample≫held-out gap (91→70%) is OVERFITTING the 3 search folds: with
# 5 accumulated folds the GA sees more diverse ICs and can't memorize them, so the
# anchor's clear in-sample edge (2.94°/2.03° vs S16 3.41°/3.05°) should survive to
# held-out. Compare held-out (report-seeds) vs folds=3 anchor (70/67.5%) AND S16
# (87/88.5%) at the SAME seeds. Args: SEED. Writes YAWF5_seed{SEED}/.
set -u
cd /Users/lacg/wnn
SEED="${1:-20260609}"
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/YawAnchorFolds5_20260626/YAWF5_seed${SEED}; mkdir -p "$DIR"
echo "[yaw-f5] $(date '+%Y-%m-%d %H:%M:%S') START obs_yaw_err folds=5 (nf=10) seed=$SEED"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
  --no-delta-control \
  --obs-yaw-err --integral-leak 0.99 --integral-scale 1.0 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
  --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
  --pop 24 --num-eval-folds 5 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
  --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed "$SEED" --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" > "$DIR/run.out" 2>&1
echo "{\"yaw_f5_done\":true,\"seed\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_f5_seed${SEED}_done.json
echo "[yaw-f5] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE seed=$SEED"
