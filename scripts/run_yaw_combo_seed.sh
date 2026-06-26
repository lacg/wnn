#!/bin/bash
# Yaw-anchor COMBO SWEEP runner: one S-series weight combo × one seed, WITH --obs-yaw-err
# (folds=3, apples-to-apples vs the blind S-sweep + the anchor-S16 baseline). Tests whether
# a different fitness-weight balance steers the GA to ROBUST (non-brittle) anchor genomes —
# the anchor is brittle under S16 weights (held-out stable bimodal ±25%). Args: COMBO SEED.
# Writes COMBO_{COMBO}_seed{SEED}/.
set -u
cd /Users/lacg/wnn
COMBO="${1:?usage: run_yaw_combo_seed.sh COMBO SEED}"
SEED="${2:-20260609}"
wt_for() {
  case "$1" in
    S01) echo "0.40 0.00 0.50 0.05 0.05";;  S02) echo "0.40 0.10 0.40 0.05 0.05";;
    S03) echo "0.35 0.20 0.35 0.05 0.05";;  S04) echo "0.30 0.30 0.30 0.05 0.05";;
    S05) echo "0.25 0.40 0.25 0.05 0.05";;  S06) echo "0.20 0.50 0.20 0.05 0.05";;
    S07) echo "0.15 0.60 0.15 0.05 0.05";;  S08) echo "0.55 0.10 0.25 0.05 0.05";;
    S09) echo "0.45 0.20 0.25 0.05 0.05";;  S10) echo "0.30 0.35 0.25 0.05 0.05";;
    S11) echo "0.15 0.50 0.25 0.05 0.05";;  S12) echo "0.30 0.10 0.50 0.05 0.05";;
    S13) echo "0.30 0.25 0.35 0.05 0.05";;  S14) echo "0.30 0.45 0.15 0.05 0.05";;
    S15) echo "0.30 0.55 0.05 0.05 0.05";;  S16) echo "0.25 0.35 0.20 0.15 0.05";;
    S17) echo "0.25 0.35 0.20 0.05 0.15";;  S18) echo "0.25 0.30 0.20 0.125 0.125";;
    *)   return 1;;
  esac
}
W=$(wt_for "$COMBO") || { echo "unknown combo $COMBO"; exit 1; }
read ERR STEADY STABLE JERK MONO <<< "$W"
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/YawComboSweep_20260626/COMBO_${COMBO}_seed${SEED}; mkdir -p "$DIR"
echo "[yaw-combo] $(date '+%Y-%m-%d %H:%M:%S') START $COMBO ($W) obs_yaw_err seed=$SEED"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
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
echo "{\"combo\":\"$COMBO\",\"seed\":$SEED,\"weights\":\"$W\",\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > "/tmp/wnn_yaw_combo_${COMBO}_seed${SEED}_done.json"
echo "[yaw-combo] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE $COMBO seed=$SEED"
