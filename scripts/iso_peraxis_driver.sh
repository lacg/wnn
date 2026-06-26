#!/bin/bash
# Integral-input A/B — ISOLATION: pin which per-axis feature cratered A1 (14%).
# A1 enabled BOTH obs-peraxis-p (proportional error) AND obs-peraxis-i (leaky
# integral). Anti-windup (A1t) was inert -> windup ruled out. This runs each
# feature ALONE at one seed (S16/absolute recipe, weights .25/.35/.20/.15/.05),
# so we can read the culprit against the anchors:
#   S16 (neither, 9 feat) = 85.4% | A1 (both, 15 feat) = ~14%
#   ISO_P (only obs-peraxis-p, 12 feat)  ISO_I (only obs-peraxis-i, 12 feat)
# If P=14 & I=85 -> proportional (dead-reckoned yaw) is the poison.
# If P=85 & I=14 -> integral is the poison. If both ~85 -> interaction only.
# ONE controller at a time; detach via scripts/detach_launch.py (PPID=1).
set -u
cd /Users/lacg/wnn
SEED=20260609
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
LOG=logs/controller/IntegralAB_20260625/driver_iso.log
mkdir -p logs/controller/IntegralAB_20260625; exec >>"$LOG" 2>&1

run_arm() {  # $1=tag (P|I)  $2=extra phased_ga obs flag
  local tag="$1" flag="$2"
  local DIR=logs/controller/IntegralAB_20260625/ISO_${tag}_seed${SEED}; mkdir -p "$DIR"
  echo "[iso] $(date '+%Y-%m-%d %H:%M:%S') START ISO_${tag} ($flag) seed=$SEED"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
    --no-delta-control \
    $flag --integral-leak 0.99 --integral-scale 1.0 \
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
  echo "{\"iso_${tag}_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_iso_${tag}_done.json
  echo "[iso] $(date '+%Y-%m-%d %H:%M:%S') ISO_${tag} COMPLETE"
}

echo "[iso-driver] $(date '+%Y-%m-%d %H:%M:%S') START isolation P-only then I-only"
run_arm P "--obs-peraxis-p"
run_arm I "--obs-peraxis-i"
echo "{\"iso_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_iso_done.json
echo "[iso-driver] $(date '+%Y-%m-%d %H:%M:%S') ISOLATION COMPLETE"
