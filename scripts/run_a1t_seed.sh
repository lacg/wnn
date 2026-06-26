#!/bin/bash
# Integral-input A/B — ARM A1t (per-axis PID + ANTI-WINDUP): identical to A1
# (S16/absolute recipe + obs-peraxis-p/-i) but with the leaky integral tamed to
# kill the windup that cratered A1 (14% stable, 12° steady offset): leak 0.99->0.80
# (shorter ~5-step memory, can't over-charge) and scale 1.0->0.25 (the integral
# feature can't dominate the thermometer). Tests whether windup, not the concept,
# was the A1 failure. Everything else IDENTICAL to A1/the absolute sweep.
# Args: SEED (one of 20260609..13). Writes A1t_seed{SEED}/ under IntegralAB dir.
set -u
cd /Users/lacg/wnn
SEED="${1:?usage: run_a1t_seed.sh SEED}"
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05   # S16 winning weights
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/IntegralAB_20260625/A1t_seed${SEED}; mkdir -p "$DIR"
echo "[a1t] $(date '+%Y-%m-%d %H:%M:%S') START A1t anti-windup seed=$SEED (leak=0.80 scale=0.25, S16 weights, absolute)"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
  --no-delta-control \
  --obs-peraxis-p --obs-peraxis-i --integral-leak 0.80 --integral-scale 0.25 \
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
echo "[a1t] $(date '+%Y-%m-%d %H:%M:%S') A1t seed=$SEED COMPLETE"
