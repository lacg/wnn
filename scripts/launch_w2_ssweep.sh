#!/bin/bash
# One combo of the steady-weight sweep (S-series). Full difficulty (d=1.0, the
# regime where the offset lives), bpf=8, short gens (round-1 cull budget), W-style
# substrate. Args: LABEL ERR STEADY STABLE JERK MONO SEED.
set -u
cd /Users/lacg/wnn
LABEL="$1"; ERR="$2"; STEADY="$3"; STABLE="$4"; JERK="$5"; MONO="$6"; SEED="$7"
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/Ssweep_20260622/${LABEL}_seed${SEED}
mkdir -p "$DIR"
echo "[ssweep] $(date '+%Y-%m-%d %H:%M:%S') START $LABEL err=$ERR steady=$STEADY stable=$STABLE jerk=$JERK mono=$MONO seed=$SEED"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --bits-per-feature 8 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --neurons-gens 15 --neurons-patience 6 --check-interval 5 \
  --memory-gens 15 --memory-patience 8 \
  --pop 24 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
  --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed "$SEED" --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[ssweep] $(date '+%Y-%m-%d %H:%M:%S') DONE $LABEL (exit $?)"
