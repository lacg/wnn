#!/bin/bash
# ABSOLUTE-control comparison for the n=5 sweep WINNER. Runs the winning combo's
# weights with --no-delta-control at the SAME 5 seeds (20260609..13) + SAME sweep
# config (15n/15m, full difficulty, bpf=8) as the delta finals → directly
# comparable to the winner's delta n=5. Separate _ABS dirs (no clobber).
# Arg: winning label (e.g. S02). base substrate otherwise identical to delta runs.
set -u
cd /Users/lacg/wnn
LAB="${1:?usage: run_absolute_winner.sh LABEL}"
declare -A WT=(
 [S01]="0.40 0.00 0.50 0.05 0.05"  [S02]="0.40 0.10 0.40 0.05 0.05"  [S04]="0.30 0.30 0.30 0.05 0.05"
 [S06]="0.20 0.50 0.20 0.05 0.05"  [S07]="0.15 0.60 0.15 0.05 0.05"  [S09]="0.45 0.20 0.25 0.05 0.05"
 [S16]="0.25 0.35 0.20 0.15 0.05"  [S18]="0.25 0.30 0.20 0.125 0.125"
)
W="${WT[$LAB]:?unknown label $LAB}"; read -r ERR STEADY STABLE JERK MONO <<<"$W"
export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DLOG=logs/controller/Ssweep_20260622/driver_absolute_${LAB}.log
mkdir -p logs/controller/Ssweep_20260622; exec >>"$DLOG" 2>&1
echo "[abs] $(date '+%Y-%m-%d %H:%M:%S') START absolute ${LAB} ($W) over 5 seeds, --no-delta-control"
for SEED in 20260609 20260610 20260611 20260612 20260613; do
  DIR=logs/controller/Ssweep_20260622/${LAB}_ABS_seed${SEED}; mkdir -p "$DIR"
  echo "[abs] $(date '+%H:%M:%S') -> ${LAB} ABS seed=$SEED"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 --bits-per-feature 8 \
    --no-delta-control \
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
done
echo "{\"abs_done\":\"$LAB\",\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_abs_${LAB}_done.json
echo "[abs] $(date '+%Y-%m-%d %H:%M:%S') ABSOLUTE ${LAB} COMPLETE (n=5)"
