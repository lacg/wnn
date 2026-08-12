#!/bin/bash
# W1 decay-law surface — training leg (02/07/2026).
# Roadmap docs/controller_research_roadmap.md W1.1: train ONE recipe at horizons
# H in {500, 1000, 2000, 4000} x 2 seeds, then eval each winner at multiples of its
# H to extract the drift-immunity law (is it always ~2.5x the trained horizon?).
# The recipe = E2 LONG's exactly (s16, immigrants 0.15) so TWO surface points are
# FREE: H=500 = E2 IMM cells; H=2000 = E2 LONG cells. This driver trains only the
# missing H=1000 and H=4000 (2 seeds each, ~2.7h and ~11h per run respectively).
# Chained behind C2K (ONE controller at a time). Marker /tmp/wnn_w1_done.json.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/W1Surface_20260702.log
exec >>"$LOG" 2>&1

echo "[w1] $(date '+%Y-%m-%d %H:%M:%S') WAITING for C2K (/tmp/wnn_c2k_done.json)"
while [ ! -f /tmp/wnn_c2k_done.json ]; do sleep 60; done
echo "[w1] $(date '+%Y-%m-%d %H:%M:%S') C2K done — starting W1 surface training (H1000/H4000)"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/W1Surface_20260702
SEEDS="20260609 20260610"
HORIZONS="1000 4000"     # H500 = E2 IMM cells (free); H2000 = E2 LONG cells (free)

run_one() {
  local h="$1" seed="$2"
  local dir="$ROOT/H${h}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[w1] $(date '+%H:%M:%S') SKIP H${h} seed=${seed}"; return; fi
  echo "[w1] $(date '+%Y-%m-%d %H:%M:%S') START H${h} seed=${seed}"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons 8 12 16 --grid-bits 24 --levels 16 --bits-per-feature 8 \
    --no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds 5 \
    --eval-episodes 100 --steps "$h" --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --immigrants 0.15 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$seed" --runs 1 \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  if [ $? -ne 0 ]; then echo "[w1] $(date '+%H:%M:%S') FAIL H${h} seed=${seed} (continuing)"
  else echo "{\"h\":${h},\"seed\":${seed}}" > "$dir/done.json"
       echo "[w1] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE H${h} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for h in $HORIZONS; do
    run_one "$h" "$seed"
  done
done
echo "{\"w1_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_w1_done.json
echo "[w1] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE — run the decay-surface eval"
echo "  (e4 solo per winner at {0.5x,1x,2.5x,5x,10x,20x} its H via E4_STEPS; free points: IMM=H500, LONG=H2000)"
