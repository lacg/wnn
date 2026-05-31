#!/usr/bin/env bash
# Plan A v3 — "Full ambition" follow-up to v2 (30/05/2026).
#
# v3 changes vs v2 (per user request after underwhelming v2 results):
#   1. --levels 12 → 64 (5× PWM resolution; target prod is 256)
#   2. --grid-state-neurons 3 4 5 → 6 9 12 15 (force bigger arch;
#      v2's sn=3 anchor insufficient memory for closed-loop integration)
#   3. --grid-bits 17 19 21 → 16 24 32 (wider bits to match wider sn,
#      need bits ≥ 2·sn + 4 for meaningful suffix)
#   4. --eval-episodes 20 → 100 (1% stability granularity; v2's 5%
#      was too coarse to differentiate near-stable controllers)
#   5. --memory-gens 200 → 600 (much bigger arch needs more cell-
#      refinement budget; user-requested)
#   6. --neurons-gens / --bits-gens / --conns-gens 100 → 300
#      (more arch search budget; user-requested)
#   7. --steps 400 → 500 (back to v1 level)
#   8. --rg-rounds 2 → 3, --rg-eps-per-round 4 → 6, --rg-eval-eps 3 → 5
#      (back to v1 inner-training depth)
#
# Patience preserves 50% semantic: target_patience = target_gens × 0.5 / 10.
#   - arch stages 300 gens → patience 15 (50%, saves ~half on plateau)
#   - memory stage 600 gens → patience 30 (50%)
#
# Cost (vs v2): per-gen ~37×, partly offset by patience-50% stops.
# Estimated wall: 24-36h total. NOT overnight — this is a multi-day run.
# Plan B chained from --save-winner (separate launch decision).
#
# Resource sharing:
#   - RAYON_NUM_THREADS=3, train-workers=3 (~9 threads max from Plan A)
#   - IDS worker has RAYON=10 (~10 threads)
#   - System headroom ~3 cores on M4 Max 16-core
#
# Per-stage save: pickles each stage's winner so reboot-mid-run only
# loses one stage of work (v1 lost 5.5h to OOM — won't happen again).

set -euo pipefail
cd "$(dirname "$0")/.."

WORKDIR="logs/controller/planAB"
mkdir -p "$WORKDIR"
TS=$(date -u +%Y%m%d_%H%M%S)
PLAN_A_LOG="$WORKDIR/planAv3_${TS}.log"
WINNER_A="$WORKDIR/winner_planAv3_${TS}.pkl"
STAGE_DIR="$WORKDIR/v3_stages_${TS}"

source wnn/bin/activate
export PYTHONPATH="/Users/lacg/wnn/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1
export RAYON_NUM_THREADS=3

echo "=========================================================="
echo "PLAN A v3: levels=64 + bigger arch + eval=100 (1% granul)"
echo "  log:           $PLAN_A_LOG"
echo "  winner pickle: $WINNER_A"
echo "  stage pickles: $STAGE_DIR/stage{1..4}_*.pkl"
echo "  ETA: 24-36h. Per-stage save → reboot survives."
echo "=========================================================="

python -u tests/run_phased_ga.py \
  --grid-state-neurons 6 9 12 15  --grid-bits 16 24 32 40 \
  --levels 64 --grid-min-suffix 4 \
  --neurons-gens 300 --neurons-patience 15 \
  --bits-gens    300 --bits-patience    15 \
  --conns-gens   300 --conns-patience   15 \
  --memory-gens  600 --memory-patience  30 \
  --pop 100 --elitism 0.2 --crossover-rate 0.5 \
  --eval-episodes 100 --steps 500 --tilt 15 \
  --universe-episodes 3 \
  --rg-rounds 3 --rg-episodes-per-round 6 --rg-eval-episodes 5 \
  --train-workers 3 \
  --fit-weight-err-sq 0.30 --fit-weight-stable 0.50 \
  --fit-weight-jerk   0.10 --fit-weight-mono   0.10 \
  --num-eval-folds 5 \
  --save-stage-checkpoints "$STAGE_DIR" \
  --base-seed 20260530 \
  --save-winner "$WINNER_A" 2>&1 | tee "$PLAN_A_LOG"

echo ""
echo "=========================================================="
echo "Plan A v3 complete."
echo "  winner: $WINNER_A"
echo "  stages: $STAGE_DIR/"
echo "  Plan B v3 launched separately when desired."
echo "=========================================================="
