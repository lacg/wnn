#!/usr/bin/env bash
# Plan A + Plan B chained launch.
#
# Plan A: full-fledge phased GA with Combo 7-equivalent weights, eval=20 to break
#         the stability granularity ceiling, bigger pop/gens, --save-winner.
# Plan B: memory-only refinement on Plan A's winner, stability-dominant fitness.
#
# Do NOT run while the 14-combo sweep is active — both contend for CPU+GPU on
# the M4 Max. Sweep completes ~03 AM UTC 30/05/2026 based on current pace.
#
# Usage:
#   ./scripts/launch_planAB.sh                # launches both, sequentially
#   ./scripts/launch_planAB.sh --plan-a-only  # just Plan A
#   ./scripts/launch_planAB.sh --plan-b-only  # just Plan B (requires Plan A winner)

set -euo pipefail
cd "$(dirname "$0")/.."

WORKDIR="logs/controller/planAB"
mkdir -p "$WORKDIR"
TS=$(date -u +%Y%m%d_%H%M%S)
PLAN_A_LOG="$WORKDIR/planA_${TS}.log"
PLAN_B_LOG="$WORKDIR/planB_${TS}.log"
WINNER_A="$WORKDIR/winner_planA_${TS}.pkl"
WINNER_B="$WORKDIR/winner_planB_${TS}.pkl"

source wnn/bin/activate
export PYTHONPATH="/Users/lacg/wnn/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1
export RAYON_NUM_THREADS=8

PLAN=${1:-both}

if [ "$PLAN" != "--plan-b-only" ]; then
  echo "=========================================================="
  echo "PLAN A: full-fledge phased GA (Combo 7 weights, eval=20)"
  echo "  log:     $PLAN_A_LOG"
  echo "  winner:  $WINNER_A"
  echo "=========================================================="
  python -u tests/run_phased_ga.py \
    --grid-state-neurons 4 6 8 --grid-bits 16 24 32 \
    --levels 12 --grid-min-suffix 4 \
    --neurons-gens 200 --neurons-patience 10 \
    --bits-gens    200 --bits-patience    10 \
    --conns-gens   200 --conns-patience   10 \
    --memory-gens  400 --memory-patience  20 \
    --pop 200 --elitism 0.2 --crossover-rate 0.5 \
    --eval-episodes 20 --steps 500 --tilt 15 \
    --universe-episodes 3 \
    --rg-rounds 3 --rg-episodes-per-round 6 --rg-eval-episodes 5 \
    --train-workers 4 \
    --fit-weight-err-sq 0.30 --fit-weight-stable 0.50 \
    --fit-weight-jerk   0.10 --fit-weight-mono   0.10 \
    --base-seed 20260530 \
    --save-winner "$WINNER_A" 2>&1 | tee "$PLAN_A_LOG"
fi

if [ "$PLAN" = "--plan-a-only" ]; then
  echo "Plan A done — exiting (--plan-a-only)."
  exit 0
fi

# Find the latest Plan A winner if --plan-b-only
if [ "$PLAN" = "--plan-b-only" ]; then
  WINNER_A=$(ls -t "$WORKDIR"/winner_planA_*.pkl 2>/dev/null | head -1)
  if [ -z "$WINNER_A" ]; then
    echo "ERROR: --plan-b-only but no winner_planA_*.pkl in $WORKDIR"; exit 1
  fi
  echo "Loading latest Plan A winner: $WINNER_A"
fi

echo ""
echo "=========================================================="
echo "PLAN B: memory-only refinement (stability-dominant, eval=20)"
echo "  load:    $WINNER_A"
echo "  log:     $PLAN_B_LOG"
echo "  winner:  $WINNER_B"
echo "=========================================================="
python -u tests/run_memory_refinement.py \
  --load-winner "$WINNER_A" \
  --memory-gens 500 --memory-patience 25 \
  --pop 200 --elitism 0.2 --crossover-rate 0.5 \
  --eval-episodes 20 --steps 500 --tilt 15 \
  --universe-episodes 3 \
  --rg-rounds 3 --rg-episodes-per-round 6 --rg-eval-episodes 5 \
  --train-workers 4 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.60 \
  --fit-weight-jerk   0.0  --fit-weight-mono   0.0 \
  --base-seed 20260531 \
  --save-winner "$WINNER_B" 2>&1 | tee "$PLAN_B_LOG"

echo ""
echo "=========================================================="
echo "Plan A + B complete."
echo "  Plan A winner: $WINNER_A"
echo "  Plan B winner: $WINNER_B"
echo "=========================================================="
