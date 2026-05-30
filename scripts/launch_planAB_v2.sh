#!/usr/bin/env bash
# Plan A v2 + Plan B chained launch (post-OOM-reboot, post-K-fold).
#
# v2 changes vs v1 (per docs/controller_kfold_design.md + 30/05/2026 review):
#   - K-fold cross-validation (--num-eval-folds 5) — prevents the single-pool
#     overfit Plan A v1 hit (12.57° training → 16.22° re-eval Stage-1 gap).
#   - Per-stage checkpoints (--save-stage-checkpoints DIR) — every stage's
#     winner pickled, so reboot-mid-run only loses one stage of work.
#   - Tight grid anchored at v1's discovered arch (sn=3, sb~18-19, ob~18-19):
#     --grid-state-neurons 3 4 5 --grid-bits 17 19 21. Hard min sn=3 enforced
#     in arch_cfg so GA mutations can't dip below the v1 baseline.
#   - Reduced budget: pop=100 (was 200), gens 100/100/100/200 (was 200/200/200/400),
#     patience 5/5/5/10 (50% semantic preserved).
#   - RAYON_NUM_THREADS=3 + --train-workers=3 (co-resident with IDS worker
#     on RAYON=10; 13 threads total of 16 cores, 3 left for system).
#   - Weights unchanged: Combo 7 = err=0.30 stable=0.50 jerk=0.10 mono=0.10.
#
# Plan B v2: same as v1 but with K-fold for the memory refinement too.

set -euo pipefail
cd "$(dirname "$0")/.."

WORKDIR="logs/controller/planAB"
mkdir -p "$WORKDIR"
TS=$(date -u +%Y%m%d_%H%M%S)
PLAN_A_LOG="$WORKDIR/planAv2_${TS}.log"
PLAN_B_LOG="$WORKDIR/planBv2_${TS}.log"
WINNER_A="$WORKDIR/winner_planAv2_${TS}.pkl"
WINNER_B="$WORKDIR/winner_planBv2_${TS}.pkl"
STAGE_DIR="$WORKDIR/v2_stages_${TS}"

source wnn/bin/activate
export PYTHONPATH="/Users/lacg/wnn/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1
export RAYON_NUM_THREADS=3

PLAN=${1:-both}

if [ "$PLAN" != "--plan-b-only" ]; then
  echo "=========================================================="
  echo "PLAN A v2: K-fold + per-stage save + tight grid + Combo 7"
  echo "  log:           $PLAN_A_LOG"
  echo "  winner pickle: $WINNER_A"
  echo "  stage pickles: $STAGE_DIR/stage{1..4}_*.pkl"
  echo "=========================================================="
  python -u tests/run_phased_ga.py \
    --grid-state-neurons 3 4 5 --grid-bits 17 19 21 \
    --levels 12 --grid-min-suffix 4 \
    --neurons-gens 100 --neurons-patience 5 \
    --bits-gens    100 --bits-patience    5 \
    --conns-gens   100 --conns-patience   5 \
    --memory-gens  200 --memory-patience  10 \
    --pop 100 --elitism 0.2 --crossover-rate 0.5 \
    --eval-episodes 20 --steps 400 --tilt 15 \
    --universe-episodes 3 \
    --rg-rounds 2 --rg-episodes-per-round 4 --rg-eval-episodes 3 \
    --train-workers 3 \
    --fit-weight-err-sq 0.30 --fit-weight-stable 0.50 \
    --fit-weight-jerk   0.10 --fit-weight-mono   0.10 \
    --num-eval-folds 5 \
    --save-stage-checkpoints "$STAGE_DIR" \
    --base-seed 20260530 \
    --save-winner "$WINNER_A" 2>&1 | tee "$PLAN_A_LOG"
fi

if [ "$PLAN" = "--plan-a-only" ]; then
  echo "Plan A v2 done — exiting (--plan-a-only)."
  exit 0
fi

if [ "$PLAN" = "--plan-b-only" ]; then
  WINNER_A=$(ls -t "$WORKDIR"/winner_planAv2_*.pkl 2>/dev/null | head -1)
  if [ -z "$WINNER_A" ]; then
    echo "ERROR: --plan-b-only but no winner_planAv2_*.pkl in $WORKDIR"; exit 1
  fi
  echo "Loading latest Plan A v2 winner: $WINNER_A"
fi

echo ""
echo "=========================================================="
echo "PLAN B v2: memory-only refinement (stability-dominant + K-fold)"
echo "  load:    $WINNER_A"
echo "  log:     $PLAN_B_LOG"
echo "  winner:  $WINNER_B"
echo "=========================================================="
python -u tests/run_memory_refinement.py \
  --load-winner "$WINNER_A" \
  --memory-gens 300 --memory-patience 15 \
  --pop 100 --elitism 0.2 --crossover-rate 0.5 \
  --eval-episodes 20 --steps 400 --tilt 15 \
  --universe-episodes 3 \
  --rg-rounds 2 --rg-episodes-per-round 4 --rg-eval-episodes 3 \
  --train-workers 3 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.60 \
  --fit-weight-jerk   0.0  --fit-weight-mono   0.0 \
  --num-eval-folds 5 \
  --base-seed 20260531 \
  --save-winner "$WINNER_B" 2>&1 | tee "$PLAN_B_LOG"

echo ""
echo "=========================================================="
echo "Plan A v2 + B v2 complete."
echo "  Plan A v2 winner: $WINNER_A"
echo "  Plan A v2 stages: $STAGE_DIR/"
echo "  Plan B v2 winner: $WINNER_B"
echo "=========================================================="
