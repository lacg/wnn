#!/bin/bash
# Steady-weight sweep ROUND-N driver: run the 8 round-1 SURVIVORS sequentially
# (one controller at a time) at a given seed, to build n=2/n=3 statistics.
# Arg: SEED (round 2 = 20260610, round 3 = 20260611). Interleaved by design
# (one of each survivor per round). Writes /tmp/wnn_ssweep_seed<SEED>_done.json.
set -u
cd /Users/lacg/wnn
SEED="${1:?usage: ssweep_round.sh SEED}"
LOG=logs/controller/Ssweep_20260622/driver_seed${SEED}.log
mkdir -p logs/controller/Ssweep_20260622
exec >>"$LOG" 2>&1
# label   err   steady stable jerk  mono   (the 8 survivors + control anchor)
COMBOS=(
  "S01 0.40 0.00 0.50 0.05 0.05"
  "S02 0.40 0.10 0.40 0.05 0.05"
  "S04 0.30 0.30 0.30 0.05 0.05"
  "S06 0.20 0.50 0.20 0.05 0.05"
  "S07 0.15 0.60 0.15 0.05 0.05"
  "S09 0.45 0.20 0.25 0.05 0.05"
  "S16 0.25 0.35 0.20 0.15 0.05"
  "S18 0.25 0.30 0.20 0.125 0.125"
)
echo "[ssweep-round] $(date '+%Y-%m-%d %H:%M:%S') ROUND seed=$SEED START (${#COMBOS[@]} survivors)"
i=0
for c in "${COMBOS[@]}"; do
  i=$((i+1))
  echo "[ssweep-round] $(date '+%H:%M:%S') ($i/${#COMBOS[@]}) -> $c  seed=$SEED"
  bash scripts/launch_w2_ssweep.sh $c $SEED
done
echo "{\"ssweep_seed_done\":$SEED,\"survivors\":${#COMBOS[@]},\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_seed${SEED}_done.json
echo "[ssweep-round] $(date '+%Y-%m-%d %H:%M:%S') ROUND seed=$SEED COMPLETE"
