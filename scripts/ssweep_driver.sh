#!/bin/bash
# Steady-weight sweep ROUND 1 driver: run 1 seed of each of the 18 S-combos
# SEQUENTIALLY (one controller at a time), full difficulty + bpf=8. After all 18,
# write the done marker; the watcher reports + we cull, then rounds 2/3 (seed+1,+2)
# run on survivors. Interleaved-by-design (round = 1 of each combo). base-seed 20260609.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/Ssweep_20260622/driver.log
mkdir -p logs/controller/Ssweep_20260622
exec >>"$LOG" 2>&1
SEED=20260609
# label   err   steady stable jerk  mono
COMBOS=(
  "S01 0.40 0.00 0.50 0.05 0.05"
  "S02 0.40 0.10 0.40 0.05 0.05"
  "S03 0.35 0.20 0.35 0.05 0.05"
  "S04 0.30 0.30 0.30 0.05 0.05"
  "S05 0.25 0.40 0.25 0.05 0.05"
  "S06 0.20 0.50 0.20 0.05 0.05"
  "S07 0.15 0.60 0.15 0.05 0.05"
  "S08 0.55 0.10 0.25 0.05 0.05"
  "S09 0.45 0.20 0.25 0.05 0.05"
  "S10 0.30 0.35 0.25 0.05 0.05"
  "S11 0.15 0.50 0.25 0.05 0.05"
  "S12 0.30 0.10 0.50 0.05 0.05"
  "S13 0.30 0.25 0.35 0.05 0.05"
  "S14 0.30 0.45 0.15 0.05 0.05"
  "S15 0.30 0.55 0.05 0.05 0.05"
  "S16 0.25 0.35 0.20 0.15 0.05"
  "S17 0.25 0.35 0.20 0.05 0.15"
  "S18 0.25 0.30 0.20 0.125 0.125"
)
echo "[ssweep-driver] $(date '+%Y-%m-%d %H:%M:%S') ROUND 1 START (${#COMBOS[@]} combos, seed $SEED)"
i=0
for c in "${COMBOS[@]}"; do
  i=$((i+1))
  echo "[ssweep-driver] $(date '+%H:%M:%S') ($i/${#COMBOS[@]}) -> $c"
  bash scripts/launch_w2_ssweep.sh $c $SEED
done
echo "{\"ssweep_r1_done\":true,\"seed\":$SEED,\"combos\":${#COMBOS[@]},\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_r1_done.json
echo "[ssweep-driver] $(date '+%Y-%m-%d %H:%M:%S') ROUND 1 COMPLETE"
