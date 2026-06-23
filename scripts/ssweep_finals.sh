#!/bin/bash
# Steady-sweep FINALS: take the n=3 picked finalists to n=5 by running 2 more
# seeds (20260612, 20260613 = rounds 4+5), INTERLEAVED by seed (all finalists at
# seed12, then all at seed13). Args: finalist labels (e.g. S02 S18). Weights are
# looked up below so the labels stay consistent with rounds 1-3.
set -u
cd /Users/lacg/wnn
declare -A WT=(
 [S01]="0.40 0.00 0.50 0.05 0.05"  [S02]="0.40 0.10 0.40 0.05 0.05"  [S04]="0.30 0.30 0.30 0.05 0.05"
 [S06]="0.20 0.50 0.20 0.05 0.05"  [S07]="0.15 0.60 0.15 0.05 0.05"  [S09]="0.45 0.20 0.25 0.05 0.05"
 [S16]="0.25 0.35 0.20 0.15 0.05"  [S18]="0.25 0.30 0.20 0.125 0.125"
)
FINALISTS=("$@")
[ ${#FINALISTS[@]} -ge 1 ] || { echo "usage: ssweep_finals.sh LABEL [LABEL...]"; exit 1; }
LOG=logs/controller/Ssweep_20260622/driver_finals.log
mkdir -p logs/controller/Ssweep_20260622
exec >>"$LOG" 2>&1
echo "[finals] $(date '+%Y-%m-%d %H:%M:%S') START finalists: ${FINALISTS[*]}"
for SEED in 20260612 20260613; do
  for lab in "${FINALISTS[@]}"; do
    [ -n "${WT[$lab]:-}" ] || { echo "[finals] unknown label $lab"; continue; }
    echo "[finals] $(date '+%H:%M:%S') -> $lab ${WT[$lab]} seed=$SEED"
    bash scripts/launch_w2_ssweep.sh $lab ${WT[$lab]} $SEED
  done
  echo "{\"finals_seed_done\":$SEED}" > /tmp/wnn_ssweep_finals_seed${SEED}_done.json
done
echo "{\"finals_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_finals_done.json
echo "[finals] $(date '+%Y-%m-%d %H:%M:%S') FINALS COMPLETE (n=5)"
