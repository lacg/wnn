#!/bin/bash
# Add the S01 control to the finals seeds (20260612, 20260613) so the control
# reaches n=5 matched with the finalists. Same config as the sweep (delta, etc.).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/Ssweep_20260622/driver_s01_finals.log
mkdir -p logs/controller/Ssweep_20260622; exec >>"$LOG" 2>&1
echo "[s01-finals] $(date '+%Y-%m-%d %H:%M:%S') START S01 control at finals seeds"
for SEED in 20260612 20260613; do
  echo "[s01-finals] $(date '+%H:%M:%S') -> S01 seed=$SEED"
  bash scripts/launch_w2_ssweep.sh S01 0.40 0.00 0.50 0.05 0.05 "$SEED"
done
echo "{\"s01_finals_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_s01_finals_done.json
echo "[s01-finals] $(date '+%Y-%m-%d %H:%M:%S') S01 finals COMPLETE (n=5 control)"
