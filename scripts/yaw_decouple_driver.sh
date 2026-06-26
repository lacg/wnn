#!/bin/bash
# Decouple-outputs A/B driver: --obs-yaw-err + --decouple-outputs at seeds 20260609, 20260610
# (the seeds with coupled-anchor 70.5%/67.5% + S16 87.2%/88.5% baselines). Sequential; runs
# ALONGSIDE the combo sweep (user-approved GPU sharing) — both time-slice the GPU.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/YawDecouple_20260626/driver.log
mkdir -p logs/controller/YawDecouple_20260626; exec >>"$LOG" 2>&1
SEEDS="20260609 20260610"
echo "[yaw-dec] $(date '+%Y-%m-%d %H:%M:%S') START decouple-outputs A/B (2 seeds)"
i=0; n=$(echo $SEEDS | wc -w | tr -d ' ')
for s in $SEEDS; do
  i=$((i+1))
  if [ -f "/tmp/wnn_yaw_dec_seed${s}_done.json" ]; then echo "[yaw-dec] ($i/$n) $s SKIP (done)"; continue; fi
  echo "[yaw-dec] $(date '+%H:%M:%S') ($i/$n) -> seed $s"
  bash scripts/run_yaw_decouple_seed.sh "$s"
done
echo "{\"yaw_decouple_done\":true,\"seeds\":$n,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_decouple_done.json
echo "[yaw-dec] $(date '+%Y-%m-%d %H:%M:%S') DECOUPLE A/B COMPLETE"
