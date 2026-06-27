#!/bin/bash
set -u
cd /Users/lacg/wnn
LOG=logs/controller/YawHighCov_20260627/driver.log
mkdir -p logs/controller/YawHighCov_20260627; exec >>"$LOG" 2>&1
echo "[yaw-hc] $(date '+%Y-%m-%d %H:%M:%S') START high-coverage A/B (grid-bits 72-88, 2 seeds)"
for s in 20260609 20260610; do
  [ -f "/tmp/wnn_yaw_hc_seed${s}_done.json" ] && { echo "skip $s"; continue; }
  echo "[yaw-hc] $(date '+%H:%M:%S') -> seed $s"; bash scripts/run_yaw_highcov_seed.sh "$s"
done
echo "{\"yaw_highcov_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_highcov_done.json
echo "[yaw-hc] $(date '+%Y-%m-%d %H:%M:%S') HIGH-COVERAGE A/B COMPLETE"
