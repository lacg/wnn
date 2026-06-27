#!/bin/bash
set -u
cd /Users/lacg/wnn
LOG=logs/controller/YawBalance_20260627/driver.log
mkdir -p logs/controller/YawBalance_20260627; exec >>"$LOG" 2>&1
echo "[yaw-bal] $(date '+%Y-%m-%d %H:%M:%S') START balance-cap A/B (2 seeds)"
for s in 20260609 20260610; do
  [ -f "/tmp/wnn_yaw_bal_seed${s}_done.json" ] && { echo "skip $s (done)"; continue; }
  echo "[yaw-bal] $(date '+%H:%M:%S') -> seed $s"; bash scripts/run_yaw_balance_seed.sh "$s"
done
echo "{\"yaw_balance_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_balance_done.json
echo "[yaw-bal] $(date '+%Y-%m-%d %H:%M:%S') BALANCE A/B COMPLETE"
