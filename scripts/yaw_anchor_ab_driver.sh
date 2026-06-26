#!/bin/bash
# Yaw-anchor A/B driver: run the obs_yaw_err arm across the 5 absolute sweep seeds
# (20260609..13), SEQUENTIALLY — one controller at a time (never contend the GPU with
# itself; never touch the IDS worker). Compares held-out stability vs blind S16 (85.4%).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/YawAnchorAB_20260626/driver.log
mkdir -p logs/controller/YawAnchorAB_20260626; exec >>"$LOG" 2>&1
SEEDS="20260609 20260610 20260611 20260612 20260613"
echo "[yaw-ab] $(date '+%Y-%m-%d %H:%M:%S') START obs_yaw_err A/B (5 seeds, sequential)"
i=0; n=$(echo $SEEDS | wc -w | tr -d ' ')
for s in $SEEDS; do
  i=$((i+1))
  echo "[yaw-ab] $(date '+%H:%M:%S') ($i/$n) -> seed $s"
  bash scripts/run_yaw_anchor_seed.sh "$s"
done
echo "{\"yaw_anchor_ab_done\":true,\"seeds\":$n,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_anchor_ab_done.json
echo "[yaw-ab] $(date '+%Y-%m-%d %H:%M:%S') A/B COMPLETE (all $n seeds)"
