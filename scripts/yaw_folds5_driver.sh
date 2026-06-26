#!/bin/bash
# Overfit-probe driver: anchor + folds=5 at 2 seeds (20260609, 20260610) — the seeds
# where we have both the folds=3 anchor (held-out 70%/67.5%) AND S16 (87%/88.5%) for a
# direct comparison. Sequential, one controller at a time; never touches the IDS worker.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/YawAnchorFolds5_20260626/driver.log
mkdir -p logs/controller/YawAnchorFolds5_20260626; exec >>"$LOG" 2>&1
SEEDS="20260609 20260610"
echo "[yaw-f5] $(date '+%Y-%m-%d %H:%M:%S') START overfit probe (folds=5, 2 seeds)"
i=0; n=$(echo $SEEDS | wc -w | tr -d ' ')
for s in $SEEDS; do
  i=$((i+1))
  echo "[yaw-f5] $(date '+%H:%M:%S') ($i/$n) -> seed $s"
  bash scripts/run_yaw_anchor_folds5_seed.sh "$s"
done
echo "{\"yaw_f5_probe_done\":true,\"seeds\":$n,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_f5_probe_done.json
echo "[yaw-f5] $(date '+%Y-%m-%d %H:%M:%S') OVERFIT PROBE COMPLETE"
