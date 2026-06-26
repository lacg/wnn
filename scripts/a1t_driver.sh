#!/bin/bash
# Integral-input A/B ARM A1t (anti-windup) driver: run A1t at all 5 sweep seeds
# sequentially (ONE controller at a time), n=5 apples-to-apples vs A1 and S16.
# Per-seed + final markers. Detach via scripts/detach_launch.py (PPID=1).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/IntegralAB_20260625/driver_a1t.log
mkdir -p logs/controller/IntegralAB_20260625; exec >>"$LOG" 2>&1
echo "[a1t-driver] $(date '+%Y-%m-%d %H:%M:%S') START A1t seeds 20260609..13"
for SEED in 20260609 20260610 20260611 20260612 20260613; do
  echo "[a1t-driver] $(date '+%H:%M:%S') -> seed=$SEED"
  bash scripts/run_a1t_seed.sh "$SEED"
  echo "{\"a1t_seed_done\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_a1t_seed_${SEED}_done.json
done
echo "{\"a1t_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_a1t_done.json
echo "[a1t-driver] $(date '+%Y-%m-%d %H:%M:%S') A1t COMPLETE (n=5)"
