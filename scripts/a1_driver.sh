#!/bin/bash
# Integral-input A/B ARM A1 driver: run A1 (per-axis PID) at all 5 sweep seeds
# sequentially (ONE controller at a time), mirroring the absolute sweep's
# 5-seed protocol so n=5 is apples-to-apples vs the S16 baseline. Marker per
# seed + a final done marker. Detach via scripts/detach_launch.py (PPID=1).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/IntegralAB_20260625/driver_a1.log
mkdir -p logs/controller/IntegralAB_20260625; exec >>"$LOG" 2>&1
echo "[a1-driver] $(date '+%Y-%m-%d %H:%M:%S') START A1 seeds 20260609..13"
for SEED in 20260609 20260610 20260611 20260612 20260613; do
  echo "[a1-driver] $(date '+%H:%M:%S') -> seed=$SEED"
  bash scripts/run_a1_seed.sh "$SEED"
  echo "{\"a1_seed_done\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_a1_seed_${SEED}_done.json
done
echo "{\"a1_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_a1_done.json
echo "[a1-driver] $(date '+%Y-%m-%d %H:%M:%S') A1 COMPLETE (n=5)"
