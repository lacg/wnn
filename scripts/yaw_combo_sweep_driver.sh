#!/bin/bash
# Yaw-anchor combo-sweep driver: all 18 S-series weight combos × seed 20260609, WITH
# --obs-yaw-err, ORDERED by `stable` weight descending (hypothesis-favored first so we get
# early signal / early cull on whether more stability pressure cures the anchor brittleness).
# Sequential, one controller at a time; never touches the IDS worker. Resume-friendly: skips
# any combo whose done-marker already exists.
set -u
cd /Users/lacg/wnn
SEED=20260609
LOG=logs/controller/YawComboSweep_20260626/driver.log
mkdir -p logs/controller/YawComboSweep_20260626; exec >>"$LOG" 2>&1
# stable-weight desc: .50 .50 .40 .35 .35 .30 .25 .25 .25 .25 .25 .20 .20 .20 .20 .15 .15 .05
ORDER="S01 S12 S02 S03 S13 S04 S08 S09 S05 S10 S11 S06 S16 S17 S18 S07 S14 S15"
echo "[combo] $(date '+%Y-%m-%d %H:%M:%S') START 18-combo anchor sweep (seed $SEED, stable-desc order)"
i=0; n=$(echo $ORDER | wc -w | tr -d ' ')
for c in $ORDER; do
  i=$((i+1))
  if [ -f "/tmp/wnn_yaw_combo_${c}_seed${SEED}_done.json" ]; then
    echo "[combo] $(date '+%H:%M:%S') ($i/$n) -> $c SKIP (already done)"; continue
  fi
  echo "[combo] $(date '+%H:%M:%S') ($i/$n) -> $c"
  bash scripts/run_yaw_combo_seed.sh "$c" "$SEED"
done
echo "{\"yaw_combo_sweep_done\":true,\"combos\":$n,\"seed\":$SEED,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_yaw_combo_sweep_done.json
echo "[combo] $(date '+%Y-%m-%d %H:%M:%S') COMBO SWEEP COMPLETE (all $n)"
