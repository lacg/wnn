#!/bin/bash
# ABSOLUTE round-1 sweep: re-run ALL 18 weight combos (S01 control + S02-S18) on the
# ABSOLUTE substrate (--no-delta-control) at seed 20260609, sequentially (one
# controller at a time). SKIP S09 (already done as the paired absolute at this seed).
# Tests whether the absolute substrate's big delta-vs-absolute win (S09: 85% vs 72%)
# holds across the whole weight space + whether S09 is still the winner on absolute.
# Each combo via run_absolute_winner.sh LABEL 20260609 → S{LABEL}_ABS_seed20260609.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/Ssweep_20260622/driver_absolute_sweep.log
mkdir -p logs/controller/Ssweep_20260622; exec >>"$LOG" 2>&1
COMBOS="S01 S02 S03 S04 S05 S06 S07 S08 S10 S11 S12 S13 S14 S15 S16 S17 S18"  # S09 skipped (done)
echo "[abs-sweep] $(date '+%Y-%m-%d %H:%M:%S') START round-1 absolute (17 combos, S09 already done, seed 20260609)"
i=0; n=$(echo $COMBOS | wc -w | tr -d ' ')
for lab in $COMBOS; do
  i=$((i+1))
  echo "[abs-sweep] $(date '+%H:%M:%S') ($i/$n) -> $lab"
  bash scripts/run_absolute_winner.sh "$lab" 20260609
done
echo "{\"abs_sweep_r1_done\":true,\"combos\":$n,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_ssweep_abs_r1_done.json
echo "[abs-sweep] $(date '+%Y-%m-%d %H:%M:%S') ABSOLUTE round-1 COMPLETE"
