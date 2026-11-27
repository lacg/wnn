#!/bin/bash
# CTRL-82 VAL pass 1: per-episode err on 100 VAL units for the three s1 winners (CPU, nice 15).
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
cd /Users/lacg/wnn/logs/controller/ctrl82
P=C-S2-d_wposx2_
while read lab tag; do
  nice -n 15 $PY eps82.py $P$tag /tmp/ctrl82/eps_$lab.npz 100 > /tmp/ctrl82/eps_$lab.log 2>&1
  tail -1 /tmp/ctrl82/eps_$lab.log
done <<'L'
gt a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_s1
lk90 a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk90_full_s1
l2b a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s1
L
