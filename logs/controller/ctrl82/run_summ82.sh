#!/bin/bash
# CTRL-82 VAL pass 3: per-episode summaries, ALL 100 eps of all 100 VAL units, the three s1 winners in parallel (CPU, nice 15).
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
cd /Users/lacg/wnn/logs/controller/ctrl82
P=C-S2-d_wposx2_
while read lab tag; do
  nice -n 15 $PY trace82.py $P$tag /tmp/ctrl82/eps_$lab.npz /tmp/ctrl82/units_all.npy /tmp/ctrl82/summ_$lab.npz summary > /tmp/ctrl82/summ_$lab.log 2>&1 &
done <<'L'
gt a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_s1
lk90 a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk90_full_s1
l2b a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s1
L
wait; tail -n 2 /tmp/ctrl82/summ_*.log
