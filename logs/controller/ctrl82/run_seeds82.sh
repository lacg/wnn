#!/bin/bash
# CTRL-82 VAL pass 4: neutral-selection roll-tail check across seeds (G+T s2,s3; lk90 s2,s3; lk85 s1-s3; l2b s2,s3), 100 VAL units each.
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
cd /Users/lacg/wnn/logs/controller/ctrl82
P=C-S2-d_wposx2_
L="gt_s2:a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_s2 gt_s3:a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_n0g3gz_full_s3
lk90_s2:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk90_full_s2 lk90_s3:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk90_full_s3
lk85_s1:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk85_full_s1 lk85_s2:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk85_full_s2
lk85_s3:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_lk85_full_s3 l2b_s2:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s2
l2b_s3:a63_rf_c200_dc_biasin_ms01_P_both_L1_yld_gt_l2b_full_s3"
for x in $L; do lab=${x%%:*}; tag=${x#*:}
  nice -n 15 $PY eps82.py $P$tag /tmp/ctrl82/eps_$lab.npz 100 > /tmp/ctrl82/eps_$lab.log 2>&1
  $PY -c "import numpy as np; u=np.load('/tmp/ctrl82/eps_$lab.npz')['units']; np.save('/tmp/ctrl82/units_$lab.npy', np.array([[x,0] for x in u]))"
done
n=0
for x in $L; do lab=${x%%:*}; tag=${x#*:}
  nice -n 15 $PY trace82.py $P$tag /tmp/ctrl82/eps_$lab.npz /tmp/ctrl82/units_$lab.npy /tmp/ctrl82/summ_$lab.npz summary > /tmp/ctrl82/summ_$lab.log 2>&1 &
  n=$((n+1)); if [ $((n % 3)) -eq 0 ]; then wait; fi
done
wait; ls /tmp/ctrl82/summ_*_s*.npz
