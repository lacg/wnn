#!/bin/zsh
# y3_gap (20/11): per-axis VAL split (tuning; decides Y3+L vs the --delta-max-yaw gain pre-check) of the Y3 and Y+F headline winners, s1-s5. nice 15, serial.
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
D=/Users/lacg/wnn/logs/controller/ctrl48
for s in 1 2 3 4 5; do
  for arm in dmy03 ffF; do
    tag=C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_${arm}_full_s$s
    PYTHONPATH=/Users/lacg/wnn/src:/Users/lacg/wnn/src/wnn nice -n 15 $PY $D/dmax_yaw/dmy_axis.py --tag $tag --genome WINNER --units val --out $D/y3_gap/${arm}_s${s}_val.npz > $D/y3_gap/${arm}_s${s}_val.log 2>&1
  done
done
echo done > $D/y3_gap/axis.done
