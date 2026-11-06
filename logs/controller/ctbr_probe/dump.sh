#!/bin/bash
# CTBR imitation probe (06/11/2026): dump the exact cascade recipe fixtures (no training).
cd /Users/lacg/wnn
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python; L=logs/controller/ctbr_probe
strip() { grep -v '^#' "$1" | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g'; }
OP31=$(strip experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt)" --imu-inclinometer --no-rotor-drag"
MA=$(strip experiments/ctrl22/ARGS_ma_anchor_lag020.txt)
d() { out=$1; shift; PYTHONPATH=src/wnn nice -n 10 $VP scripts/diag/dump_recipe_fixture.py $L/$out.txt "$@" > $L/dump_$out.out 2>&1; tail -1 $L/dump_$out.out; }
d fx_c1 $OP31 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 &
d fx_c0 $OP31 --motor-lag-s 0 --output-mode ctbr --action-repeat 10 &
d fx_s1 $MA --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 &
d fx_c1p $OP31 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --teacher pid &
d fx_c1_250 $OP31 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 4 &
wait
