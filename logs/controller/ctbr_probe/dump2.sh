#!/bin/bash
# E1 (R4 of the cascade design): the integral features with 1 s memory (--integral-leak 0.999).
cd /Users/lacg/wnn
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python; L=logs/controller/ctbr_probe
strip() { grep -v '^#' "$1" | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g'; }
OP31=$(strip experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt)" --imu-inclinometer --no-rotor-drag"
MA=$(strip experiments/ctrl22/ARGS_ma_anchor_lag020.txt)
d() { out=$1; shift; PYTHONPATH=src/wnn nice -n 10 $VP scripts/diag/dump_recipe_fixture.py $L/$out.txt "$@" > $L/dump_$out.out 2>&1; tail -1 $L/dump_$out.out; }
d fx_c1_il999 $OP31 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --integral-leak 0.999 &
d fx_s1_il999 $MA --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --integral-leak 0.999 &
d fx_c1p_il999 $OP31 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --teacher pid --integral-leak 0.999 &
wait
