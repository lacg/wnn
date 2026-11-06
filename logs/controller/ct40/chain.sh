#!/bin/bash
cd /Users/lacg/wnn
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
L=logs/controller/ct40
strip() { grep -v '^#' "$1" | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g'; }
OP31=$(strip experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt)" --imu-inclinometer --no-rotor-drag"
MA=$(strip experiments/ctrl22/ARGS_ma_anchor_lag020.txt)
run() { tag=$1; base=$2; shift 2
  case " $* " in *" --output-mode ctbr "*) base=$(echo " $base " | sed 's/ --obs-pwm / /g');; esac
  while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
  out=$L/$tag.out
  echo "$(date -u +%FT%TZ) START $tag $*" >> $L/runs.log
  PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $base "$@" > $out 2>&1 &
  pid=$!
  until grep -q "GRID WINNER" $out 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 10; done
  kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$(date -u +%FT%TZ) END $tag $(grep -a 'GRID WINNER' $out | grep -oE 'steady=[^ ]+  err=[^ ]+  stable=[^ ]+  alt=[^ ]+')" >> $L/runs.log
}
# ABI 40 grids (Luiz 06/11, option b): cascade fix pair first, motor-level gate-off, then cascade ablations.
CT="--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10"
LAW="--delta-leak 0.7 --delta-max 0.6"
run C1dg_ct40 "$OP31" $CT --ctbr-delta-law $LAW --rg-gate off
run S1dg_ct40 "$MA"   $CT --ctbr-delta-law $LAW --rg-gate off
run M2g_ct40  "$OP31" --motor-lag-s 0.20 $LAW --rg-gate off
run SM2g_ct40 "$MA"   --motor-lag-s 0.20 $LAW --rg-gate off
run C1_ct40   "$OP31" $CT
run C1g_ct40  "$OP31" $CT --rg-gate off
run C1d_ct40  "$OP31" $CT --ctbr-delta-law $LAW
run S1_ct40   "$MA"   $CT
echo "$(date -u +%FT%TZ) CT40 DONE" >> $L/runs.log
