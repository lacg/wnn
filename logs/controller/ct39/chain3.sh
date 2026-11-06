#!/bin/bash
# CTRL-27 / ABI 39 validation (MPCOF cascade teacher fixed, CTRL-33) (Luiz 06/11): pop-50 GRID ONLY, seed 31337002, stopped after
# 'GRID WINNER' — the R0-R2 launcher's protocol (logs/controller/op31_lag/run.sh). Every run trains
# with the per-episode motor asymmetry (L4C mag 0.2, ABI 38 default). One run at a time.
cd /Users/lacg/wnn
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
L=logs/controller/ct39
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
# QUAD arm (Luiz 06/11): does QUAD_WEIGHTED cells (vs BINARY antagonist decode) close the cascade imitation gap? 250 Hz + 100 Hz.
run C1_250hz_quad_ct39 "$OP31" --motor-lag-s 0.20 --output-mode ctbr --action-repeat 4 --memory-mode QUAD_WEIGHTED
run S1_250hz_quad_ct39 "$MA" --motor-lag-s 0.20 --output-mode ctbr --action-repeat 4 --memory-mode QUAD_WEIGHTED
run C1_100hz_quad_ct39 "$OP31" --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --memory-mode QUAD_WEIGHTED
run S1_100hz_quad_ct39 "$MA" --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --memory-mode QUAD_WEIGHTED
echo "$(date -u +%FT%TZ) CHAIN3 (QUAD) DONE" >> $L/runs.log
