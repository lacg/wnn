#!/bin/bash
# _op31 lag isolation (Luiz 06/11): the _op31 champion recipe on the ABI-31 plant (--imu-inclinometer
# --no-rotor-drag; xy 0, no pos inputs), ABI 37 wheel, grid stage only (stopped after GRID WINNER).
#   R1 lag 0.20 (key)   R0 lag 0 (control on ABI 37)   R2 lag 0.20 + lag law leak 0.7/dmax 0.6
cd /Users/lacg/wnn
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g')
run() { tag=$1; shift
  out=logs/controller/op31_lag/$tag.out
  echo "$(date -u +%FT%TZ) START $tag $*" >> logs/controller/op31_lag/runs.log
  PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE --imu-inclinometer --no-rotor-drag "$@" > $out 2>&1 &
  pid=$!
  until grep -q "GRID WINNER" $out 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 10; done
  kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$(date -u +%FT%TZ) END $tag $(grep -a 'GRID WINNER' $out | grep -oE 'steady=[^ ]+  err=[^ ]+  stable=[^ ]+  alt=[^ ]+')" >> logs/controller/op31_lag/runs.log
}
run R1_lag020 --motor-lag-s 0.20
run R0_lag0 --motor-lag-s 0
run R2_lag020_law --motor-lag-s 0.20 --delta-leak 0.7 --delta-max 0.6
echo "$(date -u +%FT%TZ) ALL DONE" >> logs/controller/op31_lag/runs.log
