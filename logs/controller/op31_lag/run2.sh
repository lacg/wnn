#!/bin/bash
# op31 lag isolation, part 2 (never-idle, 06/11): does memory (state neurons) recover the lag on the clean recipe?
#   R3 lag 0.20 + sn=4 (default law)   R4 lag 0.20 + sn=4 + lag law
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
  echo "$(date -u +%FT%TZ) END $tag $(grep -a 'GRID WINNER' $out | grep -oE 'sn=[0-9]+ .*seed_pop')" >> logs/controller/op31_lag/runs.log
}
run R3_lag020_sn4 --motor-lag-s 0.20 --grid-state-neurons 4 --max-state-neurons 4
run R4_lag020_sn4_law --motor-lag-s 0.20 --grid-state-neurons 4 --max-state-neurons 4 --delta-leak 0.7 --delta-max 0.6
echo "$(date -u +%FT%TZ) PART 2 DONE" >> logs/controller/op31_lag/runs.log
