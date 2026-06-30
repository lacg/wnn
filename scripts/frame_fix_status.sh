#!/bin/bash
# Status report for the frame-fix validation rounds. Reports each round that has started:
#   ROUND 1 (val):  FrameFixVal_20260627  — grid-bits 24/30, folds 3 (A/B vs the buggy baseline)
#   ROUND 2 (bits): FrameFixBits_20260627 — grid-bits 100, folds 5 (does more wiring lift it?)
# Per variant×seed: what it's about, phase+stage, stable%/err°/steady° (MEM held-out when done;
# NEURONS held-out or in-search interim while running — SRC says which), and wall duration.
# Baselines: S16 prior 87.2% | coupled-anchor 70.5% | decouple ~51% (held-out stable%).
set -u
cd /Users/lacg/wnn
VARIANTS="s16 anchor decouple peraxis tilt pwm pidmix pidmix_pwm pidmix_pwm_tilt"
SEEDS="20260609 20260610"

desc() { case "$1" in
  s16)        echo "baseline obs-OFF (9f)";;
  anchor)     echo "yaw-err COUPLED (10f)";;
  decouple)   echo "yaw-err + DECOUPLED banks (10f)";;
  peraxis)    echo "roll+pitch P+I, no-yaw (13f)";;
  tilt)       echo "tilt-to-vert P+I (11f)";;
  pwm)        echo "raw throttle accum (13f)";;
  pidmix)     echo "FULL 3-axis PID: rp P+I + yaw P+I (15f)";;
  pidmix_pwm) echo "full PID + accumulator (19f)";;
  pidmix_pwm_tilt) echo "full PID + accum + lumped tilt P+I (21f)";;
  *) echo "?";; esac; }

mins_since() { local t0; t0=$(date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s 2>/dev/null) || { echo "?"; return; }
  echo $(( ( $(date +%s) - t0 ) / 60 )); }

report_round() {
  local ROOT="$1" label="$2"
  # skip a round that hasn't produced anything yet
  ls "$ROOT"/*/run.out >/dev/null 2>&1 || { return; }
  local done_ct=0 total=0 s v
  for s in $SEEDS; do for v in $VARIANTS; do total=$((total+1)); [ -f "$ROOT/${v}_seed${s}/done.json" ] && done_ct=$((done_ct+1)); done; done
  echo "----- ${label}  (${ROOT##*/})  progress ${done_ct}/${total} -----"
  local act
  act=$(ps -eo command | grep phased_ga | grep -v grep | grep "$ROOT/" | grep -oE "save-winner [^ ]+" | head -1 | sed -E 's#.*/([a-z0-9]+)_seed([0-9]+)/.*#\1 seed\2#')
  if [ -n "$act" ]; then
    local an aseed gs gmin gmax; an=${act%% *}; aseed=$(echo "$act" | grep -oE "[0-9]+")
    gs=$(grep -oE "stable= *[0-9.]+%" "$ROOT/${an}_seed${aseed}/run.out" 2>/dev/null | grep -oE "[0-9.]+" | head -6 | sort -n)
    gmin=$(echo "$gs" | head -1); gmax=$(echo "$gs" | tail -1)
    echo ">> NOW: ${act} — $(desc "$an") | grid stable spread: ${gmin:-?}%..${gmax:-?}% (flat ⇒ memoryless)"
  fi
  printf "%-9s %-7s %-11s %-11s %-11s %-11s %-7s %-6s\n" VARIANT SEED PHASE STABLE±SD ERR±SD STEADY±SD SRC DUR
  for s in $SEEDS; do
    for v in $VARIANTS; do
      local d="$ROOT/${v}_seed${s}" out="$ROOT/${v}_seed${s}/run.out"
      if [ ! -f "$out" ]; then printf "%-9s %-7s %-11s\n" "$v" "$s" "pending"; continue; fi
      local phase stg gen line src stable err steady dur st g
      if [ -f "$d/done.json" ]; then phase="DONE"
      elif grep -q "FAIL ${v} seed=${s}" "$ROOT/driver.log" 2>/dev/null \
        || grep -q "FAIL ${v} ${ROOT##*/} seed=${s}" logs/controller/FrameFixPidmix_20260628.log 2>/dev/null; then phase="FAIL/OOM"
      else
        stg=$(grep -oE "ControllerGA-(Neurons|Memory|Connections|Bits)|STAGE [0-9]|grid" "$out" 2>/dev/null | tail -1)
        gen=$(grep -oE "Gen [0-9]+/[0-9]+" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
        case "$stg" in *Memory*) phase="run:MEM ${gen}";; *Neurons*) phase="run:NEUR ${gen}";; *) phase="run:grid";; esac
      fi
      line=$(grep "MEMORY MULTI-SEED held-out" "$out" 2>/dev/null | tail -1); src="ho-mem"
      [ -z "$line" ] && { line=$(grep "NEURONS MULTI-SEED held-out" "$out" 2>/dev/null | tail -1); src="ho-neur"; }
      if [ -n "$line" ]; then
        stable=$(echo "$line" | grep -oE "stable=[^ ,]+" | head -1 | sed 's/stable=//')
        err=$(echo "$line" | grep -oE "err=[^ ,]+" | head -1 | sed 's/err=//')
        steady=$(echo "$line" | grep -oE "steady=[^ ,]+" | head -1 | sed 's/steady=//')
      else
        g=$(grep -E "Gen [0-9]+/" "$out" 2>/dev/null | tail -1); src="search"
        stable=$(echo "$g" | grep -oE "stable=[^ ,]+" | head -1 | sed 's/stable=//')
        err=$(echo "$g" | grep -oE "err=[^ ,]+" | head -1 | sed 's/err=//'); steady="—"
      fi
      if [ "$phase" = "DONE" ]; then
        dur=$(grep "Total wall time" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+ min" | head -1 | sed 's/ min/m/')
      else
        st=$( { grep "START ${v} seed=${s}" "$ROOT/driver.log" 2>/dev/null; grep "START ${v} ${ROOT##*/} seed=${s}" logs/controller/FrameFixPidmix_20260628.log 2>/dev/null; } | tail -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
        [ -n "$st" ] && dur="$(mins_since "$st")m" || dur="—"
      fi
      printf "%-9s %-7s %-11s %-11s %-11s %-11s %-7s %-6s\n" "$v" "$s" "$phase" "${stable:-—}" "${err:-—}" "${steady:-—}" "$src" "${dur:-—}"
    done
  done
  echo
}

echo "================ FRAME-FIX VALIDATION — $(date '+%d/%m/%Y %H:%M:%S') ================"
echo "fitness weights (ALL, constant): err²=0.25  steady=0.35  stable=0.20  jerk=0.15  mono=0.05"
freepct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2}')
ctl=$(ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
ids=$(ps -eo rss,command | grep flow_runner | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
echo "mem free: ${freepct:-?}  | controller RSS: ${ctl:-0}GB  | IDS worker RSS: ${ids:-0}GB"
echo

report_round logs/controller/FrameFixVal_20260627  "ROUND 1 — grid-bits 24/30, folds 3"
report_round logs/controller/FrameFixBits_20260627 "ROUND 2 — grid-bits 100, folds 5"

echo "SRC: ho-mem=final MEMORY held-out (headline) | ho-neur=neurons-stage held-out | search=in-search interim (NOT held-out)"
echo "about: s16=$(desc s16) | anchor=$(desc anchor) | decouple=$(desc decouple) | peraxis=$(desc peraxis) | tilt=$(desc tilt) | pwm=$(desc pwm)"
