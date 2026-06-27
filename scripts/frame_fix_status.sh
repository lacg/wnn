#!/bin/bash
# Status report for the frame-fix validation sweep (logs/controller/FrameFixVal_20260627).
# Per variant×seed: what it's about, phase+stage, stable%/err°/steady° (MEM held-out when
# done; NEURONS held-out or in-search interim while running — SRC column says which), and
# wall duration. Fitness weights are constant across the sweep (shown in header).
# Baselines: S16 prior 87.2% | coupled-anchor 70.5% | decouple ~51% (held-out stable%).
set -u
cd /Users/lacg/wnn
ROOT=logs/controller/FrameFixVal_20260627
VARIANTS="s16 anchor decouple peraxis tilt pwm"
SEEDS="20260609 20260610"

# what each variant is about
desc() { case "$1" in
  s16)      echo "baseline obs-OFF (9f)";;
  anchor)   echo "yaw-err COUPLED (10f)";;
  decouple) echo "yaw-err + DECOUPLED banks (10f)";;
  peraxis)  echo "roll+pitch P+I, no-yaw (13f)";;
  tilt)     echo "tilt-to-vert P+I (11f)";;
  pwm)      echo "raw throttle accum (13f)";;
  *) echo "?";; esac; }

# minutes between a "YYYY-mm-dd HH:MM:SS" stamp and now (macOS date)
mins_since() { local t0; t0=$(date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s 2>/dev/null) || { echo "?"; return; }
  echo $(( ( $(date +%s) - t0 ) / 60 )); }

echo "================ FRAME-FIX VALIDATION — $(date '+%d/%m/%Y %H:%M:%S') ================"
echo "fitness weights (ALL, constant): err²=0.25  steady=0.35  stable=0.20  jerk=0.15  mono=0.05"
freepct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2}')
ctl=$(ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
ids=$(ps -eo rss,command | grep flow_runner | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
echo "mem free: ${freepct:-?}  | controller RSS: ${ctl:-0}GB  | IDS worker RSS: ${ids:-0}GB"

done_ct=0; total=0
for s in $SEEDS; do for v in $VARIANTS; do total=$((total+1)); [ -f "$ROOT/${v}_seed${s}/done.json" ] && done_ct=$((done_ct+1)); done; done
echo "progress: ${done_ct}/${total} runs complete"

# active run callout + grid stable% spread (the "memory is live" tell, in stable% not CE:
# pre-fix obs grid gave an IDENTICAL stable% across all 6 configs ⇒ memoryless; post-fix spreads)
act=$(ps -eo command | grep phased_ga | grep -v grep | grep -oE "save-winner [^ ]+" | head -1 | sed -E 's#.*/([a-z0-9]+)_seed([0-9]+)/.*#\1 seed\2#')
if [ -n "$act" ]; then
  an=${act%% *}; aseed=$(echo "$act" | grep -oE "[0-9]+")
  gs=$(grep -oE "stable= *[0-9.]+%" "$ROOT/${an}_seed${aseed}/run.out" 2>/dev/null | grep -oE "[0-9.]+" | head -6 | sort -n)
  gmin=$(echo "$gs" | head -1); gmax=$(echo "$gs" | tail -1)
  echo ">> NOW RUNNING: ${act}  — $(desc "$an")  | grid stable spread: ${gmin:-?}%..${gmax:-?}% (flat ⇒ memoryless)"
fi
echo

printf "%-9s %-7s %-11s %-11s %-11s %-11s %-7s %-6s\n" VARIANT SEED PHASE STABLE±SD ERR±SD STEADY±SD SRC DUR
printf "%-9s %-7s %-11s %-11s %-11s %-11s %-7s %-6s\n" ------- ---- ----- -------- ------ --------- --- ---
for s in $SEEDS; do
  for v in $VARIANTS; do
    d="$ROOT/${v}_seed${s}"; out="$d/run.out"
    if [ ! -f "$out" ]; then printf "%-9s %-7s %-11s\n" "$v" "$s" "pending"; continue; fi
    # phase + stage
    if [ -f "$d/done.json" ]; then phase="DONE"
    elif grep -q "FAIL ${v} seed=${s}" "$ROOT/driver.log" 2>/dev/null; then phase="FAIL/OOM"
    else
      stg=$(grep -oE "ControllerGA-(Neurons|Memory|Connections|Bits)|STAGE [0-9]|grid" "$out" 2>/dev/null | tail -1)
      gen=$(grep -oE "Gen [0-9]+/[0-9]+" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
      case "$stg" in *Memory*) phase="run:MEM ${gen}";; *Neurons*) phase="run:NEUR ${gen}";; *) phase="run:grid";; esac
    fi
    # metrics: prefer MEMORY held-out (final), then NEURONS held-out, then in-search gen line
    line=$(grep "MEMORY MULTI-SEED held-out" "$out" 2>/dev/null | tail -1); src="ho-mem"
    if [ -z "$line" ]; then line=$(grep "NEURONS MULTI-SEED held-out" "$out" 2>/dev/null | tail -1); src="ho-neur"; fi
    if [ -n "$line" ]; then
      # held-out line carries ±SD across the 4 report seeds — keep it (the user wants SD)
      stable=$(echo "$line" | grep -oE "stable=[^ ,]+" | head -1 | sed 's/stable=//')
      err=$(echo "$line" | grep -oE "err=[^ ,]+" | head -1 | sed 's/err=//')
      steady=$(echo "$line" | grep -oE "steady=[^ ,]+" | head -1 | sed 's/steady=//')
    else
      g=$(grep -E "Gen [0-9]+/" "$out" 2>/dev/null | tail -1); src="search"
      stable=$(echo "$g" | grep -oE "stable=[^ ,]+" | head -1 | sed 's/stable=//')
      err=$(echo "$g" | grep -oE "err=[^ ,]+" | head -1 | sed 's/err=//')
      steady="—"
    fi
    # duration: Total wall time when done, else mins since START in driver.log
    if [ "$phase" = "DONE" ]; then
      dur=$(grep "Total wall time" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+ min" | head -1 | sed 's/ min/m/')
    else
      st=$(grep "START ${v} seed=${s}" "$ROOT/driver.log" 2>/dev/null | tail -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
      [ -n "$st" ] && dur="$(mins_since "$st")m" || dur="—"
    fi
    printf "%-9s %-7s %-11s %-11s %-11s %-11s %-7s %-6s\n" "$v" "$s" "$phase" "${stable:-—}" "${err:-—}" "${steady:-—}" "$src" "${dur:-—}"
  done
done
echo
echo "SRC: ho-mem=final MEMORY held-out (headline) | ho-neur=neurons-stage held-out | search=in-search interim (NOT held-out)"
echo "about: s16=$(desc s16) | anchor=$(desc anchor) | decouple=$(desc decouple) | peraxis=$(desc peraxis) | tilt=$(desc tilt) | pwm=$(desc pwm)"
