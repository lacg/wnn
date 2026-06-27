#!/bin/bash
# Status report for the frame-fix validation sweep (logs/controller/FrameFixVal_20260627).
# Per variant×seed: phase, grid CE spread (the memoryless TELL — pre-fix obs runs gave an
# identical grid; post-fix should spread), NEURONS + MEMORY held-out stable%, wall time.
# Baselines: S16 prior 87.2% | coupled-anchor prior 70.5% | decouple prior ~51%.
set -u
cd /Users/lacg/wnn
ROOT=logs/controller/FrameFixVal_20260627
VARIANTS="s16 anchor decouple peraxis tilt pwm"
SEEDS="20260609 20260610"

echo "================ FRAME-FIX VALIDATION — $(date '+%d/%m/%Y %H:%M:%S') ================"
# memory + contention (OOM watch — the prior seed-11 run was SIGKILLed mid-MEMORY-stage)
freepct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2}')
ctl=$(ps -eo rss,command | grep phased_ga | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
ids=$(ps -eo rss,command | grep flow_runner | grep -v grep | awk '{s+=$1} END{printf "%.1f", s/1048576}')
echo "mem free: ${freepct:-?}  | controller RSS: ${ctl:-0}GB  | IDS worker RSS: ${ids:-0}GB"

done_ct=0; total=0
for s in $SEEDS; do for v in $VARIANTS; do total=$((total+1)); [ -f "$ROOT/${v}_seed${s}/done.json" ] && done_ct=$((done_ct+1)); done; done
echo "progress: ${done_ct}/${total} runs complete"
echo

printf "%-9s %-7s %-9s %-18s %-9s %-9s %-8s\n" VARIANT SEED PHASE "GRID-CE(min..max)" NEUR-HO MEM-HO WALL
printf "%-9s %-7s %-9s %-18s %-9s %-9s %-8s\n" ------- ---- ----- ----------------- ------- ------ ----
for s in $SEEDS; do
  for v in $VARIANTS; do
    d="$ROOT/${v}_seed${s}"; out="$d/run.out"
    if [ ! -f "$out" ]; then printf "%-9s %-7s %-9s\n" "$v" "$s" "—pending"; continue; fi
    if [ -f "$d/done.json" ]; then phase="DONE"; elif grep -q "FAIL ${v} seed=${s}" "$ROOT/driver.log" 2>/dev/null; then phase="FAIL/OOM"; else phase="running"; fi
    # grid CE spread = the differentiation tell
    cemin=$(grep -oE "CE= *[0-9.]+" "$out" 2>/dev/null | head -6 | grep -oE "[0-9.]+" | sort -n | head -1)
    cemax=$(grep -oE "CE= *[0-9.]+" "$out" 2>/dev/null | head -6 | grep -oE "[0-9.]+" | sort -n | tail -1)
    grid="${cemin:-?}..${cemax:-?}"
    nho=$(grep "NEURONS MULTI-SEED held-out" "$out" 2>/dev/null | tail -1 | grep -oE "stable=[0-9.]+±?[0-9.]*%?" | head -1 | sed 's/stable=//')
    mho=$(grep "MEMORY MULTI-SEED held-out" "$out" 2>/dev/null | tail -1 | grep -oE "stable=[0-9.]+±?[0-9.]*%?" | head -1 | sed 's/stable=//')
    wall=$(grep "Total wall time" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+ min" | head -1)
    printf "%-9s %-7s %-9s %-18s %-9s %-9s %-8s\n" "$v" "$s" "$phase" "$grid" "${nho:-—}" "${mho:-—}" "${wall:-—}"
  done
done
echo
echo "baselines: S16 87.2% | coupled-anchor 70.5% | decouple ~51%   (MEM-HO = held-out stable%, the headline)"
echo "TELL: pre-fix obs grid was IDENTICAL across configs (memoryless). Post-fix GRID-CE should spread."
