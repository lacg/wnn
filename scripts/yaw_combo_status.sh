#!/bin/bash
# Yaw-anchor combo-sweep STATUS REPORT. Run anytime for a ranked snapshot, or loop it
# (see --watch). Parses each combo's HELD-OUT (report-seeds MEMORY MULTI-SEED) line —
# NOT the in-sample FINAL. Ranks completed combos by held-out stable%. Baselines:
# anchor-S16 70.5% (seed09), S16 blind 87.2% (seed09). Usage: yaw_combo_status.sh [--watch]
set -u
cd /Users/lacg/wnn
DIR=logs/controller/YawComboSweep_20260626
SEED=20260609
ORDER="S01 S12 S02 S03 S13 S04 S08 S09 S05 S10 S11 S06 S16 S17 S18 S07 S14 S15"
N=$(echo $ORDER | wc -w | tr -d ' ')
wt_for() {
  case "$1" in
    S01) echo "0.40/0.00/0.50/0.05/0.05";; S02) echo "0.40/0.10/0.40/0.05/0.05";;
    S03) echo "0.35/0.20/0.35/0.05/0.05";; S04) echo "0.30/0.30/0.30/0.05/0.05";;
    S05) echo "0.25/0.40/0.25/0.05/0.05";; S06) echo "0.20/0.50/0.20/0.05/0.05";;
    S07) echo "0.15/0.60/0.15/0.05/0.05";; S08) echo "0.55/0.10/0.25/0.05/0.05";;
    S09) echo "0.45/0.20/0.25/0.05/0.05";; S10) echo "0.30/0.35/0.25/0.05/0.05";;
    S11) echo "0.15/0.50/0.25/0.05/0.05";; S12) echo "0.30/0.10/0.50/0.05/0.05";;
    S13) echo "0.30/0.25/0.35/0.05/0.05";; S14) echo "0.30/0.45/0.15/0.05/0.05";;
    S15) echo "0.30/0.55/0.05/0.05/0.05";; S16) echo "0.25/0.35/0.20/0.15/0.05";;
    S17) echo "0.25/0.35/0.20/0.05/0.15";; S18) echo "0.25/0.30/0.20/0.125/0.125";;
  esac
}

report() {
  local done_n=0 inflight="" inflight_stage=""
  echo "================================================================"
  echo " YAW-ANCHOR COMBO SWEEP — $(date '+%Y-%m-%d %H:%M:%S')  (seed $SEED, folds=3, --obs-yaw-err)"
  echo " held-out = [report-seeds] MEMORY MULTI-SEED (the trustworthy number)"
  echo " baselines: anchor-S16 70.5±17.4%   |   S16 blind 87.2±1.6%  ← beat this"
  echo "================================================================"
  printf " %-5s %-26s %12s %9s %9s\n" "combo" "weights e/st/sta/j/m" "held stbl%" "err°" "steady°"
  echo " ----- -------------------------- ------------ --------- ---------"
  # collect completed rows into a temp, sorted by stable desc
  local tmp; tmp=$(mktemp)
  for c in $ORDER; do
    local f="$DIR/COMBO_${c}_seed${SEED}/run.out"
    [ -f "$f" ] || continue
    local w; w=$(wt_for "$c")
    local hl; hl=$(grep "MEMORY MULTI-SEED held-out" "$f" 2>/dev/null | tail -1)
    if [ -n "$hl" ]; then
      done_n=$((done_n+1))
      local stbl err std
      stbl=$(echo "$hl" | grep -oE "stable=[0-9.]+±[0-9.]+" | sed 's/stable=//')
      err=$(echo  "$hl" | grep -oE "err=[0-9.]+±[0-9.]+"    | sed 's/err=//')
      std=$(echo  "$hl" | grep -oE "steady=[0-9.]+±[0-9.]+" | sed 's/steady=//')
      local sval; sval=$(echo "$stbl" | grep -oE "^[0-9.]+")
      printf "%s\t %-5s %-26s %12s %9s %9s\n" "$sval" "$c" "$w" "$stbl" "$err" "$std" >> "$tmp"
    else
      # in-flight (has run.out, no held-out yet)
      inflight="$c"
      inflight_stage=$(grep -E "STAGE [0-9]|Stage  |Gen [0-9]+/" "$f" 2>/dev/null | tail -1 | sed 's/^ *//' | cut -c1-70)
    fi
  done
  sort -t$'\t' -k1 -gr "$tmp" | cut -f2-
  rm -f "$tmp"
  echo " ----------------------------------------------------------------"
  if [ -f /tmp/wnn_yaw_combo_sweep_done.json ]; then
    echo " STATUS: SWEEP COMPLETE — all $N combos done."
  else
    echo " in-flight: ${inflight:-?}   $inflight_stage"
    local remaining=$(( N - done_n )); [ -n "$inflight" ] && remaining=$(( remaining - 1 ))
    echo " progress: $done_n/$N done, ~$remaining queued   (ETA ~$(( remaining * 55 )) min at ~55min/combo)"
  fi
  echo
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    report
    [ -f /tmp/wnn_yaw_combo_sweep_done.json ] && { echo "[watcher] sweep complete — exiting."; break; }
    sleep 1800
  done
else
  report
fi
