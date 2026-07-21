#!/usr/bin/env bash
# Phase-2 memory truth-sampler.
#
# Samples phys_footprint, NOT RSS. The 20/07 phase-2 sampler read ~11 GB RSS
# while the process was actually at 44 GB phys_footprint (peak 55 GB) — macOS
# had compressed ~21 GB out of residency, so RSS simply stopped counting it.
# phys_footprint is what jetsam charges the process, and it is the number the
# ABI-20 cell migration has to move.
#
# Follows whatever phased_ga is currently running (the runner spawns one arm at
# a time), so it survives the blind -> yaw arm handover. Logs one line per tick
# plus a running peak; never kills anything (the mem-watchdog owns that call).
set -u
OUT="${1:-/private/tmp/wnn_phase2_footprint.log}"
INTERVAL="${2:-30}"
PEAK=0
: > "$OUT"
while true; do
	# The runner wraps the arm in `/usr/bin/time -l`, so a bare pgrep on the
	# module name matches BOTH the wrapper and the real interpreter. Select the
	# PYTHON process (the wrapper reports ~944 KB and would read as "no memory
	# used at all" — the same wrapper-vs-child confusion that made the 20/07
	# snapshot list 7790=time and 7791=python separately).
	PID=""
	for p in $(pgrep -f "wnn.control.phased_ga"); do
		if ps -p "$p" -o comm= 2>/dev/null | grep -qi python; then PID="$p"; break; fi
	done
	if [ -z "$PID" ]; then
		# Runner may be between arms; keep watching unless the whole chain is done.
		[ -f /tmp/wnn_prodab_phase2_done.json ] && { echo "$(date -u +%H:%M:%S) phase2 complete, peak=${PEAK}GB" >> "$OUT"; exit 0; }
		pgrep -f run_prod_reflex_then_dfa.sh >/dev/null || { echo "$(date -u +%H:%M:%S) runner gone, peak=${PEAK}GB" >> "$OUT"; exit 0; }
		sleep "$INTERVAL"; continue
	fi
	FP=$(/usr/bin/footprint -p "$PID" 2>/dev/null | grep -oE "Footprint: [0-9.]+ [KMG]B" | head -1)
	VAL=$(echo "$FP" | awk '{v=$2; u=$3; if(u=="GB") print v; else if(u=="MB") print v/1024; else print v/1048576}')
	[ -z "$VAL" ] && { sleep "$INTERVAL"; continue; }
	PEAK=$(awk -v a="$PEAK" -v b="$VAL" 'BEGIN{print (b>a)?b:a}')
	AVAIL=$(vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);gsub(/\./,"",p); printf "%.1f",(f+i+s+p)*16384/1073741824}')
	SWAP=$(sysctl -n vm.swapusage | awk '{print $6}')
	echo "$(date -u +%H:%M:%S) pid=$PID footprint=${VAL}GB peak=${PEAK}GB avail=${AVAIL}GB swapused=${SWAP}" >> "$OUT"
	sleep "$INTERVAL"
done
