#!/usr/bin/env bash
# Sample TRUE memory (phys_footprint, not RSS) for a controller process and
# capture allocation stacks at high-water mark.
#
# RSS is the wrong metric here: during phase 2 the sampler read ~11 GB while
# phys_footprint was 44 GB, because macOS had compressed ~21 GB out of resident
# memory. footprint(8) reports what jetsam actually counts against the process.
#
# Usage: memprobe_watch.sh <pid> <trigger_GB> <outdir>
set -u
PID="$1"; TRIG="${2:-4}"; OUT="${3:-/private/tmp/memprobe}"
mkdir -p "$OUT"
LOG="$OUT/footprint.log"
CAPTURED=0
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
	# footprint prints e.g. "Footprint: 44 GB" or "Footprint: 895 MB"
	LINE=$(/usr/bin/footprint -p "$PID" 2>/dev/null | grep -oE "Footprint: [0-9.]+ [KMG]B" | head -1)
	VAL=$(echo "$LINE" | awk '{v=$2; u=$3; if(u=="GB") print v; else if(u=="MB") print v/1024; else print v/1048576}')
	[ -z "$VAL" ] && { sleep 5; continue; }
	PEAK=$(awk -v a="$PEAK" -v b="$VAL" 'BEGIN{print (b>a)?b:a}')
	echo "$(date -u +%H:%M:%S) footprint=${VAL}GB peak=${PEAK}GB" >> "$LOG"
	# Capture the stacks ONCE, at the first crossing of the trigger.
	if [ "$CAPTURED" -eq 0 ] && awk -v v="$VAL" -v t="$TRIG" 'BEGIN{exit !(v>t)}'; then
		echo "$(date -u +%H:%M:%S) CAPTURING at ${VAL}GB" >> "$LOG"
		malloc_history "$PID" -allBySize > "$OUT/allBySize.txt" 2>"$OUT/allBySize.err"
		heap "$PID" > "$OUT/heap.txt" 2>&1
		CAPTURED=1
		echo "$(date -u +%H:%M:%S) CAPTURED" >> "$LOG"
	fi
	sleep 5
done
echo "$(date -u +%H:%M:%S) process $PID exited, peak=${PEAK}GB" >> "$LOG"
