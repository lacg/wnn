#!/usr/bin/env bash
# controller_mem_sampler.sh — OOM culprit finder.
# Samples the full memory breakdown every INTERVAL seconds while a controller runs,
# so we can SEE which process/category grows toward the wall (vs. the watchdog's
# single post-hoc RSS reading). Attributes RAM to: controller (phased_ga) RSS,
# IDS worker RSS, and system categories (wired/compressed/free) from vm_stat.
#
# Usage: scripts/controller_mem_sampler.sh [INTERVAL_SEC] [OUT_CSV]
#   INTERVAL_SEC default 3 ; OUT_CSV default /private/tmp/wnn_mem_sampler.csv
# Run it detached BEFORE launching the controller; Ctrl-C or kill to stop.

INTERVAL="${1:-3}"
OUT="${2:-/private/tmp/wnn_mem_sampler.csv}"
PAGE=16384  # M-series page size (bytes)

# CSV header
echo "ts_utc,real_free_gb,wired_gb,compressed_gb,active_gb,inactive_gb,ctrl_pid,ctrl_rss_gb,worker_rss_gb,ctrl_cmd_gen" > "$OUT"
echo "[mem-sampler] armed interval=${INTERVAL}s -> $OUT" >&2

while true; do
	TS="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

	# vm_stat page counts -> GB
	read FREE SPEC WIRED COMPRESSED ACTIVE INACTIVE < <(vm_stat | awk -v p=$PAGE '
		/Pages free/                {gsub(/\./,"",$3); free=$3}
		/Pages speculative/         {gsub(/\./,"",$3); spec=$3}
		/Pages wired down/          {gsub(/\./,"",$4); wired=$4}
		/Pages occupied by compressor/ {gsub(/\./,"",$5); comp=$5}
		/Pages active/              {gsub(/\./,"",$3); act=$3}
		/Pages inactive/            {gsub(/\./,"",$3); inact=$3}
		END{print free, spec, wired, comp, act, inact}')
	REAL_FREE=$(awk -v f=$FREE -v s=$SPEC -v p=$PAGE 'BEGIN{printf "%.2f",(f+s)*p/1073741824}')
	WIRED_GB=$(awk -v w=$WIRED -v p=$PAGE 'BEGIN{printf "%.2f",w*p/1073741824}')
	COMP_GB=$(awk -v c=$COMPRESSED -v p=$PAGE 'BEGIN{printf "%.2f",c*p/1073741824}')
	ACT_GB=$(awk -v a=$ACTIVE -v p=$PAGE 'BEGIN{printf "%.2f",a*p/1073741824}')
	INACT_GB=$(awk -v i=$INACTIVE -v p=$PAGE 'BEGIN{printf "%.2f",i*p/1073741824}')

	# controller (phased_ga) — first match
	CPID="$(pgrep -f 'wnn.control.phased_ga' | head -1)"
	if [ -n "$CPID" ]; then
		CRSS=$(ps -o rss= -p "$CPID" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
		# pull current NEURONS/MEMORY gen from the newest run.out (cheap tail+grep)
		RUNOUT="$(ls -t /Users/lacg/wnn/logs/controller/c10_gran_*/**/run.out 2>/dev/null | head -1)"
		GEN="$(grep -oE 'Gen [0-9]+/[0-9]+' "$RUNOUT" 2>/dev/null | tail -1 | tr ' ' '_')"
		[ -z "$GEN" ] && GEN="$(grep -oE 'STAGE [0-9]' "$RUNOUT" 2>/dev/null | tail -1 | tr ' ' '_')"
	else
		CPID=""; CRSS="0.00"; GEN="no_ctrl"
	fi

	# IDS worker (fixed pid pattern)
	WPID="$(pgrep -f 'wnn.ram.experiments.worker' | head -1)"
	WRSS="0.00"
	[ -n "$WPID" ] && WRSS=$(ps -o rss= -p "$WPID" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')

	echo "${TS},${REAL_FREE},${WIRED_GB},${COMP_GB},${ACT_GB},${INACT_GB},${CPID},${CRSS},${WRSS},${GEN}" >> "$OUT"

	sleep "$INTERVAL"
done
