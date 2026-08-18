#!/usr/bin/env bash
# controller_mem_sampler.sh — OOM culprit finder + rc=137 ATTRIBUTION.
# Samples the full memory breakdown every INTERVAL seconds while a controller runs,
# so we can SEE which process/category grows toward the wall (vs. the watchdog's
# single post-hoc RSS reading). Attributes RAM to: controller (phased_ga) RSS,
# IDS worker RSS, and system categories (wired/compressed/free) from vm_stat.
#
# Usage: scripts/controller_mem_sampler.sh [INTERVAL_SEC] [OUT_CSV] [WATCHDOG_LOG] [RUNOUT_GLOB]
#   INTERVAL_SEC default 3 ; OUT_CSV default /private/tmp/wnn_mem_sampler.csv
# Run it detached BEFORE launching the controller; Ctrl-C or kill to stop.
# It NEVER signals anything — there is no kill/pkill/signal anywhere in this file,
# so it cannot cost a run.
#
# ---------------------------------------------------------------------------
# 11/08/2026 — three additions, all driven by the sn>0 arm's unattributed kill.
# The sn=8 run was SIGKILLed at 5h38m (rc=137), losing all five NEURONS
# generations including the only productive move of the run. The chain labelled it
# "watchdog stop", but that was a GUESS: controller_mem_watchdog.sh logs only via
# echo to stdout, and the live instance had fd 0/1/2 on /dev/null, so there was no
# record of whether the watchdog fired, whether jetsam did, or what the run-up
# looked like. rc=137 only says "someone sent SIGKILL".
#
#   1. AVAIL column. The watchdog decides on AVAIL (free+purgeable+spec+inactive),
#      NOT on the strict-free this sampler used to record. Sampling a metric the
#      killer does not use cannot explain the killer's decision, so avail_gb is now
#      the FIRST data column and uses the watchdog's exact formula.
#   2. DEATH block. On controller disappearance, a companion .deaths.log gets the
#      run-up plus the watchdog log, and states the attribution rule outright:
#      watchdog log speaks at that timestamp => the watchdog killed it; watchdog
#      silent => it did not, look outward (jetsam, OOM, chain timeout, a human).
#   3. Controller pid selection now MATCHES THE WATCHDOG VERBATIM. The old
#      `pgrep -f 'wnn.control.phased_ga' | head -1` carries the exact bug the
#      watchdog fixed on 23/07: it also matches the driver's /usr/bin/time wrapper
#      (lower pid ⇒ head -1 picks IT, so every RSS reading was the wrapper's
#      ~0.2GB) and any `grep`/`tail` a human or agent happens to be running against
#      that string. Two processes disagreeing on what "the controller" is would
#      silently destroy the attribution this file exists to provide.
INTERVAL="${1:-3}"
OUT="${2:-/private/tmp/wnn_mem_sampler.csv}"
WATCHDOG_LOG="${3:-/Users/lacg/wnn/logs/controller/mem_watchdog.log}"
RUNOUT_GLOB="${4:-/Users/lacg/wnn/logs/controller/*/*.out}"
DEATHS="${OUT%.csv}.deaths.log"
PAGE=16384   # M-series page size (bytes)
TAIL_N=30    # samples of run-up quoted into a DEATH block

mkdir -p "$(dirname "$OUT")"

# --- gauges ----------------------------------------------------------------
# AVAIL uses the watchdog's formula EXACTLY (free+inactive+speculative+purgeable),
# because it is the number every watchdog kill branch is gated on.
avail_gb() { vm_stat 2>/dev/null | awk '
	/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3}
	END{printf "%.2f",(f+i+s+p)*16384/1073741824}'; }
swap_mb() { sysctl -n vm.swapusage 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="used"){gsub(/[^0-9.]/,"",$(i+2));printf "%.0f",$(i+2)}}'; }
rss_gb()  { ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.2f",$1/1048576}'; }

# Controller = the PYTHON only. Copied verbatim from controller_mem_watchdog.sh:266
# so both processes name the same process. See note 3 in the header.
ctrl_pid() { ps -axo pid=,command= | awk '$2 !~ /\/usr\/bin\/time/ && tolower($2) ~ /python/ && /[w]nn\.control\.phased_ga/ {print $1; exit}'; }

# Biggest NON-controller RSS consumer — names an external cause when there is one.
top_other() {
	ps -axo rss=,pid=,comm= | sort -rn | awk -v skip="$1" '
		$2 != skip && $1 > 0 {n=split($3,p,"/"); printf "%s(pid %s %.1fGB)", p[n], $2, $1/1048576; exit}'
}

# Current stage/generation, for lining the memory curve up against GA progress.
ctrl_gen() {
	local runout gen
	runout="$(ls -t $RUNOUT_GLOB 2>/dev/null | head -1)"
	[ -z "$runout" ] && { echo "no_runout"; return; }
	gen="$(grep -aoE 'Gen [0-9]+/[0-9]+' "$runout" 2>/dev/null | tail -1 | tr ' ' '_')"
	[ -z "$gen" ] && gen="$(grep -aoE 'STAGE [0-9]' "$runout" 2>/dev/null | tail -1 | tr ' ' '_')"
	[ -z "$gen" ] && gen="pre_gen"
	echo "$gen"
}

# --- DEATH block: the attribution, written the tick after a controller vanishes ---
death_block() {  # $1 = pid that vanished, $2 = its last known RSS
	{
		echo "=========================================================================="
		echo "[DEATH] $(date -u +%FT%TZ) controller pid $1 vanished (last RSS=${2}GB)"
		echo "[DEATH] box NOW: avail=$(avail_gb)GB swap=$(swap_mb)MB"
		echo "[DEATH] --- last ${TAIL_N} samples (run-up) ---"
		tail -n "$TAIL_N" "$OUT" 2>/dev/null
		echo "[DEATH] --- watchdog log, last 15 lines ---"
		if [ -s "$WATCHDOG_LOG" ]; then
			tail -15 "$WATCHDOG_LOG"
			echo "[DEATH] ATTRIBUTION: watchdog log NON-EMPTY — check whether those lines are"
			echo "[DEATH]   timestamped AT this death. A kill_ctrl/pause_ctrl line now ⇒ the"
			echo "[DEATH]   watchdog did it. Only older lines ⇒ it did not."
		else
			echo "(watchdog log empty or absent: $WATCHDOG_LOG)"
			echo "[DEATH] ATTRIBUTION: watchdog SILENT ⇒ it did NOT kill this run. Look"
			echo "[DEATH]   outward: macOS jetsam, an OOM, a chain timeout, a human."
		fi
		echo "=========================================================================="
	} >> "$DEATHS"
}

# --- one CSV row -----------------------------------------------------------
sample_row() {  # $1 = ctrl pid ("" when absent)
	local cpid="$1" crss="0.00" wpid wrss="0.00"
	[ -n "$cpid" ] && crss=$(rss_gb "$cpid")
	wpid="$(pgrep -f 'wnn.ram.experiments.worker' | head -1)"
	[ -n "$wpid" ] && wrss=$(rss_gb "$wpid")
	# vm_stat page counts -> GB (one call, split in awk)
	local cats; cats=$(vm_stat | awk -v p=$PAGE '
		/Pages free/{gsub(/\./,"",$3); free=$3}
		/Pages speculative/{gsub(/\./,"",$3); spec=$3}
		/Pages wired down/{gsub(/\./,"",$4); wired=$4}
		/Pages occupied by compressor/{gsub(/\./,"",$5); comp=$5}
		/Pages active/{gsub(/\./,"",$3); act=$3}
		/Pages inactive/{gsub(/\./,"",$3); inact=$3}
		END{printf "%.2f,%.2f,%.2f,%.2f,%.2f",(free+spec)*p/1073741824,wired*p/1073741824,comp*p/1073741824,act*p/1073741824,inact*p/1073741824}')
	echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'),$(avail_gb),${cats},$(swap_mb),${cpid},${crss},${wrss},$(top_other "${cpid:-0}"),$(ctrl_gen)" >> "$OUT"
	LAST_RSS="$crss"
}

[ -s "$OUT" ] || echo "ts_utc,avail_gb,real_free_gb,wired_gb,compressed_gb,active_gb,inactive_gb,swap_mb,ctrl_pid,ctrl_rss_gb,worker_rss_gb,top_other,ctrl_gen" > "$OUT"
echo "[mem-sampler] $(date -u +%FT%TZ) armed interval=${INTERVAL}s -> $OUT (deaths -> $DEATHS, watchdog_log=$WATCHDOG_LOG) — READ-ONLY, signals nothing" >> "$DEATHS"

PREV_PID=""; LAST_RSS="0.00"
while true; do
	CPID="$(ctrl_pid)"
	# seen->gone, or seen->DIFFERENT pid (the chain's retry starts a new python, so
	# the old one died even though a controller is running again by the time we look).
	if [ -n "$PREV_PID" ] && [ "$CPID" != "$PREV_PID" ]; then
		death_block "$PREV_PID" "$LAST_RSS"
	fi
	sample_row "$CPID"
	PREV_PID="$CPID"
	sleep "$INTERVAL"
done
