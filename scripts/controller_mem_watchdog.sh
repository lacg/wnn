#!/bin/bash
# Controller memory watchdog v2 — attribution-aware two-tier kill (14/07/2026 PM).
#
# v1 (single 5GB floor) had a blind spot: it keyed on GLOBAL real-free but only ever
# killed the controller, so a transient IDS allocation spike (heavy 64-bit SP flow,
# +8GB in 60s) would SIGKILL a well-behaved controller that was flat at ~4.7GB —
# sacrificing 17 min of correct work for someone else's spike (see the 21:02 kill of
# QUAD arm 27037: RSS 4.9GB, free crashed 13.8→4.06GB in one window, recovered in 2min).
#
# v2 separates SURVIVAL from ATTRIBUTION:
#   HARD floor (default 3GB): real-free below this => SIGKILL controller + chain
#     immediately, regardless of cause. Box-survival / jetsam prevention (the 14/07
#     incident). The controller is the only thing we may kill (NEVER the IDS worker —
#     it has the paper deadline), so it is the release valve of last resort.
#   SOFT floor (default 5GB): real-free below this => kill the controller ONLY if the
#     controller is plausibly the CAUSE — its own RSS is a genuine hog (>= HOG_GB) or
#     it is climbing fast (>= CLIMB_GB over the last 2 ticks ~30s). Otherwise the
#     pressure is EXTERNAL (IDS): log it + RIDE IT OUT; the hard floor is the backstop
#     if it keeps dropping. Transient IDS spikes recover within ~2 min, so a flat
#     controller survives them instead of being collateral.
#
# Usage: controller_mem_watchdog.sh [hard_gb] [soft_gb] [hog_gb] [climb_gb]
HARD_GB="${1:-3}"      # real-free below this: always kill (survival)
SOFT_GB="${2:-5}"      # real-free below this: kill only if controller is the cause
HOG_GB="${3:-12}"      # controller RSS at/above this at soft breach = it's the hog
CLIMB_GB="${4:-1.5}"   # controller RSS rise over last 2 ticks (~30s) = runaway climb

gt() { [ "$(echo "$1 > $2" | bc 2>/dev/null)" = "1" ]; }
lt() { [ "$(echo "$1 < $2" | bc 2>/dev/null)" = "1" ]; }
real_free() { vm_stat 2>/dev/null | awk '/Pages free/{printf "%.2f",$3*16384/1073741824}'; }
kill_ctrl() {  # $1 = cpid, $2 = reason
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGKILL controller $1 (RSS=${rss}GB, free=$(real_free)GB)"
	kill -9 "$1" 2>/dev/null
	# kill the driving chain too so it can't spawn the next arm into the same wall
	pkill -9 -f "run_gran_5arm_capped|rerun_gran_all3_capped|rerun_gran_ternary_binary|granularity_ablation_chain|rerun_teacher_fulls_fixed|run_lqr_mpc_phased|task5_ensemble_hybrids_chain" 2>/dev/null
	echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
	sleep 90
}

echo "[mem-watchdog] v2 armed: HARD=${HARD_GB}GB (always-kill) SOFT=${SOFT_GB}GB (kill iff controller is cause: RSS>=${HOG_GB}GB or climb>=${CLIMB_GB}GB/2ticks)"
prev1=0; prev2=0   # controller RSS at t-1 and t-2 (GB)
while true; do
	free_gb=$(real_free)
	cpid=$(pgrep -f "wnn.control.phased_ga" | head -1)
	if [ -n "$cpid" ]; then
		rss=$(ps -o rss= -p "$cpid" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
		if lt "${free_gb:-99}" "$HARD_GB"; then
			kill_ctrl "$cpid" "HARD floor breach (free<${HARD_GB}GB)"
		elif lt "${free_gb:-99}" "$SOFT_GB"; then
			# attribution: is the controller the cause? (hog OR climbing fast)
			climb=$(echo "${rss:-0} - ${prev2:-0}" | bc 2>/dev/null)
			if gt "${rss:-0}" "$HOG_GB"; then
				kill_ctrl "$cpid" "SOFT breach + controller is HOG (RSS=${rss}GB>=${HOG_GB})"
			elif gt "${climb:-0}" "$CLIMB_GB"; then
				kill_ctrl "$cpid" "SOFT breach + controller CLIMBING (+${climb}GB/2ticks)"
			else
				echo "[mem-watchdog] $(date -u +%FT%TZ) SOFT breach (free=${free_gb}GB) but controller flat (RSS=${rss}GB, climb=+${climb}GB) — EXTERNAL pressure, riding out (hard floor=${HARD_GB}GB backstop)"
			fi
		fi
		prev2="$prev1"; prev1="${rss:-0}"
	else
		prev1=0; prev2=0
	fi
	sleep 15
done
