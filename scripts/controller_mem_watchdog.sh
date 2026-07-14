#!/bin/bash
# Controller memory watchdog v3 — attribution-aware, with GRACEFUL AUTO-PAUSE
# (14/07/2026 PM). Three responses, escalating by how bad + who caused it:
#
#   1. RIDE OUT  — external (IDS) pressure that is transient. Do nothing; the spike
#      recovers (the 21:02 IDS +8GB/60s spike recovered in 2 min). A flat controller
#      must not be collateral for someone else's transient.
#   2. GRACEFUL PAUSE (SIGTERM) — external pressure that is SUSTAINED (>=PAUSE_TICKS
#      consecutive soft breaches ~45s) or DEEP (free<PAUSE_DEEP, nearing the hard
#      floor). SIGTERM lets phased_ga dump its stage+population+cells to an
#      emergency_stage*.pkl and exit cleanly; the CHAIN (run_gran_5arm_capped) then
#      resumes that same arm via --resume-from-emergency once RAM recovers. NO chain
#      kill here — the chain must survive to resume. Zero lost steps.
#   3. SIGKILL  — box-survival or a runaway controller. HARD floor breach (free<HARD),
#      or the controller itself is the hog (RSS>=HOG) / climbing (>=CLIMB/2ticks).
#      Kills phased_ga + the driving chain (aborts the campaign; manual restart).
#
# The IDS worker is NEVER touched (paper deadline). Usage:
#   controller_mem_watchdog.sh [hard_gb] [soft_gb] [hog_gb] [climb_gb] [pause_deep_gb] [pause_ticks]
HARD_GB="${1:-3}"        # real-free below this: SIGKILL (survival)
SOFT_GB="${2:-5}"        # real-free below this: engage attribution logic
HOG_GB="${3:-12}"        # controller RSS at/above this at soft breach = it's the hog → SIGKILL
CLIMB_GB="${4:-1.5}"     # controller RSS rise over last 2 ticks (~30s) = runaway → SIGKILL
PAUSE_DEEP_GB="${5:-3.7}" # external pressure below this = graceful PAUSE immediately
PAUSE_TICKS="${6:-3}"    # consecutive external soft breaches = sustained → graceful PAUSE

CHAIN_PAT="run_gran_5arm_capped|rerun_gran_all3_capped|rerun_gran_ternary_binary|granularity_ablation_chain|rerun_teacher_fulls_fixed|run_lqr_mpc_phased|task5_ensemble_hybrids_chain"

gt() { [ "$(echo "$1 > $2" | bc 2>/dev/null)" = "1" ]; }
lt() { [ "$(echo "$1 < $2" | bc 2>/dev/null)" = "1" ]; }
real_free() { vm_stat 2>/dev/null | awk '/Pages free/{printf "%.2f",$3*16384/1073741824}'; }

kill_ctrl() {  # $1 = cpid, $2 = reason. SIGKILL + abort chain (survival / runaway).
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGKILL controller $1 (RSS=${rss}GB, free=$(real_free)GB) + chain"
	kill -9 "$1" 2>/dev/null
	pkill -9 -f "$CHAIN_PAT" 2>/dev/null
	echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
	sleep 90
}

pause_ctrl() {  # $1 = cpid, $2 = reason. SIGTERM graceful dump; chain resumes later.
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGTERM graceful PAUSE controller $1 (RSS=${rss}GB, free=$(real_free)GB); chain will resume from emergency dump when RAM recovers"
	kill -TERM "$1" 2>/dev/null
	# The dump lands at the next GA generation boundary, which for a long stage
	# (e.g. MEMORY re-eval) can be tens of seconds — so DON'T rush it. Keep waiting
	# as long as the box is SAFE (free >= HARD); only escalate to SIGKILL if free
	# actually crashes below HARD (jetsam trumps a clean dump) or the dump is truly
	# wedged (hard cap). A recovered spike (free back up) must NOT trigger a kill —
	# that was the 40s-cap bug that lost the Memory-stage QUAD at 13GB free.
	local i cap=300
	for i in $(seq 1 "$cap"); do
		kill -0 "$1" 2>/dev/null || { echo "[mem-watchdog] $(date -u +%FT%TZ) paused+dumped cleanly (${i}s); chain holds for resume"; return 0; }
		if lt "$(real_free)" "$HARD_GB"; then
			echo "[mem-watchdog] $(date -u +%FT%TZ) free<${HARD_GB}GB during dump — escalating to SIGKILL"
			kill -9 "$1" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
		fi
		sleep 1
	done
	echo "[mem-watchdog] $(date -u +%FT%TZ) graceful pause WEDGED (${cap}s, box stayed safe) — SIGKILL + abort chain"
	kill -9 "$1" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
}

echo "[mem-watchdog] v3 armed: HARD=${HARD_GB} SOFT=${SOFT_GB} HOG=${HOG_GB} CLIMB=${CLIMB_GB} | graceful-PAUSE on external pressure (deep<${PAUSE_DEEP_GB}GB or ${PAUSE_TICKS} sustained ticks)"
prev1=0; prev2=0; ext_ticks=0
while true; do
	free_gb=$(real_free)
	cpid=$(pgrep -f "wnn.control.phased_ga" | head -1)
	if [ -n "$cpid" ]; then
		rss=$(ps -o rss= -p "$cpid" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
		climb=$(echo "${rss:-0} - ${prev2:-0}" | bc 2>/dev/null)
		if lt "${free_gb:-99}" "$HARD_GB"; then
			kill_ctrl "$cpid" "HARD floor breach (free<${HARD_GB}GB)"; ext_ticks=0
		elif lt "${free_gb:-99}" "$SOFT_GB"; then
			if gt "${rss:-0}" "$HOG_GB"; then
				kill_ctrl "$cpid" "SOFT breach + controller is HOG (RSS=${rss}GB>=${HOG_GB})"; ext_ticks=0
			elif gt "${climb:-0}" "$CLIMB_GB"; then
				kill_ctrl "$cpid" "SOFT breach + controller CLIMBING (+${climb}GB/2ticks)"; ext_ticks=0
			else
				# External (IDS) pressure, controller flat. Ride out unless sustained or deep.
				ext_ticks=$((ext_ticks + 1))
				if lt "${free_gb:-99}" "$PAUSE_DEEP_GB" || [ "$ext_ticks" -ge "$PAUSE_TICKS" ]; then
					pause_ctrl "$cpid" "sustained/deep EXTERNAL pressure (free=${free_gb}GB, ext_ticks=${ext_ticks}, ctrl flat RSS=${rss}GB)"
					ext_ticks=0
				else
					echo "[mem-watchdog] $(date -u +%FT%TZ) SOFT breach (free=${free_gb}GB) ctrl flat (RSS=${rss}GB) — external, riding out (${ext_ticks}/${PAUSE_TICKS} before pause)"
				fi
			fi
		else
			ext_ticks=0
		fi
		prev2="$prev1"; prev1="${rss:-0}"
	else
		prev1=0; prev2=0; ext_ticks=0
	fi
	sleep 15
done
