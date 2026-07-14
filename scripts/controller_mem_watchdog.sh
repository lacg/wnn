#!/bin/bash
# Controller memory watchdog — AUTO-KILLS the controller if REAL free RAM drops below
# a hard floor, so a runaway can never thrash the box to 0 free / jetsam again
# (14/07/2026 incident — see feedback_kill_on_memory_climb).
#
# Keys on `vm_stat` "Pages free" (REAL free), NOT `memory_pressure` "free %" — macOS
# counts compressed memory as "free" there, which read 52-86% at 0 real free.
#
# Usage: controller_mem_watchdog.sh [floor_gb]   (default 5 GB)
FLOOR_GB="${1:-5}"
echo "[mem-watchdog] armed: SIGKILL controller if real free RAM < ${FLOOR_GB}GB (vm_stat)"
while true; do
	free_gb=$(vm_stat 2>/dev/null | awk '/Pages free/{printf "%.2f",$3*16384/1073741824}')
	cpid=$(pgrep -f "wnn.control.phased_ga" | head -1)
	if [ -n "$cpid" ] && [ "$(echo "${free_gb:-99} < ${FLOOR_GB}" | bc 2>/dev/null)" = "1" ]; then
		rss=$(ps -o rss= -p "$cpid" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
		echo "[mem-watchdog] $(date -u +%FT%TZ) REAL-FREE=${free_gb}GB < ${FLOOR_GB}GB — SIGKILL controller $cpid (RSS=${rss}GB) + chain"
		kill -9 "$cpid" 2>/dev/null
		# kill the driving chain too so it can't spawn the next arm into the same wall
		pkill -9 -f "rerun_gran_ternary_binary|granularity_ablation_chain|rerun_teacher_fulls_fixed|run_lqr_mpc_phased|task5_ensemble_hybrids_chain" 2>/dev/null
		echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
		sleep 90
	fi
	sleep 15
done
