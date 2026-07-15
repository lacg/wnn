#!/bin/bash
# Controller memory watchdog v4 — CORRECT pressure metric (14/07/2026 PM).
#
# v1-v3 keyed on `vm_stat` "Pages free" (STRICT free). That is the WRONG gauge: macOS
# parks reclaimable file cache (here ~27GB, from the IDS worker re-reading 1.8M-row
# datasets) as non-free, so strict-free read 2-4GB while the box had ~40GB effectively
# available and the compressor sat at ~1GB (zero real pressure). Result: six controller
# kills that were pure false alarms. See the 14/07 memory diagnostic.
#
# v4 gauges REAL pressure, three signals:
#   AVAIL   = free + purgeable + speculative + inactive  (reclaimable headroom, GB).
#             Normal coexistence sits ~30-40GB; a genuine broad exhaustion drops it.
#   COMPRESSOR / SWAP GROWTH = the kernel actively fighting for memory. A flat
#             compressor + zero swap growth == no pressure, no matter how low strict
#             free looks. Rising compressor or swapouts == the real jetsam precursor.
#   Controller RSS HOG / CLIMB = a controller runaway (the real 08:xx/13:xx incident
#             was RSS 33-37GB); kill it directly regardless of the box-wide numbers.
#
# Responses (unchanged from v3): ride out transient external pressure; GRACEFUL
# SIGTERM-PAUSE (dump+resume, no chain kill) on sustained/deep external pressure;
# SIGKILL only for survival (avail floor / active thrash) or a controller runaway.
# The IDS worker is NEVER touched.
#
# Usage: controller_mem_watchdog.sh [hard_avail] [soft_avail] [hog_rss] [climb] [pause_deep] [pause_ticks] [swap_grow_mb] [comp_grow_gb]
HARD_AVAIL="${1:-6}"     # available GB below this: SIGKILL (survival)
SOFT_AVAIL="${2:-10}"    # available GB below this: engage attribution
HOG_GB="${3:-28}"        # controller RSS at/above this = runaway → SIGKILL
CLIMB_GB="${4:-6}"       # controller RSS rise over last 2 ticks (~30s) = runaway → SIGKILL
PAUSE_DEEP="${5:-7.5}"   # available GB below this + external = graceful PAUSE immediately
PAUSE_TICKS="${6:-3}"    # consecutive external soft breaches = sustained → graceful PAUSE
SWAP_GROW_MB="${7:-200}" # swap-used growth per tick (MB) = active thrash → real pressure
COMP_GROW_GB="${8:-0.8}" # compressor growth per tick (GB) = real pressure

CHAIN_PAT="run_gran_5arm_capped|rerun_gran_all3_capped|rerun_gran_ternary_binary|granularity_ablation_chain|rerun_teacher_fulls_fixed|run_lqr_mpc_phased|task5_ensemble_hybrids_chain"

gt() { [ "$(echo "$1 > $2" | bc 2>/dev/null)" = "1" ]; }
lt() { [ "$(echo "$1 < $2" | bc 2>/dev/null)" = "1" ]; }
# AVAILABLE = reclaimable headroom (free + purgeable + speculative + inactive).
avail_gb() { vm_stat 2>/dev/null | awk '
	/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3}
	END{printf "%.2f",(f+i+s+p)*16384/1073741824}'; }
comp_gb()  { vm_stat 2>/dev/null | awk '/occupied by compressor/{printf "%.2f",$5*16384/1073741824}'; }
swap_mb()  { sysctl -n vm.swapusage 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="used"){gsub(/[^0-9.]/,"",$(i+2));printf "%.0f",$(i+2)}}'; }
free_gb()  { vm_stat 2>/dev/null | awk '/Pages free/{printf "%.2f",$3*16384/1073741824}'; }  # logging only

kill_ctrl() {  # $1 = cpid, $2 = reason. SIGKILL + abort chain (survival / runaway).
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGKILL controller $1 (RSS=${rss}GB, avail=$(avail_gb)GB, comp=$(comp_gb)GB, swap=$(swap_mb)MB) + chain"
	kill -9 "$1" 2>/dev/null
	pkill -9 -f "$CHAIN_PAT" 2>/dev/null
	echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
	sleep 90
}

pause_ctrl() {  # $1 = cpid, $2 = reason. SIGTERM graceful dump; chain resumes later.
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGTERM graceful PAUSE controller $1 (RSS=${rss}GB, avail=$(avail_gb)GB); chain resumes from emergency dump when memory recovers"
	kill -TERM "$1" 2>/dev/null
	# The dump lands at the next GA generation boundary — keep waiting while the box
	# is SAFE. Escalate to SIGKILL only on REAL pressure during the dump (avail below
	# hard floor OR active swap thrash), never on strict-free noise. Hard cap 300s.
	local i cap=300 sw0; sw0=$(swap_mb)
	for i in $(seq 1 "$cap"); do
		kill -0 "$1" 2>/dev/null || { echo "[mem-watchdog] $(date -u +%FT%TZ) paused+dumped cleanly (${i}s); chain holds for resume"; return 0; }
		local sw; sw=$(swap_mb)
		if lt "$(avail_gb)" "$HARD_AVAIL" || [ "$(( ${sw:-0} - ${sw0:-0} ))" -gt "$SWAP_GROW_MB" ]; then
			echo "[mem-watchdog] $(date -u +%FT%TZ) REAL pressure during dump (avail=$(avail_gb)GB, swapΔ=$(( ${sw:-0}-${sw0:-0} ))MB) — escalating to SIGKILL"
			kill -9 "$1" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
		fi
		sleep 1
	done
	echo "[mem-watchdog] $(date -u +%FT%TZ) graceful pause WEDGED (${cap}s, box stayed safe) — SIGKILL + abort chain"
	kill -9 "$1" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
}

echo "[mem-watchdog] v4 armed: metric=AVAILABLE (free+purgeable+spec+inactive). HARD=${HARD_AVAIL}GB SOFT=${SOFT_AVAIL}GB | pressure=swapΔ>${SWAP_GROW_MB}MB or compΔ>${COMP_GROW_GB}GB | runaway RSS>${HOG_GB}GB/climb>${CLIMB_GB}GB | graceful-PAUSE on sustained/deep external"
prev1=0; prev2=0; ext_ticks=0; prev_comp=$(comp_gb); prev_swap=$(swap_mb)
while true; do
	avail=$(avail_gb); comp=$(comp_gb); swap=$(swap_mb)
	comp_d=$(echo "${comp:-0} - ${prev_comp:-0}" | bc 2>/dev/null)
	swap_d=$(( ${swap:-0} - ${prev_swap:-0} ))
	# Real-pressure flag: kernel actively compressing/swapping this tick.
	pressure=0
	{ gt "${comp_d:-0}" "$COMP_GROW_GB" || [ "${swap_d:-0}" -gt "$SWAP_GROW_MB" ]; } && pressure=1
	cpid=$(pgrep -f "wnn.control.phased_ga" | head -1)
	if [ -n "$cpid" ]; then
		rss=$(ps -o rss= -p "$cpid" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
		climb=$(echo "${rss:-0} - ${prev2:-0}" | bc 2>/dev/null)
		if lt "${avail:-99}" "$HARD_AVAIL" || { [ "$pressure" = "1" ] && lt "${avail:-99}" "$SOFT_AVAIL"; }; then
			kill_ctrl "$cpid" "REAL exhaustion (avail<${HARD_AVAIL}GB or active thrash: compΔ=${comp_d}GB swapΔ=${swap_d}MB)"; ext_ticks=0
		# HOG/CLIMB are RUNAWAY backstops — but a big or fast-growing controller is only
		# a problem when it's actually eating the box. Gate them on available ALSO being
		# low (< SOFT): while available is healthy, a controller may legitimately allocate
		# fast/large into the room the box has (e.g. TERNARY's Neurons re-eval hit RSS
		# 31GB with 26GB still available + flat compressor — perfectly safe, yet the
		# ungated CLIMB rate-guard false-killed it, 15/07). A true runaway drives available
		# down, so this still catches it — just via the metric that means something.
		elif lt "${avail:-99}" "$SOFT_AVAIL" && gt "${rss:-0}" "$HOG_GB"; then
			kill_ctrl "$cpid" "controller RUNAWAY (RSS=${rss}GB>=${HOG_GB}, avail=${avail}GB)"; ext_ticks=0
		elif lt "${avail:-99}" "$SOFT_AVAIL" && gt "${climb:-0}" "$CLIMB_GB"; then
			kill_ctrl "$cpid" "controller CLIMBING (+${climb}GB/2ticks, avail=${avail}GB)"; ext_ticks=0
		elif lt "${avail:-99}" "$SOFT_AVAIL"; then
			# Low available but no active thrash → external pressure. Ride out unless sustained/deep.
			ext_ticks=$((ext_ticks + 1))
			if lt "${avail:-99}" "$PAUSE_DEEP" || [ "$ext_ticks" -ge "$PAUSE_TICKS" ]; then
				pause_ctrl "$cpid" "sustained/deep EXTERNAL pressure (avail=${avail}GB, ext_ticks=${ext_ticks}, ctrl flat RSS=${rss}GB, no thrash)"
				ext_ticks=0
			else
				echo "[mem-watchdog] $(date -u +%FT%TZ) SOFT (avail=${avail}GB) ctrl flat (RSS=${rss}GB) comp=${comp}GB swap=${swap}MB — external, riding out (${ext_ticks}/${PAUSE_TICKS})"
			fi
		else
			ext_ticks=0
		fi
		prev2="$prev1"; prev1="${rss:-0}"
	else
		prev1=0; prev2=0; ext_ticks=0
	fi
	prev_comp="$comp"; prev_swap="$swap"
	sleep 15
done
