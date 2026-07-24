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
CTRL_MIN_RSS="${9:-4}"   # controller RSS (GB) below which it CANNOT be the cause of
                         # EXTERNAL pressure — killing it frees ~nothing, so ride out
                         # instead of PAUSE/KILL (23/07/2026 fix: a 0.2GB controller was
                         # sacrificed to relieve a multi-GB IDS spike, freeing nothing and
                         # abandoning the study cell). Only the HARD survival floor
                         # (avail<HARD) ignores this — there the controller is the sole lever.

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

# The controller ($cpid) is the PYTHON; its parent is the driver's /usr/bin/time
# wrapper. Killing only the python makes /usr/bin/time exit rc=1, which the driver
# does NOT treat as a watchdog kill (it retries on 137/143) — so the run is
# ABANDONED. Both kill/pause paths must therefore SIGNAL the wrapper too so the
# driver sees 137 (SIGKILL) / 143 (SIGTERM) and its calm-gated retry fires. The
# wrapper pid MUST be captured BEFORE the python is killed: once SIGKILL lands,
# /usr/bin/time reaps the python and its ppid becomes unreachable (a post-kill
# lookup returns empty → wrapper unsignalled → rc=1 → abandonment). That was the
# 23/07 kill_ctrl bug (it looked the wrapper up AFTER the kill); pause_ctrl already
# captures first. Guarded to /usr/bin/time so we never signal the driver/shell if
# the launch path lacks the wrapper.
kill_ctrl() {  # $1 = cpid, $2 = reason. SIGKILL python+wrapper (→driver rc=137→retry) + abort chain.
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	# Capture the wrapper NOW, before the kill makes the python's ppid unreachable.
	local wrap; wrap=$(ps -o ppid= -p "$1" 2>/dev/null | tr -d ' ')
	case "$(ps -o command= -p "$wrap" 2>/dev/null)" in */usr/bin/time*) ;; *) wrap="" ;; esac
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGKILL controller $1 + /usr/bin/time wrapper (→rc=137 retry) (RSS=${rss}GB, avail=$(avail_gb)GB, comp=$(comp_gb)GB, swap=$(swap_mb)MB) + chain"
	kill -9 "$1" 2>/dev/null
	[ -n "$wrap" ] && kill -9 "$wrap" 2>/dev/null
	pkill -9 -f "$CHAIN_PAT" 2>/dev/null
	echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
	sleep 90
}

pause_ctrl() {  # $1 = cpid, $2 = reason. SIGTERM graceful dump; chain resumes later.
	local rss; rss=$(ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f",$1/1048576}')
	# Capture the /usr/bin/time wrapper NOW (its pid is unreachable once the python dies,
	# and every escalation below must signal it so the driver sees 137/143 and retries).
	local wrap; wrap=$(ps -o ppid= -p "$1" 2>/dev/null | tr -d ' ')
	case "$(ps -o command= -p "$wrap" 2>/dev/null)" in */usr/bin/time*) ;; *) wrap="" ;; esac
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGTERM graceful PAUSE controller $1 (RSS=${rss}GB, avail=$(avail_gb)GB); chain resumes from emergency dump when memory recovers"
	kill -TERM "$1" 2>/dev/null
	# The dump lands at the next GA generation boundary — keep waiting while the box
	# is SAFE. Escalate to SIGKILL only on REAL pressure during the dump (avail below
	# hard floor OR active swap thrash), never on strict-free noise. Hard cap 300s.
	local i cap=300 sw0; sw0=$(swap_mb)
	for i in $(seq 1 "$cap"); do
		kill -0 "$1" 2>/dev/null || { echo "[mem-watchdog] $(date -u +%FT%TZ) paused+dumped cleanly (${i}s); chain holds for resume"; [ -n "$wrap" ] && kill -TERM "$wrap" 2>/dev/null; return 0; }
		local sw; sw=$(swap_mb)
		if lt "$(avail_gb)" "$HARD_AVAIL" || [ "$(( ${sw:-0} - ${sw0:-0} ))" -gt "$SWAP_GROW_MB" ]; then
			echo "[mem-watchdog] $(date -u +%FT%TZ) REAL pressure during dump (avail=$(avail_gb)GB, swapΔ=$(( ${sw:-0}-${sw0:-0} ))MB) — escalating to SIGKILL (+wrapper →rc=137 retry)"
			kill -9 "$1" 2>/dev/null; [ -n "$wrap" ] && kill -9 "$wrap" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
		fi
		sleep 1
	done
	echo "[mem-watchdog] $(date -u +%FT%TZ) graceful pause WEDGED (${cap}s, box stayed safe) — SIGKILL (+wrapper →rc=137 retry) + abort chain"
	kill -9 "$1" 2>/dev/null; [ -n "$wrap" ] && kill -9 "$wrap" 2>/dev/null; pkill -9 -f "$CHAIN_PAT" 2>/dev/null; sleep 90; return 1
}

echo "[mem-watchdog] v5 armed: metric=AVAILABLE (free+purgeable+spec+inactive). HARD=${HARD_AVAIL}GB SOFT=${SOFT_AVAIL}GB | pressure=swapΔ>${SWAP_GROW_MB}MB or compΔ>${COMP_GROW_GB}GB | runaway RSS>${HOG_GB}GB/climb>${CLIMB_GB}GB | external PAUSE/KILL only when ctrl RSS>${CTRL_MIN_RSS}GB (else ride out) | kills signal the /usr/bin/time wrapper too →driver rc=137 retry"
prev1=0; prev2=0; ext_ticks=0; thrash_ticks=0; ctrl_ticks=0; prev_comp=$(comp_gb); prev_swap=$(swap_mb)
while true; do
	avail=$(avail_gb); comp=$(comp_gb); swap=$(swap_mb)
	comp_d=$(echo "${comp:-0} - ${prev_comp:-0}" | bc 2>/dev/null)
	swap_d=$(( ${swap:-0} - ${prev_swap:-0} ))
	# Real-pressure flag: kernel actively compressing/swapping this tick.
	pressure=0
	{ gt "${comp_d:-0}" "$COMP_GROW_GB" || [ "${swap_d:-0}" -gt "$SWAP_GROW_MB" ]; } && pressure=1
	# THRASH: sustained active swap growth. The AVAILABLE metric is BLIND to this —
	# it read 12.9GB (healthy) while swap exploded 1.3→47.7GB and the box stalled 12min
	# (15/07). So swap thrash must act regardless of avail. Require 2 consecutive ticks
	# (~30s) so routine one-off swapping doesn't trip it.
	if [ "${swap_d:-0}" -gt "$SWAP_GROW_MB" ]; then thrash_ticks=$((thrash_ticks + 1)); else thrash_ticks=0; fi
	# PYTHON pid ONLY (fixed 23/07/2026). Bare pgrep -f also matches the driver's
	# /usr/bin/time wrapper (lower pid ⇒ head -1 picked IT): every kill/pause hit
	# the wrapper, orphaning the python at PPID=1 still holding the memory (the
	# 23/07 double-run incidents), and every RSS read here was the wrapper's
	# ~0.2GB — so HOG/CLIMB runaway detection was blind. Killing the python makes
	# /usr/bin/time exit by itself, the driver sees rc=137, and its calm-gated
	# retry works with no orphan. ([w] bracket keeps awk from matching itself.)
	# tolower($2)~/python/ requires the executable ($2) to BE a python interpreter,
	# so a third-party process merely carrying the literal string in its argv (a
	# user/agent `grep wnn.control.phased_ga`, a `tail -f …phased_ga….out`) with a
	# lower pid cannot be mis-selected over the real controller (23/07 Dispute B).
	cpid=$(ps -axo pid=,command= | awk '$2 !~ /\/usr\/bin\/time/ && tolower($2) ~ /python/ && /[w]nn\.control\.phased_ga/ {print $1; exit}')
	if [ -n "$cpid" ]; then
		ctrl_ticks=$((ctrl_ticks + 1))
		rss=$(ps -o rss= -p "$cpid" 2>/dev/null | awk '{printf "%.2f",$1/1048576}')
		climb=$(echo "${rss:-0} - ${prev2:-0}" | bc 2>/dev/null)
		# HARD floor (avail<HARD) = survival, unconditional (controller is the sole lever).
		# The pressure+SOFT sub-condition additionally requires the controller to be big
		# enough that killing it actually relieves the deficit — a tiny controller under
		# EXTERNAL (IDS) pressure falls through to the ride-out branch below.
		if lt "${avail:-99}" "$HARD_AVAIL" || { [ "$pressure" = "1" ] && lt "${avail:-99}" "$SOFT_AVAIL" && gt "${rss:-0}" "$CTRL_MIN_RSS"; }; then
			kill_ctrl "$cpid" "REAL exhaustion (avail<${HARD_AVAIL}GB or active thrash w/ ctrl RSS=${rss}GB>${CTRL_MIN_RSS}: compΔ=${comp_d}GB swapΔ=${swap_d}MB)"; ext_ticks=0
		elif [ "${thrash_ticks:-0}" -ge 2 ]; then
			# Box overcommitted → actively swapping for ≥2 ticks, avail-blind. Only act on
			# the controller when it is big enough to BE the cause (RSS>CTRL_MIN_RSS): a
			# tiny controller does not drive the swap storm (the IDS worker does), so
			# pausing it frees nothing and just abandons the study cell — ride out instead,
			# mirroring the SOFT external-pressure branch (23/07 R5; the HARD floor above
			# still fires if avail actually collapses).
			if gt "${rss:-0}" "$CTRL_MIN_RSS"; then
				# The controller is the only lever (never touch the IDS worker), and it's
				# thrashing/stalled anyway. Graceful PAUSE (pause_ctrl escalates to SIGKILL if
				# the dump can't write while swapping). Chain resumes when the IDS flow lightens.
				pause_ctrl "$cpid" "sustained SWAP THRASH (swapΔ=${swap_d}MB × ${thrash_ticks} ticks, avail=${avail}GB blind, ctrl RSS=${rss}GB>${CTRL_MIN_RSS})"
				thrash_ticks=0; ext_ticks=0
			else
				echo "[mem-watchdog] $(date -u +%FT%TZ) SWAP THRASH (swapΔ=${swap_d}MB × ${thrash_ticks} ticks) but ctrl tiny (RSS=${rss}GB<=${CTRL_MIN_RSS}) — external, riding out"
			fi
		# HOG/CLIMB are RUNAWAY backstops — but a big or fast-growing controller is only
		# a problem when it's actually eating the box. Gate them on available ALSO being
		# low (< SOFT): while available is healthy, a controller may legitimately allocate
		# fast/large into the room the box has (e.g. TERNARY's Neurons re-eval hit RSS
		# 31GB with 26GB still available + flat compressor — perfectly safe, yet the
		# ungated CLIMB rate-guard false-killed it, 15/07). A true runaway drives available
		# down, so this still catches it — just via the metric that means something.
		elif lt "${avail:-99}" "$SOFT_AVAIL" && gt "${rss:-0}" "$HOG_GB"; then
			kill_ctrl "$cpid" "controller RUNAWAY (RSS=${rss}GB>=${HOG_GB}, avail=${avail}GB)"; ext_ticks=0
		elif [ "$ctrl_ticks" -ge 3 ] && lt "${avail:-99}" "$SOFT_AVAIL" && gt "${climb:-0}" "$CLIMB_GB"; then
			# climb = rss - prev2 (2-tick delta) is only meaningful once prev2 holds a
			# real reading — i.e. after the controller has been observed for ≥3 ticks.
			# Before that prev2 is the 0 sentinel and climb == full RSS, which
			# false-killed a legit large controller as "CLIMBING" on its first tick
			# (QA#6). HOG (absolute) + the HARD floor still cover a true early runaway.
			kill_ctrl "$cpid" "controller CLIMBING (+${climb}GB/2ticks, avail=${avail}GB)"; ext_ticks=0
		elif lt "${avail:-99}" "$SOFT_AVAIL"; then
			# Low available but no active thrash → external pressure. Ride out unless sustained/deep
			# AND the controller is big enough to matter — a tiny (< CTRL_MIN_RSS) controller under
			# external pressure is NOT the cause, so pausing/killing it frees nothing and just
			# abandons the run. Keep riding out indefinitely in that case; the HARD survival floor
			# above still fires if avail actually collapses.
			ext_ticks=$((ext_ticks + 1))
			if gt "${rss:-0}" "$CTRL_MIN_RSS" && { lt "${avail:-99}" "$PAUSE_DEEP" || [ "$ext_ticks" -ge "$PAUSE_TICKS" ]; }; then
				pause_ctrl "$cpid" "sustained/deep EXTERNAL pressure w/ ctrl RSS=${rss}GB>${CTRL_MIN_RSS} (avail=${avail}GB, ext_ticks=${ext_ticks}, no thrash)"
				ext_ticks=0
			else
				echo "[mem-watchdog] $(date -u +%FT%TZ) SOFT (avail=${avail}GB) ctrl flat (RSS=${rss}GB<=${CTRL_MIN_RSS}) comp=${comp}GB swap=${swap}MB — external, riding out (${ext_ticks})"
			fi
		else
			ext_ticks=0
		fi
		prev2="$prev1"; prev1="${rss:-0}"
	else
		prev1=0; prev2=0; ext_ticks=0; ctrl_ticks=0
	fi
	prev_comp="$comp"; prev_swap="$swap"
	sleep 15
done
