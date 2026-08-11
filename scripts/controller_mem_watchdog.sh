#!/bin/bash
# Controller memory watchdog v6 — EFFICACY-GATED kills (31/07/2026).
#
# v6 fixes the last unguarded kill path. v5 applied the CTRL_MIN_RSS efficacy guard
# ("don't kill what can't help") to the thrash branch, the soft-external branch and
# the pressure sub-clause — but deliberately EXEMPTED the HARD survival floor, on the
# reasoning "there the controller is the sole lever". That reasoning is wrong: being
# the only lever you have does not make it a lever that moves anything.
#
# MEASURED on 31/07 (vmmap --summary, not ps RSS — physical footprint INCLUDES
# compressed pages, so this is not a hidden-elephant illusion):
#     controller 73044   physical footprint 194MB (peak 299MB)
#     IDS worker child   physical footprint 2.1GB (peak 7.1GB)
# and the log audit of every HARD-floor kill this watchdog has ever performed:
#     8 × "REAL exhaustion" SIGKILL, controller RSS = 0.0 0.0 0.0 0.3 0.2 0.1 0.2 GB
# 8 of 8 freed nothing. Two on 30/07 cost the dfa1l study 10h37 of a single cell
# (8h26 + 2h11), both landing on an IDS flow transition (4732→4733→4734) where the
# worker's encode spike drove avail under the floor. Killing 0.2GB neither restores
# the floor nor protects the IDS worker from jetsam — the sacrifice buys literally
# nothing, so v6 refuses to make it.
#
# v6 changes (policy chosen by the user 31/07):
#   1. HARD floor gated on CTRL_MIN_RSS, same as every other branch.
#   2. Sub-threshold controller is NEVER SIGKILLed for EXTERNAL pressure at any
#      avail level; it emits a CRITICAL alarm and rides out. macOS decides if it
#      truly comes to jetsam. (Controller RUNAWAY via HOG/CLIMB is untouched — a
#      runaway is by definition above the threshold and IS the cause.)
#   3. Futility circuit breaker: after any kill, if avail is STILL below the floor
#      once memory settles, the kill did not fix the condition. Two such in a row
#      suppress the HARD branch for FUTILE_COOLDOWN so the watchdog stops repeating
#      an action that demonstrably does not work.
#   4. Kill/alarm reason strings name the clause that ACTUALLY fired. v5 printed
#      "avail<6GB or active thrash w/ ctrl RSS=0.15GB>4" — an OR-label asserting a
#      condition that was false, which read as though the RSS test had passed.
#
# NOTE: graceful PAUSE is not a cheaper middle ground for the dfa1l study —
# run_dfa_1layer_study.sh:103 treats rc=143 and rc=137 identically (no marker, full
# restart from STAGE 0). Only NOT acting preserves the cell.
#
# ---------------------------------------------------------------------------
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
# Usage: controller_mem_watchdog.sh [hard_avail] [soft_avail] [hog_rss] [climb] [pause_deep]
#                                   [pause_ticks] [swap_grow_mb] [comp_grow_gb] [ctrl_min_rss]
#                                   [never_kill_avail] [futile_cooldown]
# ALL ELEVEN slots are listed above and each is claimed EXACTLY ONCE. Keep it that
# way: 11/08/2026 found $10 claimed by BOTH never_kill_avail and futile_cooldown
# (see the note on FUTILE_COOLDOWN below for why that one was dangerous).
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
                         # abandoning the study cell). v6: this now gates the HARD survival
                         # floor TOO. The floor's old exemption ("sole lever") produced 8/8
                         # futile kills — see the v6 header. A controller big enough to
                         # restore the floor still gets killed there; a 0.2GB one does not.
NEVER_KILL_AVAIL="${10:-10}"  # available GB at/above which NO kill may EVER fire,
                         # whatever any branch decided. See kill_ctrl.
FUTILE_COOLDOWN="${11:-3600}"  # seconds to suppress the HARD branch after 2 consecutive
                         # ($11 since 11/08/2026 — it previously ALSO read $10, so an
                         # 11-arg launch silently set the ABSOLUTE-REFUSAL threshold to
                         # the cooldown's value: never_kill_avail=3600 would mean kills
                         # refuse to fire below 3600GB avail, i.e. the watchdog goes
                         # permanently alarm-only. Defaults were unaffected.)
                         # kills that failed to lift avail back over the floor (a kill that
                         # does not fix the condition is evidence the controller was not the
                         # cause; repeating it just burns cells).

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
	local a0; a0=$(avail_gb)
	# ABSOLUTE REFUSAL (04/08/2026, Luiz). No kill may fire while the box has room,
	# whatever branch decided to call this. Every branch is ALREADY avail-gated —
	# HARD<6, thrash<SOFT, and HOG/CLIMB were gated on <SOFT after the 15/07 false
	# kill of a legitimate 31GB allocation with 26GB still available — so today this
	# is unreachable. It exists precisely because that safety is EMERGENT from four
	# separate conditionals: one future edit to any of them reopens the hole, and
	# nothing would notice. A run is worth hours; the check costs one comparison.
	if ! lt "${a0:-0}" "$NEVER_KILL_AVAIL"; then
		echo "[mem-watchdog] $(date -u +%FT%TZ) KILL REFUSED — $2 but avail=${a0}GB >= ${NEVER_KILL_AVAIL}GB. The box has room; a kill here would destroy a run for nothing. Riding out."
		return 0
	fi
	echo "[mem-watchdog] $(date -u +%FT%TZ) $2 — SIGKILL controller $1 + /usr/bin/time wrapper (→rc=137 retry) (RSS=${rss}GB, avail=${a0}GB, comp=$(comp_gb)GB, swap=$(swap_mb)MB) + chain"
	kill -9 "$1" 2>/dev/null
	[ -n "$wrap" ] && kill -9 "$wrap" 2>/dev/null
	pkill -9 -f "$CHAIN_PAT" 2>/dev/null
	echo "[mem-watchdog] killed; sleeping 90s for memory to settle"
	sleep 90
	# FUTILITY CHECK (v6): once memory has settled, did the kill actually fix the
	# condition it was invoked for? If avail is STILL under the floor, the controller
	# was not the cause — we spent a study cell for nothing. Two in a row and we stop
	# repeating it. (Not a subshell: these assignments update the loop's globals.)
	local a1; a1=$(avail_gb)
	if lt "${a1:-99}" "$HARD_AVAIL"; then
		futile_streak=$((futile_streak + 1))
		echo "[mem-watchdog] $(date -u +%FT%TZ) FUTILE KILL #${futile_streak}: freed ${rss}GB but avail ${a0}GB→${a1}GB is still under ${HARD_AVAIL}GB — the controller was NOT the cause"
		if [ "$futile_streak" -ge 2 ]; then
			suppress_until=$(( $(date +%s) + FUTILE_COOLDOWN ))
			echo "[mem-watchdog] $(date -u +%FT%TZ) CIRCUIT BREAKER: 2 consecutive futile kills — HARD-floor kills SUPPRESSED for ${FUTILE_COOLDOWN}s (alarm-only). Runaway HOG/CLIMB detection stays armed."
		fi
	else
		[ "${futile_streak:-0}" -gt 0 ] && echo "[mem-watchdog] $(date -u +%FT%TZ) kill was EFFECTIVE (avail ${a0}GB→${a1}GB, over ${HARD_AVAIL}GB) — futile streak reset"
		futile_streak=0
	fi
}

# survival_action — the v6 EFFICACY GATE, extracted as a pure function so it can be
# proven against the historical incidents without duplicating the predicate in a test
# (see WNN_WATCHDOG_SELFTEST below). Reads only its args + the threshold globals.
# Args:  $1=avail $2=ctrl_rss $3=pressure(0|1) $4=now_epoch $5=compΔ $6=swapΔ
# Echoes: "<ACTION> <human reason>" where ACTION ∈
#   KILL             — condition tripped AND freeing this controller can plausibly fix it
#   ALARM_FUTILE     — condition tripped but ctrl RSS<=CTRL_MIN_RSS: killing frees nothing
#   ALARM_SUPPRESSED — condition tripped, ctrl big enough, but the circuit breaker is open
#   NONE             — no survival condition tripped
survival_action() {
	local avail="$1" rss="$2" pressure="$3" now_s="$4" comp_d="$5" swap_d="$6" why
	if lt "$avail" "$HARD_AVAIL"; then
		why="HARD floor (avail=${avail}GB<${HARD_AVAIL}GB)"
	elif [ "$pressure" = "1" ] && lt "$avail" "$SOFT_AVAIL"; then
		why="active thrash (compΔ=${comp_d}GB swapΔ=${swap_d}MB, avail=${avail}GB<${SOFT_AVAIL}GB)"
	else
		echo "NONE -"; return
	fi
	# THE v6 FIX: no kill unless it could work. Applies to the HARD floor too.
	gt "$rss" "$CTRL_MIN_RSS" || { echo "ALARM_FUTILE $why"; return; }
	[ "$now_s" -lt "${suppress_until:-0}" ] && { echo "ALARM_SUPPRESSED $why"; return; }
	echo "KILL $why"
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

# SELF-TEST — `WNN_WATCHDOG_SELFTEST=1 bash scripts/controller_mem_watchdog.sh`
# Exercises the REAL survival_action() (not a copy) against every historical incident
# this watchdog has acted on, so a future edit that resurrects a futile kill fails here
# instead of on a 12-hour study cell. Exits non-zero on any mismatch; never arms the loop.
if [ "${WNN_WATCHDOG_SELFTEST:-0}" = "1" ]; then
	fails=0
	check() {  # $1=expect $2=label $3..$8 = survival_action args
		local expect="$1" label="$2"; shift 2
		local got; got=$(survival_action "$@" | awk '{print $1}')
		if [ "$got" = "$expect" ]; then printf '  ok   %-46s → %s\n' "$label" "$got"
		else printf '  FAIL %-46s → %s (expected %s)\n' "$label" "$got" "$expect"; fails=$((fails+1)); fi
	}
	echo "[selftest] v6 survival_action — HARD=${HARD_AVAIL} SOFT=${SOFT_AVAIL} CTRL_MIN_RSS=${CTRL_MIN_RSS}"
	echo "[selftest] the three 30/07 futile kills that cost the dfa1l study 10h37 —"
	check ALARM_FUTILE "30/07 12:38 avail=5.85 rss=0.15"  5.85 0.15 0 0 .73  0
	check ALARM_FUTILE "30/07 21:22 avail=5.64 rss=0.12"  5.64 0.12 1 0 1.11 166
	check ALARM_FUTILE "30/07 23:51 avail=5.65 rss=0.17"  5.65 0.17 0 0 2.01 0
	echo "[selftest] the five earlier futile kills (v4 era, rss 0.0-0.3) —"
	check ALARM_FUTILE "23/07 06:42 avail=7.67 rss=0.0 (thrash)" 7.67 0.0 1 0 2.36 0
	check ALARM_FUTILE "23/07 13:29 avail=8.00 rss=0.0 (thrash)" 8.00 0.0 1 0 1.82 0
	check ALARM_FUTILE "23/07 18:09 avail=7.92 rss=0.3 (thrash)" 7.92 0.3 1 0 1.66 0
	echo "[selftest] kills that REMAIN armed (controller big enough to fix it) —"
	check KILL "floor + big ctrl  avail=5.5 rss=8.0"      5.5  8.0  0 0 1.0  0
	check KILL "floor + huge ctrl avail=4.0 rss=31.0"     4.0  31.0 0 0 2.0  0
	check KILL "thrash + big ctrl avail=9.0 rss=10.6"     9.0  10.6 1 0 0.9  618
	check KILL "boundary rss just over min (4.01)"        5.9  4.01 0 0 1.0  0
	echo "[selftest] no-action cases —"
	check NONE "healthy box          avail=30 rss=0.5"    30.0 0.5  0 0 0.1  0
	check NONE "healthy box, big ctrl avail=26 rss=31"    26.0 31.0 0 0 0.1  0
	check NONE "soft dip, no pressure avail=8 rss=0.2"    8.0  0.2  0 0 0.1  0
	check NONE "boundary rss at min (4.00) but no trip"   30.0 4.00 0 0 0.1  0
	echo "[selftest] circuit breaker (2 futile kills ⇒ HARD suppressed) —"
	suppress_until=$(( $(date +%s) + 3600 ))
	check ALARM_SUPPRESSED "breaker open, big ctrl avail=5.5 rss=8" 5.5 8.0 0 0 1.0 0
	suppress_until=0
	check KILL "breaker closed again, same inputs"        5.5  8.0  0 0 1.0  0
	if [ "$fails" = "0" ]; then echo "[selftest] ALL PASS — no futile kill survives v6"; exit 0
	else echo "[selftest] $fails FAILURE(S)"; exit 1; fi
fi

echo "[mem-watchdog] v6 armed: metric=AVAILABLE (free+purgeable+spec+inactive). HARD=${HARD_AVAIL}GB SOFT=${SOFT_AVAIL}GB | pressure=swapΔ>${SWAP_GROW_MB}MB or compΔ>${COMP_GROW_GB}GB | runaway RSS>${HOG_GB}GB/climb>${CLIMB_GB}GB | ALL external kill/pause paths (HARD floor INCLUDED, v6) require ctrl RSS>${CTRL_MIN_RSS}GB — else alarm+ride out | NEVER kill while avail>=${NEVER_KILL_AVAIL}GB (absolute) | futility breaker: 2 ineffective kills ⇒ HARD suppressed ${FUTILE_COOLDOWN}s | kills signal the /usr/bin/time wrapper too →driver rc=137 retry"
prev1=0; prev2=0; ext_ticks=0; thrash_ticks=0; ctrl_ticks=0; prev_comp=$(comp_gb); prev_swap=$(swap_mb)
futile_streak=0; suppress_until=0
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
		# HARD floor (avail<HARD) = survival. v6: EFFICACY-GATED like every other branch.
		# A kill is only justified when freeing this controller could plausibly restore
		# the floor. 8/8 historical HARD-floor kills fired at RSS<=0.3GB and freed nothing
		# (see header) — that is not survival, it is ritual. Below the threshold we alarm
		# and ride out; macOS decides if it genuinely comes to jetsam. The runaway
		# backstops (HOG/CLIMB) below are unaffected: a runaway is by definition large.
		now_s=$(date +%s)
		read -r act why <<<"$(survival_action "${avail:-99}" "${rss:-0}" "$pressure" "$now_s" "${comp_d:-0}" "${swap_d:-0}")"
		if [ "$act" != "NONE" ]; then
			case "$act" in
				KILL)             kill_ctrl "$cpid" "REAL exhaustion — ${why}, ctrl RSS=${rss}GB>${CTRL_MIN_RSS}GB so freeing it can restore the floor" ;;
				ALARM_FUTILE)     echo "[mem-watchdog] $(date -u +%FT%TZ) 🔴 CRITICAL ${why} — but ctrl RSS=${rss}GB<=${CTRL_MIN_RSS}GB frees nothing. NOT killing (futile); riding out. comp=${comp}GB swap=${swap}MB" ;;
				ALARM_SUPPRESSED) echo "[mem-watchdog] $(date -u +%FT%TZ) 🔴 CRITICAL ${why} w/ ctrl RSS=${rss}GB — kill SUPPRESSED by circuit breaker ($(( suppress_until - now_s ))s left); riding out" ;;
			esac
			ext_ticks=0
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
