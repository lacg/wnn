#!/usr/bin/env bash
# Level-triggered supervisor for the dfa1l 40-cell sweep.
#
# The companion script (dfa1l_restart_at_cell_boundary.sh) is EDGE-triggered: it
# waits for one expected transition and fires once, so it is blind to every stall
# that happens outside its window. This one is LEVEL-triggered — it periodically
# asks "is the world in the desired state?" and reconciles, which heals causes
# nobody enumerated in advance.
#
# Desired state, checked every POLL seconds:
#   * markers < 40                    (else the sweep is finished)
#   * exactly ONE driver alive
#   * at most ONE phased_ga python alive
#
# It only ever ADDS a driver when zero are running. It never performs a planned
# kill — that is the boundary watcher's job, and a supervisor that also killed
# would fight the driver it is supposed to protect.
#
# Usage: dfa1l_sweep_supervisor.sh [poll-seconds]
#
# Deliberately NOT self-healing for two cases, both of which need a human:
#   * two drivers or two cells alive  -> the double-run OOM risk; log, do nothing
#   * the IDS worker being down       -> restarting it mid-'running' CANCELS the
#                                        live flow, so that is never automatic
set -u

POLL="${1:-${SUPERVISOR_POLL:-300}}"

ROOT="${SUPERVISOR_ROOT:-/Users/lacg/wnn}"
SCRIPT="${SUPERVISOR_SCRIPT:-scripts/run_dfa_1layer_study.sh}"   # relative to ROOT
MARKDIR="${SUPERVISOR_MARKDIR:-/tmp/wnn_dfa1l}"
OUTDIR="${SUPERVISOR_OUTDIR:-$ROOT/logs/controller/dfa1l}"
DRIVER_LOG="${SUPERVISOR_DRIVER_LOG:-/private/tmp/dfa1l_driver.log}"
LOG="${SUPERVISOR_LOG:-/private/tmp/dfa1l_supervisor.log}"
STATE="${SUPERVISOR_STATE:-/private/tmp/dfa1l_supervisor_relaunches}"
TRIP="${SUPERVISOR_TRIP:-/private/tmp/dfa1l_supervisor_TRIPPED}"
IDS_WORKER="${SUPERVISOR_IDS_WORKER:-77344}"
PHASED_PAT="${SUPERVISOR_PHASED_PAT:-wnn.control.phased_ga}"
# Lock held by the boundary watcher ONLY during its kill/relaunch phase. Keying
# on the watcher PROCESS instead would be wrong: it waits for hours, so the
# supervisor would stand down for the whole cell and protect nothing.
RESTART_LOCK="${SUPERVISOR_RESTART_LOCK:-/private/tmp/dfa1l_restart.lock}"
LOCK_STALE_S="${SUPERVISOR_LOCK_STALE_S:-1800}"
TOTAL_CELLS="${SUPERVISOR_TOTAL_CELLS:-40}"
MIN_AVAIL_GB="${SUPERVISOR_MIN_AVAIL_GB:-8}"
MAX_RELAUNCH_PER_HOUR="${SUPERVISOR_MAX_RELAUNCH_PER_HOUR:-3}"
HANG_WARN_H="${SUPERVISOR_HANG_WARN_H:-6}"

log() { echo "[sweep-supervisor] $(date -u +%FT%TZ) $*" >> "$LOG"; }

marker_count() { ls -1 "$MARKDIR"/*.json 2>/dev/null | grep -v baselines.json | wc -l | tr -d ' '; }
driver_count() { ps -axo command= 2>/dev/null | grep -c "[b]ash .*$SCRIPT"; }
# Only real phased_ga PYTHONs; the /usr/bin/time wrapper is not the hog and
# counting it would make one healthy cell look like a double-run.
cell_count() {
	ps -axo pid,command 2>/dev/null | grep "$PHASED_PAT" | grep -v "/usr/bin/time" \
		| grep -v grep | grep -c python
}
avail_gb() {
	vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages speculative/{s=$3}/purgeable/{p=$3}
		END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);gsub(/\./,"",p);
		printf "%.1f",(f+i+s+p)*16384/1e9}'
}

# Crash-loop brake: an auto-relaunch turns a cell that dies in 30s into an
# overnight restart loop that burns the night and looks like progress.
relaunches_last_hour() {
	[ -f "$STATE" ] || { echo 0; return; }
	local now; now=$(date +%s)
	awk -v now="$now" '$1 > now-3600 {n++} END{print n+0}' "$STATE"
}

# Passive only — a batch eval legitimately runs ~92min, so silence is normal for
# hours. We never act on this; a wrong kill here would destroy a good cell.
hang_warn() {
	local newest age_h
	newest=$(ls -t "$OUTDIR"/*.out 2>/dev/null | head -1)
	[ -n "$newest" ] || return
	age_h=$(( ( $(date +%s) - $(stat -f %m "$newest") ) / 3600 ))
	[ "$age_h" -ge "$HANG_WARN_H" ] && \
		log "NOTE: $(basename "$newest") silent ${age_h}h (>=${HANG_WARN_H}h). Not acting — \
verify CPU before concluding anything; long batch evals are normal."
}

reconcile() {
	local markers ndriver ncell
	markers=$(marker_count)
	if [ "${markers:-0}" -ge "$TOTAL_CELLS" ]; then
		log "SWEEP COMPLETE ${markers}/${TOTAL_CELLS} — supervisor exiting"
		return 10
	fi
	# A PLANNED restart (the boundary watcher swapping stale driver code) passes
	# through a brief 0-drivers/0-cells state that looks exactly like a stall. If a
	# tick landed there we would relaunch alongside it and produce TWO drivers —
	# the very double-run this whole design exists to avoid. So a live watcher
	# suppresses reconciliation entirely; it finishes in seconds and the next tick
	# sees the healthy result.
	if [ -f "$RESTART_LOCK" ]; then
		local age; age=$(( $(date +%s) - $(stat -f %m "$RESTART_LOCK") ))
		if [ "$age" -lt "$LOCK_STALE_S" ]; then
			log "boundary restart in progress (lock ${age}s old) — standing by; its \
0-driver window is not a stall"
			return 0
		fi
		# A watcher that died mid-restart must not disable healing forever.
		log "restart lock is stale (${age}s > ${LOCK_STALE_S}s) — ignoring it and \
reconciling normally"
	fi
	ndriver=$(driver_count); ncell=$(cell_count)

	if [ "${ndriver:-0}" -gt 1 ]; then
		log "ANOMALY: ${ndriver} drivers alive — refusing to act (a second driver is \
the double-run OOM risk). Needs a human."
		return 0
	fi
	if [ "${ncell:-0}" -gt 1 ]; then
		log "ANOMALY: ${ncell} phased_ga pythons alive — refusing to act. Needs a human."
		return 0
	fi
	if [ "${ndriver:-0}" -eq 1 ]; then
		hang_warn                      # healthy: driver sequencing, nothing to do
		return 0
	fi

	# ---- ndriver == 0: the sweep is not progressing ------------------------
	if [ "${ncell:-0}" -eq 1 ]; then
		# The boundary-watcher's exit-2 shape: driver gone, cell still working.
		# Relaunching now would double-run. Let the orphan finish; the next tick
		# sees zero cells and relaunches then. Its work is lost (the driver writes
		# the marker) — R4 semantics: no marker means re-run, which is correct.
		log "driver gone but a cell is still running — waiting for the orphan to \
finish before relaunching (relaunching now would double-run)"
		return 0
	fi

	log "STALLED: 0 drivers, 0 cells, ${markers}/${TOTAL_CELLS} markers — evaluating gates"
	if ! ps -p "$IDS_WORKER" >/dev/null 2>&1; then
		log "gate: IDS worker $IDS_WORKER not alive — NOT relaunching. IDS is priority \
and restarting it mid-'running' cancels the live flow, so that stays a human call."
		return 0
	fi
	local av; av=$(avail_gb)
	if awk "BEGIN{exit !(${av:-0} < ${MIN_AVAIL_GB})}"; then
		log "gate: avail ${av}GB < ${MIN_AVAIL_GB}GB — NOT relaunching into a squeezed box; \
will retry next tick"
		return 0
	fi
	local n; n=$(relaunches_last_hour)
	if [ "${n:-0}" -ge "$MAX_RELAUNCH_PER_HOUR" ]; then
		log "BRAKE TRIPPED: ${n} relaunches in the last hour (max ${MAX_RELAUNCH_PER_HOUR}). \
Something is failing fast — stopping rather than looping all night. Supervisor exiting."
		date -u +%FT%TZ > "$TRIP"
		return 11
	fi

	cd "$ROOT" || { log "cannot cd $ROOT — exiting"; return 12; }
	nohup bash "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
	local pid=$!
	date +%s >> "$STATE"
	sleep 8
	if ps -p "$pid" -o command= 2>/dev/null | grep -q "$SCRIPT"; then
		log "RELAUNCHED driver pid=$pid (avail=${av}GB, relaunch $((n+1))/${MAX_RELAUNCH_PER_HOUR} \
this hour). It skips every cell that already has a marker."
	else
		log "relaunched driver $pid died within 8s — will retry next tick (brake counts it)"
	fi
	return 0
}

if [ -f "$TRIP" ]; then
	log "refusing to start: brake trip file $TRIP exists (from $(cat "$TRIP" 2>/dev/null)). \
Investigate, then remove it to re-arm."
	exit 11
fi
log "ARMED poll=${POLL}s markdir=$MARKDIR brake=${MAX_RELAUNCH_PER_HOUR}/h min_avail=${MIN_AVAIL_GB}GB"
while true; do
	reconcile; rc=$?
	[ "$rc" -ne 0 ] && exit "$rc"
	sleep "$POLL"
done
