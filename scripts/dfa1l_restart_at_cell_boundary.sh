#!/usr/bin/env bash
# Restart a long-running study DRIVER at its next cell boundary, so a driver that
# is holding stale in-memory bash code picks up the committed version without
# losing an in-flight cell.
#
# Why this exists: run_dfa_1layer_study.sh is read ONCE by bash at launch. Editing
# the script does nothing to a driver already running (see the memory note
# "restart process after logic edit"). The only lossless moment to restart is the
# instant a cell ends and before the next one gets far — the driver is resumable
# via per-cell markers, so a restart then re-enters the loop and skips everything
# already done.
#
# Usage: dfa1l_restart_at_cell_boundary.sh <driver-pid> <cell-python-pid> <marker> [grace-s]
#
#   driver-pid      the `bash run_dfa_1layer_study.sh` process (PPID=1)
#   cell-python-pid the LIVE phased_ga python for the current cell (NOT the
#                   /usr/bin/time wrapper — the wrapper masks the real hog)
#   marker          the marker file the current cell writes on genuine completion
#   grace-s         max seconds to wait after the python dies before killing the
#                   driver anyway (default 300). Covers the R4 no-marker paths.
#
# Fires EXACTLY ONCE, then exits. Its children reparent to init on exit, so the
# relaunched driver ends up PPID=1 without setsid (which macOS does not ship).
set -u

DRIVER_PID="${1:?driver pid required}"
CELL_PID="${2:?cell python pid required}"
MARKER="${3:?marker path required}"
GRACE="${4:-300}"

# Overridable so the kill/relaunch path can be proof-tested against fake
# processes in a sandbox instead of on the live sweep. Defaults are production.
ROOT="${WATCHER_ROOT:-/Users/lacg/wnn}"
SCRIPT="${WATCHER_SCRIPT:-scripts/run_dfa_1layer_study.sh}"   # relative to ROOT
DRIVER_LOG="${WATCHER_DRIVER_LOG:-/private/tmp/dfa1l_driver.log}"
LOG="${WATCHER_LOG:-/private/tmp/dfa1l_restart_watcher.log}"
IDS_WORKER="${WATCHER_IDS_WORKER:-77344}"  # must survive untouched — IDS is priority
PHASED_PAT="${WATCHER_PHASED_PAT:-wnn.control.phased_ga}"
# Held ONLY during the kill/relaunch phase, so the sweep supervisor stands down
# for those few seconds instead of mistaking the planned 0-driver window for a
# stall and launching a second driver. Deliberately not "while this process
# exists" — this script waits for HOURS, and that would disable the supervisor
# for the entire cell.
LOCK="${WATCHER_LOCK:-/private/tmp/dfa1l_restart.lock}"

# WHAT TO DO once the boundary is reached and the driver+cell tree are down.
#   relaunch (default) — the original behaviour: start the driver again so the
#                        sweep continues from its markers.
#   handoff            — do NOT restart the driver; run $WATCHER_HANDOFF_CMD
#                        instead. Used to stop the sweep at a clean boundary and
#                        give the box to another controller (e.g. the ceiling
#                        pipeline). The one-controller rule still holds: the same
#                        step-3 kills and step-4 gates run first, so the handoff
#                        command starts with zero phased_ga alive.
# NOTE: in handoff mode the sweep supervisor MUST be stopped first, or it will
# see 0 drivers and relaunch one alongside the handoff — the restart lock only
# suppresses it for LOCK_STALE_S (1800s), which is far shorter than a handoff run.
ON_BOUNDARY="${WATCHER_ON_BOUNDARY:-relaunch}"
HANDOFF_CMD="${WATCHER_HANDOFF_CMD:-}"
HANDOFF_LOG="${WATCHER_HANDOFF_LOG:-/private/tmp/ceiling_pipeline.log}"
SUPERVISOR_PAT="${WATCHER_SUPERVISOR_PAT:-dfa1l_sweep_supervisor.sh}"

log() { echo "[restart-watcher] $(date -u +%FT%TZ) $*" >> "$LOG"; }

# Identity-checked liveness: a bare `ps -p` would be fooled by PID reuse across a
# multi-hour wait.
driver_alive() { ps -p "$DRIVER_PID" -o command= 2>/dev/null | grep -q "$SCRIPT"; }
cell_alive()   { ps -p "$CELL_PID"   -o command= 2>/dev/null | grep -q "$PHASED_PAT"; }
# Count only real phased_ga PYTHONs; the /usr/bin/time wrapper is not the hog and
# counting it would make one healthy cell look like a double-run.
phased_count() {
	ps -axo pid,command 2>/dev/null | grep "$PHASED_PAT" | grep -v "/usr/bin/time" \
		| grep -v grep | grep -c python
}

log "ARMED driver=$DRIVER_PID cell_py=$CELL_PID marker=$(basename "$MARKER") grace=${GRACE}s"

# ---- 1. wait for the cell to END (marker-independent) ------------------------
# If the DRIVER dies first we do not abort: the cell is still doing real work and
# killing it would throw that away, while relaunching alongside it would
# double-run. The healing move is to let the orphan finish, then relaunch. Its
# result is lost either way (the driver is what writes the marker) — and R4
# semantics say an unmarked cell is simply re-run, which is correct.
orphaned=0
while cell_alive; do
	if ! driver_alive; then
		orphaned=1
		log "driver $DRIVER_PID vanished mid-cell — letting the orphaned cell finish, \
then relaunching (killing it now would waste real work; relaunching now would double-run)"
		break
	fi
	sleep 5
done
if [ "$orphaned" = "1" ]; then
	while cell_alive; do sleep 5; done
	log "orphaned cell $CELL_PID finished — proceeding to relaunch"
fi
log "cell python $CELL_PID exited — cell boundary reached"

# ---- 2. short grace for the driver's post-processing -------------------------
# On a clean run the driver now runs gran_fpga_count.py and writes the marker
# (seconds). On an R4 crash/truncation path it writes NO marker and goes straight
# to the next cell — detected by a NEW phased_ga python appearing. Either way we
# stop waiting and act.
# An orphaned cell has no driver left to write a marker or start a next cell, so
# there is nothing to wait for — skip straight to the relaunch.
[ "$orphaned" = "1" ] && GRACE=0
waited=0
while [ "$waited" -lt "$GRACE" ]; do
	[ -f "$MARKER" ] && { log "marker written — clean completion"; break; }
	if [ "$(phased_count)" -gt 0 ]; then
		log "NO marker, but a new phased_ga python already started — R4 crash path; \
acting now to avoid burning time on stale code"
		break
	fi
	sleep 5
	waited=$((waited + 5))
done
[ "$waited" -ge "$GRACE" ] && log "grace ${GRACE}s expired with no marker and no new \
cell (likely the rc=143/137 calm-wait loop) — restarting anyway"

# ---- 3. stop the driver FIRST, then any surviving cell tree ------------------
# From here on the world briefly holds zero drivers. Take the lock so the
# supervisor stands down, and release it on ANY exit path (including the aborts).
date -u +%FT%TZ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
log "took restart lock $LOCK (supervisor stands down until this clears)"

# Order is load-bearing: the driver interprets a dead child as a watchdog stop and
# would re-launch the cell we just finished (run_dfa_1layer_study.sh rc=143/137
# branch). Kill the sequencer before its children.
if driver_alive; then
	kill -9 "$DRIVER_PID" 2>/dev/null && log "killed driver $DRIVER_PID"
else
	log "driver $DRIVER_PID already gone at kill time"
fi
sleep 2
# Orphan-safe: kills the /usr/bin/time wrapper AND the python it hides. Retried,
# because a survivor is usually transient (a process wedged in uninterruptible
# I/O ignores SIGKILL until the I/O returns) rather than permanent.
for attempt in 1 2 3; do
	pkill -9 -f "$PHASED_PAT" 2>/dev/null
	sleep 3
	[ "$(phased_count)" -eq 0 ] && break
	log "kill attempt ${attempt}/3: a phased_ga python is still alive, retrying"
	sleep 5
done

# ---- 4. safety gates before relaunch ----------------------------------------
if ! ps -p "$IDS_WORKER" >/dev/null 2>&1; then
	log "ABORT: IDS worker $IDS_WORKER is not alive — NOT relaunching the controller. \
IDS is priority and restarting it mid-'running' cancels the live flow, so recovery \
here stays a human call."
	exit 3
fi
left="$(phased_count)"
if [ "${left:-0}" -ne 0 ]; then
	log "ABORT: $left phased_ga python(s) survived 3 kill attempts — NOT relaunching \
(a second cell here is the double-run OOM risk)."
	exit 4
fi

# ---- 5a. HANDOFF: give the box to another controller instead of the sweep ----
# Runs only after step 3 killed the driver+cell tree and step 4 confirmed zero
# phased_ga survivors and a live IDS worker — so the handoff command inherits a
# clean one-controller world. The command reparents to init when this exits.
if [ "$ON_BOUNDARY" = "handoff" ]; then
	[ -n "$HANDOFF_CMD" ] || { log "ABORT: WATCHER_ON_BOUNDARY=handoff but WATCHER_HANDOFF_CMD is empty"; exit 7; }
	if pgrep -f "$SUPERVISOR_PAT" >/dev/null 2>&1; then
		log "ABORT: sweep supervisor is STILL RUNNING — it would relaunch the driver \
alongside the handoff (two controllers). Stop it first, then re-arm this watcher."
		exit 8
	fi
	cd "$ROOT" || { log "ABORT: cannot cd $ROOT"; exit 5; }
	log "HANDOFF: sweep stopped at a clean boundary; starting: $HANDOFF_CMD"
	nohup bash -c "$HANDOFF_CMD" >> "$HANDOFF_LOG" 2>&1 < /dev/null &
	hp=$!
	sleep 8
	if ps -p "$hp" >/dev/null 2>&1; then
		log "HANDOFF STARTED pid=$hp (reparents to PPID=1 on our exit) → $HANDOFF_LOG"
		exit 0
	fi
	log "ABORT: handoff command did not survive 8s — check $HANDOFF_LOG. Sweep is \
stopped and NOTHING is running; needs manual attention."
	exit 9
fi

# ---- 5. relaunch; on our exit it reparents to init (PPID=1) ------------------
# Bounded retry: a transient launch failure deserves another go, but a genuinely
# broken script would loop forever, so this caps at 3 rather than persisting.
cd "$ROOT" || { log "ABORT: cannot cd $ROOT"; exit 5; }
for attempt in 1 2 3; do
	nohup bash "$SCRIPT" >> "$DRIVER_LOG" 2>&1 < /dev/null &
	new_pid=$!
	sleep 8
	if ps -p "$new_pid" -o command= 2>/dev/null | grep -q "$SCRIPT"; then
		log "RESTARTED: new driver pid=$new_pid on attempt ${attempt} (reparents to \
PPID=1 when this watcher exits). It skips every cell that already has a marker."
		exit 0
	fi
	log "relaunch attempt ${attempt}/3: driver $new_pid did not survive 8s"
	sleep 5
done
log "ABORT: driver failed to start 3 times — sweep is STALLED, needs manual attention. \
The supervisor will keep retrying on its own tick if it is armed."
exit 6
