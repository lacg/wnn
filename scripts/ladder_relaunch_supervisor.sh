#!/usr/bin/env bash
# LADDER RELAUNCH SUPERVISOR (31/08/2026) — the LAST THREE LINES of
# probe_handoff_supervisor.sh, and deliberately nothing else.
#
# WHY THIS SCRIPT EXISTS. The power outage at ~18:28Z killed the handoff
# supervisor with only its step 4 left to do: wait for the box to clear, then
# relaunch scripts/sweep_ladder_ab_chain.sh for the cull + seed 31337003. The
# handoff CANNOT simply be restarted to finish that job — its step 1 preempts
# any live controller (SIGTERM -> 60s -> SIGKILL) and its step 2 re-runs the
# b=64 smoke, so relaunching it whole would kill the b=48 long-budget run and
# then put a SECOND controller on the box. This script is the tail alone: it
# never kills anything and never starts a controller of its own.
#
# THE COMPLETION SIGNAL IS THE MARKER, NOT THE PROCESS. We wait for the b=48
# run's marker file, not merely for its process to disappear. Marker semantics
# are fixed by controller_arm_lib.sh: a marker is a CLAIM THAT THE RUN GENUINELY
# FINISHED, withheld on watchdog kill (rc=143/137), on crash (rc!=0), and on a
# clean exit that printed no MEMORY-stage triple. So "process gone, no marker"
# means the run needs a human, and this script FAILS CLOSED there — it refuses
# to relaunch the ladder and leaves the box idle to be inspected. Waiting on the
# process alone would relaunch the ladder over a crash and bury it.
#
# ONE CONTROLLER AT A TIME is preserved by wait_no_controller, a PURE WAIT that
# never escalates: the b=48 run in flight is a LEGITIMATE run and must be
# allowed to finish on its own terms.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

LOG="/private/tmp/ladder_relaunch.log"
MARKDIR="experiments/sweepladder_markers"
B48_TAG="SL_A_b48n32_cf21_brushless_L4C_g20_s31337002"
B48_MARKER="${MARKDIR}/${B48_TAG}.json"

log() { echo "[ladder-relaunch] $(date -u +%FT%TZ) $*" >> "$LOG"; }

# Match BOTH the resolved interpreter and the `/usr/bin/time -l ... venv/bin/python`
# wrapper the ladder launches runs through — the wrapper re-parents to PID 1 when its
# parent dies and would otherwise be invisible here.
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/sweep_ladder_ab_chain.sh" 2>/dev/null || true; }
b48_pids() { pgrep -f "scripts/b48_longbudget_supervisor.sh" 2>/dev/null || true; }
# $$ is this shell; a second instance would double-launch the chain.
peer_pids() { pgrep -f "scripts/ladder_relaunch_supervisor.sh" 2>/dev/null | grep -v "^$$\$" || true; }

# PURE WAIT — never escalates, never kills. See the header.
wait_no_controller() {
	local waited=0
	while [ -n "$(controller_pids)" ]; do
		sleep 10
		waited=$((waited + 10))
		[ $((waited % 600)) = 0 ] && log "still waiting for controllers to exit (${waited}s): $(controller_pids | tr '\n' ' ')"
	done
	return 0
}

log "########## ARMED — waits for $B48_MARKER, then relaunches the ladder chain ##########"

# ---- 0. single-instance and no-chain guards. Both are fail-closed: if we cannot
# be sure we are the only one about to launch, we do not launch.
if [ -n "$(peer_pids)" ]; then
	log "STOPPED: another ladder_relaunch_supervisor is already armed ($(peer_pids | tr '\n' ' ')). Refusing to double-arm."
	exit 1
fi
if [ -n "$(chain_pids)" ]; then
	log "STOPPED: the ladder chain is ALREADY RUNNING ($(chain_pids | tr '\n' ' ')). Nothing to relaunch."
	exit 1
fi

# ---- 1. wait for the b=48 long-budget run to bank its marker.
# Fail closed if its supervisor is gone with no marker: that is a kill, a crash or a
# truncated run, and a human should see it before the ladder buries it.
while [ ! -f "$B48_MARKER" ]; do
	if [ -z "$(b48_pids)" ] && [ -z "$(controller_pids)" ]; then
		log "########## STOPPED (fail-closed): b48 supervisor AND controller are gone with no marker at $B48_MARKER."
		log "That means the run was killed, crashed, or exited without a MEMORY-stage triple — it is re-runnable by design."
		log "Ladder chain NOT relaunched. Box is idle. Inspect logs/controller/sweep_ladder/${B48_TAG}.out first. ##########"
		exit 1
	fi
	sleep 60
done
log "b=48 long-budget run COMPLETE — marker banked."

# ---- 2. let it finish releasing the box. Pure wait; the run is legitimate.
wait_no_controller
log "box is clear of controllers."

# ---- 3. relaunch the ladder. It is idempotent per marker: it skips all of round 1
# (seed 31337002, b12-b36, both arms) and resumes at the cull + seed 31337003.
if [ -n "$(chain_pids)" ]; then
	log "########## STOPPED (fail-closed): a ladder chain appeared while we waited ($(chain_pids | tr '\n' ' ')). Not launching a second. ##########"
	exit 1
fi
log "relaunching ladder chain (skips the round-1 markers, resumes at cull + seed 31337003)"
nohup bash scripts/sweep_ladder_ab_chain.sh >/dev/null 2>&1 &
log "ladder chain relaunched pid=$! ########## HANDOFF COMPLETE ##########"
