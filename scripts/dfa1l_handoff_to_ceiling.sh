#!/usr/bin/env bash
# One-shot: stop the dfa1l sweep at its NEXT cell boundary and hand the box to the
# ceiling pipeline (S→B→A→C). Authorized by the user 31/07/2026 — the sweep was
# consuming seeds on a configuration the 30/07 diagnostics already flagged as
# sensor-starved (every cell runs suffix=18; the smoke test moved held-out 40%→90-95%
# at suffix 32), and dfa_10feat_BINARY_s31337003 then spent 17h to return 0% stable
# held-out. Phase S tests that lever directly.
#
# Ordering is load-bearing:
#   1. STOP THE SUPERVISOR FIRST. It auto-relaunches the driver whenever zero are
#      alive (dfa1l_sweep_supervisor.sh:12-17). The restart lock only suppresses it
#      for LOCK_STALE_S=1800s — far shorter than the pipeline — so the lock alone
#      would let a second driver start underneath the pipeline (two controllers, the
#      double-run OOM risk that the 30/07 watchdog incident showed the box cannot take).
#      The watcher's own handoff gate re-checks this and REFUSES (exit 8) if missed.
#   2. Arm the boundary watcher in handoff mode. It waits for the in-flight cell to
#      finish (so its marker is not lost), kills driver-then-cell-tree, verifies the
#      IDS worker is alive and zero phased_ga survive, then starts the pipeline.
#
# Resuming the sweep later: the markers ARE the resume state, so
#   nohup bash scripts/run_dfa_1layer_study.sh >> /private/tmp/dfa1l_driver.log 2>&1 &
# re-enters the loop and skips every completed cell. Re-arm the supervisor separately.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

DRIVER_PID="${1:?driver pid required}"
CELL_PID="${2:?live cell python pid required}"
MARKER="${3:?marker path of the in-flight cell required}"
PHASE="${4:-all}"

log() { echo "[handoff] $(date -u +%FT%TZ) $*"; }

# ---- 1. stop the supervisor (must precede arming; see header) ----------------
if pgrep -f dfa1l_sweep_supervisor.sh >/dev/null 2>&1; then
	pkill -f dfa1l_sweep_supervisor.sh
	sleep 2
	if pgrep -f dfa1l_sweep_supervisor.sh >/dev/null 2>&1; then
		log "ABORT: supervisor would not die — not arming (it would relaunch the driver)"
		exit 2
	fi
	log "supervisor stopped"
else
	log "supervisor already stopped"
fi

# ---- 2. arm the boundary watcher in handoff mode -----------------------------
export WATCHER_ON_BOUNDARY=handoff
export WATCHER_HANDOFF_CMD="bash scripts/run_ceiling_pipeline.sh ${PHASE}"
export WATCHER_HANDOFF_LOG=/private/tmp/ceiling_pipeline.log
log "arming boundary watcher: driver=$DRIVER_PID run_id=$CELL_PID marker=$(basename "$MARKER") phase=$PHASE"
exec bash scripts/dfa1l_restart_at_cell_boundary.sh "$DRIVER_PID" "$CELL_PID" "$MARKER" 300
