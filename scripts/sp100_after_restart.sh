#!/usr/bin/env bash
# SP100 GATE — create the 500-flow cohort ONLY after the worker restart succeeds.
#
# WHY THIS ORDER IS LOAD-BEARING. The worker-restart watcher
# (worker_restart_at_idle.sh) fires only at 0 running + 0 queued flows; queueing
# 500 flows before it fires would BLOCK the restart forever and the whole cohort
# would run on the pre-8b839a30 positional-rank optimizer — the exact thing the
# "100 clean" decision exists to avoid. So: watcher exits -> verify it logged a
# SUCCESSFUL restart -> only then POST the flows. If the watcher aborted (flow
# appeared last-moment, TERM ignored, ceiling), NOTHING is created and a human
# reads the log. Armed by Luiz's "let's arm a worker restart to gate it".
set -u

LOG="/private/tmp/sp100_gate.log"
WLOG="/private/tmp/worker_restart_watcher.log"
VP="/Users/lacg/wnn-venv/bin/python"
WATCH_PID="${SP100_WATCH_PID:?set SP100_WATCH_PID to the watcher pid}"
CEIL="${SP100_WAIT_CEIL:-259200}"   # 72h

log() { echo "[sp100] $(date -u +%FT%TZ) $*" >> "$LOG"; }

log "########## ARMED — gate on watcher pid $WATCH_PID, then create 500 SP100 flows ##########"
waited=0
while kill -0 "$WATCH_PID" 2>/dev/null; do
	[ $((waited % 3600)) -eq 0 ] && log "waiting for watcher to exit (${waited}s)"
	sleep 120
	waited=$((waited + 120))
	if [ "$waited" -ge "$CEIL" ]; then
		log "ABORT: watcher still alive after ${waited}s — creating nothing."
		exit 3
	fi
done

if ! grep -q "tie-aware ranking live" "$WLOG" 2>/dev/null; then
	log "ESCALATION: watcher exited WITHOUT a successful restart — creating nothing. Read $WLOG"
	exit 4
fi
log "watcher reports a successful worker restart — creating the cohort"

cd /Users/lacg/wnn || exit 1
"$VP" scripts/create_sp100_cohorts.py >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
	log "ESCALATION: creation script rc=$rc — cohort NOT fully queued, read this log"
	exit 5
fi
log "########## SP100 COHORT QUEUED (500 flows, verified) ##########"
