#!/usr/bin/env bash
# WORKER RESTART AT IDLE — pick up the compute_ranks tie fix (8b839a30) for IDS.
#
# WHY. The IDS worker holds its Python code in memory; the 09/08/2026 fractional-
# rank fix only reaches IDS when the worker restarts. Restarting mid-cohort would
# split the SP-abl cohort across two ranking rules (internal-consistency breach),
# and killing a RUNNING flow is forbidden outright — so this waits for a FULL
# drain (0 running AND 0 queued across ALL flows, not just abl) and only then
# cycles the worker. Authorized by Luiz 09/08/2026 ("arm both").
#
# The relaunch reproduces the live worker's captured context exactly:
#   cwd /Users/lacg/wnn · PYTHONPATH=/Users/lacg/wnn/src/wnn ·
#   /Users/lacg/wnn-venv/bin/python -u -B -m wnn.ram.experiments.worker
#     --url https://localhost:3000 --no-ssl-verify  >> /tmp/wnn_worker.log
set -u

DB="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
LOG="/private/tmp/worker_restart_watcher.log"
VP="/Users/lacg/wnn-venv/bin/python"
WLOG="/tmp/wnn_worker.log"
CEIL="${WRK_WAIT_CEIL:-172800}"   # 48h — a queue that never drains needs a human

log() { echo "[wrk-restart] $(date -u +%FT%TZ) $*" >> "$LOG"; }
busy() { sqlite3 "$DB" "select count(*) from flows where status in ('running','queued');" 2>/dev/null || echo 999; }
worker_pid() { pgrep -f "wnn.ram.experiments.worker" | head -1; }

log "########## ARMED — restart worker when flows hit 0 running + 0 queued ##########"
waited=0
while [ "$(busy)" -gt 0 ]; do
	[ $((waited % 1800)) -eq 0 ] && log "waiting: $(busy) flows running/queued (${waited}s)"
	sleep 120
	waited=$((waited + 120))
	if [ "$waited" -ge "$CEIL" ]; then
		log "ABORT: queue not drained after ${waited}s — leaving the worker alone."
		exit 3
	fi
done

# Re-check right before acting: a flow enqueued during the last sleep wins.
if [ "$(busy)" -gt 0 ]; then log "flow appeared at the last moment — restart aborted, re-run me"; exit 4; fi

PID="$(worker_pid)"
if [ -z "$PID" ]; then log "no worker process found — nothing to restart"; exit 5; fi

log "queue drained — SIGTERM worker pid $PID"
kill -TERM "$PID"
for _ in $(seq 1 30); do
	kill -0 "$PID" 2>/dev/null || break
	sleep 2
done
if kill -0 "$PID" 2>/dev/null; then
	log "worker ignored TERM for 60s — NOT escalating to KILL (a human should look)"
	exit 6
fi
log "worker $PID exited cleanly"

cd /Users/lacg/wnn || { log "ABORT: bad cwd"; exit 7; }
PYTHONPATH=/Users/lacg/wnn/src/wnn nohup "$VP" -u -B -m wnn.ram.experiments.worker \
	--url https://localhost:3000 --no-ssl-verify >> "$WLOG" 2>&1 &
NEW=$!
disown
sleep 5
if kill -0 "$NEW" 2>/dev/null; then
	log "new worker pid $NEW alive (ppid $(ps -o ppid= -p "$NEW" | tr -d ' ')) — tie-aware ranking live for IDS"
else
	log "ESCALATION: new worker died within 5s — check $WLOG"
	exit 8
fi
