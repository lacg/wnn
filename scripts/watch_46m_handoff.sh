#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# watch_46m_handoff.sh — overnight handoff for the 46M flow (id 2748).
#
# Polls the flow's status in the dashboard DB. When it COMPLETES:
#   1. stop the worker            (pkill -f wnn.ram.experiments.worker)
#   2. rebuild the accelerator    (maturin develop --release  ← bakes in the
#                                  held axonogenesis getters; old .so kept if it fails)
#   3. requeue flow 2748 FROM THE BEGINNING  (POST /api/flows/2748/restart from_beginning)
#   4. restart the worker         (./start-worker.sh --tls — picks up the requeued flow)
#   5. (re)check the dashboard; only flag if it is down (does NOT blindly relaunch it)
#
# Safe by design: NOT set -e (one failed step never aborts the rest); a failed
# maturin leaves the previous .so in place so the worker still restarts; every
# step is timestamped to the log. Trigger is status='completed' only — a
# failed/cancelled flow logs and exits WITHOUT auto-requeuing (avoids a loop).
#
# Usage:   nohup scripts/watch_46m_handoff.sh > watcher_46m.out 2>&1 &
# Tail:    tail -f watcher_46m.out
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="/Users/lacg/wnn"
DB="$REPO/db/wnn.db"
FLOW_ID="${FLOW_ID:-2748}"
URL="${URL:-https://localhost:3000}"
POLL_SECS="${POLL_SECS:-30}"
ACCEL_DIR="$REPO/src/wnn/ram/strategies/accelerator"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

flow_status() { sqlite3 "$DB" "SELECT status FROM flows WHERE id=$FLOW_ID;" 2>/dev/null; }
dashboard_up() { curl -sk --max-time 5 -o /dev/null -w "%{http_code}" "$URL/api/flows" 2>/dev/null | grep -q "^2"; }

log "watcher armed for flow $FLOW_ID (poll ${POLL_SECS}s, url $URL)"
log "initial flow status: $(flow_status)"

# 1. Poll until the flow completes (or ends some other way).
while true; do
	st="$(flow_status)"
	case "$st" in
		completed)  log "flow $FLOW_ID COMPLETED — starting handoff"; break ;;
		failed|cancelled)
			log "flow $FLOW_ID ended status=$st — NOT auto-requeuing (handoff skipped). Exiting."
			exit 0 ;;
		"")         log "WARN: flow $FLOW_ID not found in DB (status empty) — retrying" ;;
	esac
	sleep "$POLL_SECS"
done

# 2. Stop the worker and wait for it to actually exit.
log "stopping worker (pkill -f wnn.ram.experiments.worker)"
pkill -f "wnn.ram.experiments.worker" 2>/dev/null && log "  signalled" || log "  no worker process found"
for _ in $(seq 1 30); do
	pgrep -f "wnn.ram.experiments.worker" >/dev/null 2>&1 || break
	sleep 1
done
pgrep -f "wnn.ram.experiments.worker" >/dev/null 2>&1 \
	&& log "  WARN: worker still alive after 30s; continuing anyway" \
	|| log "  worker stopped"

# 3. Rebuild the accelerator (the held 4b-5 maturin build). Old .so survives a failure.
log "rebuilding accelerator: maturin develop --release"
(
	cd "$ACCEL_DIR" || exit 1
	unset CONDA_PREFIX
	# shellcheck disable=SC1091
	source "$REPO/wnn/bin/activate"
	maturin develop --release
)
if [ $? -eq 0 ]; then log "  maturin rebuild OK (axonogenesis getters now live)"; else log "  maturin rebuild FAILED — keeping previous .so (worker still restartable)"; fi

# 4. Requeue flow 2748 from the beginning (needs the dashboard up).
if dashboard_up; then
	log "requeuing flow $FLOW_ID from the beginning"
	resp="$(curl -sk -X POST "$URL/api/flows/$FLOW_ID/restart" \
		-H 'Content-Type: application/json' \
		-d '{"from_beginning": true, "start_from_experiment": null}' 2>&1)"
	log "  /restart response: $resp"
else
	log "  WARN: dashboard not responding at $URL — CANNOT requeue via API."
	log "        Restart the dashboard, then: curl -sk -X POST $URL/api/flows/$FLOW_ID/restart -d '{\"from_beginning\":true}'"
fi

# 5. Restart the worker (TLS, matching how it was running).
log "restarting worker (./start-worker.sh --tls)"
( cd "$REPO" && ./start-worker.sh --tls ) 2>&1 | sed 's/^/  /'

# 6. Dashboard health note (do NOT blindly relaunch — it stays up across a worker restart).
if dashboard_up; then
	log "dashboard healthy at $URL"
else
	log "WARN: dashboard DOWN — relaunch manually (binary: /Volumes/.../cargo-target/release/wnn-dashboard)"
fi

sleep 8
log "handoff complete. flow $FLOW_ID status now: $(flow_status)"
log "worker: $(pgrep -f 'wnn.ram.experiments.worker' | head -1 || echo 'NOT RUNNING')"
