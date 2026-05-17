#!/bin/bash
# Watches flow 2674. When it reaches terminal status, restarts the worker
# (to pick up commits f554509f + 94ff5519) and requeues any flow that was
# running at the moment of restart.
#
# Usage:
#   nohup bash scripts/watch_2674_restart.sh >> watcher.log 2>&1 &

set -u

TARGET_FLOW=2674
WORKER_PID=6443
API=https://localhost:3000
LOG=/Users/lacg/wnn/watcher.log
POLL_INTERVAL=30
SHUTDOWN_WAIT_SECONDS=120  # max time to wait for graceful SIGINT exit

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

get_status() {
	local fid="$1"
	curl -sk "$API/api/flows/$fid" 2>/dev/null \
		| python3 -c "import json,sys
try:
    print(json.load(sys.stdin).get('status','unknown'))
except Exception:
    print('unknown')"
}

get_running_ids() {
	curl -sk "$API/api/flows" 2>/dev/null \
		| python3 -c "import json,sys
try:
    d=json.load(sys.stdin)
    print(' '.join(str(f['id']) for f in d if f.get('status')=='running'))
except Exception:
    pass"
}

log "==== Watcher started (target=$TARGET_FLOW, worker_pid=$WORKER_PID) ===="

# 1. Poll until target flow reaches terminal status
while true; do
	status=$(get_status "$TARGET_FLOW")
	if [ "$status" = "completed" ] || [ "$status" = "error" ] || [ "$status" = "failed" ] || [ "$status" = "stopped" ]; then
		log "Flow $TARGET_FLOW reached terminal status: $status"
		break
	fi
	log "Flow $TARGET_FLOW status=$status (polling again in ${POLL_INTERVAL}s)"
	sleep "$POLL_INTERVAL"
done

# 2. Give the worker a few seconds to either idle or pick up the next flow,
#    then snapshot what's running so we can requeue it after the restart.
sleep 5
RUNNING_BEFORE=$(get_running_ids)
log "Running flows before SIGINT: [$RUNNING_BEFORE]"

# 3. Send SIGINT to the worker
if ps -p "$WORKER_PID" > /dev/null 2>&1; then
	log "Sending SIGINT to worker PID $WORKER_PID"
	kill -INT "$WORKER_PID"
	waited=0
	while ps -p "$WORKER_PID" > /dev/null 2>&1; do
		if [ "$waited" -ge "$SHUTDOWN_WAIT_SECONDS" ]; then
			log "WARNING: worker still alive after ${SHUTDOWN_WAIT_SECONDS}s; sending SIGTERM"
			kill -TERM "$WORKER_PID" 2>/dev/null || true
			sleep 5
			break
		fi
		sleep 2
		waited=$((waited + 2))
	done
	if ps -p "$WORKER_PID" > /dev/null 2>&1; then
		log "ERROR: worker $WORKER_PID still alive after SIGTERM. Aborting watcher."
		exit 1
	fi
	log "Worker $WORKER_PID stopped after ${waited}s"
else
	log "Worker PID $WORKER_PID was not running at SIGINT time"
fi

# 4. Requeue any flow that was running (skip the target — it's already terminal)
for fid in $RUNNING_BEFORE; do
	if [ "$fid" = "$TARGET_FLOW" ]; then
		continue
	fi
	log "Requeuing flow $fid (from_beginning=false → resume from checkpoint)"
	resp=$(curl -sk -X POST "$API/api/flows/$fid/restart" \
		-H "Content-Type: application/json" \
		-d '{"from_beginning": false}' 2>&1)
	log "  /restart response: $resp"
done

# 5. Restart the worker with the same env the prior PID had
log "Starting new worker..."
cd /Users/lacg/wnn || { log "ERROR: cd /Users/lacg/wnn failed"; exit 1; }

VIRTUAL_ENV=/Users/lacg/wnn-venv \
PATH=/Users/lacg/wnn-venv/bin:$PATH \
WNN_HYBRID_SPEED_RATIO=3 \
WNN_GPU_BATCHED_TRAIN_TRACE=1 \
	nohup /Users/lacg/wnn-venv/bin/python -u -m wnn.ram.experiments.worker \
		--url https://localhost:3000 --no-ssl-verify >> worker.out 2>&1 &
NEW_PID=$!
log "New worker started, PID=$NEW_PID"
log "==== Watcher complete ===="
