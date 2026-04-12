#!/usr/bin/env bash
# ============================================================================
# Post-PUB50 Watcher: orchestrates #31 → #23 → #25 in sequence
#
# Waits for the PUB50 8b CIC-IoT batch to finish, then queues:
#   Stage 1: 46M 6n×16b eval (flow 1231, ~30 min)
#   Stage 2: UNSW random mini-sweep (flows 1219-1230, ~4h)
#   Stage 3: UNSW temporal mini-sweep (flows 1232-1243, ~1h)
#
# Usage:
#   nohup bash scripts/watcher_post_pub50.sh > /tmp/watcher_post_pub50.out 2>&1 &
# ============================================================================

set -euo pipefail

DB="db/wnn.db"
LOG="/tmp/watcher_post_pub50.log"

# Configurable sleep durations
INITIAL_SLEEP="${INITIAL_SLEEP:-18000}"  # 5h default
POLL_INTERVAL=1800                       # 30 min between polls
LAST_FLOW_POLL=60                        # 60s when last flow is running

# Flow ID ranges
PUB50_FIRST=711
PUB50_LAST=730  # r001 is the last to complete (lowest ID, worker goes ID DESC)

EVAL46M_FLOW=1231
UNSW_RANDOM_FIRST=1219
UNSW_RANDOM_LAST=1230
UNSW_TEMPORAL_FIRST=1232
UNSW_TEMPORAL_LAST=1243

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

wait_for_all_completed() {
	# Wait for all flows in range [$1, $2] to be completed
	local first=$1 last=$2 label=$3 poll=$4
	while true; do
		local remaining=$(sqlite3 "$DB" "
			SELECT COUNT(*) FROM flows
			WHERE id BETWEEN $first AND $last
			  AND status IN ('queued','running');
		")
		if [ "$remaining" -eq 0 ]; then
			log "  All $label flows completed!"
			return 0
		fi
		local running=$(sqlite3 "$DB" "
			SELECT id, name FROM flows
			WHERE id BETWEEN $first AND $last AND status='running'
			LIMIT 1;
		" 2>/dev/null || echo "?")
		log "  $label: $remaining remaining, running: $running"
		sleep "$poll"
	done
}

queue_range() {
	local first=$1 last=$2 label=$3
	local count=$(sqlite3 "$DB" "
		UPDATE flows SET status='queued'
		WHERE id BETWEEN $first AND $last AND status='pending';
		SELECT changes();
	")
	log "  Queued $count $label flows (IDs $first..$last)"
}

# ============================================================================
log "==== Post-PUB50 Watcher started ===="
log "Strategy: sleep ${INITIAL_SLEEP}s, then poll every ${POLL_INTERVAL}s"
log "  Stage 1: 46M 6n×16b eval (flow $EVAL46M_FLOW)"
log "  Stage 2: UNSW random mini-sweep (flows $UNSW_RANDOM_FIRST..$UNSW_RANDOM_LAST)"
log "  Stage 3: UNSW temporal mini-sweep (flows $UNSW_TEMPORAL_FIRST..$UNSW_TEMPORAL_LAST)"

# Phase 1: Wait for PUB50 8b to finish
initial_hours=$(awk "BEGIN {printf \"%.1f\", ${INITIAL_SLEEP}/3600}")
log "Sleeping ${initial_hours}h..."
sleep "${INITIAL_SLEEP}"

# Poll until last PUB50 flow is running
log "Polling every ${POLL_INTERVAL}s for PUB50 completion..."
while true; do
	pub50_remaining=$(sqlite3 "$DB" "
		SELECT COUNT(*) FROM flows
		WHERE name LIKE 'PUB50-ciciot-random-%'
		  AND name NOT LIKE '%-2b-%'
		  AND status IN ('queued','running');
	")

	if [ "$pub50_remaining" -eq 0 ]; then
		log "PUB50 8b batch COMPLETE!"
		break
	elif [ "$pub50_remaining" -eq 1 ]; then
		log "Last PUB50 flow running — switching to ${LAST_FLOW_POLL}s polling"
		while true; do
			still_going=$(sqlite3 "$DB" "
				SELECT COUNT(*) FROM flows
				WHERE name LIKE 'PUB50-ciciot-random-%'
				  AND name NOT LIKE '%-2b-%'
				  AND status IN ('queued','running');
			")
			if [ "$still_going" -eq 0 ]; then
				log "PUB50 8b batch COMPLETE!"
				break 2
			fi
			sleep "$LAST_FLOW_POLL"
		done
	else
		log "PUB50: $pub50_remaining flows remaining"
		sleep "$POLL_INTERVAL"
	fi
done

# ============================================================================
# Stage 1: Queue 46M 6n×16b eval
# ============================================================================
log ""
log "==== STAGE 1: 46M 6n×16b first-stage filter eval ===="
queue_range $EVAL46M_FLOW $EVAL46M_FLOW "46M eval"
wait_for_all_completed $EVAL46M_FLOW $EVAL46M_FLOW "46M eval" 60

# ============================================================================
# Stage 2: Queue UNSW random mini-sweep
# ============================================================================
log ""
log "==== STAGE 2: UNSW random mini-sweep (12 flows) ===="
queue_range $UNSW_RANDOM_FIRST $UNSW_RANDOM_LAST "UNSW random mini-sweep"
wait_for_all_completed $UNSW_RANDOM_FIRST $UNSW_RANDOM_LAST "UNSW random" 300

# ============================================================================
# Stage 3: Queue UNSW temporal mini-sweep
# ============================================================================
log ""
log "==== STAGE 3: UNSW temporal mini-sweep (12 flows) ===="
queue_range $UNSW_TEMPORAL_FIRST $UNSW_TEMPORAL_LAST "UNSW temporal mini-sweep"
wait_for_all_completed $UNSW_TEMPORAL_FIRST $UNSW_TEMPORAL_LAST "UNSW temporal" 300

# ============================================================================
log ""
log "==== ALL STAGES COMPLETE ===="
log "  PUB50 8b: done"
log "  46M 6n×16b: done"
log "  UNSW random mini-sweep: done"
log "  UNSW temporal mini-sweep: done"
log ""
log "Next: review mini-sweep results and fire 110-run batches"

# Desktop notification
osascript -e 'display notification "PUB50 + 46M eval + both UNSW mini-sweeps complete!" with title "WNN Watcher" sound name "Glass"' 2>/dev/null || true
