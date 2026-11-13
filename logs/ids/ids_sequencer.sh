#!/bin/bash
# IDS release sequencer (Luiz 13/11/2026). Replaces ids57_a1_before_carryover.sh + ids56_theta_then_backlog.sh.
# Releases flows ONE AT A TIME in an explicit order: a flow is POST /resume'd only when every earlier flow in the
# order file is terminal (completed/failed/cancelled). Because at most one listed flow is ever queued, the worker's
# FIFO min(id) admission cannot reorder them. Never PATCHes status (re-queue = /resume only, per the 09/11 rule).
# A flow that is PAUSED (e.g. a cost stop on Luiz's order) is NOT terminal: the sequencer waits on it, it never
# skips it. A line "WAIT <id>" only waits for that flow (never resumes it — used for flows already queued/running,
# so a cost-stop pause Luiz orders is respected). The final line may be "BACKLOG <file>": all paused flows listed
# there are resumed together (id order = FIFO) once everything above is terminal.
# Usage: ids_sequencer.sh <order-file>    (one flow id per line; '#' comments allowed)
P=/Users/lacg/wnn
D="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API=https://127.0.0.1:3000/api/flows
ORDER=$1
LOG=$P/logs/ids/ids_sequencer.log
log() { echo "$(date -u +%FT%TZ) [ids-seq] $*" >> $LOG; }
status() { sqlite3 "$D" "select status from flows where id=$1"; }
terminal() { case "$(status $1)" in completed|failed|cancelled) return 0 ;; *) return 1 ;; esac; }
resume()
{
	local c t
	for t in 1 2 3 4 5 6; do
		c=$(curl -sk -o /dev/null -w "%{http_code}" -X POST $API/$1/resume)
		[ "$c" = "200" ] && return 0
		sleep 5
	done
	log "!! POST resume $1 failed (http $c)"
}
wait_terminal() { until terminal $1; do sleep 60; done; log "$1 terminal: $(status $1)"; [ "$(status $1)" = "failed" ] && log "!! flow $1 FAILED (sequence continues)"; }
[ -f "$ORDER" ] || { echo "no order file $ORDER"; exit 1; }
log "armed: $(grep -cvE '^\s*(#|$)' "$ORDER") entries from $ORDER"
grep -vE '^\s*(#|$)' "$ORDER" | while read -r id file; do
	if [ "$id" = "WAIT" ]; then
		terminal $file || wait_terminal $file
		continue
	fi
	if [ "$id" = "BACKLOG" ]; then
		for b in $(sort -n "$file"); do [ "$(status $b)" = "paused" ] && resume $b; done
		log "BACKLOG released: $(sqlite3 "$D" "select count(*) from flows where status='queued' and id in ($(paste -sd, "$file"))") queued from $file"
		continue
	fi
	if ! terminal $id; then
		[ "$(status $id)" = "paused" ] && { resume $id; log "released $id"; }
		wait_terminal $id
	fi
done
log "SEQUENCE DONE"
