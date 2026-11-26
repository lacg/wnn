#!/bin/bash
# CTRL-78 waiter (Luiz 25/11/2026 "queue 2"): start ab_leak78.sh when the n=3 CTRL-77 factorial logs 'F77 DONE seeds [1 2 3]'.
# Stands down on 'F77 ABORT'. Lines 'L78Q: ...' in ctrl48/runs.log. Never touches a running flight (pure wait).
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
# The match is ANCHORED on "<timestamp> F77 ..." so no L78Q line (which quotes the words) can satisfy it — the first
# version matched its own 'armed' line and started the driver at once (25/11, killed in preflight, nothing flown).
DONE_RE='^[0-9TZ:-]+ F77 DONE seeds \[1 2 3\]'
ABORT_RE='^[0-9TZ:-]+ F77 ABORT'
grep -qE "$DONE_RE" $RUNS && { log "L78Q: refused — the n=3 factorial is already done; start ab_leak78.sh by hand"; exit 1; }
log "L78Q: armed (pid $$) — waits for the n=3 factorial to finish, then ab_leak78.sh (lk85, lk90 x s1-s3)"
while :; do
	grep -qE "$DONE_RE" $RUNS && break
	if grep -qE "$ABORT_RE" $RUNS; then log "L78Q: F77 ABORT seen — standing down, CTRL-78 NOT started"; exit 1; fi
	sleep 60
done
log "L78Q: the factorial finished — starting ab_leak78.sh"
exec logs/controller/ctrl63/ab_leak78.sh
