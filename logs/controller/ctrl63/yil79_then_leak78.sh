#!/bin/bash
# Queue after the n=3 CTRL-77 factorial (Luiz 25/11/2026: "fly the ladder before the leak block"):
#   F77 DONE (n=3)  ->  ab_yil79.sh (CTRL-79, yil x s1-s3)  ->  ab_leak78.sh (CTRL-78, lk85 lk90 x s1-s3).
# Pure waits; never touches a running flight. Lines 'Q79:' in ctrl48/runs.log. Fails CLOSED: stands down on
# 'F77 ABORT', and does not start the leak block unless the ladder block logged 'Y79 DONE'.
# Every match is ANCHORED on "<timestamp> <block> ..." so a queue line that quotes the words cannot satisfy it.
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
F_DONE='^[0-9TZ:-]+ F77 DONE seeds \[1 2 3\]'
F_ABORT='^[0-9TZ:-]+ F77 ABORT'
Y_DONE='^[0-9TZ:-]+ Y79 DONE seeds \[1 2 3\]'
grep -qE "$F_DONE" $RUNS && { log "Q79: refused — the n=3 factorial is already done; start the blocks by hand"; exit 1; }
log "Q79: armed (pid $$) — after the n=3 factorial: ab_yil79.sh (ladder), then ab_leak78.sh (leak)"
while :; do
	grep -qE "$F_DONE" $RUNS && break
	if grep -qE "$F_ABORT" $RUNS; then log "Q79: the factorial aborted — standing down, nothing started"; exit 1; fi
	sleep 60
done
log "Q79: the factorial finished — starting ab_yil79.sh"
logs/controller/ctrl63/ab_yil79.sh
if ! grep -qE "$Y_DONE" $RUNS; then log "Q79: the ladder block did not finish cleanly — standing down, leak block NOT started"; exit 1; fi
log "Q79: the ladder block finished — starting ab_leak78.sh"
exec logs/controller/ctrl63/ab_leak78.sh
