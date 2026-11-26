#!/bin/bash
# CTRL-78 waiter (Luiz 25/11/2026 "queue 2"): start ab_leak78.sh when the n=3 CTRL-77 factorial logs 'F77 DONE seeds [1 2 3]'.
# Stands down on 'F77 ABORT'. Lines 'L78Q: ...' in ctrl48/runs.log. Never touches a running flight (pure wait).
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
log "L78Q: armed (pid $$) — waits for 'F77 DONE seeds [1 2 3]', then ab_leak78.sh (lk85, lk90 x s1-s3)"
while :; do
	grep -q "F77 DONE seeds \[1 2 3\]" $RUNS && break
	if grep -q "F77 ABORT" $RUNS; then log "L78Q: F77 ABORT seen — standing down, CTRL-78 NOT started"; exit 1; fi
	sleep 60
done
log "L78Q: F77 DONE seen — starting ab_leak78.sh"
exec logs/controller/ctrl63/ab_leak78.sh
