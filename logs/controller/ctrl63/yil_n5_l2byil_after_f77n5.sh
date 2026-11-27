#!/bin/bash
# CTRL-79 amendment 1 (Luiz 26/11/2026): after the CTRL-77 n=5 stage (s4-s5) finishes, fly yil s4-s5, then the stacking
# arm l2byil s1-s3, both via ab_yil79.sh (one controller at a time, ABI 63). Pure waits; 'Y79Q:' lines. Fails closed:
# only NEW runs.log lines (after arming) count, and any ABORT / stand-down stops it.
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
START=$(wc -l < $RUNS)
new() { tail -n +$((START + 1)) $RUNS; }
GO='^[0-9TZ:-]+ (F77 DONE seeds \[4 5\]|N5Q: CTRL-78 done, but no CTRL-77 arm meets)'
STOP='^[0-9TZ:-]+ (L78 ABORT|F77 ABORT|Y79 ABORT|N5Q: .*standing down)'
log "Y79Q: armed (pid $$) — after CTRL-77 n=5: yil s4-s5, then l2byil s1-s3 (ab_yil79.sh)"
while :; do
	new | grep -qE "$GO" && break
	if new | grep -qE "$STOP"; then log "Y79Q: an earlier block aborted or stood down — standing down, nothing started"; exit 1; fi
	sleep 60
done
log "Y79Q: CTRL-77 n=5 finished — starting yil s4-s5"
Y79_SEEDS="4 5" Y79_ARMS=yil bash logs/controller/ctrl63/ab_yil79.sh \
	|| { log "Y79Q: yil s4-s5 did not finish cleanly — l2byil NOT started"; exit 1; }
log "Y79Q: yil s4-s5 done — starting l2byil s1-s3"
Y79_SEEDS="1 2 3" Y79_ARMS=l2byil exec bash logs/controller/ctrl63/ab_yil79.sh
