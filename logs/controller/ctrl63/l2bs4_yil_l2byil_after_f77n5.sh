#!/bin/bash
# Replaces yil_n5_l2byil_after_f77n5.sh (27/11/2026): l2b s4 died on a transient Metal GPU error (rc=1, no result);
# its .out/ckpt are archived to ctrl48/void/ (precedent: N0 s9, Luiz 24/11 'a different tag just confuses') and it
# re-flies on its plain tag. After the CTRL-77 n=5 stage finishes: l2b s4 re-fly, then yil s4-s5, then l2byil s1-s3.
# Pure waits; 'Y79Q:' lines. Only NEW runs.log lines count; any ABORT / stand-down stops it.
cd /Users/lacg/wnn || exit 1
RUNS=logs/controller/ctrl48/runs.log
log() { echo "$(date -u +%FT%TZ) $*" >> $RUNS; }
START=$(wc -l < $RUNS)
new() { tail -n +$((START + 1)) $RUNS; }
GO='^[0-9TZ:-]+ F77 DONE seeds \[4 5\]'
STOP='^[0-9TZ:-]+ (F77 ABORT|Y79 ABORT|L78 ABORT)'
log "Y79Q: re-armed (pid $$) — after CTRL-77 n=5: l2b s4 RE-FLY (GPU error, void archived), then yil s4-s5, then l2byil s1-s3"
while :; do
	new | grep -qE "$GO" && break
	if new | grep -qE "$STOP"; then log "Y79Q: an earlier block aborted — standing down, nothing started"; exit 1; fi
	sleep 60
done
log "Y79Q: CTRL-77 n=5 finished — re-flying l2b s4"
F77_SEEDS=4 F77_ARMS=l2b bash logs/controller/ctrl63/ab_fact77.sh \
	|| { log "Y79Q: l2b s4 re-fly did not finish cleanly — yil / l2byil NOT started"; exit 1; }
log "Y79Q: l2b s4 done — starting yil s4-s5"
Y79_SEEDS="4 5" Y79_ARMS=yil bash logs/controller/ctrl63/ab_yil79.sh \
	|| { log "Y79Q: yil s4-s5 did not finish cleanly — l2byil NOT started"; exit 1; }
log "Y79Q: yil s4-s5 done — starting l2byil s1-s3"
Y79_SEEDS="1 2 3" Y79_ARMS=l2byil exec bash logs/controller/ctrl63/ab_yil79.sh
