#!/bin/bash
# CTRL-57+58 A1+A2 waiter (Luiz 21/11 ~07:40 EST: run A1+A2 after A1). Waits for the A1 block's
# exact end line, then flies ab_a1a2.sh. Gives up (no flight) if the A1 driver ABORTED.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
END_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z A1 DONE$'
STOP_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z A1 ABORT:'
say() { echo "$(date -u +%FT%TZ) A1A2Q: $*" >> $RUNS; }
say "armed (pid $$): waits for the A1 block's end line, then A1+A2 s1-s5; stands down if A1 aborts"
until grep -qE "$END_RE" $RUNS; do
	grep -qE "$STOP_RE" $RUNS && { say "A1 aborted; A1+A2 NOT launched (main session decides)"; exit 1; }
	sleep 60
done
sleep 30
say "A1 block end line seen; launching ab_a1a2.sh"
exec bash $L/ab_a1a2.sh
