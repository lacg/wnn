#!/bin/bash
# CTRL-57 A1 waiter (Luiz 20/11 ~20:00 EST: "queue A1 soon after A2, adopting A2 or not"). Waits for the A2 block's
# exact end line, then flies ab_a1.sh. Gives up (no flight) if the A2 chain REFUSED or the A2 driver ABORTED.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
RUNS=$L/runs.log
END_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z A2 DONE$'
STOP_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z (A2 ABORT:|A2 CHAIN: REFUSED)'
say() { echo "$(date -u +%FT%TZ) A1Q: $*" >> $RUNS; }
say "armed (pid $$): waits for the A2 block's end line, then A1 s1-s5 (prereg docs/ctrl57_a1_prereg.md); stands down if A2 aborts or its chain refuses"
until grep -qE "$END_RE" $RUNS; do
	grep -qE "$STOP_RE" $RUNS && { say "A2 aborted or its chain refused; A1 NOT launched (main session decides)"; exit 1; }
	sleep 60
done
sleep 30
say "A2 block end line seen; launching ab_a1.sh"
exec bash $L/ab_a1.sh
