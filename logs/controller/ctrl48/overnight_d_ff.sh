#!/bin/bash
# Overnight 18->19/11 (Luiz ~21:00 EST: "D is pre-approved; do everything I need to decide by the morning").
# After the CTRL-58 ABI 59 chain (PID $CHAIN_PID) ends:
#   1. FF block on base D, arm F only, s1-s10 (prereg docs/ctrl48_yawff_prereg.md Addendum 2) on whatever ABI is installed (59 if the gate passed, else 58);
#   2. in parallel, offline scoring of every n10 stage candidate (s6-s10, 6 lanes, nice 10), then the pooled n=10 verdict table
#      logs/controller/ctrl48/ymd_stage/ymd_table_n10.txt (docs/ctrl48_n10_ymd_prereg.md, fitness-only re-pick).
# Log lines go to runs.log as 'OVERNIGHT:'.
# ABI rule (Luiz 18/11 ~21:15 EST): the FF block flies on ABI 59. If the chain does NOT leave 59 installed (gate/test failure),
# HOLD: the scoring still runs, the FF waits for $GO (written by Claude at a tick): "59" after the fix is installed,
# or "58" only if the fix needs Luiz's answer.
cd /Users/lacg/wnn
L=logs/controller/ctrl48; S=$L/ymd_stage; RUNS=$L/runs.log
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
CHAIN_PID=${CHAIN_PID:-12515}
GO=$L/overnight_ff_go
LANES=${LANES:-6}
say() { echo "$(date -u +%FT%TZ) OVERNIGHT: $*" >> $RUNS; }

say "armed (pid $$): waiting for N10 DONE + the CTRL-58 chain (pid $CHAIN_PID) to end; then FF on base D (arm F, s1-s10) + n10 offline scoring"
until grep -q " N10 DONE" $RUNS && ! kill -0 $CHAIN_PID 2>/dev/null; do sleep 60; done
ABI=$($VENV/bin/python -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null)
CH=$(grep "CTRL58 CHAIN:" $RUNS | tail -1 | cut -c1-200)
say "chain ended; last chain line: ${CH#* CTRL58 CHAIN: }; installed controller ABI = $ABI"
case $ABI in 58|59) ;; *) say "ABORT: installed controller ABI '$ABI' is neither 58 nor 59 — nothing launched"; exit 1 ;; esac

# 2. offline scoring lanes (CPU scorer; independent of the flight, results are deterministic)
rm -f $S/lane_n10_*.sh
awk -v n=$LANES -v d=$S '{ print > (d "/lane_n10_" (NR % n) ".sh") }' $S/cmds_n10.txt
( for f in $S/lane_n10_*.sh; do
	( while read -r c; do o=$(echo "$c" | grep -oE -- '--out [^ ]+' | cut -d' ' -f2); [ -s "$o" ] || bash -c "$c"; done < $f ) &
  done; wait
  n=$(ls $S/scores/n10*_s*_*.npz 2>/dev/null | wc -l | tr -d ' ')
  say "n10 offline scoring done: $n/270 npz"
  PYTHONPATH=src/wnn $VENV/bin/python $S/ymd_table.py 1 2 3 4 5 6 7 8 9 10 > $S/ymd_table_n10.txt 2>&1 \
	&& say "n10 verdict table written: $S/ymd_table_n10.txt ($(grep -A3 'PREREG VERDICT' $S/ymd_table_n10.txt | grep 'D-Y' | tr -s ' ' | cut -c1-160))" \
	|| say "n10 verdict table FAILED (see $S/ymd_table_n10.txt)" ) &
SCORE=$!

# 1. FF block on base D, arm F, s1-s10 — on ABI 59; HOLD otherwise until $GO says which ABI
if [ "$ABI" != 59 ]; then
	say "HOLD: ABI 59 is NOT installed after the chain (ABI $ABI) — FF waits for the fix; Claude writes $GO = 59 (fixed) or 58 (fix needs Luiz)"
	until [ -s $GO ]; do sleep 60; done
	WANT=$(tr -d ' \n' < $GO)
	ABI=$($VENV/bin/python -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null)
	[ "$WANT" = "$ABI" ] || { say "ABORT: go file says ABI '$WANT' but ABI $ABI is installed — FF not launched"; wait $SCORE; exit 1; }
	say "go file: ABI $WANT"
fi
say "launching FF base D arm F s1-s10 on ABI $ABI"
ABI=$ABI FF_BASE=D FF_ARMS=F FF_SEEDS="1 2 3 4 5 6 7 8 9 10" bash $L/ab_yawff.sh
say "FF driver exited rc=$?"
wait $SCORE
say "done"
