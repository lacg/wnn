#!/bin/bash
# CTRL-48 fail-free STRESS TEST (Luiz 14/11/2026; plan docs/ctrl48_stress_test_plan.md): re-score each run's PUBLISHED
# controller on N test units (99990101..99990100+N, x100 episodes) instead of 35. No training: --recalc-headline under
# rule P re-picks the same genome (the parser REFUSES a run whose pick differs from the published headline), then scores
# it on the N units. Each report block also carries the MPCOF cascade row on the SAME episodes -> MPCOF fails for free.
# Usage: stress_test.sh <run-list> <n_units>.  Progress: runs.log 'STRESS k/N'. Ends 'STRESS DONE'.
cd /Users/lacg/wnn
L=logs/controller/ctrl48; O=experiments/ctrl_stress; VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
LIST=$1; U=$2
[ -f "$LIST" ] && [ -n "$U" ] || { echo "usage: $0 <run-list> <n_units>"; exit 2; }
PYTHONPATH=src/wnn $VENV/bin/python -c "import ram_controller as c, wnn.control._accel; assert c.ABI_VERSION == 50" \
	|| { echo "$(date -u +%FT%TZ) STRESS ABORT: ram_controller is not ABI 50" >> $L/runs.log; exit 1; }
TAGS=(); while IFS= read -r x; do TAGS+=("$x"); done < <(grep -vE '^[[:space:]]*(#|$)' $LIST)
N=${#TAGS[@]}; k=0
echo "$(date -u +%FT%TZ) STRESS START $N controllers x $U test units x 100 episodes, rule P pick (list $LIST)" >> $L/runs.log
for t in "${TAGS[@]}"; do
	k=$((k+1)); out=$O/$t.u$U.log
	grep -q '^\[recalc\] PICK' $out 2>/dev/null && continue
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	eval "set -- $($VENV/bin/python $L/rebuild_argv.py $L/$t.out stress $L/ckpt_$t $U --val-units 20 --select-tail p99)" \
		|| { echo "$(date -u +%FT%TZ) STRESS $k/$N $t ARGV-FAIL" >> $L/runs.log; continue; }
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) STRESS $k/$N $t rc=$rc $(grep -a '^\[recalc\] PICK' $out | sed -E 's/.*(headline=[^ ]+).*report\(([0-9]+) units\): (.*)/\1 units=\2 \3/')" >> $L/runs.log
done
echo "$(date -u +%FT%TZ) STRESS DONE ($N controllers x $U units)" >> $L/runs.log
