#!/bin/bash
# CTRL-48 PART 1 — the PRE-REGISTERED tail-aware selection test (docs/ctrl_select_tail_prereg.md, Luiz 10/11,
# approved to run 12/11). Re-ranks saved checkpoints with --recalc-headline; trains nothing, writes no checkpoint.
#   B  = 5 val units, no tail term (today's rule; must re-pick the original headline)   PRIMARY baseline
#   P  = --val-units 20 --select-tail p99                                                 PRIMARY rule
#   S20 = --val-units 20 alone ; S5p = --select-tail p99 alone                            descriptive only
# Each run's argv = its own .out [argv] line (rebuild_argv.py: report->test rename, --save-* dropped).
# Usage: select_tail_recalc.sh <run-list file, one tag per line> [rules, default "B P S20 S5p"]
# Progress: logs/controller/ctrl48/runs.log RECALC lines (k/N); outputs experiments/ctrl_select_tail/<tag>.<rule>.log
cd /Users/lacg/wnn
L=logs/controller/ctrl48
O=experiments/ctrl_select_tail
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
LIST=$1; RULES=${2:-"B P S20 S5p"}
[ -f "$LIST" ] || { echo "usage: $0 <run-list> [rules]"; exit 2; }
mkdir -p $O
$VENV/bin/python -c "import ram_controller as c; assert c.ABI_VERSION == 49" || { echo "$(date -u +%FT%TZ) SELECTTAIL ABORT: ram_controller is not ABI 49" >> $L/runs.log; exit 1; }
rule_flags() { case $1 in B) echo "";; P) echo "--val-units 20 --select-tail p99";; S20) echo "--val-units 20";; S5p) echo "--select-tail p99";; esac; }
TAGS=(); while IFS= read -r x; do TAGS+=("$x"); done < <(grep -vE '^[[:space:]]*(#|$)' $LIST)
N=$(( ${#TAGS[@]} * $(echo $RULES | wc -w) )); k=0
echo "$(date -u +%FT%TZ) SELECTTAIL START ${#TAGS[@]} runs x rules [$RULES] = $N re-picks (list $LIST)" >> $L/runs.log
for r in $RULES; do               # rule-major: every B and P lands before any descriptive secondary
	for t in "${TAGS[@]}"; do
		k=$((k+1)); out=$O/$t.$r.log
		grep -q '^\[recalc\] PICK' $out 2>/dev/null && continue
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		eval "set -- $($VENV/bin/python $L/rebuild_argv.py $L/$t.out recalc $L/ckpt_$t $(rule_flags $r))" || { echo "$(date -u +%FT%TZ) RECALC $k/$N $t $r ARGV-FAIL" >> $L/runs.log; continue; }
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) RECALC $k/$N $t $r rc=$rc $(grep -a '^\[recalc\] PICK' $out | sed -E 's/.*(headline=[^ ]+).*report\(([0-9]+) units\): (.*)/\1 \3/')" >> $L/runs.log
	done
done
$VENV/bin/python $L/select_tail_result.py $LIST && git add -f docs/ctrl_select_tail_result.md $O \
	&& git commit -q -m "CTRL-48 part 1: tail-aware selection prereg readout (docs/ctrl_select_tail_result.md)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exc8QwHhxZqBnX4jPhZ3aZ" -- docs/ctrl_select_tail_result.md $O && git push -q
echo "$(date -u +%FT%TZ) SELECTTAIL DONE (docs/ctrl_select_tail_result.md)" >> $L/runs.log
