#!/bin/bash
# CTRL-48 block 7 (Luiz approved 12/11): ONE arm x 3 seeds, paired vs the SAME seed's block-6 rf_c200_dc control
# (already flown on ABI 49 — NOT re-flown).
#   rf_c200_dc_biasin_ms01 = DC + --obs-att-bias-pred --write-rule vote --vote-share 0.1
#   Hypothesis: does combining the best-steady arm (biasin, 0.85° n=3) with the fewest-fails arm (ms01, 1/21000)
#   keep both gains?
# Recipe = the block-6 dc control's OWN argv (its .out [argv] line via rebuild_argv.py fly: report->test rename,
# new save paths) + the two stacked flags — identical to block 6 otherwise, --test-seeds 99990101..99990135.
# END lines in the block-6 format; ends 'BLOCK7 DONE' (or 'BLOCK7 ABORT: reason'). Seeds s1..s3, 35 CF2.1 units.
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
abort() { echo "$(date -u +%FT%TZ) BLOCK7 ABORT: $1" >> $L/runs.log; exit 1; }
$VENV/bin/python -c "import ram_controller as c; assert c.ABI_VERSION == 49" || abort "ram_controller is not ABI 49"
row() # headline 35-unit row: runner-up HEADLINE-<g> MULTI-SEED, else the #0 winner's stage row (old + new log tag)
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -aE "\[(test|report)-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
EXTRA="--obs-att-bias-pred --write-rule vote --vote-share 0.1"
for s in 1 2 3; do
	tag=C-S2-d_wposx2_a49_rf_c200_dc_biasin_ms01_full_s$s
	[ -f $L/$tag.done ] && continue
	argv=$($VENV/bin/python $L/rebuild_argv.py $L/C-S2-d_wposx2_a49_rf_c200_dc_full_s$s.out fly \
		$L/ckpt_$tag $L/${tag}_winner.yaml.gz $EXTRA) || abort "argv rebuild failed for $tag"
	echo "$argv" | grep -q -- "--base-seed 3133700$s " || abort "$tag: control argv is not seed 3133700$s"
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	echo "$(date -u +%FT%TZ) START $tag $EXTRA on the dc recipe (posx2, ABI 49, seed 3133700$s, 35 units)" >> $L/runs.log
	eval "set -- $argv"
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga "$@" > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
echo "$(date -u +%FT%TZ) BLOCK7 DONE" >> $L/runs.log
