#!/bin/bash
# CTRL-48 block 5 — the dres-failure fix ideas (Luiz 09/11; docs/ctrl_dres_failures.md, offline sizing
# docs/ctrl_dres_fix_offline_tests.md). Every arm = rf_c200_dres + ONE change, paired vs the SAME seed's
# C-S2-d_wposx2_a48_rf_c200_dres_full_sN (block 4's ABI 48 control). Same recipe/flags as ab_block4.sh.
#   oq02          #1  --threshold-outer-quantile 0.02 (offline: removes far/near aliasing on att-err, v_xy, cresid;
#                     ~+25% coarser near zero). The a45 oq02 arm in block 3 used the pre-fix scorer + ABI 45.
#   cent_oq02     #1  --calib-ladder centred + outer q 0.02 (offline: same de-aliasing at ~+10% coarser near zero)
#   ct30          #5  --threshold-calib-tilt 30 (closes the August tilt-5 vs tilt-30 question; offline weaker than #1)
#   tvxy045       #4  --train-init-vxy 0.45 (training-only initial sideways velocity; scored episodes unchanged)
# #3 (tail-aware selection) needs no GA run: --recalc-headline --select-tail on saved checkpoints.
# #2 (student refit) is NOT flown: the collector refuses this plant and phased_ga silently keeps the teacher ladder (bug).
# Waits 'BLOCK4 DONE' (aborts on 'BLOCK4 ABORT'). Seeds s1..s3 SEED-MAJOR, 35 CF2.1 units. ABI 48 (deployed by block 4).
cd /Users/lacg/wnn
L=logs/controller/ctrl48
VENV=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv
until grep -qE "BLOCK4 (DONE|ABORT)" $L/runs.log 2>/dev/null; do sleep 60; done
grep -q "BLOCK4 ABORT" $L/runs.log && { echo "$(date -u +%FT%TZ) BLOCK5 ABORT: block 4 aborted" >> $L/runs.log; exit 1; }
$VENV/bin/python -c "import ram_controller as c; assert c.ABI_VERSION == 48" || { echo "$(date -u +%FT%TZ) BLOCK5 ABORT: ram_controller is not ABI 48" >> $L/runs.log; exit 1; }
row() # headline 35-unit row: runner-up HEADLINE-<g> MULTI-SEED, else the #0 winner's stage row
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -a "\[report-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//; s/--base-seed [0-9]+//')
DRES="--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --obs-cascade-dhat-residual"
ARMS=(
	"rf_c200_dres_oq02|$DRES --threshold-outer-quantile 0.02"
	"rf_c200_dres_cent_oq02|$DRES --calib-ladder centred --threshold-outer-quantile 0.02"
	"rf_c200_dres_ct30|$DRES --threshold-calib-tilt 30"
	"rf_c200_dres_tvxy045|$DRES --train-init-vxy 0.45"
)
for s in 1 2 3; do
	for a in "${ARMS[@]}"; do
		name=${a%%|*}; flags=${a#*|}
		tag=C-S2-d_wposx2_a48_${name}_full_s$s
		[ -f $L/$tag.done ] && continue
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		echo "$(date -u +%FT%TZ) START $tag $flags (posx2, ABI 48, seed 3133700$s, 35 units)" >> $L/runs.log
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l $VENV/bin/python -u -m wnn.control.phased_ga $MA \
			--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
			--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
			$flags --report-seeds $(echo {99990101..99990135}) \
			--base-seed 3133700$s --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
echo "$(date -u +%FT%TZ) BLOCK5 DONE" >> $L/runs.log
