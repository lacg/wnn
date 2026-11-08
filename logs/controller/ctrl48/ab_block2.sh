#!/bin/bash
# CTRL-48 A/B block 2 (Luiz 08/11: "A/B for each change"): one change per arm on ABI 45, paired vs the SAME seed's
# C-S2-d_wposx2_a44_base_full_sN (ABI-45 flags off == ABI 44, pinned). Seeds s1..s3 seed-major, 35 CF2.1 units.
# Order: foundation (200-episode calibration) first, then the hidden-state blocks the CTRL-49 diagnosis points to, then B/C/A-variants.
until grep -q "BLOCK1 (base, rf) DONE" /Users/lacg/wnn/logs/controller/ctrl48/runs.log 2>/dev/null; do sleep 60; done
cd /Users/lacg/wnn
L=logs/controller/ctrl48
mkdir -p $L
row() # headline 35-unit row: runner-up HEADLINE-<g> MULTI-SEED, else the #0 winner's stage row
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -a "\[report-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//; s/--base-seed [0-9]+//')
ARMS=(
	"c200|--calib-episodes 200"
	"c200_rateI|--calib-episodes 200 --obs-rate-i"
	"c200_dres|--calib-episodes 200 --obs-cascade-dhat-residual"
	"c200_leak|--calib-episodes 200 --integral-leak 0.999"
	"dropconst|--drop-constant-features"
	"dropacc|--drop-features acc_x,acc_y"
	"c200_tmix|--calib-episodes 200 --calib-regime train_mix"
	"c200_centred|--calib-episodes 200 --calib-ladder centred"
	"c200_dhat|--calib-episodes 200 --obs-cascade-dhat"
)
for s in 1 2 3; do
	for a in "${ARMS[@]}"; do
		name=${a%%|*}; flags=${a#*|}
		tag=C-S2-d_wposx2_a45_${name}_full_s$s
		[ -f $L/$tag.done ] && continue
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		echo "$(date -u +%FT%TZ) START $tag ${flags:-(baseline)} (posx2, ABI 45, seed 3133700$s, 35 units)" >> $L/runs.log
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
			--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
			--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
			$flags --report-seeds $(echo {99990101..99990135}) \
			--base-seed 3133700$s --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
done
echo "$(date -u +%FT%TZ) BLOCK2 (encoding fixes A-D) DONE" >> $L/runs.log
