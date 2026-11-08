#!/bin/bash
# CTRL-48 block 2 RE-BASED ON RECEPTIVE FIELDS (Luiz 08/11): every arm = rf (axis, cross 0.2) + ONE change, paired vs the SAME seed's C-S2-d_wposx2_a44_rf_full_sN; rank on err/steady (stable saturated). Order = the validity-audit ranking. Waits for block 1b.
#   base = posx2 cascade recipe, ABI 44 (bank-segment neurogenesis), all new flags off. This is the pairing
#          control for every later arm: ABI-45 flags off is pinned bit-identical to ABI 44.
#   rf   = base + --receptive-fields axis --rf-cross 0.2 (CTRL-45, family from output bank)
# New seed set s1..s3 (base seeds 31337001..003), SEED-MAJOR, reported over the 35 CF2.1 units.
cd /Users/lacg/wnn
until grep -q "BLOCK1b (rf-cross dose) DONE" logs/controller/ctrl48/runs.log 2>/dev/null; do sleep 60; done
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
	"rf_dropconst|--receptive-fields axis --rf-cross 0.2 --drop-constant-features"
	"rf_c200|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200"
	"rf_c200_leak|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --integral-leak 0.999"
	"rf_c200_rateI|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --obs-rate-i"
	"rf_c200_dres|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --obs-cascade-dhat-residual"
	"rf_dropacc|--receptive-fields axis --rf-cross 0.2 --drop-features acc_x,acc_y"
	"rf_b32|--receptive-fields axis --rf-cross 0.2 --grid-bits 32"
	"rf_c200_tmix|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --calib-regime train_mix"
	"rf_c200_centred|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --calib-ladder centred"
	"rf_c200_dhat|--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --obs-cascade-dhat"
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
echo "$(date -u +%FT%TZ) BLOCK2rf DONE" >> $L/runs.log
