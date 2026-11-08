#!/bin/bash
# Replication (Luiz 07/11, option A): seeds s3 + s4 for the whole posx2 set — baseline, leak999, rateI, dhat,
# leak999+rateI (CTRL-42), leak999+rateI+dhat (CTRL-43) — so every arm reaches n=3 with s2. SEED-MAJOR (all six on
# s3, then all six on s4) so each finished seed is a complete paired block. Same recipe, posx2 weights, ABI 43,
# reported over the same 35 CF2.1 units. Waits for the CTRL-43 triple (s2) to END.
cd /Users/lacg/wnn
L=logs/controller/ctrl37_38
until grep -q "END C-S2-d_wposx2_leak999_rateI_dhat_full_s2" $L/runs.log 2>/dev/null; do sleep 60; done
eval "$(sed -n '/^row()/,/^}/p' $L/chain.sh)"
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//; s/--base-seed [0-9]+//')
ARMS=(
	"base|"
	"leak999|--integral-leak 0.999"
	"rateI|--obs-rate-i"
	"dhat|--obs-cascade-dhat"
	"leak999_rateI|--integral-leak 0.999 --obs-rate-i"
	"leak999_rateI_dhat|--integral-leak 0.999 --obs-rate-i --obs-cascade-dhat"
)
for s in 3 4; do
	for a in "${ARMS[@]}"; do
		name=${a%%|*}; flags=${a#*|}
		[ "$name" = base ] && tag=C-S2-d_wposx2_full_s$s || tag=C-S2-d_wposx2_${name}_full_s$s
		[ -f $L/$tag.done ] && continue
		while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
		echo "$(date -u +%FT%TZ) START $tag ${flags:-(baseline)} (posx2, seed 3133700$s, 35 units)" >> $L/runs.log
		PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
			--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
			--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
			$flags --report-seeds $(echo {99990101..99990135}) \
			--base-seed 3133700$s --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
		[ $rc -eq 0 ] && touch $L/$tag.done
	done
	echo "$(date -u +%FT%TZ) REPLICATE seed s$s DONE" >> $L/runs.log
done
echo "$(date -u +%FT%TZ) REPLICATE DONE" >> $L/runs.log
