#!/bin/bash
# CTRL-37 + CTRL-38 on the posx2 WEIGHTS (Luiz 07/11: "go with posx2 weights for CTRL-37 and CTRL-38").
# Control = C-S2-d_wposx2_full_s2 over 35 CF2.1 units: 83.1±26.3% / 3.67° / 2.20° / pos 0.431 (590/3500).
# Arms (same recipe + posx2 weights .2557/.2045/.3580/.1818, s31337002, ABI 43, reported over 35 units):
#   leak999 (--integral-leak 0.999) · rateI (--obs-rate-i) · dhat (--obs-cascade-dhat).
# The control-weights leak999 (done) and rateI (pid 99722, its waiter was stopped) stay as a bonus set vs the
# 82.2% control; this script first writes rateI's END line when that process exits.
cd /Users/lacg/wnn
L=logs/controller/ctrl37_38
mkdir -p $L
row() # $1 = .out — the headline's 35-unit row (runner-up: HEADLINE-<g>; #0 winner: its stage's pool[0] row)
{
	local g r
	g=$(grep -a 'stage-select\] HEADLINE stage' $1 | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	r=$(grep -a "HEADLINE-$g MULTI-SEED" $1 | tail -1)
	[ -z "$r" ] && r=$(grep -a "\[report-seeds\] ${g%%#*} MULTI-SEED" $1 | tail -1)
	echo "genome=$g $(echo "$r" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')"
}
while kill -0 99722 2>/dev/null; do sleep 30; done
R=logs/controller/ctrl38/C-S2-d_rateI_full_s2.out
echo "$(date -u +%FT%TZ) END C-S2-d_rateI_full_s2 (CONTROL weights) $(row $R)" >> logs/controller/ctrl38/runs.log
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
for a in "leak999 --integral-leak 0.999" "rateI --obs-rate-i" "dhat --obs-cascade-dhat"; do
	set -- $a
	name=$1; shift
	tag=C-S2-d_wposx2_${name}_full_s2
	[ -f $L/$tag.done ] && continue
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	echo "$(date -u +%FT%TZ) START $tag $* (posx2 weights, 35 units)" >> $L/runs.log
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
		--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
		--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
		"$@" --report-seeds $(echo {99990101..99990135}) \
		--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
echo "$(date -u +%FT%TZ) CTRL-37/38 posx2 DONE" >> $L/runs.log
