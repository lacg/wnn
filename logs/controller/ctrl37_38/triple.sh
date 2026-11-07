#!/bin/bash
# CTRL-43 TRIPLE (Luiz 07/11): --integral-leak 0.999 + --obs-rate-i + --obs-cascade-dhat, posx2 weights, s31337002, ABI 43,
# reported over 35 CF2.1 units. Hypothesis: the two disturbance-memory routes are complementary (leak999 kept pos +
# best steady; rateI best stable/err but cost pos), so together they beat either alone. Pair vs posx2 83.1% and the
# posx2 single arms. Waits for the posx2 chain (ctrl37_38/chain.sh) to finish.
cd /Users/lacg/wnn
L=logs/controller/ctrl37_38
until [ -f $L/C-S2-d_wposx2_leak999_rateI_full_s2.done ] || grep -q "END C-S2-d_wposx2_leak999_rateI_full_s2" $L/runs.log 2>/dev/null; do sleep 60; done
eval "$(sed -n '/^row()/,/^}/p' $L/chain.sh)"
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
tag=C-S2-d_wposx2_leak999_rateI_dhat_full_s2
[ -f $L/$tag.done ] && exit 0
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) START $tag --integral-leak 0.999 --obs-rate-i --obs-cascade-dhat (posx2 weights, 35 units)" >> $L/runs.log
PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
	--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
	--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
	--integral-leak 0.999 --obs-rate-i --obs-cascade-dhat --report-seeds $(echo {99990101..99990135}) \
	--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
[ $rc -eq 0 ] && touch $L/$tag.done
