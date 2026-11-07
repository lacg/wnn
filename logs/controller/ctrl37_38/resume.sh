#!/bin/bash
# Restores the posx2 queue after the mistaken drop (Luiz 07/11 "no, don't stop"): waits for the still-running
# C-S2-d_wposx2_leak999_full_s2 (pid 48553, its chain.sh was killed), writes its END line, then flies the posx2
# rateI and dhat arms, then writes "CTRL-37/38 posx2 DONE" (which starts combo.sh -> triple.sh).
cd /Users/lacg/wnn
L=logs/controller/ctrl37_38
eval "$(sed -n '/^row()/,/^}/p' $L/chain.sh)"
while kill -0 48553 2>/dev/null; do sleep 30; done
t=C-S2-d_wposx2_leak999_full_s2
r=$(row $L/$t.out)
echo "$(date -u +%FT%TZ) END $t (pid 48553) $r" >> $L/runs.log
grep -aq "stage-select\] HEADLINE stage" $L/$t.out && touch $L/$t.done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
for a in "rateI --obs-rate-i" "dhat --obs-cascade-dhat"; do
	set -- $a
	tag=C-S2-d_wposx2_${1}_full_s2; flag=$2
	[ -f $L/$tag.done ] && continue
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	echo "$(date -u +%FT%TZ) START $tag $flag (posx2 weights, 35 units)" >> $L/runs.log
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
		--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
		--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
		$flag --report-seeds $(echo {99990101..99990135}) \
		--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
	rc=$?
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
echo "$(date -u +%FT%TZ) CTRL-37/38 posx2 DONE" >> $L/runs.log
