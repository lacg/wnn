#!/bin/bash
# posx2 d-hat single arm (never ran on any weights), then unblocks combo.sh ("CTRL-37/38 posx2 DONE").
# Replaces the posx2 chain after Luiz dropped the leak999/rateI reruns (07/11).
cd /Users/lacg/wnn
L=logs/controller/ctrl37_38
eval "$(sed -n '/^row()/,/^}/p' $L/chain.sh)"
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
tag=C-S2-d_wposx2_dhat_full_s2
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) START $tag --obs-cascade-dhat (posx2 weights, 35 units)" >> $L/runs.log
PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
	--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
	--fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 \
	--obs-cascade-dhat --report-seeds $(echo {99990101..99990135}) \
	--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $tag rc=$rc $(row $L/$tag.out)" >> $L/runs.log
[ $rc -eq 0 ] && touch $L/$tag.done
echo "$(date -u +%FT%TZ) CTRL-37/38 posx2 DONE" >> $L/runs.log
