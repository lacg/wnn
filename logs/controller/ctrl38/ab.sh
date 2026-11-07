#!/bin/bash
# CTRL-38 A/B (Luiz 07/11: "both, as an A/B"): does a disturbance input close the unit-to-unit motor-wear tail?
# C-S2-d cascade recipe, s31337002, control weights, ABI 41. Arms: rateI (--obs-rate-i, firmware rate-PID I/i_limit)
# and dhat (--obs-cascade-dhat, the MPCOF observer's d-hat). Reported over 35 Crazyflie 2.1 units (99990101..135).
# Control = C-S2-d_full_s2 re-scored over the same 35 units: 82.2±25.4% / 3.71° / 2.29° (623/3500), identical under
# ABI 40 and 41. Waits for CTRL-37 (leak999) to finish.
cd /Users/lacg/wnn
L=logs/controller/ctrl38
until [ -f logs/controller/ctrl30/C-S2-d_leak999_full_s2.done ] || grep -q "END C-S2-d_leak999_full_s2" logs/controller/ctrl30/runs.log 2>/dev/null; do sleep 60; done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
for a in "rateI --obs-rate-i" "dhat --obs-cascade-dhat"; do
	set -- $a
	tag=C-S2-d_${1}_full_s2
	[ -f $L/$tag.done ] && continue
	while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
	echo "$(date -u +%FT%TZ) START $tag $2 (35 units)" >> $L/runs.log
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
		--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
		$2 --report-seeds $(echo {99990101..99990135}) \
		--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
	rc=$?
	# A runner-up headline prints its own HEADLINE-<genome> MULTI-SEED row; a #0 winner IS its stage's pool[0],
	# whose row is `[report-seeds] <STAGE> MULTI-SEED`.
	g=$(grep -a 'stage-select\] HEADLINE stage' $L/$tag.out | grep -oE 'genome=[^ ]+' | cut -d= -f2)
	row=$(grep -a "HEADLINE-$g MULTI-SEED" $L/$tag.out | tail -1)
	[ -z "$row" ] && row=$(grep -a "\[report-seeds\] ${g%%#*} MULTI-SEED" $L/$tag.out | tail -1)
	echo "$(date -u +%FT%TZ) END $tag rc=$rc genome=$g $(echo "$row" | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
echo "$(date -u +%FT%TZ) CTRL-38 AB DONE" >> $L/runs.log
