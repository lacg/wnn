#!/bin/bash
# Airframe-tail lever (b) (Luiz 07/11): the C-S2-d cascade recipe with --integral-leak 0.999 (the peraxis_i
# attitude-integral features remember ~1 s instead of ~0.1 s; cascade design doc risk R4). Same seed s31337002,
# control weights (.28125/.225/.39375/.10). Reported over 35 AIRFRAMES (report seeds 99990101..135) — compare with
# the control's 35-airframe re-score (logs/controller/ctrl30/rescore35/control.out, 82.2% from the probe).
# Waits for CTRL-30 round A.
cd /Users/lacg/wnn
L=logs/controller/ctrl30
until grep -q "CTRL-30 ROUND A DONE" $L/runs.log 2>/dev/null; do sleep 60; done
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
tag=C-S2-d_leak999_full_s2
[ -f $L/$tag.done ] && exit 0
echo "$(date -u +%FT%TZ) START $tag --integral-leak 0.999 (35 airframes)" >> $L/runs.log
PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
	--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
	--integral-leak 0.999 --report-seeds $(echo {99990101..99990135}) \
	--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $tag rc=$rc $(grep -a '\[stage-select\] HEADLINE' $L/$tag.out | tail -2 | tr '\n' ' ' | cut -c1-400)" >> $L/runs.log
[ $rc -eq 0 ] && touch $L/$tag.done
