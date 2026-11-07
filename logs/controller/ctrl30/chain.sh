#!/bin/bash
# CTRL-30 fitness-weight re-sweep on the CHOSEN architecture (Luiz 07/11: cascade, CTRL-27 closed).
# Recipe = C-S2-d_full_s2 exactly (stage-2 plant, lag 0.20, cascade CTBR @100 Hz, delta law 0.7/0.6, gate off,
# MPCOF teacher, BINARY, zscore, ABI 40, s31337002); only the 4 rank weights change (alt stays 0, decided 28/10).
# Control = C-S2-d_full_s2 itself (same seed + recipe, NOT re-flown): 97.4%/2.86°/1.64°/alt 0.186/pos 0.409.
# Design: one-at-a-time x2 upweight (renormalised) per term + pos dropped + equal weights. n=1 = direction;
# arms that beat the control holistically go to n=3 (s3, s4) paired with the control.
# Waits for CTRL-31 (score-only re-runs) to finish.
cd /Users/lacg/wnn
L=logs/controller/ctrl30
until grep -q "CTRL-31 DONE" logs/controller/ct40/runs.log 2>/dev/null; do sleep 60; done
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g')
#     tag      err_sq  stable steady pos   (control .28125/.225/.39375/.10)
ARMS=(
	"errx2    0.4390 0.1756 0.3073 0.0780"
	"stabx2   0.2296 0.3673 0.3214 0.0816"
	"steadyx2 0.2018 0.1614 0.5650 0.0717"
	"posx2    0.2557 0.2045 0.3580 0.1818"
	"pos0     0.3125 0.2500 0.4375 0.0000"
	"equal    0.2500 0.2500 0.2500 0.2500"
)
for a in "${ARMS[@]}"; do
	set -- $a
	tag=C-S2-d_w${1}_full_s2
	[ -f $L/$tag.done ] && continue
	echo "$(date -u +%FT%TZ) START $tag err_sq=$2 stable=$3 steady=$4 pos=$5" >> $L/runs.log
	PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
		--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
		--fit-weight-err-sq $2 --fit-weight-stable $3 --fit-weight-steady $4 --fit-weight-pos $5 \
		--base-seed 31337002 --save-stage-checkpoints $L/ckpt_$tag --save-winner $L/${tag}_winner.yaml.gz > $L/$tag.out 2>&1
	rc=$?
	# The canonical headline is the [stage-select] line (a #0 winner prints no HEADLINE-* MULTI-SEED line).
	echo "$(date -u +%FT%TZ) END $tag rc=$rc $(grep -a '\[stage-select\] HEADLINE' $L/$tag.out | tail -2 | tr '\n' ' ' | cut -c1-400)" >> $L/runs.log
	[ $rc -eq 0 ] && touch $L/$tag.done
done
echo "$(date -u +%FT%TZ) CTRL-30 ROUND A DONE" >> $L/runs.log
