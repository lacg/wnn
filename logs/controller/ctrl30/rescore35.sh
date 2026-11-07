#!/bin/bash
# CTRL-30 re-score over 35 AIRFRAMES (Luiz 07/11). Each report seed = one random ±20% motor-asym airframe
# (evaluator.disturbance_stream); the 5 standard report seeds drew lenient airframes (control 97.4% there vs
# 82.2% over 35 — experiments/ctrl_airframe_tail/). Score-only: --recalc-headline on each run's checkpoints with
# the run's OWN weights (stage-select reuses the fitness calculator), --report-seeds 99990101..99990135.
# Polls the round-A .done markers; control first. The control must reproduce 82.2% (623/3500 fails, the probe).
cd /Users/lacg/wnn
L=logs/controller/ctrl30
mkdir -p $L/rescore35
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g; s/--report-seeds( [0-9]+)+//')
SEEDS=$(echo {99990101..99990135})
#     tag      ckpt_dir                                         err_sq  stable steady pos
RUNS=(
	"control  logs/controller/ct40/full/ckpt_C-S2-d_full_s2    0.28125 0.225  0.39375 0.1"
	"errx2    $L/ckpt_C-S2-d_werrx2_full_s2                    0.4390 0.1756 0.3073 0.0780"
	"stabx2   $L/ckpt_C-S2-d_wstabx2_full_s2                   0.2296 0.3673 0.3214 0.0816"
	"steadyx2 $L/ckpt_C-S2-d_wsteadyx2_full_s2                 0.2018 0.1614 0.5650 0.0717"
	"posx2    $L/ckpt_C-S2-d_wposx2_full_s2                    0.2557 0.2045 0.3580 0.1818"
	"pos0     $L/ckpt_C-S2-d_wpos0_full_s2                     0.3125 0.2500 0.4375 0.0000"
	"equal    $L/ckpt_C-S2-d_wequal_full_s2                    0.2500 0.2500 0.2500 0.2500"
)
ready() { [ "$1" = control ] || [ -f $L/C-S2-d_w${1}_full_s2.done ]; }
left=${#RUNS[@]}
while [ $left -gt 0 ]; do
	left=0
	for r in "${RUNS[@]}"; do
		set -- $r
		out=$L/rescore35/$1.out
		[ -f $out.done ] && continue
		if ! ready $1; then left=$((left + 1)); continue; fi
		echo "$(date -u +%FT%TZ) RESCORE35 START $1" >> $L/runs.log
		PYTHONPATH=src/wnn nice -n 5 /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
			--motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
			--fit-weight-err-sq $3 --fit-weight-stable $4 --fit-weight-steady $5 --fit-weight-pos $6 \
			--base-seed 31337002 --report-seeds $SEEDS --recalc-headline $2 > $out 2>&1
		rc=$?
		echo "$(date -u +%FT%TZ) RESCORE35 END $1 rc=$rc $(grep -a 'stage-select\] HEADLINE stage' $out | grep -oE 'genome=[^ ]+') $(grep -a 'HEADLINE.*MULTI-SEED' $out | tail -1 | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+|pos=[^ ]+|stable_fail=[^ ]+' | tr '\n' ' ')" >> $L/runs.log
		[ $rc -eq 0 ] && touch $out.done
	done
	[ $left -gt 0 ] && sleep 120
done
echo "$(date -u +%FT%TZ) RESCORE35 DONE" >> $L/runs.log
