#!/bin/bash
# CTRL-31 τ-mismatch (Luiz 07/11): the C-S2-d_full_s2 cascade (TRAINED at motor lag 0.20) re-scored on plants at
# other lags via --recalc-headline (score-only: rebuilds the stage candidates from the checkpoints, re-runs the val
# selection, reports on the 5 report seeds). 0.20 = control (must reproduce 97.4%/2.86°). Waits for the motor twin.
cd /Users/lacg/wnn
L=logs/controller/ct40
until grep -q "END M-S2-d_full_s2" $L/runs.log 2>/dev/null; do sleep 60; done
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g')
for T in 0.20 0.15 0.26 0.30 0.10; do
  tag=C-S2-d_tau${T}_s2
  echo "$(date -u +%FT%TZ) START $tag (CTRL-31 recalc, trained 0.20)" >> $L/runs.log
  PYTHONPATH=src/wnn nice -n 5 /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
    --motor-lag-s $T --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
    --base-seed 31337002 --recalc-headline $L/full/ckpt_C-S2-d_full_s2 > $L/full/$tag.out 2>&1
  echo "$(date -u +%FT%TZ) END $tag rc=$? $(grep -a 'HEADLINE.*MULTI-SEED' $L/full/$tag.out | tail -1 | grep -oE 'stable=[^ ]+  err=[^ ]+  steady=[^ ]+|alt=[^ ]+' | tr '\n' ' ')" >> $L/runs.log
done
echo "$(date -u +%FT%TZ) CTRL-31 DONE" >> $L/runs.log
