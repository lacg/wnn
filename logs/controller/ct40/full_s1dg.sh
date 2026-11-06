#!/bin/bash
# Full pipeline (grid -> NEURONS -> MEMORY, report seeds) on the S1dg recipe — the real stage-2 headline
# (Luiz 06/11: plan step 2). Waits for the ct40 grid set to finish. Cascade, stage-2 plant, lag 0.20,
# delta law 0.7/0.6, gate off, MPCOF teacher, per-episode asym training, ABI 40, s31337002.
cd /Users/lacg/wnn
L=logs/controller/ct40
until grep -q "CT40 DONE" $L/runs.log 2>/dev/null; do sleep 30; done
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 30; done
MA=$(grep -v '^#' experiments/ctrl22/ARGS_ma_anchor_lag020.txt | tr '\n' ' ' | sed -E 's/--save-stage-checkpoints [^ ]+//g; s/--save-winner [^ ]+//g; s/--motor-lag-s [^ ]+//g; s/ --obs-pwm / /g')
tag=S1dg_ct40_full_s31337002
echo "$(date -u +%FT%TZ) START $tag (full pipeline)" >> $L/runs.log
PYTHONPATH=src/wnn nice -n 5 /usr/bin/time -l /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga $MA \
  --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off \
  --base-seed 31337002 --save-stage-checkpoints $L/full/ckpt_$tag --save-winner $L/full/${tag}_winner.yaml.gz > $L/full/$tag.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $tag rc=$rc $(grep -a 'HEADLINE.*MULTI-SEED' $L/full/$tag.out | tail -1 | cut -c1-260)" >> $L/runs.log
