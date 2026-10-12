#!/usr/bin/env bash
# Tilt-width rule stress read (CTRL-25, experiments/ctrl_tilt_width_rule.json): every _tw/_op31
# MEMORY winner, s31337002-05, scored at tilt 15 and 30 with thresholds held at the trained 5°.
cd /Users/lacg/wnn; export PYTHONPATH=/Users/lacg/wnn/src/wnn:
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
for t in 15 30; do for arm in op31 tw15 tw10 tw7; do
	out=experiments/tilt_stress/stress_${arm}_tilt${t}.json
	if python3 -c "import json,sys;sys.exit(0 if json.load(open('$out'))['meta']['complete'] else 1)" 2>/dev/null; then echo "skip $out"; continue; fi
	echo "[stress $(date -u +%FT%TZ)] $arm tilt $t"
	nice -n 5 $PY -u scripts/rescore_winners.py --glob "logs/controller/sweep_ladder/SL_C_b24n256_cf21_brushless_L4C_g10_s3133700[2-5]_${arm}_winner.yaml.gz" \
		--airframe cf21_brushless --translation --disturbance L4C --tilt $t --trained-tilt 5 --out $out
done; done
echo "[stress $(date -u +%FT%TZ)] STRESS DONE"
