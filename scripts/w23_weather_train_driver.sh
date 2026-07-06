#!/bin/bash
# W2.3 — train-under-weather (06/07/2026, Luiz GO after the W2.2 audit).
# Motivation: EVERY clean-trained WNN scores 0% at L2 (where memoryless PD holds
# 84%) and loses 9-63pp at L1 (where PID/PD hold 100%) — weather shifts inputs
# off the trained distribution. W2.3 asks: does putting the calibrated weather
# IN the training loop teach the GA weather-robust cells (and/or integrator-like
# state)? Success gate (plan w2_disturbances.md): weather-trained beats
# clean-trained under weather WITHOUT losing the clean score.
# Cells: PWM2K recipe (the reproducible family) @2000 + --disturbance L1,
# 2 seeds. L2 arm deferred until the L1 readout (L2 may be untrainable — even
# PID+ holds only 27% at L3; PD 84% at L2 bounds what memoryless can do).
# Chained on /tmp/wnn_w22_done.json (ONE controller job at a time).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/W23Weather_20260706.log
exec >>"$LOG" 2>&1

echo "[w23] $(date '+%Y-%m-%d %H:%M:%S') WAITING for W2.2 (/tmp/wnn_w22_done.json)"
while [ ! -f /tmp/wnn_w22_done.json ]; do sleep 60; done
echo "[w23] $(date '+%Y-%m-%d %H:%M:%S') W2.2 done — starting weather training"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/W23Weather_20260706
SEEDS="20260609 20260610"

run_one() {
	local name="$1" seed="$2"
	local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
	if [ -f "$dir/done.json" ]; then echo "[w23] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
	echo "[w23] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed}"
	$PY -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 --levels 16 --bits-per-feature 8 \
		--no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
		--neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
		--pop 24 --num-eval-folds 5 \
		--eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
		--rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
		--immigrants 0.15 --obs-pwm \
		--disturbance L1 \
		--fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
		--fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
		--report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
	if [ $? -ne 0 ]; then echo "[w23] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
	else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"disturbance\":\"L1\"}" > "$dir/done.json"
		echo "[w23] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

for seed in $SEEDS; do
	run_one "PWM2K_L1" "$seed"
done
echo "{\"w23_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_w23_done.json
echo "[w23] $(date '+%Y-%m-%d %H:%M:%S') DRIVER COMPLETE"
