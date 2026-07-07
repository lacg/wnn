#!/bin/bash
# W2.3-L2 — train-under-weather, L2 arm (07/07/2026; plan w2_disturbances.md
# prescribed "L2 if L1 is free" — the L1 arm cleared its gate decisively:
# w23_pwm2k_L1_s09 ho-mem 93.5 under L1, fresh 90.2@L1 / 86.2 clean / 57.2@L2).
# THE question this arm answers (E5 go/no-go): can training UNDER L2 (3x bias,
# the zone where memoryless PD caps at 84% and stock PID needs its integrator)
# push a WNN past PD's 84 — i.e., does the GA discover integrator-like state
# when the training distribution DEMANDS it? L1-trained already transfers
# 57.2 to L2 (vs clean-trained 0.0), so the machinery is learnable in part.
# Cells: PWM2K recipe @2000 + --disturbance L2, 2 seeds, serial.
# Controller slot free at launch (W2.3-L1 marker fired 02:27Z) — no parking.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/W23WeatherL2_20260707.log
exec >>"$LOG" 2>&1

echo "[w23l2] $(date '+%Y-%m-%d %H:%M:%S') starting L2 weather training"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/W23WeatherL2_20260707
SEEDS="20260609 20260610"

run_one() {
	local name="$1" seed="$2"
	local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
	if [ -f "$dir/done.json" ]; then echo "[w23l2] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
	echo "[w23l2] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed}"
	$PY -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 --levels 16 --bits-per-feature 8 \
		--no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
		--neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
		--pop 24 --num-eval-folds 5 \
		--eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
		--rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
		--immigrants 0.15 --obs-pwm \
		--disturbance L2 \
		--fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
		--fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
		--report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
	if [ $? -ne 0 ]; then echo "[w23l2] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
	else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"disturbance\":\"L2\"}" > "$dir/done.json"
		echo "[w23l2] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

for seed in $SEEDS; do
	run_one "PWM2K_L2" "$seed"
done
echo "{\"w23_l2_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_w23_l2_done.json
echo "[w23l2] $(date '+%Y-%m-%d %H:%M:%S') DRIVER COMPLETE"
