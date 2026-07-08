#!/bin/bash
# E5 MEMORY-ONLY L1→L2 (08/07/2026, Luiz's variant). E5.2 ran NEURONS→MEMORY under
# L2 and the NEURONS stage was its weak part (s09 arch-search drifted to a flat 15%
# roll → fresh-eval degraded to 29% @L2). This arm asks: does FREEZING the proven L1
# architecture and fine-tuning ONLY the memory (cells) under L2 beat the neurons+memory
# recipe? Uses the new `--seed-winner-stage memory` flag → skip NEURONS/BITS/CONNECTIONS,
# warm-start MEMORY from the L1 winner's spec + cells + FULL population under --disturbance L2.
#
# Cells: each seed fine-tunes from its OWN L1 winner (paired vs E5.2 neurons+memory):
#   s09 ← W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz
#   s10 ← W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz
# Rulers @L2 (FRESH-eval, apples-to-apples): from-scratch L2 = 19.5 (beat = curriculum
# helps); L1-transfer = 57.2 (beat = fine-tuning adds value); E5.2 neurons+memory fresh
# pooled = 46.1 (s09 29 / s10 63.2 — BEAT THIS = memory-only > neurons+memory); PD = 84
# (memoryless ceiling / success); PID+ = 99.8 (analytic ceiling). Smoke hit 57.5 @L2.
set -u
cd /Users/lacg/wnn
LOG=logs/controller/E5MemOnly_20260707.log
exec >>"$LOG" 2>&1

echo "[e5m] $(date '+%Y-%m-%d %H:%M:%S') starting L1→L2 MEMORY-ONLY (arch frozen)"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=8
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/E5MemOnly_20260707

run_one() {
	local name="$1" seed="$2" l1win="$3"
	local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
	if [ -f "$dir/done.json" ]; then echo "[e5m] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
	if [ ! -f "$l1win" ]; then echo "[e5m] $(date '+%H:%M:%S') MISSING L1 winner $l1win — skip ${name}"; return; fi
	echo "[e5m] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} ← $l1win (MEMORY-only)"
	$PY -u -m wnn.control.phased_ga \
		--seed-winner "$l1win" \
		--seed-winner-stage memory \
		--disturbance L2 \
		--no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
		--lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
		--check-interval 5 --memory-gens 15 --memory-patience 8 \
		--pop 24 --num-eval-folds 5 \
		--eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
		--rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
		--immigrants 0.15 --obs-pwm \
		--fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
		--fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
		--report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
	if [ $? -ne 0 ]; then echo "[e5m] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
	else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"disturbance\":\"L2\",\"mode\":\"memory_only\"}" > "$dir/done.json"
		echo "[e5m] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

run_one MEMONLY_L2 20260609 "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz"
run_one MEMONLY_L2 20260610 "logs/controller/W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz"

echo "{\"e5memonly_done\":true,\"ts\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"}" > /tmp/wnn_e5memonly_done.json
echo "[e5m] $(date '+%Y-%m-%d %H:%M:%S') ALL MEMORY-ONLY CELLS DONE"
