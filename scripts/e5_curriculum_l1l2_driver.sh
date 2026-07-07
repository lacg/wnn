#!/bin/bash
# E5.2 — L1→L2 curriculum (07/07/2026). W2.3-L2 showed from-scratch L2 training
# FAILS (ho 2.8/16.5; L1-trained @L2 = 57.2 beats L2-trained @L2 = 19.5 by 3×)
# because under L2 the population never flies during search → no gradient to
# integral action. The fix Luiz proposed: seed the L2 GA with the L1 winner's
# FULL flying population (via phased_ga --seed-winner), so the search starts
# from a healthy rain-trained population (~57 @L2) and refines under the storm.
# Smoke (2-gen/20-ep): warm-started in-search stable 20→70% (vs from-scratch
# 0-9%), ho 47.5 @L2. This arm runs the full recipe.
#
# Cells: each seed fine-tunes from its OWN L1 winner (paired comparison):
#   s09 ← W23Weather_20260706/PWM2K_L1_seed20260609/winner.yaml.gz
#   s10 ← W23Weather_20260706/PWM2K_L1_seed20260610/winner.yaml.gz
# Grid skipped (architecture comes from the L1 winner); NEURONS + MEMORY under L2.
# Rulers @L2: from-scratch L2-trained = 19.5 (beat this = curriculum helps);
# L1-trained transfer = 57.2 (beat this = fine-tuning adds value); PD = 84
# (success — first learned controller past the memoryless ceiling in the
# integrator zone); PID+ = 99.8 (analytic ceiling).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/E5Curriculum_20260707.log
exec >>"$LOG" 2>&1

echo "[e5c] $(date '+%Y-%m-%d %H:%M:%S') starting L1→L2 curriculum"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/E5Curriculum_20260707
L1ROOT=logs/controller/W23Weather_20260706

run_one() {
	local name="$1" seed="$2" l1win="$3"
	local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
	if [ -f "$dir/done.json" ]; then echo "[e5c] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
	if [ ! -f "$l1win" ]; then echo "[e5c] $(date '+%H:%M:%S') MISSING L1 winner $l1win — skip ${name}"; return; fi
	echo "[e5c] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} ← $l1win"
	$PY -u -m wnn.control.phased_ga \
		--seed-winner "$l1win" \
		--disturbance L2 \
		--no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
		--neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
		--pop 24 --num-eval-folds 5 \
		--eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
		--rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
		--immigrants 0.15 --obs-pwm \
		--fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
		--fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
		--report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
	if [ $? -ne 0 ]; then echo "[e5c] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
	else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"disturbance\":\"L2\",\"curriculum\":\"L1\"}" > "$dir/done.json"
		echo "[e5c] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

run_one "CURRIC_L2" "20260609" "$L1ROOT/PWM2K_L1_seed20260609/winner.yaml.gz"
run_one "CURRIC_L2" "20260610" "$L1ROOT/PWM2K_L1_seed20260610/winner.yaml.gz"

echo "{\"e5c_done\":true,\"ts\":\"$(date -u +%FT%TZ)\"}" > /tmp/wnn_e5c_done.json
echo "[e5c] $(date '+%Y-%m-%d %H:%M:%S') DRIVER COMPLETE"
