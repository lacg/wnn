#!/usr/bin/env bash
# Rerun of the 2 runs that crashed in the ABI-9 deploy window (L2s5 + L3s2).
set -uo pipefail
PROJ=/Users/lacg/wnn
cd "$PROJ"
source wnn/bin/activate
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}" WNN_CONTROLLER_GPU_EVAL=0 RAYON_NUM_THREADS=4
run() {
	local dir="$PROJ/logs/controller/hex_${1}_20260712"
	mkdir -p "$dir"
	echo "[hex-driver2] START $1 seed=$2 ($(date -u +%FT%TZ))"
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections \
		--neurons-gens 20 --neurons-patience 3 --memory-gens 40 --memory-patience 4 \
		--pop 20 --num-eval-folds 5 --check-interval 2 \
		--eval-episodes 30 --memory-eval-episodes 50 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 50 --holdout-pop-sample 4 \
		--base-seed "$2" --runs 1 --universe-episodes 8 \
		--geometry canted-hex --geometry-cant 20 \
		--geometry-tilt-err "$3" --geometry-pos-err "$4" --rotor-asym "$5" \
		> "$dir/run.out" 2>&1
	echo "[hex-driver2] END $1 rc=$? ($(date -u +%FT%TZ))"
}
run L2s5 31337005 5.0 0.008 0.10
run L3s2 31337002 10.0 0.015 0.20
printf '{"done":"%s","runs":["L2s5","L3s2"]}\n' "$(date -u +%FT%TZ)" > /tmp/wnn_hex_confirm_done.json
