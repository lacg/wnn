#!/usr/bin/env bash
# Effort-excess confirmation (Luiz mandate pattern): 3 more seeds of the
# octo-L2 effort-weighted screening, serial, CPU RAYON=4.
set -uo pipefail
PROJ=/Users/lacg/wnn
cd "$PROJ"
source wnn/bin/activate
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}" WNN_CONTROLLER_GPU_EVAL=0 RAYON_NUM_THREADS=4
for SEED in 31337003 31337004 31337005; do
	dir="$PROJ/logs/controller/octo_effort_s${SEED}_20260712"
	mkdir -p "$dir"
	echo "[effort-confirm] START seed=$SEED ($(date -u +%FT%TZ))"
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections \
		--neurons-gens 20 --neurons-patience 3 --memory-gens 40 --memory-patience 4 \
		--pop 20 --num-eval-folds 5 --check-interval 2 \
		--eval-episodes 30 --memory-eval-episodes 50 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.35 --fit-weight-stable 0.25 --fit-weight-jerk 0.15 \
		--fit-weight-mono 0.05 --fit-weight-effort 0.20 \
		--report-seed 99990101 --report-episodes 50 --holdout-pop-sample 4 \
		--base-seed "$SEED" --runs 1 --universe-episodes 8 \
		--geometry octo-x --geometry-tilt-err 5.0 --geometry-pos-err 0.008 --rotor-asym 0.10 \
		> "$dir/run.out" 2>&1
	echo "[effort-confirm] END seed=$SEED rc=$? ($(date -u +%FT%TZ))"
	grep -E "RESULT —|vs alloc-LQR  \(held" "$dir/run.out" | tail -2
done
printf '{"done":"%s"}\n' "$(date -u +%FT%TZ)" > /tmp/wnn_effort_confirm_done.json
