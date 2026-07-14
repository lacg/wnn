#!/usr/bin/env bash
# Launch a phased_ga controller run with the locked C10 recipe, detached (PPID=1).
# ONE place for the recipe — pass only the run-specific bits as args (no copy-paste).
#
# Usage: launch_controller_phased.sh <run_name> <base_seed> [extra phased_ga args...]
#   delta + H2 obs : launch_controller_phased.sh c10_delta_H2_tiltPI_20260618 31337002 --obs-tilt-p --obs-tilt-i
#   absolute       : launch_controller_phased.sh c10_absolute_20260618        31337002 --no-delta-control
#   plain delta    : launch_controller_phased.sh c10_delta_baseline_20260618  31337002
#
# Recipe = the magnitude-aware C10 config (grid 8/12/16 × 24/30, skip bits+conns,
# lamarckian, magnitude-aware patience, 100/200 eps, C10 weights, report-seed 99990101).
# Coexists with the XDS worker via RAYON_NUM_THREADS=3.
set -euo pipefail
if [ "$#" -lt 2 ]; then
	echo "usage: $0 <run_name> <base_seed> [extra phased_ga args...]" >&2
	exit 2
fi
RUN_NAME="$1"; BASE_SEED="$2"; shift 2
PROJ=/Users/lacg/wnn
DIR="$PROJ/logs/controller/${RUN_NAME}/seed0_base${BASE_SEED}"
mkdir -p "$DIR"
cd "$PROJ"
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$PROJ/wnn/bin/activate"
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=3
python "$PROJ/scripts/detach_launch.py" "$DIR/run.out" "$PROJ" -- \
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 5 --memory-gens 120 --memory-patience 8 \
		--pop 50 --num-eval-folds 5 --check-interval 5 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed "$BASE_SEED" --runs 1 \
		--save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
		"$@"
echo "launched run=$RUN_NAME base=$BASE_SEED extra=[$*] -> $DIR/run.out"
