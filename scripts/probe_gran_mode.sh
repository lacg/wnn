#!/bin/bash
# probe_gran_mode.sh — INSTRUMENTED OOM probe, parameterized by memory mode.
# Same short 8x8 recipe as probe_gran_quad.sh but takes the mode as $1 so we can
# compare Σcells growth across QUAD_WEIGHTED / TERNARY / BINARY at matched gens.
# Hypothesis (Luiz, 14/07): TERNARY/BINARY store ~3x QUAD's cells (kill log:
# QUAD ~10GB vs TERNARY 33GB / BINARY 37GB) — a mode-specific write/fire bug.
# Runs UNDER controller_mem_watchdog.sh. ONE controller at a time.
#
# Usage: probe_gran_mode.sh <QUAD_WEIGHTED|TERNARY|BINARY> [sampler_csv]
set -u

MODE="${1:?usage: probe_gran_mode.sh <MODE> [sampler_csv]}"
case "$MODE" in
	QUAD_WEIGHTED) TAG=quad ;;
	TERNARY)       TAG=ternary ;;
	BINARY)        TAG=binary ;;
	*) echo "bad mode $MODE (QUAD_WEIGHTED|TERNARY|BINARY)"; exit 2 ;;
esac

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
DIR="$PROJ/logs/controller/c10_gran_${TAG}_PROBE_20260714/seed_base31337002_SCREENING_p32"
mkdir -p "$DIR"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

echo "[probe:$TAG] START $(date -u +%FT%TZ) -> $DIR/run.out"
python -u -m wnn.control.phased_ga \
	--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
	--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
	--neurons-gens 8 --neurons-patience 3 --memory-gens 8 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
	--base-seed 31337002 --runs 1 --teacher lqr \
	--memory-mode "$MODE" \
	--save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
	> "$DIR/run.out" 2>&1
RC=$?
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"rc\": $RC, \"mode\": \"$MODE\", \"kind\": \"gran_probe_${TAG}_8x8\"}" > /tmp/wnn_gran_probe_${TAG}_done.json
echo "[probe:$TAG] END rc=$RC $(date -u +%FT%TZ)"
