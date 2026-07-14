#!/bin/bash
# probe_gran_quad.sh — INSTRUMENTED OOM probe (14/07/2026).
# Exact OOM recipe (pop=50, steps=2000, QUAD, lamarckian, folds=5, caps sn24/on128)
# but SHORT (neurons-gens 8, memory-gens 8) so we watch the memory climb quickly.
# The mem-sampler (started separately) records real-free/wired/ctrl-RSS/worker-RSS every
# 3s; the new Σcells log in generic_ga.py reports population-total written cells per gen.
# Goal: confirm RSS tracks Σcells (Lamarckian accumulation) before choosing a fix.
# Runs UNDER the existing controller_mem_watchdog.sh (5GB floor). Marker: wnn_gran_probe_done.json.
set -u

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
DIR="$PROJ/logs/controller/c10_gran_quad_PROBE_20260714/seed_base31337002_SCREENING_p32"
mkdir -p "$DIR"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

echo "[probe] START $(date -u +%FT%TZ) -> $DIR/run.out"
python -u -m wnn.control.phased_ga \
	--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
	--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
	--neurons-gens 8 --neurons-patience 3 --memory-gens 8 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
	--base-seed 31337002 --runs 1 --teacher lqr \
	--memory-mode QUAD_WEIGHTED \
	--save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
	> "$DIR/run.out" 2>&1
RC=$?
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"rc\": $RC, \"kind\": \"gran_probe_quad_8x8\"}" > /tmp/wnn_gran_probe_done.json
echo "[probe] END rc=$RC $(date -u +%FT%TZ)"
