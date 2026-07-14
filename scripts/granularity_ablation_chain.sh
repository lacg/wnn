#!/bin/bash
# Task #9 — controller granularity ablation (QUAD vs TERNARY vs BINARY), armed
# 13/07/2026 (Luiz approved). Chains AFTER the task-5 hybrid chain finishes.
#
# Isolates the memory-cell FORMAT: three arms differ ONLY by --memory-mode, all
# on the identical seed-31337002 fast-patience LQR-teacher screening recipe (the
# same recipe as teacher_seed_pairs_waiter.sh / the task-5 hybrids, so numbers
# are directly comparable to the teacher screening table). All three RE-RUN on
# current code — the pre-existing 20260708 QUAD screening predates today's
# mode-awareness (52ad3de9) + saver (9788d1df) commits, so it is NOT reused as
# the QUAD baseline (confound-free).
#
# CPU-pinned (WNN_CONTROLLER_GPU_EVAL=0) to coexist with the IDS/GPU wave; the
# split-trainer GPU path is bit-exact anyway (parity-proven incl. T/B).
# Markers: /tmp/wnn_granularity_ablation_done.json at the end.
set -u

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
STAMP="20260713"
LOGDIR="$PROJ/logs/controller"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

log() { echo "[granularity] $1 $(date -u +%FT%TZ)"; }

# ---- 0. wait for the task-5 hybrid chain to finish (frees the 10 threads) ----
log "waiting for /tmp/wnn_task5_chain_done.json"
until [ -f /tmp/wnn_task5_chain_done.json ]; do sleep 120; done
log "task-5 chain done — starting granularity ablation"

run_arm() {  # $1 = mode (QUAD_WEIGHTED|TERNARY|BINARY), $2 = short tag
	local mode="$1" tag="$2"
	local dir="$LOGDIR/c10_gran_${tag}_$STAMP/seed_base31337002_SCREENING_p32"
	mkdir -p "$dir"
	log "===== START arm mode=$mode -> $dir/run.out ====="
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed 31337002 --runs 1 --teacher lqr \
		--memory-mode "$mode" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	log "===== END arm mode=$mode rc=$? ====="
}

run_arm QUAD_WEIGHTED quad
run_arm TERNARY       ternary
run_arm BINARY        binary

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"arms\": [\"quad\", \"ternary\", \"binary\"], \"teacher\": \"lqr\", \"seed\": 31337002}" > /tmp/wnn_granularity_ablation_done.json
log "ALL GRANULARITY ABLATION ARMS DONE"
