#!/bin/bash
# Granularity ablation — FULL 3-arm rerun on the fixed+capped recipe (14/07/2026).
# Supersedes the steps=1000 pass 1 (QUAD ok but at the old steps; TERNARY jetsam;
# BINARY odd-levels). New recipe: --steps 2000 + --max-state-neurons 24
# --max-output-neurons 128 (the OOM fix, commit 2b1af957) so per-genome memory stays
# bounded. Fresh STAMP so nothing collides with the pass-1 dirs.
#
# Memory-guarded per arm; runs UNDER scripts/controller_mem_watchdog.sh (which
# auto-SIGKILLs on real-free-RAM floor). Marker: /tmp/wnn_granularity_ablation_done.json.
set -u

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
STAMP="20260714"
LOGDIR="$PROJ/logs/controller"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

log() { echo "[gran-all3] $1 $(date -u +%FT%TZ)"; }
real_free() { vm_stat 2>/dev/null | awk '/Pages free/{printf "%.1f",$3*16384/1073741824}'; }
guard() {   # wait for ≥12GB REAL free (vm_stat, not memory_pressure %), up to 60 min
	local tag="$1"
	for _ in $(seq 1 60); do
		local f; f=$(real_free)
		if [ "$(echo "${f:-0} >= 12" | bc 2>/dev/null)" = "1" ]; then log "$tag: real-free=${f}GB — launching"; return 0; fi
		log "$tag: waiting for memory (real-free=${f:-?}GB, need ≥12GB)…"; sleep 60
	done
	log "$tag: guard timed out (real-free=$(real_free)GB) — proceeding"
}

run_arm() {  # $1 = mode, $2 = tag
	local mode="$1" tag="$2"
	local dir="$LOGDIR/c10_gran_${tag}_$STAMP/seed_base31337002_SCREENING_p32"
	mkdir -p "$dir"
	guard "$tag"
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

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"arms\": [\"quad\", \"ternary\", \"binary\"], \"stamp\": \"$STAMP\", \"recipe\": \"steps2000_sn24_on128\", \"teacher\": \"lqr\", \"seed\": 31337002}" \
	> /tmp/wnn_granularity_ablation_done.json
log "ALL 3 GRANULARITY ARMS DONE (capped recipe)"
