#!/bin/bash
# Rerun the granularity ablation's TERNARY + BINARY arms (14/07/2026). The first
# pass: QUAD rc=0 (kept); TERNARY rc=137 (SIGKILL mid-NEURONS — jetsam under a
# memory peak); BINARY rc=1 (odd levels_per_motor, fixed in db8a705e). This reruns
# only the two failed arms on the fixed code, with a pre-arm MEMORY GUARD (wait for
# ≥45% free, the run_lqr_mpc_phased.sh pattern) so a memory peak can't jetsam them.
# QUAD is NOT rerun (its rc=0 result stands). Writes the REAL granularity marker at
# the end (the first pass wrote it falsely despite 2 failures).
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

log() { echo "[gran-rerun] $1 $(date -u +%FT%TZ)"; }
mem_free() { memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1; }
guard() {   # wait for ≥45% free (check every 60s, up to 60 min, then proceed w/ warning)
	local tag="$1"
	for _ in $(seq 1 60); do
		local f; f=$(mem_free)
		if [ "${f:-0}" -ge 45 ]; then log "$tag: memory clear (free=${f}%) — launching"; return 0; fi
		log "$tag: waiting for memory (free=${f:-?}%, need ≥45%)…"; sleep 60
	done
	log "$tag: guard timed out (free=$(mem_free)%) — proceeding anyway"
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
		--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed 31337002 --runs 1 --teacher lqr \
		--memory-mode "$mode" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	log "===== END arm mode=$mode rc=$? ====="
}

run_arm TERNARY ternary
run_arm BINARY  binary

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"arms\": [\"quad(pass1)\", \"ternary(rerun)\", \"binary(rerun)\"], \"teacher\": \"lqr\", \"seed\": 31337002}" \
	> /tmp/wnn_granularity_ablation_done.json
log "GRANULARITY TERNARY+BINARY RERUN DONE (QUAD from pass 1)"
