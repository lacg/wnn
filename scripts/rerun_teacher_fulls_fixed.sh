#!/bin/bash
# Teacher-fulls RERUN on the post-unification (fixed) code (Luiz order 14/07/2026).
# The 20260708/20260710 LQR/MPC/PID fulls ran on the pre-unification code (poisoned
# controller grid: single-winner seeding + CE ranking, fixed by WS1/WS1b). Re-run all
# three teacher fulls on current code so the teacher table + everything built on it
# (ensemble, hybrids) rests on confound-free winners.
#
# EXACT full-patience C10 recipe from run_lqr_mpc_phased.sh (neurons-patience 5,
# memory-patience 8, check-interval 5) — the ONLY differences vs the granularity
# screening are the patience/interval (full, not p32). New STAMP so the fresh
# winners never collide with the old-code dirs.
#
# CHAINS AFTER the granularity ablation (one controller at a time — waits for
# /tmp/wnn_granularity_ablation_done.json). CPU-pinned (WNN_CONTROLLER_GPU_EVAL=0),
# RAYON=10 — coexists with the GPU-bound IDS wave.
# Marker: /tmp/wnn_teacher_fulls_rerun_done.json at the end.
set -u

PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
BASE_SEED=31337002                 # seed-matched across teachers (and to the old fulls)
STAMP=20260714
LOGDIR="$PROJ/logs/controller"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

log() { echo "[teacher-fulls-rerun] $1 $(date -u +%FT%TZ)"; }

# ---- 0. one controller at a time: wait for the granularity ablation to finish ----
log "waiting for /tmp/wnn_granularity_ablation_done.json (one controller at a time)"
until [ -f /tmp/wnn_granularity_ablation_done.json ]; do sleep 120; done
log "granularity ablation done — starting teacher-fulls rerun"

run_teacher() {   # $1 = teacher (lqr|mpc|pid)
	local teacher="$1"
	local dir="$LOGDIR/c10_${teacher}_teacher_${STAMP}/seed0_base${BASE_SEED}"
	mkdir -p "$dir"
	log "===== START teacher=${teacher} -> ${dir}/run.out ====="
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 5 --memory-gens 120 --memory-patience 8 \
		--pop 50 --num-eval-folds 5 --check-interval 5 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed "$BASE_SEED" --runs 1 --teacher "$teacher" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	log "===== END teacher=${teacher} rc=$? ====="
}

run_teacher lqr
run_teacher mpc
run_teacher pid

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"teachers\": [\"lqr\", \"mpc\", \"pid\"], \"stamp\": \"$STAMP\", \"seed\": $BASE_SEED}" \
	> /tmp/wnn_teacher_fulls_rerun_done.json
log "ALL TEACHER-FULLS RERUN DONE"
