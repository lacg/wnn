#!/bin/bash
# Task #5 chain (13/07/2026): after the pid@31337004 winner recovery finishes,
# run (1) the ensemble RERUN on the full-patience winners, then (2) the three
# Phase-3 hybrid-teacher SCREENINGS sequentially (fast patience, seed 31337002,
# same recipe as teacher_seed_pairs_waiter.sh so all screenings are comparable).
# Markers: /tmp/wnn_ensemble_fulls_done.json after (1),
#          /tmp/wnn_task5_chain_done.json at the very end.
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

log() { echo "[task5-chain] $1 $(date -u +%FT%TZ)"; }

# ---- 0. wait for the pid@04 winner recovery (frees its 10 rayon threads) ----
log "waiting for /tmp/wnn_pid04_winner_recovery_done.json"
until [ -f /tmp/wnn_pid04_winner_recovery_done.json ]; do sleep 60; done
log "recovery marker present — starting ensemble rerun"

# ---- 1. ensemble RERUN on the FULL-patience winners (supersedes 11/07 scr run)
python -u scripts/ensemble_teachers.py \
  --winners \
    lqr="$LOGDIR/c10_lqr_teacher_20260708/seed0_base31337002/winner.yaml.gz" \
    mpc="$LOGDIR/c10_mpc_teacher_20260708/seed0_base31337002/winner.yaml.gz" \
    pid="$LOGDIR/c10_pid_teacher_20260710/seed0_base31337002/winner.yaml.gz" \
  --pairs --agg both --steps 1000 --episodes 100 \
  --seeds 99990001,99990101,12345,67890 \
  > "$LOGDIR/ensemble_fulls_$STAMP.log" 2>&1
rc=$?
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"rc\": $rc}" > /tmp/wnn_ensemble_fulls_done.json
log "ensemble rerun done rc=$rc"

# ---- 2. Phase-3 hybrid screenings (sequential; same recipe as seed-pairs) ----
run_hybrid() {  # $1 = dir tag, $2.. = extra phased_ga args
	local tag="$1"; shift
	local dir="$LOGDIR/c10_hyb_${tag}_$STAMP/seed_base31337002_SCREENING_p32"
	mkdir -p "$dir"
	log "===== START hybrid $tag -> $dir/run.out ====="
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed 31337002 --runs 1 \
		"$@" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	log "===== END hybrid $tag rc=$? ====="
}

# A. blended labels (top hypothesis: label-diversity regularizer)
run_hybrid "blend_lqr_pid"  --teacher lqr --teacher-blend "lqr,pid"
# B. curriculum: LQR rounds 0-3, PID rounds 4-7 (8 DAGGER rounds/train)
run_hybrid "curr_lqr2pid"   --teacher lqr --teacher-schedule "lqr,lqr,lqr,lqr,pid"
# C. warm-start: seed pop+memory from the LQR-full winner, evolve under PID labels
run_hybrid "warm_lqr2pid"   --teacher pid \
  --seed-winner "$LOGDIR/c10_lqr_teacher_20260708/seed0_base31337002/winner.yaml.gz"

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"runs\": [\"ensemble_fulls\", \"blend_lqr_pid\", \"curr_lqr2pid\", \"warm_lqr2pid\"]}" > /tmp/wnn_task5_chain_done.json
log "ALL TASK-5 CHAIN RUNS DONE"
