#!/bin/bash
# Post-reboot controller recovery (13/07/2026). The 09:38 macOS-update reboot
# killed the task-5 chain (blend+curr DONE on disk, warm interrupted) and the
# parked granularity chain (never started). This ONE self-contained script
# finishes everything remaining, sequentially, CPU-pinned (coexist with IDS):
#   1. warm hybrid (last task-5 piece)      -> /tmp/wnn_task5_chain_done.json
#   2. granularity ablation QUAD/TERN/BIN   -> /tmp/wnn_granularity_ablation_done.json
#   3. 6-member diverse committee eval       -> /tmp/wnn_committee_done.json
set -u

PROJ="/Users/lacg/wnn"
PY="$PROJ/wnn/bin/python"          # canonical venv (symlink -> volume venv; accel ABI6 + controller ABI12)
STAMP="20260713"
LOGDIR="$PROJ/logs/controller"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
cd "$PROJ"

log() { echo "[recovery] $1 $(date -u +%FT%TZ)"; }

# Shared screening recipe (seed 31337002 fast-patience) — identical to the
# teacher screenings + the pre-reboot chains, so all numbers stay comparable.
run_phased() {  # $1 = save dir, $2.. = extra args
	local dir="$1"; shift
	mkdir -p "$dir"
	"$PY" -u -m wnn.control.phased_ga \
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
}

# ---- 1. warm hybrid (only missing task-5 piece; blend+curr already on disk) --
WARM="$LOGDIR/c10_hyb_warm_lqr2pid_$STAMP/seed_base31337002_SCREENING_p32"
if [ -f "$WARM/winner.yaml.gz" ]; then
	log "warm hybrid already complete — skipping"
else
	log "===== START warm hybrid ====="
	run_phased "$WARM" --teacher pid \
		--seed-winner "$LOGDIR/c10_lqr_teacher_20260708/seed0_base31337002/winner.yaml.gz"
	log "===== END warm hybrid rc=$? ====="
fi
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"runs\": [\"blend\", \"curr\", \"warm\"]}" > /tmp/wnn_task5_chain_done.json
log "task-5 marker written"

# ---- 2. granularity ablation (QUAD / TERNARY / BINARY, --teacher lqr) --------
for arm in "QUAD_WEIGHTED:quad" "TERNARY:ternary" "BINARY:binary"; do
	mode="${arm%%:*}"; tag="${arm#*:}"
	dir="$LOGDIR/c10_gran_${tag}_$STAMP/seed_base31337002_SCREENING_p32"
	if [ -f "$dir/winner.yaml.gz" ]; then log "gran $tag already done — skipping"; continue; fi
	log "===== START gran arm mode=$mode ====="
	run_phased "$dir" --teacher lqr --memory-mode "$mode"
	log "===== END gran arm mode=$mode rc=$? ====="
done
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"arms\": [\"quad\", \"ternary\", \"binary\"], \"teacher\": \"lqr\", \"seed\": 31337002}" > /tmp/wnn_granularity_ablation_done.json
log "granularity marker written"

# ---- 3. 6-member diverse committee (teacher x format x seed) -----------------
# ternary/binary members were just produced by stage 2; verify before including.
G="$LOGDIR"
COMMITTEE=(
	"lqr=$G/c10_lqr_teacher_20260708/seed0_base31337002/winner.yaml.gz"
	"mpc=$G/c10_mpc_teacher_20260708/seed0_base31337002/winner.yaml.gz"
	"pid=$G/c10_pid_teacher_20260710/seed0_base31337002/winner.yaml.gz"
	"lqr04=$G/c10_lqr_teacher_20260711/seed_base31337004_SCREENING_p32/winner.yaml.gz"
	"ternary=$G/c10_gran_ternary_$STAMP/seed_base31337002_SCREENING_p32/winner.yaml.gz"
	"binary=$G/c10_gran_binary_$STAMP/seed_base31337002_SCREENING_p32/winner.yaml.gz"
)
members=()
for m in "${COMMITTEE[@]}"; do [ -f "${m#*=}" ] && members+=("$m") || log "committee member MISSING, skipping: ${m%%=*}"; done
log "===== START committee eval (${#members[@]} members) ====="
"$PY" -u scripts/ensemble_teachers.py --winners "${members[@]}" \
	--agg both --steps 1000 --episodes 100 --seeds 99990001,99990101,12345,67890 \
	> "$LOGDIR/committee_6member_$STAMP.log" 2>&1
log "===== END committee eval rc=$? ====="
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"members\": ${#members[@]}}" > /tmp/wnn_committee_done.json
log "ALL POST-REBOOT CONTROLLER RECOVERY DONE"
