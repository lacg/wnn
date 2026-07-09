#!/usr/bin/env bash
# Sequential phased-GA runs with the OPTIMAL-CONTROL teachers (Task #1: close the
# WNN→LQR gap). Runs teacher=lqr to completion, then teacher=mpc, using the EXACT
# locked C10 recipe from launch_controller_phased.sh (lines 30-38) so the results are
# directly comparable to the PID-teacher baseline (Lamarckian 3.76°/88°). Each run is
# memory-guarded vs the live IDS worker (wait for ≥45% free before starting — a prior
# concurrent controller run was jetsam-killed at a worker memory peak). Coexists at
# RAYON_NUM_THREADS=3. This driver runs the two phased_ga jobs in the FOREGROUND
# (blocking, sequential); launch IT detached via detach_launch.py.
set -uo pipefail

PROJ=/Users/lacg/wnn
BASE_SEED=31337002                         # same seed for lqr+mpc → seed-matched compare
STAMP=20260708
MARKER=/tmp/wnn_lqrmpc_phased_done.json
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=3
# CPU eval (09/07): the IDS worker owns the GPU (non-preemptible kernels starve the
# controller's command buffer for tens of minutes). Score on CPU to coexist without
# GPU-wait. Flip to 1 (or unset) once the controller has the GPU to itself.
export WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$PROJ/wnn/bin/activate"
cd "$PROJ"

mem_free() { memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1; }

guard() {   # wait for ≥45% free (check every 60s, up to 90 min, then proceed w/ warning)
	local tag="$1"
	for _ in $(seq 1 90); do
		local f; f=$(mem_free)
		if [ "${f:-0}" -ge 45 ]; then
			echo "[lqrmpc] $tag: memory clear (free=${f}%) — launching" >&2 ; return 0
		fi
		echo "[lqrmpc] $tag: waiting for memory (free=${f:-?}%, need ≥45%)…" >&2
		sleep 60
	done
	echo "[lqrmpc] $tag: guard timed out after 90min (free=$(mem_free)%) — proceeding anyway (RAYON=3 coexists)" >&2
}

run_teacher() {   # $1 = teacher name
	local teacher="$1"
	local name="c10_${teacher}_teacher_${STAMP}"
	local dir="$PROJ/logs/controller/${name}/seed0_base${BASE_SEED}"
	mkdir -p "$dir"
	guard "$teacher"
	echo "[lqrmpc] ===== START teacher=${teacher} $(date -u +%FT%TZ) -> ${dir}/run.out =====" >&2
	# EXACT C10 recipe (mirror of launch_controller_phased.sh 30-38) + --teacher.
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 5 --memory-gens 120 --memory-patience 8 \
		--pop 50 --num-eval-folds 5 --check-interval 5 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed "$BASE_SEED" --runs 1 \
		--teacher "$teacher" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	local rc=$?
	echo "[lqrmpc] ===== END teacher=${teacher} rc=${rc} $(date -u +%FT%TZ) =====" >&2
	echo "$dir/run.out"
}

echo "[lqrmpc] driver START $(date -u +%FT%TZ)  base_seed=${BASE_SEED}  teachers: lqr, mpc"
LQR_LOG=$(run_teacher lqr)
MPC_LOG=$(run_teacher mpc)
echo "[lqrmpc] driver END $(date -u +%FT%TZ)"
printf '{"done":"%s","lqr_log":"%s","mpc_log":"%s"}\n' "$(date -u +%FT%TZ)" "$LQR_LOG" "$MPC_LOG" > "$MARKER"
