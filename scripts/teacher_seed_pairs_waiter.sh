#!/usr/bin/env bash
# Multi-seed teacher screening pairs (Luiz order 11/07): after the PID-FULL run
# finishes, run 4 sequential SCREENING runs — LQR + PID at base seeds 31337003
# and 31337004 (interleaved teacher-first per feedback_sweeps_always_interleave)
# — to close the n=1 search-seed gap in the teacher comparison (screening trio
# @31337002: LQR 76%/3.93 > MPC 71%/4.06 >> PID 40%/6.63). One run at a time
# (max-2-heavy-runners rule), each memory-guarded. ~4-5h per run.
# Launch DETACHED via scripts/detach_launch.py (PPID=1).
set -uo pipefail

PIDFULL_WAITER="${1:?usage: teacher_seed_pairs_waiter.sh <pid_full_waiter_pid>}"
PROJ=/Users/lacg/wnn
STAMP=20260711
PIDFULL_MARKER=/tmp/wnn_pid_full_done.json
MARKER=/tmp/wnn_seed_pairs_done.json

echo "[seed-pairs] waiting for PID-FULL to finish ($(date -u +%FT%TZ))"
while [ ! -f "$PIDFULL_MARKER" ]; do
	if ! kill -0 "$PIDFULL_WAITER" 2>/dev/null && ! pgrep -f "control.phased_ga" >/dev/null; then
		echo "[seed-pairs] pid-full waiter gone without marker (crash path?) — proceeding anyway $(date -u +%FT%TZ)"
		break
	fi
	sleep 300
done
echo "[seed-pairs] PID-FULL chain finished $(date -u +%FT%TZ)"

mem_free() { memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1; }

guard() {
	for _ in $(seq 1 90); do
		local f; f=$(mem_free)
		if [ "${f:-0}" -ge 45 ]; then return 0; fi
		echo "[seed-pairs] waiting for memory (free=${f:-?}%, need >=45%)"; sleep 60
	done
	echo "[seed-pairs] memory guard timed out — proceeding anyway"
}

export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$PROJ/wnn/bin/activate"
cd "$PROJ"

run_screening() {   # $1 = teacher, $2 = base seed
	local teacher="$1" seed="$2"
	local dir="$PROJ/logs/controller/c10_${teacher}_teacher_${STAMP}/seed_base${seed}_SCREENING_p32"
	mkdir -p "$dir"
	guard
	echo "[seed-pairs] ===== START teacher=${teacher} seed=${seed} $(date -u +%FT%TZ) -> $dir/run.out ====="
	python -u -m wnn.control.phased_ga \
		--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
		--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
		--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
		--base-seed "$seed" --runs 1 \
		--teacher "$teacher" \
		--save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" \
		> "$dir/run.out" 2>&1
	echo "[seed-pairs] ===== END teacher=${teacher} seed=${seed} rc=$? $(date -u +%FT%TZ) ====="
}

# Interleaved: seed-matched pair at 03 first, then the pair at 04 — so a partial
# drain still yields one complete new LQR-vs-PID comparison.
run_screening lqr 31337003
run_screening pid 31337003
run_screening lqr 31337004
run_screening pid 31337004

printf '{"done":"%s","runs":["lqr@31337003","pid@31337003","lqr@31337004","pid@31337004"]}\n' \
	"$(date -u +%FT%TZ)" > "$MARKER"
echo "[seed-pairs] ALL 4 SCREENING RUNS DONE $(date -u +%FT%TZ)"
