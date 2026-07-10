#!/usr/bin/env bash
# PID-teacher SCREENING interleave (10/07/2026, Luiz order: "after the lqr and
# before mpc"). The driver run_lqr_mpc_phased.sh (bash, DRIVER_PID) launches MPC
# immediately after LQR, so it was SIGSTOPped while its LQR phased_ga child runs
# free (a stopped parent doesn't affect a running child). When LQR's phased_ga
# exits, THIS waiter runs the PID screening (exact SCREENING_p32 recipe → honest
# PID baseline + third ensemble member for the hybrid roadmap), then SIGCONTs
# the driver, which resumes exactly where it was: launching MPC (its own memory
# guard re-checks first). Nothing is ever killed.
# Launch DETACHED via scripts/detach_launch.py (PPID=1).
set -uo pipefail

DRIVER_PID="${1:?usage: pid_interleave_waiter.sh <driver_pid>}"
PROJ=/Users/lacg/wnn
BASE_SEED=31337002
DIR="$PROJ/logs/controller/c10_pid_teacher_20260710/seed0_base${BASE_SEED}_SCREENING_p32"
MARKER=/tmp/wnn_pid_screen_done.json

# GUARANTEE the driver is never left stopped, even if this script dies.
trap 'kill -CONT "$DRIVER_PID" 2>/dev/null' EXIT

# 1. Wait for the LQR phased_ga to finish (cmdline match, PID-reuse-proof).
echo "[pid-interleave] waiting for LQR phased_ga to finish ($(date -u +%FT%TZ))"
while pgrep -f "control.phased_ga.*--teacher lqr" >/dev/null; do sleep 60; done
echo "[pid-interleave] LQR phased_ga finished $(date -u +%FT%TZ)"

# 2. Memory guard (same ≥45%-free rule as the driver; 90 min cap).
mem_free() { memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1; }
for _ in $(seq 1 90); do
	f=$(mem_free)
	if [ "${f:-0}" -ge 45 ]; then break; fi
	echo "[pid-interleave] waiting for memory (free=${f:-?}%, need >=45%)"; sleep 60
done

# 3. PID screening — EXACT screening recipe (patience 3/2, check-2, RAYON=10,
# CPU-eval; mirrors c10_{lqr,mpc}_teacher SCREENING_p32 for seed-matched compare).
mkdir -p "$DIR"
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$PROJ/wnn/bin/activate"
cd "$PROJ"
echo "[pid-interleave] ===== START teacher=pid SCREENING $(date -u +%FT%TZ) -> $DIR/run.out ====="
python -u -m wnn.control.phased_ga \
	--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
	--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
	--neurons-gens 60 --neurons-patience 3 --memory-gens 120 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
	--base-seed "$BASE_SEED" --runs 1 \
	--teacher pid \
	--save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
	> "$DIR/run.out" 2>&1
rc=$?
echo "[pid-interleave] ===== END teacher=pid rc=$rc $(date -u +%FT%TZ) ====="
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%FT%TZ)" "$rc" "$DIR/run.out" > "$MARKER"

# 4. Resume the driver -> MPC launches next (trap also covers failure paths).
kill -CONT "$DRIVER_PID" 2>/dev/null && echo "[pid-interleave] driver $DRIVER_PID resumed -> MPC next"
