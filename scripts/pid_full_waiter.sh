#!/usr/bin/env bash
# FULL-PATIENCE PID-teacher run, chained AFTER the LQR→PID-screening→MPC driver
# finishes (Luiz order 10/07: "arm it with full patience after the mpc").
# Waits for the driver's end marker (/tmp/wnn_lqrmpc_phased_done.json — written
# as its last act), with a fallback: if the driver is gone AND no controller
# phased_ga is running but the marker never appeared (crash path), proceed
# anyway after logging. Then memory-guard and run teacher=pid on the EXACT
# full-patience recipe of run_lqr_mpc_phased.sh (patience 5/8, check-5,
# RAYON=10, CPU-eval, seed 31337002) → seed-matched pair for LQR-full 86%/3.56°.
# Launch DETACHED via scripts/detach_launch.py (PPID=1). Max-2-heavy-runners
# rule holds: this starts only when the chain's controller slot is free.
set -uo pipefail

DRIVER_PID="${1:?usage: pid_full_waiter.sh <driver_pid>}"
PROJ=/Users/lacg/wnn
BASE_SEED=31337002
CHAIN_MARKER=/tmp/wnn_lqrmpc_phased_done.json
DIR="$PROJ/logs/controller/c10_pid_teacher_20260710/seed0_base${BASE_SEED}"
MARKER=/tmp/wnn_pid_full_done.json

echo "[pid-full] waiting for driver chain to finish ($(date -u +%FT%TZ))"
while [ ! -f "$CHAIN_MARKER" ]; do
	if ! kill -0 "$DRIVER_PID" 2>/dev/null && ! pgrep -f "control.phased_ga" >/dev/null; then
		echo "[pid-full] driver gone without marker (crash path?) — proceeding anyway $(date -u +%FT%TZ)"
		break
	fi
	sleep 120
done
echo "[pid-full] chain finished $(date -u +%FT%TZ)"

# Memory guard (same >=45%-free rule as the driver; 90 min cap).
mem_free() { memory_pressure 2>/dev/null | grep -i 'free perc' | grep -oE '[0-9]+' | head -1; }
for _ in $(seq 1 90); do
	f=$(mem_free)
	if [ "${f:-0}" -ge 45 ]; then break; fi
	echo "[pid-full] waiting for memory (free=${f:-?}%, need >=45%)"; sleep 60
done

# FULL-PATIENCE recipe — mirror of run_lqr_mpc_phased.sh run_teacher(), teacher=pid.
mkdir -p "$DIR"
export PYTHONPATH="$PROJ/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$PROJ/wnn/bin/activate"
cd "$PROJ"
echo "[pid-full] ===== START teacher=pid FULL-PATIENCE $(date -u +%FT%TZ) -> $DIR/run.out ====="
python -u -m wnn.control.phased_ga \
	--grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
	--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
	--neurons-gens 60 --neurons-patience 5 --memory-gens 120 --memory-patience 8 \
	--pop 50 --num-eval-folds 5 --check-interval 5 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 --steps 1000 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
	--base-seed "$BASE_SEED" --runs 1 \
	--teacher pid \
	--save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
	> "$DIR/run.out" 2>&1
rc=$?
echo "[pid-full] ===== END teacher=pid rc=$rc $(date -u +%FT%TZ) ====="
printf '{"done":"%s","rc":%d,"log":"%s"}\n' "$(date -u +%FT%TZ)" "$rc" "$DIR/run.out" > "$MARKER"
