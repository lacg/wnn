#!/usr/bin/env bash
# #13: is the phase-2 GRID stage reproducible run-to-run?
#
# Evidence so far: grid point 5 (sn=16, b=24, BINARY, base-seed 31337002) scored
# CE=1.31 / 70% stable in the production phase-2 run, but CE=374 / 0% and
# CE=391 / 0% in two MallocStackLogging probes — and even "healthy" probe points
# sat 5-10x above production CE with nominally identical flags. MSL reshuffles
# allocation timing and therefore rayon interleaving; BINARY training runs
# WITHOUT WNN_ORDER_INDEPENDENT_TRAIN, so write order can differ. If identical
# reruns diverge HERE (no MSL, same seed, same wheel), grid results carry huge
# scheduling-injected variance and point 5 was never special.
#
# Three identical runs, production phase-2 recipe VERBATIM, each killed right
# after the 6-point grid table completes. One controller at a time (IDS first).
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
export WNN_STATE_SPLIT="${SPLIT:-1}"
PY="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$PY" ] || PY="python"

ARGS="--levels 16 --skip-stages bits,connections --lamarckian \
--saturation-grow-gain 1.0 --neurons-gens 60 --neurons-patience 3 \
--memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 \
--check-interval 2 --magnitude-aware-patience --eval-episodes 100 \
--memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 \
--fit-weight-mono 0.1 --report-seed 99990101 --report-episodes 100 \
--holdout-pop-sample 8 --base-seed 31337002 --runs 1 --teacher lqr \
--disturbance L2 --memory-mode BINARY --grid-state-neurons 8 12 16 \
--grid-bits 24 30 --max-state-neurons 24 --max-output-neurons 128"

RUNS="${RUNS:-3}"
TAG="${TAG:-}"
for i in $(seq 1 "$RUNS"); do
	OUT="/private/tmp/grid_det${TAG}_run${i}.out"
	: > "$OUT"
	echo "[grid-det] run $i starting $(date -u +%FT%TZ)"
	$PY -u -m wnn.control.phased_ga $ARGS > "$OUT" 2>&1 &
	PID=$!
	# Wait for the 6th grid line (or process death), then stop the run — the
	# GA stages are not under test.
	while kill -0 "$PID" 2>/dev/null && ! grep -q "\[grid  6/ 6\]" "$OUT"; do
		sleep 15
	done
	sleep 3
	kill -TERM "$PID" 2>/dev/null
	sleep 8
	kill -KILL "$PID" 2>/dev/null
	wait "$PID" 2>/dev/null
	echo "[grid-det] run $i done"
done

{
	echo "=== grid determinism probe: 3 identical runs, base-seed 31337002 ==="
	for i in $(seq 1 "$RUNS"); do
		echo "--- run $i ---"
		grep "\[grid" "/private/tmp/grid_det${TAG}_run${i}.out"
	done
	echo "--- production reference (prodab_dfa_split_blind_b31337002.out) ---"
	grep "\[grid" logs/controller/prodab_dfa_split_blind_b31337002.out 2>/dev/null
} > "/private/tmp/grid_det${TAG}_summary.txt"
echo done > "/tmp/wnn_grid_det${TAG}_done.marker"
