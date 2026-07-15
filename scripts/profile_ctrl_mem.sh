#!/bin/bash
# profile_ctrl_mem.sh MODE POP [safety_gb] — profile controller memory PER STAGE at a
# tiny population, to tell whether the TERNARY balloon scales with pop (per-genome cell
# cost) or is a FIXED allocation (a bug). Same 30-bit config that ballooned. Detached,
# self-killing at safety_gb so it can never thrash the box alongside the IDS worker.
set -u
MODE="${1:?usage: profile_ctrl_mem.sh MODE POP [safety_gb]}"
POP="${2:?need POP}"
SAFETY="${3:-18}"
PROJ="/Users/lacg/wnn"
VENV="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv"
DIR="$PROJ/scratchpad/memprofile_${MODE}_p${POP}"
mkdir -p "$DIR"

export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=8 WNN_CONTROLLER_GPU_EVAL=0
unset CONDA_PREFIX || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJ"

python -u -m wnn.control.phased_ga \
	--grid-state-neurons 16 --grid-bits 30 --levels 16 \
	--skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
	--neurons-gens 2 --neurons-patience 1 --memory-gens 2 --memory-patience 1 \
	--pop "$POP" --num-eval-folds 2 --check-interval 1 --magnitude-aware-patience \
	--eval-episodes 60 --memory-eval-episodes 60 --steps 1000 --max-state-neurons 24 --max-output-neurons 128 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 15 --holdout-pop-sample 3 \
	--base-seed 31337002 --runs 1 --teacher lqr --memory-mode "$MODE" \
	> "$DIR/run.out" 2>&1 &
PG=$!

echo "ts,rss_gb,stage" > "$DIR/mem.csv"
PEAK=0; PEAKSTAGE="?"
while kill -0 "$PG" 2>/dev/null; do
	R=$(ps -o rss= -p "$PG" 2>/dev/null | awk '{print $1+0}')
	CH=$(pgrep -P "$PG" | head -1); CR=$(ps -o rss= -p "${CH:-0}" 2>/dev/null | awk '{print $1+0}')
	GB=$(echo "scale=2;(${R:-0}+${CR:-0})/1048576" | bc)
	STAGE=$(grep -oE "STAGE [0-9]|grid +[0-9]+/|Building population|Re-eval streaming|MEMORY|NEURONS" "$DIR/run.out" 2>/dev/null | tail -1 | tr -d '\n')
	echo "$(date +%s),${GB},${STAGE}" >> "$DIR/mem.csv"
	if (( $(echo "${GB} > ${PEAK}" | bc -l) )); then PEAK="$GB"; PEAKSTAGE="$STAGE"; fi
	if (( $(echo "${GB} > ${SAFETY}" | bc -l) )); then echo "[profile] SAFETY-KILL at ${GB}GB (stage=$STAGE)" >> "$DIR/run.out"; kill -9 "$PG" 2>/dev/null; break; fi
	sleep 2
done
RC=$?
echo "{\"mode\":\"$MODE\",\"pop\":$POP,\"peak_gb\":$PEAK,\"peak_stage\":\"$PEAKSTAGE\",\"safety_gb\":$SAFETY}" > "$DIR/peak.json"
echo "[profile] DONE mode=$MODE pop=$POP peak=${PEAK}GB @ ${PEAKSTAGE}" >> "$DIR/run.out"
