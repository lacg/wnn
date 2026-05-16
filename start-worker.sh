#!/bin/bash
# Start a single worker - prevents duplicates
# Usage: ./start-worker.sh [--tls]

cd "$(dirname "$0")"

# Parse args
USE_TLS=false
for arg in "$@"; do
    case $arg in
        --tls) USE_TLS=true ;;
    esac
done

# Check if worker already running
EXISTING=$(pgrep -f "wnn.ram.experiments.worker" | head -1)
if [ -n "$EXISTING" ]; then
    echo "Worker already running (PID $EXISTING)"
    echo "To restart: kill $EXISTING && ./start-worker.sh"
    exit 0
fi

# Start worker
source wnn/bin/activate
# NOTE: Do NOT add src/wnn to PYTHONPATH - it shadows HuggingFace's tokenizers package

if [ "$USE_TLS" = true ]; then
    URL="https://localhost:3000"
    EXTRA_ARGS="--no-ssl-verify"
    echo "Starting worker with TLS..."
else
    URL="http://localhost:3000"
    EXTRA_ARGS=""
    echo "Starting worker..."
fi

# Default behavior (15/05/2026):
#   - B11 affinity routing: always on (per-batch CPU/GPU decision).
#   - B12 hybrid CPU+GPU split: on by default (set WNN_HYBRID=0 to disable).
#   - B13 per-shape adaptive state: always on.
#
# WNN_GPU_BATCHED_TRAIN env var (and legacy WNN_OPTION_B alias) is now an
# OVERRIDE, not a switch:
#   - unset (default): B11 affinity decides per batch (the right thing)
#   - "off"/"0"/"false": force CPU baseline always
#   - "force"/"always": force GPU always (debug/benchmarking)
#
# Only forward when set AND non-empty — Rust's `std::env::var().is_ok()`
# treats empty-string as "set" which previously enabled the path
# accidentally when callers passed-through unset values.
# EXPERIMENT (16/05/2026): B12 hybrid speed_ratio cap 1.5 → 3 (middle ground).
# Prior 1.5 (default) almost never fires; 10 (over-relaxed) fires even at extreme
# disparity and showed +14-19% regression at 125n/175n shapes. Try 3 to allow
# hybrid for moderate disparity while falling back to GPU-only for extreme cases.
# WNN_GPU_BATCHED_TRAIN_TRACE=1 logs each B11/B12/B14 decision so we can SEE when
# hybrid actually engages.
export WNN_HYBRID_SPEED_RATIO="${WNN_HYBRID_SPEED_RATIO:-3}"
export WNN_GPU_BATCHED_TRAIN_TRACE="${WNN_GPU_BATCHED_TRAIN_TRACE:-1}"

GPU_BATCHED="${WNN_GPU_BATCHED_TRAIN:-${WNN_OPTION_B:-}}"
if [ -n "${GPU_BATCHED}" ]; then
    nohup env WNN_GPU_BATCHED_TRAIN="${GPU_BATCHED}" WNN_HYBRID_SPEED_RATIO="${WNN_HYBRID_SPEED_RATIO}" WNN_GPU_BATCHED_TRAIN_TRACE="${WNN_GPU_BATCHED_TRAIN_TRACE}" python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) — GPU batched-train OVERRIDE: ${GPU_BATCHED} | HYBRID_SPEED_RATIO=${WNN_HYBRID_SPEED_RATIO} | TRACE=${WNN_GPU_BATCHED_TRAIN_TRACE}"
else
    nohup env WNN_HYBRID_SPEED_RATIO="${WNN_HYBRID_SPEED_RATIO}" WNN_GPU_BATCHED_TRAIN_TRACE="${WNN_GPU_BATCHED_TRAIN_TRACE}" python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) — B11 affinity routing + B12 hybrid (defaults) | HYBRID_SPEED_RATIO=${WNN_HYBRID_SPEED_RATIO} | TRACE=${WNN_GPU_BATCHED_TRAIN_TRACE}"
fi
echo "Logs: tail -f worker.out"
