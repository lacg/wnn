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

# GPU batched-train (marker-FSM Metal kernel) is opt-in via
# WNN_GPU_BATCHED_TRAIN=1. Enables single-dispatch GPU training for an
# entire batch of genomes, with parity-verified correctness. Disabled by
# default. Example: `WNN_GPU_BATCHED_TRAIN=1 ./start-worker.sh --tls`.
#
# WNN_OPTION_B is accepted as a backward-compatible alias.
#
# Only forward when set AND non-empty — Rust's `std::env::var().is_ok()`
# treats empty-string as "set" which previously enabled the path
# accidentally when callers passed-through unset values.
GPU_BATCHED="${WNN_GPU_BATCHED_TRAIN:-${WNN_OPTION_B:-}}"
if [ -n "${GPU_BATCHED}" ]; then
    nohup env WNN_GPU_BATCHED_TRAIN="${GPU_BATCHED}" python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) WNN_GPU_BATCHED_TRAIN=${GPU_BATCHED}"
else
    nohup python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) GPU batched-train: OFF (baseline path)"
fi
echo "Logs: tail -f worker.out"
