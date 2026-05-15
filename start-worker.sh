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

# Option B (marker-FSM Metal training kernel) is opt-in via WNN_OPTION_B.
# Set to "1" to enable batched-genome GPU training for single- and multi-
# cluster flows. Disabled by default; set in env to override (e.g.
# `WNN_OPTION_B=1 ./start-worker.sh --tls`).
#
# BUG fixed 15/05/2026: previously passed `WNN_OPTION_B=""` when caller
# left it unset, which Rust's `std::env::var().is_ok()` treats as "set"
# and enabled Option B accidentally. Now we only forward the env var
# when caller has it set to non-empty.
if [ -n "${WNN_OPTION_B}" ]; then
    nohup env WNN_OPTION_B="${WNN_OPTION_B}" python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) WNN_OPTION_B=${WNN_OPTION_B}"
else
    nohup python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
    echo "Worker started (PID $!) WNN_OPTION_B=unset (baseline path)"
fi
echo "Logs: tail -f worker.out"
