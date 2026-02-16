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

nohup python -u -m wnn.ram.experiments.worker --url "$URL" $EXTRA_ARGS > worker.out 2>&1 &
echo "Worker started (PID $!)"
echo "Logs: tail -f worker.out"
