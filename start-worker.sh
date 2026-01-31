#!/bin/bash
# Start a single worker - prevents duplicates

cd "$(dirname "$0")"

# Check if worker already running
EXISTING=$(pgrep -f "wnn.ram.experiments.worker" | head -1)
if [ -n "$EXISTING" ]; then
    echo "Worker already running (PID $EXISTING)"
    echo "To restart: kill $EXISTING && ./start-worker.sh"
    exit 0
fi

# Start worker
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

echo "Starting worker..."
nohup python -u -m wnn.ram.experiments.worker --url http://localhost:3000 > worker.out 2>&1 &
echo "Worker started (PID $!)"
echo "Logs: tail -f worker.out"
