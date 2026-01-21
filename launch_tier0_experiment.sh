#!/bin/bash
# Wait for current experiment to finish, then launch tier0 bits_first experiment

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

# Find and wait for current python experiment
CURRENT_PID=$(ps aux | grep "run_coarse_fine_search.py\|run_phased_search.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$CURRENT_PID" ]; then
    echo "$(date): Waiting for PID $CURRENT_PID to finish..."
    while kill -0 "$CURRENT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "$(date): Previous experiment finished!"
else
    echo "$(date): No current experiment found, starting immediately..."
fi

# Wait a few seconds for cleanup
sleep 5

# Launch tier0 bits_first experiment
echo "$(date): Launching tier0 bits_first experiment..."
PYTHONUNBUFFERED=1 nohup python -u run_phased_search.py \
    --tier-config "100,15,20;400,10,12;rest,5,8" \
    --phase-order bits_first \
    --tier0-only \
    --ga-gens 100 \
    --ts-iters 200 \
    --patience 10 \
    --population 50 \
    --neighbors 50 \
    --context 4 \
    --train-tokens 200000 \
    --eval-tokens 50000 \
    --output experiments/tier0_bits_first_$(date +%Y%m%d_%H%M%S).json \
    > nohup_tier0.out 2>&1 &

NEW_PID=$!
echo "$(date): Started tier0 experiment with PID $NEW_PID"
echo "$(date): Output in nohup_tier0.out"
echo "$(date): Monitor with: tail -f nohup_tier0.out"
