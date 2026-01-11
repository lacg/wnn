#!/bin/bash
# Monitor five_tier_balanced and auto-start two_tier_20bit when complete

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

LOG_FILE="nohup_five_tier_balanced.out"

echo "[$(date)] Monitor started - waiting for five_tier_balanced to complete..."

while true; do
    # Check if experiment completed (look for final PPL line)
    if grep -q "Overall PPL:" "$LOG_FILE" 2>/dev/null; then
        RESULT=$(grep "Overall PPL:" "$LOG_FILE" | tail -1)
        echo "[$(date)] five_tier_balanced completed!"
        echo "$RESULT"
        break
    fi

    # Check if process died without completing
    if ! pgrep -f "five_tier_balanced" > /dev/null && ! pgrep -f "tiered.*50,27,21" > /dev/null; then
        # Process not running, check if it's in the log
        if grep -q "five_tier_balanced" "$LOG_FILE" 2>/dev/null; then
            # Check for errors
            if grep -q "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null; then
                echo "[$(date)] five_tier_balanced failed with error!"
                tail -20 "$LOG_FILE"
                exit 1
            fi
        fi
    fi

    sleep 600  # Check every 10 minutes
done

echo "[$(date)] Starting two_tier_20bit experiment with weekend mode (1000 gens/iters, patience 5)..."

PYTHONUNBUFFERED=1 python -u tests/ramlm_full_benchmark.py \
    --mode full \
    --full-data \
    --tiered "500,15,20;rest,5,8" \
    --context 4 \
    --optimize \
    --strategy GA,TS \
    --ga-gens 1000 \
    --ts-iters 1000 \
    --patience 5 \
    --per-tier
