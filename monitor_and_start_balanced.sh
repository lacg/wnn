#!/bin/bash
# Monitor five_tier_gradient and auto-start five_tier_balanced when complete
# Using direct run (not sweep) to ensure weekend mode parameters work

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

echo "[$(date)] Monitor started - waiting for five_tier_gradient to complete..."

# Wait for completion signal in nohup output
while true; do
    if grep -q "five_tier_gradient" nohup_five_tier.out && grep -q "Completed in" nohup_five_tier.out; then
        echo "[$(date)] five_tier_gradient completed!"
        grep "Overall PPL" nohup_five_tier.out | tail -1
        break
    fi
    sleep 60
done

echo "[$(date)] Starting five_tier_balanced experiment with weekend mode (1000 gens/iters, patience 5)..."

# Run directly (not via sweep) to ensure parameters work
PYTHONUNBUFFERED=1 python -u tests/ramlm_full_benchmark.py \
    --mode full \
    --full-data \
    --tiered "50,27,21;450,23,20;10000,5,12;10000,5,10;rest,5,8" \
    --context 8 \
    --optimize \
    --strategy GA,TS \
    --ga-gens 1000 \
    --ts-iters 1000 \
    --patience 5 \
    --per-tier \
    > nohup_five_tier_balanced.out 2>&1

echo "[$(date)] five_tier_balanced experiment completed!"
echo "Results:"
tail -30 nohup_five_tier_balanced.out
