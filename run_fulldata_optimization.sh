#!/bin/bash
# Full-data optimization: test if using ALL training/eval data during
# optimization reduces the overfitting gap seen in v2/v3.
#
# Key change from v3: --train-parts 1 --eval-parts 1
# This means EVERY genome evaluation uses ALL training data and ALL test data.
#
# Trade-off: ~36x slower per generation, so we use fewer generations.
# With 50 GA gens and 50 TS iters, this should run in ~12-18 hours.
#
# Expected result: the gap between sampled CE and full eval CE should shrink.

set -e

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

echo "=========================================="
echo "Full-Data Optimization: BitwiseRAMLM v4"
echo "train-parts=1, eval-parts=1 (NO sampling)"
echo "=========================================="
echo ""

python -u run_bitwise_optimization.py \
	--context 4 \
	--rate 0.25 \
	--train-parts 1 \
	--eval-parts 1 \
	--ga-gens 50 \
	--population 50 \
	--ts-iters 50 \
	--neighbors 30 \
	--patience 3 \
	--check-interval 5 \
	--output experiments/bitwise_fulldata_v4.json \
	2>&1

echo ""
echo "Full-data optimization complete!"
echo "Results in experiments/bitwise_fulldata_v4.json"
