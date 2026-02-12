#!/bin/bash
# N-gram sweep: grid search + init population for each context size
# Context=4 is the existing baseline (skip, already have results)
#
# Runs Phase 1 (grid search) for each context size to find the best
# (neurons, bits) config, then evaluates the init population on full data.
#
# Expected runtime: ~2-4 hours total (Phase 1 is fast, ~3-5 min each)

set -e

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

echo "=========================================="
echo "N-gram Sweep: BitwiseRAMLM Grid Search"
echo "Context sizes: 2, 3, 5, 6, 7, 8, 16"
echo "=========================================="
echo ""

for ctx in 2 3 5 6 7 8 16; do
	output="experiments/bitwise_ngram_ctx${ctx}.json"

	echo "──────────────────────────────────────"
	echo "Context size: ${ctx}  →  ${output}"
	echo "──────────────────────────────────────"

	# Run just Phase 1 (grid search) + create init population + full eval
	python -u run_bitwise_optimization.py \
		--context "${ctx}" \
		--rate 0.25 \
		--phase 1 \
		--top-k 3 \
		--output "${output}" \
		2>&1

	echo ""
	echo "Context ${ctx} done. Results in ${output}"
	echo ""
done

echo "=========================================="
echo "N-gram sweep complete!"
echo "Results in experiments/bitwise_ngram_ctx*.json"
echo "=========================================="
