#!/usr/bin/env python3
"""Quick comparison: how does train_parts affect CE/Acc for ctx=2, 200n, 18b?

Tests train_parts={1, 4, 10, 20} with eval_parts=1 (full eval).
No optimization — just trains one uniform config and evaluates.
"""

import sys
import time

# Load data once
from run_bitwise_optimization import load_wikitext2_tokens
train_tokens, test_tokens, val_tokens, vocab_size = load_wikitext2_tokens(logger=lambda m: sys.stdout.write(f"{m}\n") or sys.stdout.flush())

from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

CONTEXT = 2
NEURONS = 200
BITS = 18
RATE = 0.25
NUM_CLUSTERS = 16
TOTAL_INPUT_BITS = CONTEXT * NUM_CLUSTERS

train_parts_list = [1, 4, 10, 20]

print(f"\ntrain={len(train_tokens)}, eval(test)={len(test_tokens)}")
print(f"Config: ctx={CONTEXT}, neurons={NEURONS}, bits={BITS}, rate={RATE}")
print(f"{'train_parts':>12} {'train_examples':>15} {'CE':>8} {'Acc':>8} {'Time':>8}")
print("-" * 60)
sys.stdout.flush()

for tp in train_parts_list:
	evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=CONTEXT,
		neurons_per_cluster=NEURONS,
		bits_per_neuron=BITS,
		num_parts=tp,
		num_eval_parts=1,  # always full eval
		memory_mode=2,  # QUAD_WEIGHTED
		neuron_sample_rate=RATE,
	)

	genome = ClusterGenome.create_uniform(
		num_clusters=NUM_CLUSTERS,
		bits=BITS,
		neurons=NEURONS,
		total_input_bits=TOTAL_INPUT_BITS,
	)

	t0 = time.time()
	results = evaluator.evaluate_batch([genome])
	elapsed = time.time() - t0

	ce, acc = results[0]
	train_examples = len(train_tokens) // tp
	print(f"{tp:>12} {train_examples:>15,} {ce:>8.4f} {acc*100:>7.2f}% {elapsed:>7.1f}s")
	sys.stdout.flush()

print("\nDone.")
