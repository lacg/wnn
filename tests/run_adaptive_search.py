#!/usr/bin/env python3
"""
Run adaptive architecture search on WikiText-2 dataset.

This script uses genetic algorithm to evolve per-cluster (bits, neurons)
configurations for the AdaptiveClusteredRAM language model.

Uses subsampled data (200K train, 50K eval) for faster fitness evaluation.
Expected runtime: 1-2 hours.

For full evaluation after finding best genome, use the full dataset.
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnn import Logger
from wnn.tokenizers import TokenizerFactory, TokenizerType
from wnn.ram.strategies.connectivity.adaptive_cluster import (
	run_architecture_search,
	GenomeInitStrategy,
)


def load_wikitext2(logger: Logger, train_limit: int = None, val_limit: int = None):
	"""Load WikiText-2 dataset with GPT-2 tokenization."""
	from datasets import load_dataset

	logger("Loading WikiText-2 dataset...")
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	tokenizer = TokenizerFactory.create(TokenizerType.GPT2)
	vocab_size = tokenizer.vocab_size

	def tokenize_split(split_name: str, limit: int = None) -> list[int]:
		text = " ".join(dataset[split_name]["text"])
		tokens = tokenizer.encode(text)
		if limit:
			tokens = tokens[:limit]
		return tokens

	train_tokens = tokenize_split("train", train_limit)
	val_tokens = tokenize_split("validation", val_limit)

	logger(f"  Train: {len(train_tokens):,} tokens")
	logger(f"  Validation: {len(val_tokens):,} tokens")
	logger(f"  Vocab size: {vocab_size:,}")

	return train_tokens, val_tokens, vocab_size


def main():
	# Create logger
	logger = Logger("adaptive_search")

	start_time = time.time()
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	logger.header("Adaptive Architecture Search - WikiText-2")
	logger(f"Log file: {logger.log_file}")
	logger()

	# Load data with subsampling for faster architecture search
	# Full data is too slow (~10+ min per genome evaluation)
	# Subsampled data gives proxy fitness that correlates with full performance
	train_tokens, val_tokens, vocab_size = load_wikitext2(
		logger,
		train_limit=200_000,   # Subsample: 200K tokens (~10x faster)
		val_limit=50_000,      # Subsample: 50K tokens (~5x faster)
	)

	# Compute token frequencies for FREQUENCY_SCALED initialization
	logger()
	logger("Computing token frequencies...")
	counts = Counter(train_tokens)
	token_frequencies = [counts.get(i, 0) for i in range(vocab_size)]
	cluster_order = sorted(range(vocab_size), key=lambda t: -counts.get(t, 0))

	# Log frequency distribution
	freq_sorted = sorted(token_frequencies, reverse=True)
	logger(f"  Top 10 token freqs: {freq_sorted[:10]}")
	logger(f"  Tokens with freq > 1000: {sum(1 for f in freq_sorted if f > 1000):,}")
	logger(f"  Tokens with freq > 100: {sum(1 for f in freq_sorted if f > 100):,}")
	logger(f"  Tokens with freq > 0: {sum(1 for f in freq_sorted if f > 0):,}")

	# Run architecture search with settings for multi-hour run
	logger()
	logger("Starting architecture search...")
	logger("  Expected runtime: 1-2 hours (with subsampled data)")
	logger()

	result = run_architecture_search(
		train_tokens=train_tokens,
		eval_tokens=val_tokens,
		vocab_size=vocab_size,
		context_size=4,
		token_frequencies=token_frequencies,
		cluster_order=cluster_order,

		# GA parameters (tuned for ~1-2 hour run with subsampled data)
		population_size=10,      # Smaller population for faster iteration
		generations=50,          # Enough for convergence
		patience=10,             # Wait for improvement

		# Architecture bounds
		min_bits=4,              # Minimum addressable
		max_bits=20,             # Maximum for frequent tokens
		min_neurons=1,           # Single neuron minimum
		max_neurons=15,          # Reasonable upper bound
		phase=2,                 # Optimize both bits and neurons

		# Other settings
		init_strategy=GenomeInitStrategy.FREQUENCY_SCALED,
		empty_value=0.0,         # Best performing EMPTY value
		seed=42,
		logger=logger,           # Pass our Logger (callable)
	)

	# Calculate runtime
	elapsed = time.time() - start_time
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = int(elapsed % 60)

	# Save results
	output_dir = Path(__file__).parent.parent / "experiments"
	output_dir.mkdir(exist_ok=True)

	# Save genome as JSON
	genome_path = output_dir / f"adaptive_genome_{timestamp}.json"
	genome_data = {
		"timestamp": timestamp,
		"runtime_seconds": elapsed,
		"initial_fitness": result.initial_fitness,
		"final_fitness": result.final_fitness,
		"generations_run": result.generations_run,
		"early_stopped": result.early_stopped,
		"config": {
			"bits_per_cluster": result.best_genome.bits_per_cluster,
			"neurons_per_cluster": result.best_genome.neurons_per_cluster,
		},
		"stats": result.best_genome.stats(),
		"history": [(gen, best, avg) for gen, best, avg in result.history],
	}

	with open(genome_path, "w") as f:
		json.dump(genome_data, f, indent=2)

	# Print summary
	logger()
	logger.header("FINAL RESULTS")
	logger(f"  Runtime: {hours}h {minutes}m {seconds}s")
	logger(f"  Generations: {result.generations_run}")
	logger(f"  Early stopped: {result.early_stopped}")
	logger()
	logger(f"  Initial CE: {result.initial_fitness:.4f}")
	logger(f"  Final CE: {result.final_fitness:.4f}")
	improvement = (1 - result.final_fitness / result.initial_fitness) * 100
	logger(f"  Improvement: {improvement:.2f}%")
	logger()
	stats = result.best_genome.stats()
	logger(f"  Best genome:")
	logger(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	logger(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")
	logger(f"    Total memory: {stats['total_memory_cells']:,} cells")
	logger()
	logger(f"  Results saved to: {genome_path}")
	logger.separator("=")


if __name__ == "__main__":
	main()
