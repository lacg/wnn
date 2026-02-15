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
import math
from datetime import datetime
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnn import Logger
from wnn.tokenizers import TokenizerFactory, TokenizerType
from wnn.ram.strategies.connectivity.adaptive_cluster import (
	run_architecture_search,
	run_architecture_tabu_search,
	run_connectivity_optimization,
	evaluate_genome_with_accuracy,
	GenomeInitStrategy,
)
from wnn.ram.core.reporting import OptimizationResultsTable


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

	# Common parameters
	seed = int(time.time())
	context_size = 4
	min_bits = 8
	max_bits = 25
	min_neurons = 3
	max_neurons = 33
	empty_value = 0.0

	# =========================================================================
	# Phase 1a: GA for Architecture Search
	# =========================================================================
	logger()
	logger.header("Phase 1a: GA Architecture Search")
	logger("  Expected runtime: 3-5 hours")
	logger()

	ga_result = run_architecture_search(
		train_tokens=train_tokens,
		eval_tokens=val_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		token_frequencies=token_frequencies,
		cluster_order=cluster_order,

		# GA parameters
		population_size=30,      # Diverse population for better exploration
		generations=50,          # Enough for convergence
		patience=10,             # Wait for improvement

		# Architecture bounds
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=2,                 # Optimize both bits and neurons

		# Other settings
		init_strategy=GenomeInitStrategy.FREQUENCY_SCALED,
		empty_value=empty_value,
		seed=seed,
		logger=logger,
	)

	phase1a_elapsed = time.time() - start_time
	phase1a_ce = ga_result.final_fitness
	logger(f"[Phase 1a] Complete in {phase1a_elapsed/3600:.1f}h, CE={phase1a_ce:.4f}")

	# =========================================================================
	# Phase 1b: Tabu Search for Architecture Refinement
	# =========================================================================
	logger()
	logger.header("Phase 1b: TS Architecture Refinement")
	logger()

	ts_result = run_architecture_tabu_search(
		initial_genome=ga_result.best_genome,
		initial_fitness=ga_result.final_fitness,
		train_tokens=train_tokens,
		eval_tokens=val_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		cluster_order=cluster_order,

		# TS parameters
		iterations=100,          # Local search iterations
		neighbors_per_iter=30,   # Neighbors to explore (consistent with GA pop)
		patience=15,             # More patience for refinement

		# Architecture bounds (same as GA)
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=2,

		# Other
		empty_value=empty_value,
		seed=seed + 1000,
		logger=logger,

		# Population seeding: use GA's final population as initial neighbors
		initial_neighbors=ga_result.final_population,
	)

	phase1b_elapsed = time.time() - start_time
	phase1b_ce = ts_result.final_fitness
	phase1_improvement = (1 - phase1b_ce / ga_result.initial_fitness) * 100
	logger(f"[Phase 1b] Complete at {phase1b_elapsed/3600:.1f}h total, CE={phase1b_ce:.4f}")
	logger(f"[Phase 1] Total architecture improvement: {phase1_improvement:.2f}%")

	# =========================================================================
	# Phase 2: Connectivity Optimization
	# =========================================================================
	logger()
	logger.header("Phase 2: Connectivity Optimization")
	logger()

	conn_result = run_connectivity_optimization(
		genome=ts_result.best_genome,
		genome_fitness=ts_result.final_fitness,
		train_tokens=train_tokens,
		eval_tokens=val_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		cluster_order=cluster_order,
		token_frequencies=token_frequencies,

		# GA parameters (Phase 2a)
		ga_population=20,
		ga_generations=30,
		ga_patience=5,

		# TS parameters (Phase 2b)
		ts_iterations=50,
		ts_neighbors=30,
		ts_patience=5,

		# Architecture bounds (same as Phase 1)
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=2,

		# Other
		empty_value=empty_value,
		seed=seed + 2000,
		logger=logger,

		# Population seeding: use TS's final population as initial population
		initial_population=ts_result.final_population,
	)

	# =========================================================================
	# Final Results
	# =========================================================================
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
		# Phase 1a: GA
		"phase1a": {
			"initial_fitness": ga_result.initial_fitness,
			"final_fitness": ga_result.final_fitness,
			"generations": ga_result.iterations_run,
			"early_stopped": ga_result.early_stopped,
		},
		# Phase 1b: TS
		"phase1b": {
			"initial_fitness": ts_result.initial_fitness,
			"final_fitness": ts_result.final_fitness,
			"iterations": ts_result.iterations_run,
			"early_stopped": ts_result.early_stopped,
		},
		# Phase 2: GA→TS Architecture Refinement
		"phase2": {
			"phase1b_baseline": conn_result.initial_fitness,
			"after_ga": conn_result.phase2_baseline,
			"after_ts": conn_result.final_fitness,
			"ga_improvement_pct": conn_result.ga_improvement_pct,
			"ts_improvement_pct": conn_result.ts_improvement_pct,
			"total_improvement_pct": conn_result.total_improvement_pct,
			"ga_generations": conn_result.ga_iterations,
			"ts_iterations": conn_result.ts_iterations,
		},
		# Final genome (from Phase 2 TS)
		"genome": {
			"bits_per_neuron": conn_result.final_population[0].bits_per_neuron if conn_result.final_population else ts_result.best_genome.bits_per_neuron,
			"neurons_per_cluster": conn_result.final_population[0].neurons_per_cluster if conn_result.final_population else ts_result.best_genome.neurons_per_cluster,
		},
		"stats": ts_result.best_genome.stats(),
		# History from all phases
		"ga_history": ga_result.history,  # [(iteration, best_fitness)]
		"ts_history": ts_result.history,  # [(iteration, best_fitness)]
	}

	with open(genome_path, "w") as f:
		json.dump(genome_data, f, indent=2)

	# Print summary
	logger()
	logger.header("FINAL RESULTS (All Phases)")
	logger(f"  Total Runtime: {hours}h {minutes}m {seconds}s")
	logger()

	# Phase 1a results
	logger("  Phase 1a (GA Architecture):")
	logger(f"    Generations: {ga_result.iterations_run}")
	logger(f"    Initial CE: {ga_result.initial_fitness:.4f}")
	logger(f"    Final CE: {ga_result.final_fitness:.4f}")
	ga_improvement = (1 - ga_result.final_fitness / ga_result.initial_fitness) * 100
	logger(f"    Improvement: {ga_improvement:.2f}%")
	logger()

	# Phase 1b results
	logger("  Phase 1b (TS Architecture Refinement):")
	logger(f"    Iterations: {ts_result.iterations_run}")
	logger(f"    Input CE: {ts_result.initial_fitness:.4f}")
	logger(f"    Output CE: {ts_result.final_fitness:.4f}")
	ts_improvement = (1 - ts_result.final_fitness / ts_result.initial_fitness) * 100
	logger(f"    Improvement: {ts_improvement:.2f}%")
	logger()

	# Phase 2 results (GA→TS)
	logger("  Phase 2a (GA Architecture Refinement):")
	logger(f"    Generations: {conn_result.ga_iterations}")
	logger(f"    Input CE: {conn_result.initial_fitness:.4f}")
	logger(f"    Output CE: {conn_result.phase2_baseline:.4f}")
	logger(f"    Improvement: {conn_result.ga_improvement_pct:.2f}%")
	logger()

	logger("  Phase 2b (TS Architecture Refinement):")
	logger(f"    Iterations: {conn_result.ts_iterations}")
	logger(f"    Input CE: {conn_result.phase2_baseline:.4f}")
	logger(f"    Output CE: {conn_result.final_fitness:.4f}")
	logger(f"    Improvement: {conn_result.ts_improvement_pct:.2f}%")
	logger()

	logger(f"  Total Phase 2 improvement: {conn_result.total_improvement_pct:.2f}%")
	logger()

	# Best Architecture stats (from Phase 2 TS best genome)
	best_genome = conn_result.final_population[0] if conn_result.final_population else ts_result.best_genome
	stats = best_genome.stats()
	logger("  Best Architecture (after all phases):")
	logger(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	logger(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")
	logger(f"    Total memory: {stats['total_memory_cells']:,} cells")
	logger()

	# Comparison table using unified OptimizationResultsTable (with accuracy)
	logger()
	comparison = OptimizationResultsTable("Validation")
	comparison.add_stage("Initial (FREQUENCY_SCALED)", ce=ga_result.initial_fitness, accuracy=ga_result.initial_accuracy)
	comparison.add_stage("After Phase 1a (GA)", ce=ga_result.final_fitness, accuracy=ga_result.final_accuracy)
	comparison.add_stage("After Phase 1b (TS)", ce=ts_result.final_fitness, accuracy=ts_result.final_accuracy)
	comparison.add_stage("After Phase 2a (GA)", ce=conn_result.phase2_baseline, accuracy=conn_result.ga_final_accuracy)
	comparison.add_stage("After Phase 2b (TS)", ce=conn_result.final_fitness, accuracy=conn_result.final_accuracy)
	comparison.print(logger)
	logger()
	logger(f"  Results saved to: {genome_path}")
	logger.separator("=")


if __name__ == "__main__":
	main()
