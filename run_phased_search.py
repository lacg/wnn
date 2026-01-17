#!/usr/bin/env python3
"""
Phased Architecture Search with Per-Iteration Token Rotation

Demonstrates the three-phase optimization approach:
1. Phase 1: Optimize neurons only (bits fixed at default_bits)
2. Phase 2: Optimize bits only (neurons from Phase 1)
3. Phase 3: Optimize connections only (architecture from Phase 2)

Each phase uses GA then TS refinement.

Per-iteration rotation: Each generation/iteration uses a different 1/3 of
training tokens (cycling randomly through thirds). This acts as a regularizer
that forces genomes to generalize across all data subsets.

Full evaluation: Uses all tokens for final evaluation after Phase 3b.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wnn.logger import Logger
from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.core.reporting import OptimizationResultsTable
from wnn.ram.architecture import CachedEvaluator


# Global logger instance
logger: Optional[Logger] = None


def log(msg: str):
	"""Log wrapper that uses the global logger."""
	if logger:
		logger(msg)
	else:
		print(msg)


def run_phase(
	phase_name: str,
	strategy_type: OptimizerStrategyType,
	evaluator: CachedEvaluator,
	num_clusters: int,
	token_frequencies: list[int],
	total_input_bits: int,
	optimize_bits: bool,
	optimize_neurons: bool,
	optimize_connections: bool,
	default_bits: int,
	default_neurons: int,
	initial_genome: Optional[ClusterGenome],
	initial_fitness: Optional[float] = None,  # Required for TS
	initial_population: Optional[list[ClusterGenome]] = None,  # Seed population for GA/TS
	**kwargs,
):
	"""Run a single optimization phase with per-iteration token rotation."""
	log(f"\n{'='*60}")
	log(f"  {phase_name}")
	log(f"{'='*60}")
	log(f"  optimize_bits={optimize_bits}, optimize_neurons={optimize_neurons}")
	log(f"  optimize_connections={optimize_connections}")
	log(f"  Per-iteration rotation: {evaluator.num_parts} subsets")
	if initial_genome:
		log(f"  Starting from previous best: {initial_genome}")
	if initial_population:
		log(f"  Seeding from {len(initial_population)} genomes from previous phase")
	log("")

	strategy = OptimizerStrategyFactory.create(
		strategy_type=strategy_type,
		num_clusters=num_clusters,
		optimize_bits=optimize_bits,
		optimize_neurons=optimize_neurons,
		optimize_connections=optimize_connections,
		default_bits=default_bits,
		default_neurons=default_neurons,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		batch_evaluator=evaluator,  # Pass cached evaluator
		logger=log,
		**kwargs,
	)

	# Run optimization
	if initial_genome is not None:
		if strategy_type == OptimizerStrategyType.ARCHITECTURE_GA:
			# GA: seed with population from previous phase (or just best genome)
			seed_pop = initial_population if initial_population else [initial_genome]
			result = strategy.optimize(
				evaluate_fn=None,  # Not needed - batch_evaluator is used
				initial_population=seed_pop,
			)
		else:
			# TS: seed with neighbors from previous phase
			result = strategy.optimize(
				evaluate_fn=None,
				initial_genome=initial_genome,
				initial_fitness=initial_fitness,
				initial_neighbors=initial_population,  # Reuse as neighbors
			)
	else:
		result = strategy.optimize(evaluate_fn=None)

	log(f"\n{phase_name} Result:")
	log(f"  Best fitness (CE): {result.final_fitness:.4f}")
	log(f"  Best genome: {result.best_genome}")
	log(f"  Generations/Iterations: {result.iterations_run}")

	return result


def main():
	global logger

	parser = argparse.ArgumentParser(description="Phased Architecture Search")
	parser.add_argument("--train-tokens", type=int, default=200000, help="Total training tokens")
	parser.add_argument("--eval-tokens", type=int, default=50000, help="Eval tokens")
	parser.add_argument("--context", type=int, default=4, help="Context size")
	parser.add_argument("--ga-gens", type=int, default=50, help="GA generations per phase")
	parser.add_argument("--ts-iters", type=int, default=100, help="TS iterations per phase")
	parser.add_argument("--population", type=int, default=30, help="GA population size")
	parser.add_argument("--neighbors", type=int, default=20, help="TS neighbors per iteration")
	parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
	parser.add_argument("--default-bits", type=int, default=8, help="Default bits for Phase 1")
	parser.add_argument("--default-neurons", type=int, default=5, help="Default neurons for Phase 2")
	parser.add_argument("--token-parts", type=int, default=3, help="Number of token subsets (3=thirds)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for rotation (None=time-based)")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file")
	args = parser.parse_args()

	# Determine seed: time-based if not specified
	import time
	rotation_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)

	# Setup logger using the project's Logger class
	logger = Logger(name="phased_search")

	logger.header("Phased Architecture Search with Per-Iteration Rotation")
	log(f"  Log file: {logger.log_file}")
	log(f"  Total train tokens: {args.train_tokens:,}")
	log(f"  Per-iteration rotation: 1/{args.token_parts} (~{args.train_tokens // args.token_parts:,} tokens per iteration)")
	log(f"  Rotation seed: {rotation_seed}" + (" (time-based)" if args.seed is None else " (explicit)"))
	log(f"  Eval tokens: {args.eval_tokens:,}")
	log(f"  Context: {args.context}")
	log(f"  GA generations: {args.ga_gens}, population: {args.population}")
	log(f"  TS iterations: {args.ts_iters}, neighbors: {args.neighbors}")
	log(f"  Patience: {args.patience}")
	log(f"  Default bits (Phase 1): {args.default_bits}")
	log(f"  Default neurons (Phase 2): {args.default_neurons}")
	log("")

	# Load data
	log("Loading WikiText-2 dataset...")
	try:
		from datasets import load_dataset
		import tiktoken

		dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
		text = " ".join([t for t in dataset["text"] if t.strip()])

		enc = tiktoken.get_encoding("gpt2")
		all_tokens = enc.encode(text)

		train_tokens = all_tokens[:args.train_tokens]
		eval_tokens = all_tokens[args.train_tokens:args.train_tokens + args.eval_tokens]
		vocab_size = enc.n_vocab

		log(f"  Train: {len(train_tokens):,} tokens")
		log(f"  Eval: {len(eval_tokens):,} tokens")
		log(f"  Vocab: {vocab_size:,}")

	except ImportError as e:
		log(f"ERROR: Missing dependencies: {e}")
		log("Please install: pip install datasets tiktoken")
		return 1

	# Compute token frequencies from full training data
	log("\nComputing token frequencies from training data...")
	from collections import Counter
	freq_counter = Counter(train_tokens)
	token_frequencies = [freq_counter.get(i, 0) for i in range(vocab_size)]
	log(f"  Tokens with freq > 0: {sum(1 for f in token_frequencies if f > 0):,}")

	# Create cluster ordering (sorted by frequency, most frequent first)
	cluster_order = sorted(range(vocab_size), key=lambda i: -token_frequencies[i])

	# Create cached evaluator (holds all tokens in Rust, zero-copy per iteration)
	log("\nCreating cached evaluator with per-iteration rotation...")
	evaluator = CachedEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=args.context,
		cluster_order=cluster_order,
		num_parts=args.token_parts,
		num_negatives=5,
		empty_value=0.0,
		seed=rotation_seed,
	)
	log(f"  {evaluator}")

	# Compute input bits
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = args.context * bits_per_token
	log(f"\nInput encoding: {args.context} tokens × {bits_per_token} bits = {total_input_bits} bits")

	results = {}

	def print_progress(title: str, stages: list):
		"""Print progress table with current stages."""
		table = OptimizationResultsTable(title)
		for name, result in stages:
			table.add_stage(name, ce=result.final_fitness, accuracy=result.final_accuracy)
		log("")
		table.print(log)

	# =========================================================================
	# Phase 1a: GA Neurons Only
	# =========================================================================
	result_p1_ga = run_phase(
		phase_name="Phase 1a: GA Neurons Only",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=False,
		optimize_neurons=True,
		optimize_connections=False,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=None,
		generations=args.ga_gens,
		population_size=args.population,
		patience=args.patience,
		initial_threshold=None,
	)
	results["phase1_ga"] = {
		"fitness": result_p1_ga.final_fitness,
		"generations": result_p1_ga.iterations_run,
	}
	print_progress("After Phase 1a", [("Phase 1a (GA Neurons)", result_p1_ga)])

	# =========================================================================
	# Phase 1b: TS Neurons Only
	# =========================================================================
	result_p1_ts = run_phase(
		phase_name="Phase 1b: TS Neurons Only (refine)",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=False,
		optimize_neurons=True,
		optimize_connections=False,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=result_p1_ga.best_genome,
		initial_fitness=result_p1_ga.final_fitness,
		initial_population=result_p1_ga.final_population,
		iterations=args.ts_iters,
		neighbors_per_iter=args.neighbors,
		total_neighbors_size=args.population,
		patience=args.patience,
		initial_threshold=result_p1_ga.final_threshold,
	)
	results["phase1_ts"] = {
		"fitness": result_p1_ts.final_fitness,
		"iterations": result_p1_ts.iterations_run,
	}
	print_progress("After Phase 1b", [
		("Phase 1a (GA Neurons)", result_p1_ga),
		("Phase 1b (TS Neurons)", result_p1_ts),
	])

	# =========================================================================
	# Phase 2a: GA Bits Only
	# =========================================================================
	result_p2_ga = run_phase(
		phase_name="Phase 2a: GA Bits Only",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=True,
		optimize_neurons=False,
		optimize_connections=False,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=result_p1_ts.best_genome,
		initial_population=result_p1_ts.final_population,
		generations=args.ga_gens,
		population_size=args.population,
		patience=args.patience,
		initial_threshold=result_p1_ts.final_threshold,
	)
	results["phase2_ga"] = {
		"fitness": result_p2_ga.final_fitness,
		"generations": result_p2_ga.iterations_run,
	}
	print_progress("After Phase 2a", [
		("Phase 1a (GA Neurons)", result_p1_ga),
		("Phase 1b (TS Neurons)", result_p1_ts),
		("Phase 2a (GA Bits)", result_p2_ga),
	])

	# =========================================================================
	# Phase 2b: TS Bits Only
	# =========================================================================
	result_p2_ts = run_phase(
		phase_name="Phase 2b: TS Bits Only (refine)",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=True,
		optimize_neurons=False,
		optimize_connections=False,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=result_p2_ga.best_genome,
		initial_fitness=result_p2_ga.final_fitness,
		initial_population=result_p2_ga.final_population,
		iterations=args.ts_iters,
		neighbors_per_iter=args.neighbors,
		total_neighbors_size=args.population,
		patience=args.patience,
		initial_threshold=result_p2_ga.final_threshold,
	)
	results["phase2_ts"] = {
		"fitness": result_p2_ts.final_fitness,
		"iterations": result_p2_ts.iterations_run,
	}
	print_progress("After Phase 2b", [
		("Phase 1a (GA Neurons)", result_p1_ga),
		("Phase 1b (TS Neurons)", result_p1_ts),
		("Phase 2a (GA Bits)", result_p2_ga),
		("Phase 2b (TS Bits)", result_p2_ts),
	])

	# =========================================================================
	# Phase 3a: GA Connections Only
	# =========================================================================
	result_p3_ga = run_phase(
		phase_name="Phase 3a: GA Connections Only",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_GA,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=False,
		optimize_neurons=False,
		optimize_connections=True,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=result_p2_ts.best_genome,
		initial_population=result_p2_ts.final_population,
		generations=args.ga_gens,
		population_size=args.population,
		patience=args.patience,
		initial_threshold=result_p2_ts.final_threshold,
	)
	results["phase3_ga"] = {
		"fitness": result_p3_ga.final_fitness,
		"generations": result_p3_ga.iterations_run,
	}
	print_progress("After Phase 3a", [
		("Phase 1a (GA Neurons)", result_p1_ga),
		("Phase 1b (TS Neurons)", result_p1_ts),
		("Phase 2a (GA Bits)", result_p2_ga),
		("Phase 2b (TS Bits)", result_p2_ts),
		("Phase 3a (GA Conns)", result_p3_ga),
	])

	# =========================================================================
	# Phase 3b: TS Connections Only
	# =========================================================================
	result_p3_ts = run_phase(
		phase_name="Phase 3b: TS Connections Only (refine)",
		strategy_type=OptimizerStrategyType.ARCHITECTURE_TS,
		evaluator=evaluator,
		num_clusters=vocab_size,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,
		optimize_bits=False,
		optimize_neurons=False,
		optimize_connections=True,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		initial_genome=result_p3_ga.best_genome,
		initial_fitness=result_p3_ga.final_fitness,
		initial_population=result_p3_ga.final_population,
		iterations=args.ts_iters,
		neighbors_per_iter=args.neighbors,
		total_neighbors_size=args.population,
		patience=args.patience,
		initial_threshold=result_p3_ga.final_threshold,
	)
	results["phase3_ts"] = {
		"fitness": result_p3_ts.final_fitness,
		"iterations": result_p3_ts.iterations_run,
	}

	# =========================================================================
	# Final Evaluation with FULL training tokens
	# =========================================================================
	log(f"\n{'='*60}")
	log("  Final Evaluation with FULL Training Data")
	log(f"{'='*60}")
	log(f"  Using all {len(train_tokens):,} training tokens for final evaluation")

	# Evaluate the final best genome with full data
	final_ce, final_acc = evaluator.evaluate_single_full(result_p3_ts.best_genome)
	log(f"\n  Final genome: {result_p3_ts.best_genome}")
	log(f"  Final CE (full data): {final_ce:.4f}")
	log(f"  Final Accuracy: {final_acc:.2%}")
	log(f"  Final PPL: {2.71828 ** final_ce:.1f}")

	# =========================================================================
	# Final Summary
	# =========================================================================
	log("")
	log("=" * 78)
	log("  FINAL RESULTS")
	log("=" * 78)
	comparison = OptimizationResultsTable("Complete Phased Search (Per-Iteration Rotation)")
	comparison.add_stage("Initial", ce=result_p1_ga.initial_fitness, accuracy=result_p1_ga.initial_accuracy)
	comparison.add_stage("Phase 1a (GA Neurons)", ce=result_p1_ga.final_fitness, accuracy=result_p1_ga.final_accuracy)
	comparison.add_stage("Phase 1b (TS Neurons)", ce=result_p1_ts.final_fitness, accuracy=result_p1_ts.final_accuracy)
	comparison.add_stage("Phase 2a (GA Bits)", ce=result_p2_ga.final_fitness, accuracy=result_p2_ga.final_accuracy)
	comparison.add_stage("Phase 2b (TS Bits)", ce=result_p2_ts.final_fitness, accuracy=result_p2_ts.final_accuracy)
	comparison.add_stage("Phase 3a (GA Conns)", ce=result_p3_ga.final_fitness, accuracy=result_p3_ga.final_accuracy)
	comparison.add_stage("Phase 3b (TS Conns)", ce=result_p3_ts.final_fitness, accuracy=result_p3_ts.final_accuracy)
	comparison.add_stage("Final (Full Data)", ce=final_ce, accuracy=final_acc)
	comparison.print(log)
	log("")
	log(f"Final best genome: {result_p3_ts.best_genome}")

	# Save results
	if args.output:
		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		results["final"] = {
			"fitness": final_ce,
			"accuracy": final_acc,
			"genome_stats": result_p3_ts.best_genome.stats(),
		}
		results["args"] = vars(args)
		results["timestamp"] = datetime.now().isoformat()
		results["per_iteration_rotation"] = {
			"num_parts": args.token_parts,
			"tokens_per_part": len(train_tokens) // args.token_parts,
		}

		with open(output_path, "w") as f:
			json.dump(results, f, indent=2, default=str)
		log(f"\nResults saved to: {output_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
