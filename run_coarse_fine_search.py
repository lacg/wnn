#!/usr/bin/env python3
"""
Coarse-to-Fine Phased Architecture Search

Implements a multi-pass approach where:
- Pass 1 (coarse): Low patience for quick exploration
- Pass 2+ (fine): Higher patience (doubled each pass) for refinement

Each pass runs through all phases (neurons -> bits -> connections).
Later passes can be seeded from previous pass results.

Usage:
  # First pass (coarse exploration)
  python run_coarse_fine_search.py --pass 1 --base-patience 2

  # Second pass (refinement, patience=4)
  python run_coarse_fine_search.py --pass 2 --base-patience 2 --seed-from results_pass1.json

  # Third pass if needed (patience=8)
  python run_coarse_fine_search.py --pass 3 --base-patience 2 --seed-from results_pass2.json
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
	log("")
	log(f"{'='*60}")
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

	log("")
	log(f"{phase_name} Result:")
	log(f"  Best fitness (CE): {result.final_fitness:.4f}")
	log(f"  Best genome: {result.best_genome}")
	log(f"  Generations/Iterations: {result.iterations_run}")

	return result


def load_seed_genome(seed_file: str) -> Optional[ClusterGenome]:
	"""Load a genome from a previous run's JSON output."""
	try:
		with open(seed_file, 'r') as f:
			data = json.load(f)

		# Look for the final genome stats
		if 'final' in data and 'genome_stats' in data['final']:
			stats = data['final']['genome_stats']
			genome = ClusterGenome(
				bits_per_cluster=stats.get('bits_per_cluster', []),
				neurons_per_cluster=stats.get('neurons_per_cluster', []),
				connections=stats.get('connections'),
			)
			return genome
	except Exception as e:
		log(f"Warning: Could not load seed genome from {seed_file}: {e}")

	return None


def main():
	global logger

	parser = argparse.ArgumentParser(description="Coarse-to-Fine Phased Architecture Search")

	# Pass configuration
	parser.add_argument("--pass", dest="pass_num", type=int, default=1,
						help="Pass number (1=coarse, 2+=fine). Patience doubles each pass.")
	parser.add_argument("--base-patience", type=int, default=2,
						help="Base patience for pass 1 (default: 2)")
	parser.add_argument("--seed-from", type=str, default=None,
						help="JSON file from previous pass to seed from")

	# Data configuration
	parser.add_argument("--train-tokens", type=int, default=200000, help="Total training tokens")
	parser.add_argument("--eval-tokens", type=int, default=50000, help="Eval tokens")
	parser.add_argument("--context", type=int, default=4, help="Context size")

	# Optimization configuration
	parser.add_argument("--ga-gens", type=int, default=100, help="GA generations per phase")
	parser.add_argument("--ts-iters", type=int, default=200, help="TS iterations per phase")
	parser.add_argument("--population", type=int, default=50, help="GA population size")
	parser.add_argument("--neighbors", type=int, default=50, help="TS neighbors per iteration")

	# Architecture defaults
	parser.add_argument("--default-bits", type=int, default=8, help="Default bits for Phase 1")
	parser.add_argument("--default-neurons", type=int, default=5, help="Default neurons for Phase 2")

	# Other settings
	parser.add_argument("--token-parts", type=int, default=3, help="Number of token subsets (3=thirds)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for rotation (None=time-based)")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file (auto-generated if not specified)")

	args = parser.parse_args()

	# Calculate patience based on pass number: base_patience * 2^(pass-1)
	patience = args.base_patience * (2 ** (args.pass_num - 1))

	# Auto-generate output filename if not specified
	if args.output is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		args.output = f"results_pass{args.pass_num}_{timestamp}.json"

	# Determine seed: time-based if not specified
	import time
	rotation_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)

	# Setup logger using the project's Logger class
	logger = Logger(name=f"coarse_fine_pass{args.pass_num}")

	logger.header(f"Coarse-to-Fine Search - Pass {args.pass_num}")
	log(f"  Log file: {logger.log_file}")
	log(f"  Pass: {args.pass_num} (patience: {args.base_patience} * 2^{args.pass_num - 1} = {patience})")
	if args.seed_from:
		log(f"  Seeding from: {args.seed_from}")
	log(f"  Total train tokens: {args.train_tokens:,}")
	log(f"  Per-iteration rotation: 1/{args.token_parts} (~{args.train_tokens // args.token_parts:,} tokens per iteration)")
	log(f"  Rotation seed: {rotation_seed}" + (" (time-based)" if args.seed is None else " (explicit)"))
	log(f"  Eval tokens: {args.eval_tokens:,}")
	log(f"  Context: {args.context}")
	log(f"  GA generations: {args.ga_gens}, population: {args.population}")
	log(f"  TS iterations: {args.ts_iters}, neighbors: {args.neighbors}")
	log(f"  Patience: {patience}")
	log(f"  Default bits (Phase 1): {args.default_bits}")
	log(f"  Default neurons (Phase 2): {args.default_neurons}")
	log(f"  Output: {args.output}")
	log("")

	# Load seed genome if specified
	seed_genome = None
	if args.seed_from:
		seed_genome = load_seed_genome(args.seed_from)
		if seed_genome:
			log(f"Loaded seed genome: {seed_genome}")
		else:
			log("No seed genome loaded, starting fresh")

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
	log("")
	log("Computing token frequencies from training data...")
	from collections import Counter
	freq_counter = Counter(train_tokens)
	token_frequencies = [freq_counter.get(i, 0) for i in range(vocab_size)]
	log(f"  Tokens with freq > 0: {sum(1 for f in token_frequencies if f > 0):,}")

	# Create cluster ordering (sorted by frequency, most frequent first)
	cluster_order = sorted(range(vocab_size), key=lambda i: -token_frequencies[i])

	# Create cached evaluator (holds all tokens in Rust, zero-copy per iteration)
	log("")
	log("Creating cached evaluator with per-iteration rotation...")
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
		log_path=logger.log_file,  # Pass to Rust for offspring logging
	)
	log(f"  {evaluator}")

	# Compute input bits
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = args.context * bits_per_token
	log("")
	log(f"Input encoding: {args.context} tokens × {bits_per_token} bits = {total_input_bits} bits")

	results = {
		"pass": args.pass_num,
		"patience": patience,
		"base_patience": args.base_patience,
	}

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
		initial_genome=seed_genome,
		generations=args.ga_gens,
		population_size=args.population,
		patience=patience,
		initial_threshold=None,
	)
	results["phase1_ga"] = {
		"fitness": result_p1_ga.final_fitness,
		"accuracy": result_p1_ga.final_accuracy,
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
		patience=patience,
		initial_threshold=result_p1_ga.final_threshold,
	)
	results["phase1_ts"] = {
		"fitness": result_p1_ts.final_fitness,
		"accuracy": result_p1_ts.final_accuracy,
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
		patience=patience,
		initial_threshold=result_p1_ts.final_threshold,
	)
	results["phase2_ga"] = {
		"fitness": result_p2_ga.final_fitness,
		"accuracy": result_p2_ga.final_accuracy,
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
		patience=patience,
		initial_threshold=result_p2_ga.final_threshold,
	)
	results["phase2_ts"] = {
		"fitness": result_p2_ts.final_fitness,
		"accuracy": result_p2_ts.final_accuracy,
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
		patience=patience,
		initial_threshold=result_p2_ts.final_threshold,
	)
	results["phase3_ga"] = {
		"fitness": result_p3_ga.final_fitness,
		"accuracy": result_p3_ga.final_accuracy,
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
		patience=patience,
		initial_threshold=result_p3_ga.final_threshold,
	)
	results["phase3_ts"] = {
		"fitness": result_p3_ts.final_fitness,
		"accuracy": result_p3_ts.final_accuracy,
		"iterations": result_p3_ts.iterations_run,
	}

	# =========================================================================
	# Final Evaluation with FULL training tokens
	# =========================================================================
	log("")
	log(f"{'='*60}")
	log("  Final Evaluation with FULL Training Data")
	log(f"{'='*60}")
	log(f"  Using all {len(train_tokens):,} training tokens for final evaluation")

	# Evaluate the final best genome with full data
	final_ce, final_acc = evaluator.evaluate_single_full(result_p3_ts.best_genome)
	log("")
	log(f"  Final genome: {result_p3_ts.best_genome}")
	log(f"  Final CE (full data): {final_ce:.4f}")
	log(f"  Final Accuracy: {final_acc:.2%}")
	log(f"  Final PPL: {2.71828 ** final_ce:.1f}")

	# =========================================================================
	# Final Summary
	# =========================================================================
	log("")
	log("=" * 78)
	log(f"  FINAL RESULTS - Pass {args.pass_num} (patience={patience})")
	log("=" * 78)
	comparison = OptimizationResultsTable(f"Coarse-to-Fine Pass {args.pass_num} (patience={patience})")
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
	log("")
	log(f"Results saved to: {output_path}")

	# Print next pass command
	if args.pass_num < 3:
		next_patience = args.base_patience * (2 ** args.pass_num)
		log("")
		log(f"To run pass {args.pass_num + 1} (patience={next_patience}):")
		log(f"  python run_coarse_fine_search.py --pass {args.pass_num + 1} --base-patience {args.base_patience} --seed-from {args.output}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
