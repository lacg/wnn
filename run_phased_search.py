#!/usr/bin/env python3
"""
Phased Architecture Search

Demonstrates the three-phase optimization approach:
1. Phase 1: Optimize neurons only (bits fixed at default_bits)
2. Phase 2: Optimize bits only (neurons from Phase 1)
3. Phase 3: Optimize connections only (architecture from Phase 2)

Each phase uses GA then TS refinement.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
from wnn.ram.strategies.connectivity.adaptive_cluster import (
	ClusterGenome,
	RustParallelEvaluator,
	EvaluatorConfig,
)


def log(msg: str):
	"""Log with timestamp."""
	timestamp = datetime.now().strftime("%H:%M:%S")
	print(f"{timestamp} | {msg}")


def run_phase(
	phase_name: str,
	strategy_type: OptimizerStrategyType,
	evaluator: RustParallelEvaluator,
	num_clusters: int,
	token_frequencies: list[int],
	total_input_bits: int,
	optimize_bits: bool,
	optimize_neurons: bool,
	optimize_connections: bool,
	default_bits: int,
	default_neurons: int,
	initial_genome: Optional[ClusterGenome],
	**kwargs,
):
	"""Run a single optimization phase."""
	log(f"\n{'='*60}")
	log(f"  {phase_name}")
	log(f"{'='*60}")
	log(f"  optimize_bits={optimize_bits}, optimize_neurons={optimize_neurons}")
	log(f"  optimize_connections={optimize_connections}")
	if initial_genome:
		log(f"  Starting from previous best: {initial_genome}")
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
		batch_evaluator=evaluator,  # Strategy uses this for hybrid Metal/CPU batch eval
		logger=log,
		**kwargs,
	)

	# Run optimization - strategy internally uses batch_evaluator for hybrid Metal/CPU eval
	if initial_genome is not None:
		if strategy_type == OptimizerStrategyType.ARCHITECTURE_GA:
			result = strategy.optimize(
				evaluate_fn=None,  # Not needed - batch_evaluator is used
				initial_population=[initial_genome],
			)
		else:
			result = strategy.optimize(
				evaluate_fn=None,
				initial_genome=initial_genome,
			)
	else:
		result = strategy.optimize(evaluate_fn=None)

	log(f"\n{phase_name} Result:")
	log(f"  Best fitness (CE): {result.best_fitness:.4f}")
	log(f"  Best genome: {result.best_genome}")
	log(f"  Generations/Iterations: {result.generations}")

	return result


def main():
	parser = argparse.ArgumentParser(description="Phased Architecture Search")
	parser.add_argument("--train-tokens", type=int, default=200000, help="Training tokens")
	parser.add_argument("--eval-tokens", type=int, default=50000, help="Eval tokens")
	parser.add_argument("--context", type=int, default=4, help="Context size")
	parser.add_argument("--ga-gens", type=int, default=50, help="GA generations per phase")
	parser.add_argument("--ts-iters", type=int, default=100, help="TS iterations per phase")
	parser.add_argument("--population", type=int, default=30, help="GA population size")
	parser.add_argument("--default-bits", type=int, default=8, help="Default bits for Phase 1")
	parser.add_argument("--default-neurons", type=int, default=5, help="Default neurons for Phase 2")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file")
	args = parser.parse_args()

	log("=" * 60)
	log("  Phased Architecture Search")
	log("=" * 60)
	log(f"  Train tokens: {args.train_tokens:,}")
	log(f"  Eval tokens: {args.eval_tokens:,}")
	log(f"  Context: {args.context}")
	log(f"  GA generations: {args.ga_gens}")
	log(f"  TS iterations: {args.ts_iters}")
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

	# Compute token frequencies
	log("\nComputing token frequencies...")
	from collections import Counter
	freq_counter = Counter(train_tokens)
	token_frequencies = [freq_counter.get(i, 0) for i in range(vocab_size)]
	log(f"  Tokens with freq > 0: {sum(1 for f in token_frequencies if f > 0):,}")

	# Create evaluator config
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = args.context * bits_per_token

	log(f"\nInput encoding: {args.context} tokens × {bits_per_token} bits = {total_input_bits} bits")

	# Create cluster ordering (sorted by frequency, most frequent first)
	cluster_order = sorted(range(vocab_size), key=lambda i: -token_frequencies[i])

	config = EvaluatorConfig(
		vocab_size=vocab_size,
		context_size=args.context,
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		cluster_order=cluster_order,
		global_top_k=min(1000, vocab_size),
	)
	evaluator = RustParallelEvaluator(config)

	results = {}

	# =========================================================================
	# Phase 1: Optimize neurons only (bits fixed)
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
	)
	results["phase1_ga"] = {
		"fitness": result_p1_ga.best_fitness,
		"generations": result_p1_ga.generations,
	}

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
		iterations=args.ts_iters,
	)
	results["phase1_ts"] = {
		"fitness": result_p1_ts.best_fitness,
		"iterations": result_p1_ts.generations,
	}

	# =========================================================================
	# Phase 2: Optimize bits only (neurons from Phase 1)
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
		generations=args.ga_gens,
		population_size=args.population,
	)
	results["phase2_ga"] = {
		"fitness": result_p2_ga.best_fitness,
		"generations": result_p2_ga.generations,
	}

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
		iterations=args.ts_iters,
	)
	results["phase2_ts"] = {
		"fitness": result_p2_ts.best_fitness,
		"iterations": result_p2_ts.generations,
	}

	# =========================================================================
	# Phase 3: Optimize connections only (architecture from Phase 2)
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
		generations=args.ga_gens,
		population_size=args.population,
	)
	results["phase3_ga"] = {
		"fitness": result_p3_ga.best_fitness,
		"generations": result_p3_ga.generations,
	}

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
		iterations=args.ts_iters,
	)
	results["phase3_ts"] = {
		"fitness": result_p3_ts.best_fitness,
		"iterations": result_p3_ts.generations,
	}

	# =========================================================================
	# Summary
	# =========================================================================
	log("\n" + "=" * 60)
	log("  PHASED SEARCH SUMMARY")
	log("=" * 60)
	log(f"\nPhase 1 (Neurons): {result_p1_ga.best_fitness:.4f} → {result_p1_ts.best_fitness:.4f}")
	log(f"Phase 2 (Bits):    {result_p2_ga.best_fitness:.4f} → {result_p2_ts.best_fitness:.4f}")
	log(f"Phase 3 (Conns):   {result_p3_ga.best_fitness:.4f} → {result_p3_ts.best_fitness:.4f}")
	log(f"\nFinal best fitness (CE): {result_p3_ts.best_fitness:.4f}")
	log(f"Final best genome: {result_p3_ts.best_genome}")

	# Save results
	if args.output:
		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		results["final"] = {
			"fitness": result_p3_ts.best_fitness,
			"genome_stats": result_p3_ts.best_genome.stats(),
		}
		results["args"] = vars(args)
		results["timestamp"] = datetime.now().isoformat()

		with open(output_path, "w") as f:
			json.dump(results, f, indent=2, default=str)
		log(f"\nResults saved to: {output_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
