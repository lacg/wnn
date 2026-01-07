#!/usr/bin/env python3
"""
RAMLM Full Benchmark - Rust-Accelerated Training with GA/TS Optimization

This benchmark tests the core RAMLM using:
- GPT-2 tokenizer (50k vocab)
- WikiText-2 full dataset
- Rust-accelerated training (3.35x faster than PyTorch)
- ConnectivityOptimizer with GA/TS for connectivity optimization

Usage:
	# Quick test (smaller data, fewer iterations)
	python tests/ramlm_full_benchmark.py --mode fast

	# Full benchmark with optimization
	python tests/ramlm_full_benchmark.py --mode full --optimize

	# Full data with GA then TS optimization
	python tests/ramlm_full_benchmark.py --full-data --optimize --strategy GA,TS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch

from wnn.ram.core import AccelerationMode, OptimizationMethod
from wnn.ram.core.models.ramlm import RAMLM
from wnn.ram.strategies.connectivity import ConnectivityOptimizer, OptimizationConfig
from wnn.tokenizers import TokenizerType, TokenizerFactory


# ============================================================================
# RUST ACCELERATOR CHECK
# ============================================================================
try:
	import ram_accelerator
	RUST_AVAILABLE = True
	RUST_CPU_CORES = ram_accelerator.cpu_cores()
	METAL_AVAILABLE = ram_accelerator.metal_available()
except ImportError:
	RUST_AVAILABLE = False
	RUST_CPU_CORES = 0
	METAL_AVAILABLE = False


@dataclass
class BenchmarkConfig:
	"""Configuration for RAMLM benchmark."""
	# Model architecture
	neurons_per_cluster: int = 3
	bits_per_neuron: int = 8
	context_size: int = 4  # 4-gram context

	# Training
	global_top_k: int = 100
	batch_size: int = 500

	# Data
	train_limit: Optional[int] = None
	test_limit: Optional[int] = None
	val_limit: Optional[int] = None

	# Optimization
	optimize: bool = False
	strategy: str = "TS"  # GA, TS, SA, or combinations like GA,TS

	# TS parameters
	ts_iterations: int = 10
	ts_neighbors: int = 30

	# GA parameters
	ga_population: int = 20
	ga_generations: int = 50


def load_wikitext2(tokenizer_type: TokenizerType = TokenizerType.GPT2,
					train_limit: Optional[int] = None,
					test_limit: Optional[int] = None,
					val_limit: Optional[int] = None):
	"""Load WikiText-2 dataset with GPT-2 tokenization."""
	try:
		from datasets import load_dataset
	except ImportError:
		print("ERROR: datasets library not installed. Run: pip install datasets")
		sys.exit(1)

	print("Loading WikiText-2 dataset...")
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	# Get GPT-2 tokenizer
	print(f"Using {tokenizer_type.name} tokenizer...")
	if tokenizer_type == TokenizerType.GPT2:
		tokenizer = TokenizerFactory.create(TokenizerType.GPT2)
		vocab_size = tokenizer.vocab_size
	else:
		raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")

	# Tokenize datasets
	def tokenize_split(split_name: str, limit: Optional[int] = None) -> list[int]:
		text = " ".join(dataset[split_name]["text"])
		tokens = tokenizer.encode(text)
		if limit:
			tokens = tokens[:limit]
		return tokens

	train_tokens = tokenize_split("train", train_limit)
	test_tokens = tokenize_split("test", test_limit)
	val_tokens = tokenize_split("validation", val_limit)

	print(f"  Train: {len(train_tokens):,} tokens")
	print(f"  Test: {len(test_tokens):,} tokens")
	print(f"  Validation: {len(val_tokens):,} tokens")
	print(f"  Vocab size: {vocab_size:,}")

	return train_tokens, test_tokens, val_tokens, vocab_size


def print_header(title: str):
	"""Print a formatted header."""
	print()
	print("=" * 70)
	print(f"  {title}")
	print("=" * 70)


def print_section(title: str):
	"""Print a section divider."""
	print()
	print("-" * 50)
	print(f"  {title}")
	print("-" * 50)


def run_benchmark(config: BenchmarkConfig):
	"""Run the RAMLM benchmark."""
	print_header("RAMLM Full Benchmark")

	# Print configuration
	print("\nConfiguration:")
	print(f"  neurons_per_cluster: {config.neurons_per_cluster}")
	print(f"  bits_per_neuron: {config.bits_per_neuron}")
	print(f"  context_size: {config.context_size}")
	print(f"  global_top_k: {config.global_top_k}")
	print(f"  batch_size: {config.batch_size}")
	print(f"  optimize: {config.optimize}")
	if config.optimize:
		print(f"  strategy: {config.strategy}")
	print(f"  Rust available: {RUST_AVAILABLE}")
	print(f"  CPU cores: {RUST_CPU_CORES}")
	print(f"  Metal available: {METAL_AVAILABLE}")

	# Load data
	print_section("Loading Data")
	train_tokens, test_tokens, val_tokens, vocab_size = load_wikitext2(
		TokenizerType.GPT2,
		train_limit=config.train_limit,
		test_limit=config.test_limit,
		val_limit=config.val_limit,
	)

	# Compute global top-k from training data
	print_section("Computing Global Top-K")
	token_counts = Counter(train_tokens)
	global_top_k_tokens = [t for t, _ in token_counts.most_common(config.global_top_k)]
	print(f"  Top-{config.global_top_k} tokens cover {sum(token_counts[t] for t in global_top_k_tokens)/len(train_tokens)*100:.1f}% of training data")

	# Create RAMLM
	print_section("Creating RAMLM Model")
	model = RAMLM(
		vocab_size=vocab_size,
		context_size=config.context_size,
		neurons_per_cluster=config.neurons_per_cluster,
		bits_per_neuron=config.bits_per_neuron,
	)
	# Set global top-k
	model._global_top_k = global_top_k_tokens

	print(f"  Total neurons: {model.layer.total_neurons:,}")
	print(f"  Total input bits: {model.total_input_bits}")
	print(f"  Memory addresses per neuron: {2**config.bits_per_neuron}")

	# Initial training (no optimization)
	print_section("Initial Training (Rust-Accelerated)")
	start = time.perf_counter()

	if RUST_AVAILABLE:
		stats = model.train_epoch_fast_rust(
			train_tokens,
			global_top_k=config.global_top_k,
			batch_size=config.batch_size,
			verbose=True,
		)
		backend = "Rust"
	else:
		stats = model.train_epoch_fast(
			train_tokens,
			global_top_k=config.global_top_k,
			batch_size=config.batch_size,
			verbose=True,
		)
		backend = "PyTorch"

	train_time = time.perf_counter() - start
	print(f"\n  Backend: {backend}")
	print(f"  Training time: {train_time:.2f}s")
	print(f"  Modified cells: {stats['modified']:,}")
	print(f"  Examples: {stats['examples']:,}")
	print(f"  Throughput: {stats['examples']/train_time:,.0f} examples/sec")

	# Evaluate on validation set
	print_section("Initial Evaluation (Validation Set)")
	start = time.perf_counter()
	val_stats = model.evaluate_fast(
		val_tokens,
		batch_size=config.batch_size * 2,
		backend=AccelerationMode.AUTO,
		verbose=True,
	)
	eval_time = time.perf_counter() - start

	print(f"\n  Evaluation time: {eval_time:.2f}s")
	print(f"  Cross-entropy: {val_stats['cross_entropy']:.4f}")
	print(f"  Perplexity: {val_stats['perplexity']:.2f}")
	print(f"  Accuracy: {val_stats['accuracy']:.2%}")

	# Connectivity optimization
	if config.optimize:
		print_section("Connectivity Optimization")

		# Parse strategy sequence
		strategies = [s.strip().upper() for s in config.strategy.split(',')]

		for strategy_name in strategies:
			print(f"\n  Running {strategy_name} optimization...")

			# Map strategy name to enum
			method_map = {
				"GA": OptimizationMethod.GENETIC_ALGORITHM,
				"TS": OptimizationMethod.TABU_SEARCH,
				"SA": OptimizationMethod.SIMULATED_ANNEALING,
			}

			if strategy_name not in method_map:
				print(f"  WARNING: Unknown strategy {strategy_name}, skipping")
				continue

			method = method_map[strategy_name]

			# Create optimizer config
			opt_config = OptimizationConfig(
				method=method,
				ts_iterations=config.ts_iterations,
				ts_neighbors_per_iter=config.ts_neighbors,
				ga_population_size=config.ga_population,
				ga_generations=config.ga_generations,
				verbose=True,
			)

			optimizer = ConnectivityOptimizer(config=opt_config)

			# Define train/eval functions for optimizer
			def train_fn():
				if RUST_AVAILABLE:
					model.train_epoch_fast_rust(
						train_tokens,
						global_top_k=config.global_top_k,
						batch_size=config.batch_size,
						verbose=False,
					)
				else:
					model.train_epoch_fast(
						train_tokens,
						global_top_k=config.global_top_k,
						batch_size=config.batch_size,
						verbose=False,
					)

			def eval_fn() -> float:
				stats = model.evaluate_fast(
					val_tokens[:min(50000, len(val_tokens))],  # Use subset for faster eval
					batch_size=config.batch_size * 2,
					backend=AccelerationMode.AUTO,
					verbose=False,
				)
				return stats['cross_entropy']

			# Run optimization
			start = time.perf_counter()
			result = optimizer.optimize(
				model=model,
				train_fn=train_fn,
				eval_fn=eval_fn,
			)
			opt_time = time.perf_counter() - start

			print(f"\n  {strategy_name} Results:")
			print(f"    Time: {opt_time:.1f}s")
			print(f"    Initial CE: {result.initial_cross_entropy:.4f}")
			print(f"    Final CE: {result.final_cross_entropy:.4f}")
			print(f"    Improvement: {result.improvement_percent:.2f}%")
			print(f"    Iterations: {result.iterations_run}")

	# Final evaluation on test set
	print_section("Final Evaluation (Test Set)")
	start = time.perf_counter()
	test_stats = model.evaluate_fast(
		test_tokens,
		batch_size=config.batch_size * 2,
		backend=AccelerationMode.AUTO,
		verbose=True,
	)
	eval_time = time.perf_counter() - start

	print(f"\n  Evaluation time: {eval_time:.2f}s")
	print(f"  Cross-entropy: {test_stats['cross_entropy']:.4f}")
	print(f"  Perplexity: {test_stats['perplexity']:.2f}")
	print(f"  Accuracy: {test_stats['accuracy']:.2%}")

	# Summary
	print_header("Summary")
	print(f"Model: RAMLM (vocab={vocab_size}, neurons={config.neurons_per_cluster}/cluster, bits={config.bits_per_neuron})")
	print(f"Data: WikiText-2 (train={len(train_tokens):,}, val={len(val_tokens):,}, test={len(test_tokens):,})")
	print(f"Backend: {backend} ({'Rust + rayon parallel' if RUST_AVAILABLE else 'PyTorch vectorized'})")
	print()
	print("Results:")
	print(f"  Validation PPL: {val_stats['perplexity']:.2f}")
	print(f"  Validation Acc: {val_stats['accuracy']:.2%}")
	print(f"  Test PPL: {test_stats['perplexity']:.2f}")
	print(f"  Test Acc: {test_stats['accuracy']:.2%}")

	if config.optimize:
		print(f"  Optimization: {config.strategy}")

	print()
	print("=" * 70)

	return {
		"val_stats": val_stats,
		"test_stats": test_stats,
		"train_time": train_time,
		"model": model,
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="RAMLM Full Benchmark with Rust Acceleration and GA/TS Optimization"
	)

	# Mode selection
	parser.add_argument("--mode", type=str, default="fast",
		choices=["fast", "full", "overnight"],
		help="Benchmark mode: fast (quick test), full (standard), overnight (thorough)")

	# Data options
	parser.add_argument("--full-data", action="store_true",
		help="Use full WikiText-2 dataset instead of subsampled")

	# Model architecture
	parser.add_argument("--neurons", type=int, default=None,
		help="Neurons per cluster (default: 3 fast, 4 full, 6 overnight)")
	parser.add_argument("--bits", type=int, default=None,
		help="Bits per neuron (default: 8 fast, 10 full, 12 overnight)")
	parser.add_argument("--context", type=int, default=4,
		help="Context size in tokens (default: 4)")

	# Training
	parser.add_argument("--top-k", type=int, default=100,
		help="Global top-k for FALSE training (default: 100)")
	parser.add_argument("--batch-size", type=int, default=500,
		help="Batch size for training (default: 500)")

	# Optimization
	parser.add_argument("--optimize", action="store_true",
		help="Run connectivity optimization after initial training")
	parser.add_argument("--strategy", type=str, default="TS",
		help="Optimization strategy: GA, TS, SA, or comma-separated sequence like GA,TS (default: TS)")

	# Optimization parameters
	parser.add_argument("--ts-iters", type=int, default=None,
		help="Tabu Search iterations (default: 5 fast, 10 full, 20 overnight)")
	parser.add_argument("--ts-neighbors", type=int, default=30,
		help="Tabu Search neighbors per iteration (default: 30)")
	parser.add_argument("--ga-pop", type=int, default=None,
		help="GA population size (default: 10 fast, 20 full, 30 overnight)")
	parser.add_argument("--ga-gens", type=int, default=None,
		help="GA generations (default: 20 fast, 50 full, 100 overnight)")

	args = parser.parse_args()

	# Configure based on mode
	config = BenchmarkConfig()
	config.context_size = args.context
	config.global_top_k = args.top_k
	config.batch_size = args.batch_size
	config.optimize = args.optimize
	config.strategy = args.strategy

	if args.mode == "fast":
		# Quick test with smaller data
		config.neurons_per_cluster = args.neurons or 3
		config.bits_per_neuron = args.bits or 8
		config.train_limit = None if args.full_data else 50000
		config.test_limit = None if args.full_data else 10000
		config.val_limit = None if args.full_data else 10000
		config.ts_iterations = args.ts_iters or 5
		config.ga_population = args.ga_pop or 10
		config.ga_generations = args.ga_gens or 20

	elif args.mode == "full":
		# Standard benchmark
		config.neurons_per_cluster = args.neurons or 4
		config.bits_per_neuron = args.bits or 10
		config.train_limit = None if args.full_data else 200000
		config.test_limit = None if args.full_data else 50000
		config.val_limit = None if args.full_data else 50000
		config.ts_iterations = args.ts_iters or 10
		config.ga_population = args.ga_pop or 20
		config.ga_generations = args.ga_gens or 50

	elif args.mode == "overnight":
		# Thorough benchmark (always uses full data)
		config.neurons_per_cluster = args.neurons or 6
		config.bits_per_neuron = args.bits or 12
		config.train_limit = None  # Full data
		config.test_limit = None
		config.val_limit = None
		config.ts_iterations = args.ts_iters or 20
		config.ga_population = args.ga_pop or 30
		config.ga_generations = args.ga_gens or 100

	# Override optimization parameters if specified
	config.ts_neighbors = args.ts_neighbors

	# Run benchmark
	results = run_benchmark(config)
