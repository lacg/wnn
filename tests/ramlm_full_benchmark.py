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
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

from wnn.ram.core import AccelerationMode, OptimizationMethod
from wnn.ram.core.models.ramlm import RAMLM
from wnn.ram.strategies.connectivity import (
	OverfittingMonitor,
	GeneticAlgorithmStrategy,
	GeneticAlgorithmConfig,
	TabuSearchStrategy,
	TabuSearchConfig,
	HEALTHY_THRESHOLD,
	WARNING_THRESHOLD,
	SEVERE_THRESHOLD,
	CRITICAL_THRESHOLD,
)
from wnn.tokenizers import TokenizerType, TokenizerFactory


# ============================================================================
# LOGGING SETUP
# ============================================================================
LOG_FILENAME = None

def setup_logging(log_dir: str = None) -> str:
	"""Setup logging to both console and file. Returns log filename."""
	global LOG_FILENAME

	if log_dir is None:
		# Use date-based folder structure: wnn/logs/YYYY/MM/DD/
		script_dir = os.path.dirname(os.path.abspath(__file__))
		project_root = os.path.dirname(script_dir)  # Go up from tests/ to wnn/
		now = datetime.now()
		log_dir = os.path.join(project_root, "logs", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
		os.makedirs(log_dir, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	LOG_FILENAME = os.path.join(log_dir, f"ramlm_benchmark_{timestamp}.log")

	# Create logger
	logger = logging.getLogger('ramlm_benchmark')
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	# Timestamp format for log entries
	timestamp_format = '%(asctime)s | %(message)s'
	date_format = '%H:%M:%S'

	# File handler (with timestamps)
	file_handler = logging.FileHandler(LOG_FILENAME)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(logging.Formatter(timestamp_format, datefmt=date_format))
	logger.addHandler(file_handler)

	# Console handler (with timestamps)
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(logging.Formatter(timestamp_format, datefmt=date_format))
	logger.addHandler(console_handler)

	return LOG_FILENAME


def log(message: str = ""):
	"""Log message to both console and file."""
	logger = logging.getLogger('ramlm_benchmark')
	logger.info(message)


def log_separator(char: str = "=", width: int = 70):
	"""Log a separator line."""
	log(char * width)


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


# ============================================================================
# PARALLEL TIERED CANDIDATE EVALUATOR
# ============================================================================
class ParallelTieredEvaluator:
	"""
	Prepares data and evaluates GA/TS candidates in parallel using Rust.

	Supports two modes:
	- CPU-only (16 cores via rayon): ~16x speedup
	- Hybrid CPU+GPU (16 + 40 cores): ~32x speedup on M4 Max

	The hybrid mode uses:
	- CPU for training (DashMap is optimal for parallel writes)
	- GPU for evaluation (Metal binary search on sparse memory)
	"""

	def __init__(
		self,
		model,
		train_tokens: list[int],
		eval_tokens: list[int],
		global_top_k: list[int],
		tier_configs: list[tuple[int, int, int]],  # [(end_cluster, neurons, bits), ...]
		logger=None,
		use_hybrid: bool = True,  # Use hybrid CPU+GPU if available
	):
		"""
		Initialize evaluator with pre-encoded training/eval data.

		Args:
			model: RAMLM model (for encoding)
			train_tokens: Training token sequence
			eval_tokens: Evaluation token sequence
			global_top_k: Global top-k token IDs for negative sampling
			tier_configs: Tier configurations [(end_cluster, neurons_per_cluster, bits_per_neuron), ...]
			logger: Optional logging function
			use_hybrid: Use hybrid CPU+GPU evaluation if available (default True)
		"""
		self.model = model
		self.tier_configs = tier_configs
		self.num_tiers = len(tier_configs)
		self.log = logger or print

		# Check hybrid availability
		self.use_hybrid = use_hybrid and RUST_AVAILABLE and hasattr(ram_accelerator, 'evaluate_candidates_parallel_hybrid')
		if self.use_hybrid:
			# Verify sparse Metal is available
			self.use_hybrid = ram_accelerator.sparse_metal_available()

		mode_str = "hybrid CPU+GPU (56 cores)" if self.use_hybrid else "CPU-only (16 cores)"
		self.log(f"  Preparing parallel evaluator ({mode_str})...")

		# Pre-encode training data
		self._prepare_data(train_tokens, eval_tokens, global_top_k)

	def _prepare_data(self, train_tokens: list[int], eval_tokens: list[int], global_top_k: list[int]):
		"""Pre-encode all training and evaluation data."""
		# Encode training contexts
		train_bits = self.model.encode_sequence(train_tokens)  # [num_train, total_input_bits]
		self.num_train = train_bits.shape[0]
		self.total_input_bits = train_bits.shape[1]

		# Flatten to list for Rust
		self.train_input_bits = train_bits.flatten().bool().tolist()

		# Training targets
		self.train_true_clusters = train_tokens[self.model.context_size:]

		# Training negatives (expand global_top_k for each example)
		self.num_negatives = len(global_top_k)
		self.train_false_clusters = []
		for i in range(self.num_train):
			self.train_false_clusters.extend(global_top_k)

		# Encode evaluation contexts
		eval_bits = self.model.encode_sequence(eval_tokens)  # [num_eval, total_input_bits]
		self.num_eval = eval_bits.shape[0]
		self.eval_input_bits = eval_bits.flatten().bool().tolist()

		# Evaluation targets
		self.eval_targets = eval_tokens[self.model.context_size:]

		# Tier configs for Rust (flattened)
		self.tier_configs_flat = []
		for end_cluster, neurons, bits in self.tier_configs:
			self.tier_configs_flat.extend([end_cluster, neurons, bits])

		# Compute connection size per candidate
		self.conn_size_per_candidate = 0
		for end_cluster, neurons, bits in self.tier_configs:
			# Each tier contributes: num_clusters_in_tier * neurons * bits
			start = 0 if len(self.tier_configs_flat) <= 3 else self.tier_configs[self.tier_configs.index((end_cluster, neurons, bits)) - 1][0] if self.tier_configs.index((end_cluster, neurons, bits)) > 0 else 0
			# Actually, we need cumulative neuron count * bits
			pass  # Will compute below

		# Compute total neurons and connection size
		self.num_clusters = self.model.vocab_size
		total_neurons = 0
		start_cluster = 0
		for end_cluster, neurons_per_cluster, bits in self.tier_configs:
			num_tier_clusters = end_cluster - start_cluster
			total_neurons += num_tier_clusters * neurons_per_cluster
			start_cluster = end_cluster

		# Connection size: sum of (neurons_in_tier * bits_per_neuron) for all tiers
		# For simplicity, use the model's flattened connection size
		# Each candidate's connections are flattened: [neuron_0_connections, neuron_1_connections, ...]
		# where each neuron has bits_per_neuron connections

		# For tiered, we need variable-length per neuron based on tier
		# But for Rust, we flatten with proper indexing
		# The Rust code expects connections laid out per-neuron with each neuron's bits_per_neuron
		self._compute_conn_layout()

		self.log(f"    Train: {self.num_train:,} examples, {len(self.train_input_bits):,} bits")
		self.log(f"    Eval: {self.num_eval:,} examples")
		self.log(f"    Negatives: {self.num_negatives}")
		self.log(f"    Total neurons: {total_neurons:,}")
		self.log(f"    Connection size per candidate: {self.conn_size_per_candidate:,}")

	def _compute_conn_layout(self):
		"""Compute connection layout for tiered architecture."""
		# For each neuron, we need bits_per_neuron connections
		# Neurons are organized by tier: tier0 neurons, then tier1 neurons, etc.
		self.conn_size_per_candidate = 0
		start_cluster = 0
		for end_cluster, neurons_per_cluster, bits_per_neuron in self.tier_configs:
			num_tier_clusters = end_cluster - start_cluster
			num_tier_neurons = num_tier_clusters * neurons_per_cluster
			self.conn_size_per_candidate += num_tier_neurons * bits_per_neuron
			start_cluster = end_cluster

	def evaluate_candidates(self, candidates: list[torch.Tensor]) -> list[float]:
		"""
		Evaluate multiple candidate connectivities in parallel.

		Args:
			candidates: List of connectivity tensors

		Returns:
			List of cross-entropy scores (lower is better)
		"""
		if not RUST_AVAILABLE:
			raise RuntimeError("Rust accelerator not available for parallel evaluation")

		num_candidates = len(candidates)

		# Flatten all candidates into single array
		# Each candidate's connections need to be laid out correctly for tiered architecture
		candidates_flat = []
		for conn in candidates:
			candidates_flat.extend(conn.flatten().tolist())

		# Call Rust parallel evaluation (hybrid or CPU-only)
		if self.use_hybrid:
			results = ram_accelerator.evaluate_candidates_parallel_hybrid(
				candidates_flat,
				num_candidates,
				self.conn_size_per_candidate,
				self.train_input_bits,
				self.train_true_clusters,
				self.train_false_clusters,
				self.eval_input_bits,
				self.eval_targets,
				self.tier_configs_flat,
				self.num_tiers,
				self.num_train,
				self.num_eval,
				self.total_input_bits,
				self.num_clusters,
				self.num_negatives,
			)
		else:
			results = ram_accelerator.evaluate_candidates_parallel_tiered(
				candidates_flat,
				num_candidates,
				self.conn_size_per_candidate,
				self.train_input_bits,
				self.train_true_clusters,
				self.train_false_clusters,
				self.eval_input_bits,
				self.eval_targets,
				self.tier_configs_flat,
				self.num_tiers,
				self.num_train,
				self.num_eval,
				self.total_input_bits,
				self.num_clusters,
				self.num_negatives,
			)

		return results


@dataclass
class BenchmarkConfig:
	"""Configuration for RAMLM benchmark."""
	# Model architecture (uniform mode)
	neurons_per_cluster: int = 3
	bits_per_neuron: int = 8
	context_size: int = 4  # 4-gram context

	# Tiered architecture (overrides neurons_per_cluster/bits_per_neuron when set)
	# Format: [(count, neurons, bits), ...] where count=None means "rest"
	# Example: [(100, 11, 8), (400, 7, 8), (None, 5, 8)]
	tiers: Optional[list[tuple[Optional[int], int, int]]] = None

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
	ts_early_stop_pct: float = 0.5  # TS is focused, higher threshold (0.5%)

	# GA parameters
	ga_population: int = 20
	ga_generations: int = 50
	ga_early_stop_pct: float = 0.05  # GA needs diversity, lower threshold (0.05%)

	# Early stopping (checks every 5 gens/iters, patience=1 means 10 total)
	early_stop_patience: int = 1

	# Optimization subset sizes - multi-window sampling for diversity
	# Both train and eval use windows sampled from across the entire corpus
	opt_window_size: int = 200       # Size of each window (tokens)
	opt_train_windows: int = 50      # 50 windows × 200 = 10k tokens for training
	opt_eval_windows: int = 20       # 20 windows × 200 = 4k tokens for evaluation

	# Use windowed 10k pretrain instead of full 2.4M initial training
	# Faster optimization iterations, full retrain at end with best connectivity
	windowed_pretrain: bool = False

	# Per-tier metrics tracking (for tiered architectures)
	per_tier: bool = False

	# Acceleration mode
	cpu_only: bool = False  # Force CPU-only (disable hybrid CPU+GPU)


def load_wikitext2(tokenizer_type: TokenizerType = TokenizerType.GPT2,
					train_limit: Optional[int] = None,
					test_limit: Optional[int] = None,
					val_limit: Optional[int] = None):
	"""Load WikiText-2 dataset with GPT-2 tokenization."""
	try:
		from datasets import load_dataset
	except ImportError:
		log("ERROR: datasets library not installed. Run: pip install datasets")
		sys.exit(1)

	log("Loading WikiText-2 dataset...")
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	# Get GPT-2 tokenizer
	log(f"Using {tokenizer_type.name} tokenizer...")
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

	log(f"  Train: {len(train_tokens):,} tokens")
	log(f"  Test: {len(test_tokens):,} tokens")
	log(f"  Validation: {len(val_tokens):,} tokens")
	log(f"  Vocab size: {vocab_size:,}")

	return train_tokens, test_tokens, val_tokens, vocab_size


def log_header(title: str):
	"""Log a formatted header."""
	log()
	log_separator("=")
	log(f"  {title}")
	log_separator("=")


def log_section(title: str):
	"""Log a section divider."""
	log()
	log("-" * 50)
	log(f"  {title}")
	log("-" * 50)


def run_benchmark(config: BenchmarkConfig):
	"""Run the RAMLM benchmark."""
	log_header("RAMLM Full Benchmark")

	# Log configuration
	log("\nConfiguration:")
	if config.tiers:
		log(f"  architecture: Tiered")
		for i, (count, neurons, bits) in enumerate(config.tiers):
			count_str = str(count) if count is not None else "rest"
			log(f"    tier {i}: {count_str} clusters × {neurons} neurons × {bits} bits")
	else:
		log(f"  architecture: Uniform")
		log(f"  neurons_per_cluster: {config.neurons_per_cluster}")
		log(f"  bits_per_neuron: {config.bits_per_neuron}")
	log(f"  context_size: {config.context_size}")
	log(f"  global_top_k: {config.global_top_k}")
	log(f"  batch_size: {config.batch_size}")
	log(f"  optimize: {config.optimize}")
	if config.optimize:
		log(f"  strategy: {config.strategy}")
	log(f"  Rust available: {RUST_AVAILABLE}")
	log(f"  CPU cores: {RUST_CPU_CORES}")
	log(f"  Metal available: {METAL_AVAILABLE}")

	# Load data
	log_section("Loading Data")
	train_tokens, test_tokens, val_tokens, vocab_size = load_wikitext2(
		TokenizerType.GPT2,
		train_limit=config.train_limit,
		test_limit=config.test_limit,
		val_limit=config.val_limit,
	)

	# Compute global top-k from training data
	log_section("Computing Global Top-K")
	token_counts = Counter(train_tokens)
	global_top_k_tokens = [t for t, _ in token_counts.most_common(config.global_top_k)]
	log(f"  Top-{config.global_top_k} tokens cover {sum(token_counts[t] for t in global_top_k_tokens)/len(train_tokens)*100:.1f}% of training data")

	# Create RAMLM
	log_section("Creating RAMLM Model")

	if config.tiers:
		# Tiered architecture: assign clusters by frequency
		# cluster_order maps logical tier position → physical cluster (token ID)
		# So logical position 0 (tier 0) → most frequent token
		# Include all tokens in vocab, even those not in training data
		seen_tokens = set(t for t, _ in token_counts.most_common())
		unseen_tokens = [t for t in range(vocab_size) if t not in seen_tokens]
		cluster_order = [t for t, _ in token_counts.most_common()] + unseen_tokens

		model = RAMLM(
			vocab_size=vocab_size,
			context_size=config.context_size,
			tiers=config.tiers,
			cluster_order=cluster_order,
		)
		log(f"  Architecture: Tiered ({len(model.layer.tier_configs)} tiers)")
		log(f"  Total neurons: {model.layer.total_neurons:,}")
		log(f"  Total memory cells: {model.layer.total_memory_cells:,}")
		for i, tc in enumerate(model.layer.tier_configs):
			log(f"    Tier {i}: {tc.cluster_count:,} clusters × {tc.neurons_per_cluster} neurons × {tc.bits_per_neuron} bits")
	else:
		# Uniform architecture
		model = RAMLM(
			vocab_size=vocab_size,
			context_size=config.context_size,
			neurons_per_cluster=config.neurons_per_cluster,
			bits_per_neuron=config.bits_per_neuron,
		)
		log(f"  Architecture: Uniform")
		log(f"  Total neurons: {model.layer.total_neurons:,}")
		log(f"  Memory addresses per neuron: {2**config.bits_per_neuron}")

	# Set global top-k
	model._global_top_k = global_top_k_tokens
	log(f"  Total input bits: {model.total_input_bits}")

	# Determine backend for logging
	backend = "Rust" if RUST_AVAILABLE else "PyTorch"
	val_stats = None  # Will be set if initial training runs

	# Initial training (use windowed pretrain if optimizing for faster iterations)
	if config.windowed_pretrain:
		log_section("Using Windowed Pretrain (10k tokens)")
		log(f"  (Will pretrain on {config.opt_train_windows}×{config.opt_window_size}={config.opt_train_windows * config.opt_window_size:,} tokens, full retrain after optimization)")
		train_time = 0
	else:
		log_section("Initial Training (Rust-Accelerated)")
		total_examples = len(train_tokens) - config.context_size
		log(f"Training on {total_examples:,} examples (batch_size={config.batch_size})...")
		start = time.perf_counter()

		if RUST_AVAILABLE:
			stats = model.train_epoch_fast_rust(
				train_tokens,
				global_top_k=config.global_top_k,
				batch_size=config.batch_size,
				verbose=True,  # Show progress
			)
		else:
			stats = model.train_epoch_fast(
				train_tokens,
				global_top_k=config.global_top_k,
				batch_size=config.batch_size,
				verbose=True,  # Show progress
			)

		train_time = time.perf_counter() - start
		log(f"  Backend: {backend}")
		log(f"  Training time: {train_time:.1f}s")
		log(f"  Modified cells: {stats['modified']:,}")
		log(f"  Examples: {stats['examples']:,}")
		log(f"  Throughput: {stats['examples']/train_time:,.0f} examples/sec")

		# Evaluate on validation set
		log_section("Initial Evaluation (Validation Set)")
		val_examples = len(val_tokens) - config.context_size
		log(f"Evaluating on {val_examples:,} examples...")
		start = time.perf_counter()
		val_stats = model.evaluate_fast(
			val_tokens,
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,  # Suppress print, we log ourselves
		)
		eval_time = time.perf_counter() - start

		log(f"  Evaluation time: {eval_time:.2f}s")
		log(f"  Cross-entropy: {val_stats['cross_entropy']:.4f}")
		log(f"  Perplexity: {val_stats['perplexity']:.2f}")
		log(f"  Accuracy: {val_stats['accuracy']:.2%}")

	# Connectivity optimization
	if config.optimize:
		log_section("Connectivity Optimization")
		import random as rand_module

		# Parse strategy sequence
		strategies = [s.strip().upper() for s in config.strategy.split(',')]

		# Get model dimensions - handle both uniform and tiered
		total_input_bits = model.total_input_bits

		# For tiered models, use JOINT optimization (all tiers together)
		# This is ~3x faster than per-tier optimization because we only train once per evaluation
		if config.tiers:
			num_neurons = model.layer.total_neurons
			bits_per_tier = [tc.bits_per_neuron for tc in model.layer.tier_configs]
			is_variable_bits = len(set(bits_per_tier)) > 1

			if is_variable_bits:
				log(f"Joint tiered optimization: {len(model.layer.tier_configs)} tiers → {num_neurons:,} neurons (variable bits: {bits_per_tier})")
			else:
				log(f"Joint tiered optimization: {len(model.layer.tier_configs)} tiers → {num_neurons:,} neurons × {bits_per_tier[0]} bits")

			# Find max widths for alignment
			max_clusters = max(tc.cluster_count for tc in model.layer.tier_configs)
			max_neurons_pc = max(tc.neurons_per_cluster for tc in model.layer.tier_configs)
			max_total = max(tc.total_neurons for tc in model.layer.tier_configs)
			cw = len(f"{max_clusters:,}")  # cluster width
			nw = len(f"{max_neurons_pc}")   # neurons per cluster width
			tw = len(f"{max_total:,}")      # total neurons width
			for tier_idx, tc in enumerate(model.layer.tier_configs):
				log(f"  Tier {tier_idx}: {tc.cluster_count:>{cw},} clusters × {tc.neurons_per_cluster:>{nw}} neurons × {tc.bits_per_neuron} bits = {tc.total_neurons:>{tw},} neurons")

			# Helper functions to get/set combined connectivity across all tiers
			# FLATTEN to 1D to handle variable bits_per_neuron across tiers
			def get_combined_connections() -> torch.Tensor:
				"""Concatenate all tier connectivities into one flattened tensor."""
				return torch.cat([mem.connections.flatten() for mem in model.layer.tier_memories])

			def set_combined_connections(combined: torch.Tensor) -> None:
				"""Split flattened tensor and reshape each tier's connectivity."""
				offset = 0
				for tc, mem in zip(model.layer.tier_configs, model.layer.tier_memories):
					size = tc.total_neurons * tc.bits_per_neuron
					mem.connections = combined[offset:offset + size].view(tc.total_neurons, tc.bits_per_neuron).clone()
					offset += size

			is_tiered = True
			# For GA/TS mutation bounds, use max bits (connections are bounded by total_input_bits anyway)
			bits_per_neuron = max(bits_per_tier)
		else:
			# Uniform model - single layer
			num_neurons = model.layer.total_neurons
			bits_per_neuron = model.layer.bits_per_neuron
			bits_per_tier = [bits_per_neuron]
			is_tiered = False
			is_variable_bits = False

		# Current connections (combined for tiered, direct for uniform)
		if is_tiered:
			current_connections = get_combined_connections()
		else:
			current_connections = model.layer.memory.connections.clone()

		# ========================================================================
		# Create optimization subsets using multi-window sampling
		# Both train and eval sample windows from across the ENTIRE corpus
		# This ensures diversity and avoids distribution mismatch
		# ========================================================================
		corpus_end = len(train_tokens) - config.opt_window_size - 10
		total_windows_needed = config.opt_train_windows + config.opt_eval_windows

		sampling_rng = rand_module.Random(42)  # Consistent seed for reproducibility

		if corpus_end > config.opt_window_size * total_windows_needed:
			# Sample all windows at once, then split between train and eval
			all_window_starts = sampling_rng.sample(range(0, corpus_end), total_windows_needed)
			all_window_starts.sort()  # Cache-friendly access

			# Split: first N for train, rest for eval (both from same distribution)
			train_starts = all_window_starts[:config.opt_train_windows]
			eval_starts = all_window_starts[config.opt_train_windows:]

			opt_train_tokens = []
			for start in train_starts:
				opt_train_tokens.extend(train_tokens[start:start + config.opt_window_size])

			opt_eval_tokens = []
			for start in eval_starts:
				opt_eval_tokens.extend(train_tokens[start:start + config.opt_window_size])

			log(f"Multi-window train: {config.opt_train_windows} windows × {config.opt_window_size} = {len(opt_train_tokens):,} tokens")
			log(f"Multi-window eval: {config.opt_eval_windows} windows × {config.opt_window_size} = {len(opt_eval_tokens):,} tokens")
		else:
			# Fallback if corpus too small - use sequential
			train_size = config.opt_train_windows * config.opt_window_size
			eval_size = config.opt_eval_windows * config.opt_window_size
			opt_train_tokens = train_tokens[:train_size]
			opt_eval_tokens = train_tokens[train_size:train_size + eval_size]
			log(f"Sequential train: {len(opt_train_tokens):,} tokens (corpus too small for multi-window)")
			log(f"Sequential eval: {len(opt_eval_tokens):,} tokens")

		# Create evaluation function (handles both tiered and uniform)
		def evaluate_connectivity(conn: torch.Tensor) -> float:
			"""Train model with connectivity and return cross-entropy."""
			if is_tiered:
				set_combined_connections(conn)
			else:
				model.layer.memory.connections = conn
			model.reset_memory()

			# Train on multi-window subset (10k tokens by default)
			if RUST_AVAILABLE:
				model.train_epoch_fast_rust(
					opt_train_tokens,
					global_top_k=config.global_top_k,
					batch_size=config.batch_size,
					verbose=False,
				)
			else:
				model.train_epoch_fast(
					opt_train_tokens,
					global_top_k=config.global_top_k,
					batch_size=config.batch_size,
					verbose=False,
				)

			# Evaluate on multi-window sampled tokens
			stats = model.evaluate_fast(
				opt_eval_tokens,
				batch_size=config.batch_size * 2,
				backend=AccelerationMode.AUTO,
				verbose=False,
			)
			return stats['cross_entropy']

		# Create parallel evaluator for tiered architectures (HUGE speedup)
		parallel_evaluator = None
		if is_tiered and RUST_AVAILABLE and hasattr(ram_accelerator, 'evaluate_candidates_parallel_tiered'):
			try:
				# Build tier configs for Rust: [(end_cluster, neurons_per_cluster, bits_per_neuron), ...]
				rust_tier_configs = []
				for tc in model.layer.tier_configs:
					rust_tier_configs.append((tc.cluster_end, tc.neurons_per_cluster, tc.bits_per_neuron))

				parallel_evaluator = ParallelTieredEvaluator(
					model=model,
					train_tokens=opt_train_tokens,
					eval_tokens=opt_eval_tokens,
					global_top_k=list(model._global_top_k) if model._global_top_k else list(range(config.global_top_k)),
					tier_configs=rust_tier_configs,
					logger=log,
					use_hybrid=not config.cpu_only,  # Use hybrid unless --cpu-only
				)
				mode_cores = "16 CPU" if config.cpu_only else "56 CPU+GPU"
				log(f"  Parallel evaluator ready ({mode_cores} cores)")
			except Exception as e:
				log(f"  WARNING: Failed to create parallel evaluator: {e}")
				log(f"  Falling back to sequential evaluation")
				parallel_evaluator = None

		# Batch evaluation function
		def batch_evaluate(connectivities: list, **_kwargs) -> list:
			# Use parallel evaluator if available (16x speedup for tiered architectures)
			if parallel_evaluator is not None and len(connectivities) > 1:
				return parallel_evaluator.evaluate_candidates(connectivities)
			# Fallback to sequential evaluation
			return [evaluate_connectivity(conn) for conn in connectivities]

		# If using windowed pretrain, train on 10k subset first for baseline
		# This gives us a real baseline instead of random (PPL = vocab_size)
		if config.windowed_pretrain:
			log("")
			log(f"Pretraining on {len(opt_train_tokens):,} tokens for baseline...")
			start = time.perf_counter()
			if RUST_AVAILABLE:
				pretrain_stats = model.train_epoch_fast_rust(
					opt_train_tokens,
					global_top_k=config.global_top_k,
					batch_size=config.batch_size,
					verbose=False,
				)
			else:
				pretrain_stats = model.train_epoch_fast(
					opt_train_tokens,
					global_top_k=config.global_top_k,
					batch_size=config.batch_size,
					verbose=False,
				)
			pretrain_time = time.perf_counter() - start
			log(f"  Pretrain time: {pretrain_time:.1f}s, modified {pretrain_stats['modified']:,} cells")

		# Compute baselines for overfitting detection
		log("")
		log(f"Computing baselines for overfitting detection...")
		baseline_train_stats = model.evaluate_fast(
			opt_train_tokens[:min(20000, len(opt_train_tokens))],
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_train_ce = baseline_train_stats['cross_entropy']
		baseline_train_ppl = baseline_train_stats['perplexity']

		# Validation baseline: evaluate on validation subset
		baseline_val_stats = model.evaluate_fast(
			val_tokens[:min(50000, len(val_tokens))],
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_val_ce = baseline_val_stats['cross_entropy']
		baseline_val_ppl = baseline_val_stats['perplexity']

		# Optimization eval baseline: evaluate on opt_eval_tokens (same as GA fitness)
		# This should match the GA initial best error
		baseline_opt_eval_stats = model.evaluate_fast(
			opt_eval_tokens,
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_opt_eval_ce = baseline_opt_eval_stats['cross_entropy']
		baseline_opt_eval_ppl = baseline_opt_eval_stats['perplexity']

		global_baseline_ratio = baseline_val_ce / baseline_train_ce if baseline_train_ce > 0 else 1.0

		log("")
		log(f"Optimization baselines:")
		log(f"  Train CE: {baseline_train_ce:.4f} (PPL: {baseline_train_ppl:.2f}) - opt_train_tokens")
		log(f"  Opt Eval CE: {baseline_opt_eval_ce:.4f} (PPL: {baseline_opt_eval_ppl:.2f}) - opt_eval_tokens (GA fitness)")
		log(f"  Val CE: {baseline_val_ce:.4f} (PPL: {baseline_val_ppl:.2f}) - val_tokens")
		log(f"  Val/Train ratio: {global_baseline_ratio:.2f}x")

		log("")
		log(f"Optimization config:")
		log(f"  GA: {config.ga_population} pop × {config.ga_generations} max gens, early_stop <{config.ga_early_stop_pct}%")
		log(f"  TS: {config.ts_neighbors} neighbors × {config.ts_iterations} max iters, early_stop <{config.ts_early_stop_pct}%")
		log(f"  Early stop patience: {config.early_stop_patience} (checks every 5 = {(config.early_stop_patience+1)*5} gens/iters)")
		log(f"  Overfitting thresholds: <{HEALTHY_THRESHOLD}% healthy, >{WARNING_THRESHOLD}% warn, >{SEVERE_THRESHOLD}% severe, >{CRITICAL_THRESHOLD}% stop")

		# Create overfitting monitor with validation function
		def val_eval_fn(conn: torch.Tensor) -> float:
			if is_tiered:
				set_combined_connections(conn)
			else:
				model.layer.memory.connections = conn
			model.reset_memory()
			if RUST_AVAILABLE:
				model.train_epoch_fast_rust(opt_train_tokens, global_top_k=config.global_top_k, batch_size=config.batch_size, verbose=False)
			else:
				model.train_epoch_fast(opt_train_tokens, global_top_k=config.global_top_k, batch_size=config.batch_size, verbose=False)
			return model.evaluate_fast(val_tokens[:min(50000, len(val_tokens))], batch_size=config.batch_size * 2, backend=AccelerationMode.AUTO, verbose=False)['cross_entropy']

		overfitting_monitor = OverfittingMonitor(
			validation_fn=val_eval_fn,
			grace_checks=1,
			logger=log,
			global_baseline_ratio=global_baseline_ratio,
		)

		# Run optimization strategies (joint for tiered, same code for uniform)
		arch_label = "Joint tiered" if is_tiered else "Uniform"
		for strategy_name in strategies:
			log("")
			if is_tiered and is_variable_bits:
				log(f"[{arch_label}] Running {strategy_name} optimization ({num_neurons:,} neurons, bits={bits_per_tier})...")
			else:
				bits_str = bits_per_tier[0] if is_tiered else bits_per_neuron
				log(f"[{arch_label}] Running {strategy_name} optimization ({num_neurons:,} neurons × {bits_str} bits)...")

			start = time.perf_counter()

			if strategy_name == "GA":
				ga_config = GeneticAlgorithmConfig(
					population_size=config.ga_population,
					generations=config.ga_generations,
					mutation_rate=0.01,
					crossover_rate=0.7,
					elitism=max(2, config.ga_population // 10),
					early_stop_patience=config.early_stop_patience,
					early_stop_threshold_pct=config.ga_early_stop_pct,
				)
				strategy_obj = GeneticAlgorithmStrategy(config=ga_config, seed=42, verbose=True, logger=log)
				result = strategy_obj.optimize(
					current_connections,
					evaluate_connectivity,
					total_input_bits,
					num_neurons,
					bits_per_neuron,
					batch_evaluate,
					overfitting_callback=overfitting_monitor,
				)

			elif strategy_name == "TS":
				ts_config = TabuSearchConfig(
					iterations=config.ts_iterations,
					neighbors_per_iter=config.ts_neighbors,
					early_stop_patience=config.early_stop_patience,
					early_stop_threshold_pct=config.ts_early_stop_pct,
				)
				strategy_obj = TabuSearchStrategy(config=ts_config, seed=42, verbose=True, logger=log)
				result = strategy_obj.optimize(
					current_connections,
					evaluate_connectivity,
					total_input_bits,
					num_neurons,
					bits_per_neuron,
					batch_evaluate,
					overfitting_callback=overfitting_monitor,
				)

			else:
				log(f"  WARNING: Unknown strategy {strategy_name}, skipping")
				continue

			opt_time = time.perf_counter() - start

			# Update current connections (for next strategy in sequence)
			current_connections = result.optimized_connections.clone()
			if is_tiered:
				set_combined_connections(current_connections)
			else:
				model.layer.memory.connections = current_connections

			# Report results for this strategy
			improvement = ((result.initial_error - result.final_error) / result.initial_error * 100) if result.initial_error > 0 else 0
			log("")
			log(f"[{arch_label}] {strategy_name} Results:")
			log(f"    Time: {opt_time:.1f}s")
			log(f"    Initial CE: {result.initial_error:.4f}")
			log(f"    Final CE: {result.final_error:.4f}")
			log(f"    Improvement: {improvement:.2f}%")
			log(f"    Iterations: {result.iterations_run}")
			if result.early_stopped_overfitting:
				log(f"    Early stopped: overfitting detected")

		# Retrain on FULL data with optimized connectivity
		log("")
		log(f"Retraining on full data with optimized connectivity...")
		model.reset_memory()
		start = time.perf_counter()
		if RUST_AVAILABLE:
			model.train_epoch_fast_rust(
				train_tokens,
				global_top_k=config.global_top_k,
				batch_size=config.batch_size,
				verbose=True,
			)
		else:
			model.train_epoch_fast(
				train_tokens,
				global_top_k=config.global_top_k,
				batch_size=config.batch_size,
				verbose=True,
			)
		retrain_time = time.perf_counter() - start
		log(f"  Full retraining time: {retrain_time:.1f}s")

	# Final evaluation on validation set (always evaluate, even if windowed pretrain was used)
	log_section("Final Evaluation (Validation Set)")
	val_examples = len(val_tokens) - config.context_size
	log(f"Evaluating on {val_examples:,} examples...")
	start = time.perf_counter()
	final_val_stats = model.evaluate_fast(
		val_tokens,
		batch_size=config.batch_size * 2,
		backend=AccelerationMode.AUTO,
		verbose=False,
		per_tier=config.per_tier,
	)
	val_eval_time = time.perf_counter() - start

	log(f"  Evaluation time: {val_eval_time:.2f}s")
	log(f"  Cross-entropy: {final_val_stats['cross_entropy']:.4f}")
	log(f"  Perplexity: {final_val_stats['perplexity']:.2f}")
	log(f"  Accuracy: {final_val_stats['accuracy']:.2%}")

	# Final evaluation on test set
	log_section("Final Evaluation (Test Set)")
	test_examples = len(test_tokens) - config.context_size
	log(f"Evaluating on {test_examples:,} examples...")
	start = time.perf_counter()
	test_stats = model.evaluate_fast(
		test_tokens,
		batch_size=config.batch_size * 2,
		backend=AccelerationMode.AUTO,
		verbose=False,
		per_tier=config.per_tier,
	)
	eval_time = time.perf_counter() - start

	log(f"  Evaluation time: {eval_time:.2f}s")
	log(f"  Cross-entropy: {test_stats['cross_entropy']:.4f}")
	log(f"  Perplexity: {test_stats['perplexity']:.2f}")
	log(f"  Accuracy: {test_stats['accuracy']:.2%}")

	# Summary
	log_header("Summary")
	if config.tiers:
		tier_summary = ", ".join(f"{tc.cluster_count}×{tc.neurons_per_cluster}n" for tc in model.layer.tier_configs)
		log(f"Model: RAMLM Tiered (vocab={vocab_size}, tiers=[{tier_summary}], total_neurons={model.layer.total_neurons:,})")
	else:
		log(f"Model: RAMLM (vocab={vocab_size}, neurons={config.neurons_per_cluster}/cluster, bits={config.bits_per_neuron})")
	log(f"Data: WikiText-2 (train={len(train_tokens):,}, val={len(val_tokens):,}, test={len(test_tokens):,})")
	log(f"Backend: {backend} ({'Rust + rayon parallel' if RUST_AVAILABLE else 'PyTorch vectorized'})")
	log()
	log("Results:")

	# Show before/after comparison if we have initial stats
	if val_stats is not None and config.optimize:
		# Validation PPL
		val_ppl_before = val_stats['perplexity']
		val_ppl_after = final_val_stats['perplexity']
		val_ppl_delta = (val_ppl_after - val_ppl_before) / val_ppl_before * 100
		val_ppl_sign = "+" if val_ppl_delta >= 0 else ""
		log(f"  Validation PPL: {val_ppl_before:.2f} → {val_ppl_after:.2f} ({val_ppl_sign}{val_ppl_delta:.2f}%)")

		# Validation Acc
		val_acc_before = val_stats['accuracy']
		val_acc_after = final_val_stats['accuracy']
		val_acc_delta = (val_acc_after - val_acc_before) / val_acc_before * 100 if val_acc_before > 0 else 0
		val_acc_sign = "+" if val_acc_delta >= 0 else ""
		log(f"  Validation Acc: {val_acc_before:.2%} → {val_acc_after:.2%} ({val_acc_sign}{val_acc_delta:.2f}%)")
	else:
		log(f"  Validation PPL: {final_val_stats['perplexity']:.2f}")
		log(f"  Validation Acc: {final_val_stats['accuracy']:.2%}")

	log(f"  Test PPL: {test_stats['perplexity']:.2f}")
	log(f"  Test Acc: {test_stats['accuracy']:.2%}")

	if config.optimize:
		log(f"  Optimization: {config.strategy}")

	# Per-tier breakdown (if enabled and tiered)
	if config.per_tier and 'by_tier' in test_stats:
		log()
		log("Per-Tier Test Results:")
		log("=" * 70)

		# Header
		log(f"{'Tier':<8} {'Clusters':>10} {'Neurons':>8} {'Data %':>8} {'PPL':>12} {'Accuracy':>10}")
		log("-" * 70)

		for ts in test_stats['by_tier']:
			tier_name = ts['name'].replace('tier_', 'Tier ')
			log(f"{tier_name:<8} {ts['cluster_count']:>10,} {ts['neurons_per_cluster']:>8} "
				f"{ts['data_pct']:>7.1f}% {ts['perplexity']:>12.1f} {ts['accuracy']:>9.2%}")

		log("-" * 70)
		log(f"{'TOTAL':<8} {sum(ts['cluster_count'] for ts in test_stats['by_tier']):>10,} "
			f"{'':>8} {'100.0':>7}% {test_stats['perplexity']:>12.1f} {test_stats['accuracy']:>9.2%}")

	log()
	log_separator("=")
	log(f"Log file: {LOG_FILENAME}")
	log_separator("=")

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
		help="Neurons per cluster (default: 3 fast, 4 full, 6 overnight). Ignored if --tiered is set.")
	parser.add_argument("--bits", type=int, default=None,
		help="Bits per neuron (default: 8 fast, 10 full, 12 overnight). Ignored if --tiered is set.")
	parser.add_argument("--context", type=int, default=4,
		help="Context size in tokens (default: 4)")
	parser.add_argument("--tiered", type=str, default=None,
		help="Tiered architecture spec: 'count1,neurons1,bits1;count2,neurons2,bits2;...' "
		     "Use 'rest' for remaining clusters. Example: '100,11,8;400,7,8;rest,5,8'")

	# Training
	parser.add_argument("--top-k", type=int, default=100,
		help="Global top-k for FALSE training (default: 100)")
	parser.add_argument("--batch-size", type=int, default=500,
		help="Batch size for training (default: 500)")

	# Optimization
	parser.add_argument("--optimize", action="store_true",
		help="Run connectivity optimization after initial training")
	parser.add_argument("--windowed-pretrain", action="store_true",
		help="Use windowed 10k pretrain instead of full 2.4M initial training (faster optimization)")
	parser.add_argument("--strategy", type=str, default="GA,TS",
		help="Optimization strategy: GA, TS, SA, or comma-separated sequence like GA,TS (default: GA,TS)")

	# Optimization parameters
	parser.add_argument("--ts-iters", type=int, default=None,
		help="Tabu Search iterations (default: 5 fast, 10 full, 20 overnight)")
	parser.add_argument("--ts-neighbors", type=int, default=30,
		help="Tabu Search neighbors per iteration (default: 30)")
	parser.add_argument("--ga-pop", type=int, default=None,
		help="GA population size (default: 10 fast, 20 full, 30 overnight)")
	parser.add_argument("--ga-gens", type=int, default=None,
		help="GA generations (default: 20 fast, 50 full, 100 overnight)")

	# Evaluation options
	parser.add_argument("--per-tier", action="store_true",
		help="Track per-tier PPL and accuracy (only for tiered architectures)")

	# Acceleration options
	parser.add_argument("--cpu-only", action="store_true",
		help="Force CPU-only evaluation (disable hybrid CPU+GPU). Default uses hybrid if available.")

	args = parser.parse_args()

	# Setup logging first
	log_file = setup_logging()
	print(f"\n{'='*70}")
	print(f"LOG FILE: {log_file}")
	print(f"{'='*70}")
	print(f"Tail command: tail -f \"{log_file}\"")
	print(f"{'='*70}\n")

	# Log session start
	log_separator()
	log("RAMLM BENCHMARK - SESSION START")
	log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	log_separator()

	# Configure based on mode
	config = BenchmarkConfig()
	config.context_size = args.context
	config.global_top_k = args.top_k
	config.batch_size = args.batch_size
	config.optimize = args.optimize
	config.strategy = args.strategy
	config.windowed_pretrain = args.windowed_pretrain
	config.per_tier = args.per_tier
	config.cpu_only = args.cpu_only

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
		config.neurons_per_cluster = args.neurons or 8
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

	# Parse tiered architecture if specified
	if args.tiered:
		tiers = []
		for tier_spec in args.tiered.split(';'):
			parts = tier_spec.strip().split(',')
			if len(parts) != 3:
				print(f"ERROR: Invalid tier spec '{tier_spec}'. Expected 'count,neurons,bits'")
				sys.exit(1)
			count_str, neurons_str, bits_str = parts
			count = None if count_str.lower() == 'rest' else int(count_str)
			neurons = int(neurons_str)
			bits = int(bits_str)
			tiers.append((count, neurons, bits))
		config.tiers = tiers

	# Log configuration
	log(f"Mode: {args.mode.upper()}")
	log(f"Full data: {args.full_data}")
	log(f"Optimize: {args.optimize}")
	if args.optimize:
		log(f"Strategy: {args.strategy}")
	if config.tiers:
		log(f"Tiered: {config.tiers}")

	# Run benchmark
	session_start = time.perf_counter()
	results = run_benchmark(config)
	session_duration = time.perf_counter() - session_start

	# Log total duration
	minutes = int(session_duration // 60)
	seconds = int(session_duration % 60)
	log(f"Total Duration: {minutes}m {seconds}s")

	# Log session end
	log_separator()
	log("SESSION COMPLETE")
	log(f"Log file: {LOG_FILENAME}")
	log_separator()
