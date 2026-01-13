#!/usr/bin/env python3
"""
RAMLM Full Benchmark - Rust-Accelerated Training with GA/TS Optimization

This benchmark tests the core RAMLM using:
- GPT-2 tokenizer (50k vocab)
- WikiText-2 full dataset
- Rust-accelerated training (3.35x faster than PyTorch)
- ConnectivityOptimizer with GA/TS for connectivity optimization

Usage (Single Run):
	# Quick test (smaller data, fewer iterations)
	python tests/ramlm_full_benchmark.py --mode fast

	# Full benchmark with optimization
	python tests/ramlm_full_benchmark.py --mode full --optimize

	# Full data with GA then TS optimization
	python tests/ramlm_full_benchmark.py --full-data --optimize --strategy GA,TS

Usage (Sweep Mode - Multiple Experiments):
	# Quick sweep (~4-6 hours, 4 experiments, GA+TS enabled by default)
	python tests/ramlm_full_benchmark.py --sweep --set quick

	# Standard sweep (~8-10 hours, 6 experiments)
	python tests/ramlm_full_benchmark.py --sweep --set standard

	# Extended sweep (~16-20 hours, 10 experiments)
	python tests/ramlm_full_benchmark.py --sweep --set extended

	# Run specific experiments
	python tests/ramlm_full_benchmark.py --sweep --experiments tier0_16bit,balanced_14bit

	# Disable optimization in sweep mode
	python tests/ramlm_full_benchmark.py --sweep --set quick --no-optimize
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import logging
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from wnn.ram.core import AccelerationMode, OptimizationMethod, TierResultsTable
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
from wnn.ram.strategies.connectivity.per_cluster import (
	FitnessMode,
	TierOptConfig,
	IncrementalEvaluator,
	PerClusterOptimizer,
	RustPerClusterOptimizer,
	create_rust_optimizer,
	_check_rust_per_cluster,
)
from wnn.tokenizers import TokenizerType, TokenizerFactory


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_fitness_mode(mode_str: str) -> FitnessMode:
	"""Convert fitness mode string to FitnessMode enum."""
	mode_map = {
		"SIMPLE": FitnessMode.SIMPLE,
		"CE": FitnessMode.CROSS_ENTROPY,
		"PENALIZE": FitnessMode.PENALIZE_HIGH_VOTES,
		"ACCURACY": FitnessMode.ACCURACY,
	}
	return mode_map.get(mode_str.upper(), FitnessMode.SIMPLE)


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


def log(message: str = "", flush: bool = True):
	"""Log message to both console and file."""
	logger = logging.getLogger('ramlm_benchmark')
	logger.info(message)
	if flush:
		for handler in logger.handlers:
			handler.flush()


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
	# Train from wiki2 train, eval/dev from wiki2 test (different distribution)
	opt_window_size: int = 200       # Size of each window (tokens)
	opt_train_windows: int = 50      # 50 windows × 200 = 10k tokens for training (from wiki2 train)
	opt_eval_windows: int = 20       # 20 windows × 200 = 4k tokens for optimizer fitness (from wiki2 test)
	opt_dev_windows: int = 20        # 20 windows × 200 = 4k tokens for overfitting detection (from wiki2 test, different windows)

	# Use windowed 10k pretrain instead of full 2.4M initial training
	# Faster optimization iterations, full retrain at end with best connectivity
	windowed_pretrain: bool = False

	# Per-tier metrics tracking (for tiered architectures)
	per_tier: bool = False

	# Acceleration mode
	cpu_only: bool = False  # Force CPU-only (disable hybrid CPU+GPU)

	# Iterative refinement with grouped optimization
	iterative_passes: int = 1   # Number of optimization passes (2-3 for iterative refinement)
	group_size: int = 10        # Group size for joint optimization of tier 0/1 (0 disables)

	# Fitness function for PC optimization
	fitness_mode: str = "SIMPLE"  # SIMPLE, CE, or PENALIZE


# ============================================================================
# Sweep Mode - Multiple Experiment Support
# ============================================================================

@dataclass
class SweepExperiment:
	"""Configuration for a single sweep experiment."""
	name: str
	tiered: str
	context: int = 4
	description: str = ""
	priority: int = 1  # 1=quick, 2=standard, 3=extended


@dataclass
class SweepTierResult:
	"""Results for a single tier in sweep output."""
	tier: int
	name: str
	clusters: int
	neurons: int
	bits: int
	data_pct: float
	ppl: float
	accuracy: float


@dataclass
class SweepOptResult:
	"""Results from optimization phase."""
	strategy: str
	initial_ce: float
	final_ce: float
	improvement_pct: float
	iterations: int


@dataclass
class SweepResult:
	"""Results from a single sweep experiment."""
	name: str
	config: str
	context: int
	overall_ppl: float
	overall_accuracy: float
	tier_results: list[SweepTierResult]
	train_time: float
	eval_time: float
	timestamp: str
	initial_val_ppl: Optional[float] = None
	final_val_ppl: Optional[float] = None
	val_ppl_improvement_pct: Optional[float] = None
	optimization_results: Optional[list[SweepOptResult]] = None
	total_ce_improvement_pct: Optional[float] = None


def define_sweep_experiments() -> list[SweepExperiment]:
	"""Define sweep experiments with priority levels.

	Priority levels:
	- 1 = quick set (4 experiments, ~4-6 hours)
	- 2 = standard set (6 experiments, ~8-10 hours)
	- 3 = extended set (10 experiments, ~16-20 hours)
	"""
	experiments = []

	# Priority 1: Quick set (~4-6 hours)
	experiments.append(SweepExperiment(
		name="tier0_16bit",
		tiered="100,15,16;400,10,10;rest,5,8",
		description="Tier0: 15 neurons, 16 bits (SPARSE) - best capacity boost",
		priority=1,
	))
	experiments.append(SweepExperiment(
		name="balanced_14bit",
		tiered="100,12,14;400,8,12;rest,5,10",
		description="Balanced SPARSE: 14→12→10 bits gradient",
		priority=1,
	))
	experiments.append(SweepExperiment(
		name="neurons_20_tier0",
		tiered="100,20,12;400,12,10;rest,7,8",
		description="High neurons: 20→12→7 with moderate bits",
		priority=1,
	))
	experiments.append(SweepExperiment(
		name="context_6_sparse",
		tiered="100,12,14;400,8,10;rest,5,8",
		context=6,
		description="Context 6 with SPARSE tier0/tier1",
		priority=1,
	))

	# Priority 2: Standard set (~8-10 hours total)
	experiments.append(SweepExperiment(
		name="tier0_18bit",
		tiered="100,15,18;400,10,12;rest,5,8",
		description="Tier0: 18 bits (262K addresses per neuron)",
		priority=2,
	))
	experiments.append(SweepExperiment(
		name="neurons_25_gradient",
		tiered="100,25,14;400,15,10;rest,7,8",
		description="Max neurons: 25→15→7 gradient",
		priority=2,
	))

	# Priority 3: Extended/weekend set (~16-20 hours total)
	experiments.append(SweepExperiment(
		name="tier0_20bit",
		tiered="100,15,20;400,10,12;rest,5,8",
		description="Tier0: 20 bits (1M addresses per neuron)",
		priority=3,
	))
	experiments.append(SweepExperiment(
		name="all_sparse_16bit",
		tiered="100,15,16;400,12,16;rest,8,16",
		description="All tiers 16-bit SPARSE",
		priority=3,
	))
	experiments.append(SweepExperiment(
		name="context_8_high_cap",
		tiered="100,15,12;400,10,10;rest,7,8",
		context=8,
		description="Context 8 with high capacity",
		priority=3,
	))
	experiments.append(SweepExperiment(
		name="extreme_tier0",
		tiered="100,30,16;400,12,12;rest,5,8",
		context=5,
		description="Extreme: 30 neurons, 16 bits tier0",
		priority=3,
	))

	# Priority 4: Asymmetric capacity experiments (based on insight that
	# frequent tokens benefit from more capacity, rare tokens need minimal)
	experiments.append(SweepExperiment(
		name="asymmetric_extreme_t0",
		tiered="100,25,24;400,8,10;rest,5,8",
		context=8,
		description="Extreme tier0: 25n×24b, minimal tier1/2",
		priority=4,
	))
	experiments.append(SweepExperiment(
		name="asymmetric_expanded_t0",
		tiered="200,25,20;300,8,10;rest,4,8",
		context=8,
		description="Expanded tier0: 200 tokens, 25n×20b",
		priority=4,
	))
	experiments.append(SweepExperiment(
		name="two_tier_simple",
		tiered="500,15,16;rest,4,6",
		context=8,
		description="Two-tier: 500 frequent vs rest rare",
		priority=4,
	))

	# Priority 5: Fine-grained 5-tier architecture
	experiments.append(SweepExperiment(
		name="five_tier_gradient",
		tiered="50,37,22;50,31,20;400,11,12;20000,7,10;rest,3,11",
		context=8,
		description="5-tier: 50/50/400/20K/rest with gradient neurons",
		priority=5,
	))
	experiments.append(SweepExperiment(
		name="five_tier_balanced",
		tiered="50,27,21;450,23,20;10000,5,12;10000,5,10;rest,5,8",
		context=8,
		description="5-tier: 50/450/10K/10K/rest with balanced neurons",
		priority=5,
	))

	# Priority 6: Minimal tier architecture (opposite of 5-tier)
	experiments.append(SweepExperiment(
		name="two_tier_20bit",
		tiered="500,15,20;rest,5,8",
		context=4,
		description="Two-tier: 500 tokens at 15n×20b, rest at 5n×8b",
		priority=6,
	))

	return experiments


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
			per_tier=config.per_tier,
		)
		eval_time = time.perf_counter() - start

		log(f"  Evaluation time: {eval_time:.2f}s")
		log(f"  Cross-entropy: {val_stats['cross_entropy']:.4f}")
		log(f"  Perplexity: {val_stats['perplexity']:.2f}")
		log(f"  Accuracy: {val_stats['accuracy']:.2%}")

		# Per-tier breakdown for initial evaluation (if enabled and tiered)
		if config.per_tier and 'by_tier' in val_stats:
			log()
			TierResultsTable.from_stats("Initial Validation", val_stats).print(log)

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

			# Compute neuron_offsets for GA crossover to respect neuron boundaries
			# neuron_offsets[i] = starting connection index for neuron i
			neuron_offsets = [0]
			for tc in model.layer.tier_configs:
				for _ in range(tc.total_neurons):
					neuron_offsets.append(neuron_offsets[-1] + tc.bits_per_neuron)
		else:
			# Uniform model - single layer
			num_neurons = model.layer.total_neurons
			bits_per_neuron = model.layer.bits_per_neuron
			bits_per_tier = [bits_per_neuron]
			is_tiered = False
			is_variable_bits = False
			neuron_offsets = None  # Not needed for uniform architectures

		# Current connections (combined for tiered, direct for uniform)
		if is_tiered:
			current_connections = get_combined_connections()
		else:
			current_connections = model.layer.memory.connections.clone()

		# ========================================================================
		# Create optimization subsets using multi-window sampling
		# - opt_train: from wiki2 TRAIN (what we train on)
		# - opt_eval: from wiki2 TEST (optimizer fitness - different distribution)
		# - opt_dev: from wiki2 TEST (overfitting detection - different windows)
		# ========================================================================
		sampling_rng = rand_module.Random(42)  # Consistent seed for reproducibility

		# Sample opt_train windows from TRAIN corpus
		train_corpus_end = len(train_tokens) - config.opt_window_size - 10
		if train_corpus_end > config.opt_window_size * config.opt_train_windows:
			train_starts = sampling_rng.sample(range(0, train_corpus_end), config.opt_train_windows)
			train_starts.sort()  # Cache-friendly access
			opt_train_tokens = []
			for start in train_starts:
				opt_train_tokens.extend(train_tokens[start:start + config.opt_window_size])
		else:
			train_size = config.opt_train_windows * config.opt_window_size
			opt_train_tokens = train_tokens[:train_size]

		# Sample opt_eval and opt_dev windows from TEST corpus (different distribution!)
		test_corpus_end = len(test_tokens) - config.opt_window_size - 10
		total_test_windows = config.opt_eval_windows + config.opt_dev_windows

		if test_corpus_end > config.opt_window_size * total_test_windows:
			# Sample all test windows at once, then split between eval and dev
			all_test_starts = sampling_rng.sample(range(0, test_corpus_end), total_test_windows)
			all_test_starts.sort()  # Cache-friendly access

			# Split: first N for eval (optimizer fitness), rest for dev (overfitting detection)
			eval_starts = all_test_starts[:config.opt_eval_windows]
			dev_starts = all_test_starts[config.opt_eval_windows:]

			opt_eval_tokens = []
			for start in eval_starts:
				opt_eval_tokens.extend(test_tokens[start:start + config.opt_window_size])

			opt_dev_tokens = []
			for start in dev_starts:
				opt_dev_tokens.extend(test_tokens[start:start + config.opt_window_size])
		else:
			# Fallback if test corpus too small - use sequential
			eval_size = config.opt_eval_windows * config.opt_window_size
			dev_size = config.opt_dev_windows * config.opt_window_size
			opt_eval_tokens = test_tokens[:eval_size]
			opt_dev_tokens = test_tokens[eval_size:eval_size + dev_size]

		log(f"Multi-window train: {config.opt_train_windows} windows × {config.opt_window_size} = {len(opt_train_tokens):,} tokens (from wiki2 train)")
		log(f"Multi-window eval: {config.opt_eval_windows} windows × {config.opt_window_size} = {len(opt_eval_tokens):,} tokens (from wiki2 test - optimizer fitness)")
		log(f"Multi-window dev: {config.opt_dev_windows} windows × {config.opt_window_size} = {len(opt_dev_tokens):,} tokens (from wiki2 test - overfitting detection)")

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

		# Baseline on opt_train (from wiki2 train)
		baseline_train_stats = model.evaluate_fast(
			opt_train_tokens[:min(20000, len(opt_train_tokens))],
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_train_ce = baseline_train_stats['cross_entropy']
		baseline_train_ppl = baseline_train_stats['perplexity']

		# Baseline on opt_eval (from wiki2 test - optimizer fitness target)
		baseline_opt_eval_stats = model.evaluate_fast(
			opt_eval_tokens,
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_opt_eval_ce = baseline_opt_eval_stats['cross_entropy']
		baseline_opt_eval_ppl = baseline_opt_eval_stats['perplexity']

		# Baseline on opt_dev (from wiki2 test - overfitting detection, different windows)
		baseline_opt_dev_stats = model.evaluate_fast(
			opt_dev_tokens,
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_opt_dev_ce = baseline_opt_dev_stats['cross_entropy']
		baseline_opt_dev_ppl = baseline_opt_dev_stats['perplexity']

		# Baseline on val_tokens (from wiki2 validation - final evaluation only)
		baseline_val_stats = model.evaluate_fast(
			val_tokens[:min(50000, len(val_tokens))],
			batch_size=config.batch_size * 2,
			backend=AccelerationMode.AUTO,
			verbose=False,
		)
		baseline_val_ce = baseline_val_stats['cross_entropy']
		baseline_val_ppl = baseline_val_stats['perplexity']

		# Baseline ratio: opt_dev vs opt_eval (both from test, should be ~1.0)
		# If this ratio increases, optimizer is overfitting to opt_eval
		global_baseline_ratio = baseline_opt_dev_ce / baseline_opt_eval_ce if baseline_opt_eval_ce > 0 else 1.0

		log("")
		log(f"Optimization baselines:")
		log(f"  Train CE: {baseline_train_ce:.4f} (PPL: {baseline_train_ppl:.2f}) - opt_train (wiki2 train)")
		log(f"  Eval CE: {baseline_opt_eval_ce:.4f} (PPL: {baseline_opt_eval_ppl:.2f}) - opt_eval (wiki2 test - optimizer fitness)")
		log(f"  Dev CE: {baseline_opt_dev_ce:.4f} (PPL: {baseline_opt_dev_ppl:.2f}) - opt_dev (wiki2 test - overfitting detection)")
		log(f"  Val CE: {baseline_val_ce:.4f} (PPL: {baseline_val_ppl:.2f}) - val (wiki2 validation - final eval only)")
		log(f"  Dev/Eval ratio: {global_baseline_ratio:.2f}x (should stay ~1.0, increase = overfitting to opt_eval)")

		log("")
		log(f"Optimization config:")
		log(f"  GA: {config.ga_population} pop × {config.ga_generations} max gens, early_stop <{config.ga_early_stop_pct}%")
		log(f"  TS: {config.ts_neighbors} neighbors × {config.ts_iterations} max iters, early_stop <{config.ts_early_stop_pct}%")
		log(f"  Early stop patience: {config.early_stop_patience} (checks every 5 = {(config.early_stop_patience+1)*5} gens/iters)")
		log(f"  Overfitting thresholds: <{HEALTHY_THRESHOLD}% healthy, >{WARNING_THRESHOLD}% warn, >{SEVERE_THRESHOLD}% severe, >{CRITICAL_THRESHOLD}% stop")

		# Create overfitting monitor with dev evaluation function
		# This evaluates on opt_dev (different windows from wiki2 test) to detect
		# if we're overfitting to opt_eval (the optimizer's fitness target)
		def dev_eval_fn(conn: torch.Tensor) -> float:
			if is_tiered:
				set_combined_connections(conn)
			else:
				model.layer.memory.connections = conn
			model.reset_memory()
			if RUST_AVAILABLE:
				model.train_epoch_fast_rust(opt_train_tokens, global_top_k=config.global_top_k, batch_size=config.batch_size, verbose=False)
			else:
				model.train_epoch_fast(opt_train_tokens, global_top_k=config.global_top_k, batch_size=config.batch_size, verbose=False)
			# Evaluate on opt_dev (different windows from wiki2 test)
			return model.evaluate_fast(opt_dev_tokens, batch_size=config.batch_size * 2, backend=AccelerationMode.AUTO, verbose=False)['cross_entropy']

		overfitting_monitor = OverfittingMonitor(
			validation_fn=dev_eval_fn,
			grace_checks=1,
			logger=log,
			global_baseline_ratio=global_baseline_ratio,
		)

		# Run optimization strategies (joint for tiered, same code for uniform)
		arch_label = "Joint tiered" if is_tiered else "Uniform"
		prev_optimizer_final_error: float | None = None  # For GA → TS chaining
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
					neuron_offsets=neuron_offsets,  # For tiered crossover at neuron boundaries
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
					initial_error_hint=prev_optimizer_final_error,  # Inherit from GA if available
				)

			elif strategy_name == "PC":
				# ================================================================
				# PER-CLUSTER OPTIMIZATION (replaces global GA/TS)
				# Each cluster is optimized independently with discriminative fitness
				# Uses Rust accelerator when available (16x faster)
				# Uses GLOBAL CE fitness (softmax over all 50K clusters) for true PPL optimization
				# ================================================================
				use_rust_pc = _check_rust_per_cluster()
				log(f"  Setting up per-cluster optimization (Rust: {use_rust_pc})...")

				# Encode tokens to binary contexts
				log("  Encoding contexts to binary...")
				train_contexts = model.encode_sequence(opt_train_tokens)  # [N, total_input_bits]
				train_targets = torch.tensor(opt_train_tokens[model.context_size:], dtype=torch.long)
				eval_contexts = model.encode_sequence(opt_eval_tokens)
				eval_targets = torch.tensor(opt_eval_tokens[model.context_size:], dtype=torch.long)

				log(f"    Train: {train_contexts.shape[0]:,} examples × {train_contexts.shape[1]} bits")
				log(f"    Eval: {eval_contexts.shape[0]:,} examples")

				# Result adapter for compatibility with rest of benchmark
				from dataclasses import dataclass as dc
				@dc
				class PCResultAdapter:
					optimized_connections: torch.Tensor
					initial_error: float
					final_error: float
					iterations_run: int
					early_stopped_overfitting: bool

				total_clusters_optimized = 0
				all_tier_summaries = {}

				if is_tiered:
					# Process each tier separately (different bits_per_neuron per tier)
					log(f"  Processing {len(model.layer.tier_configs)} tiers...")

					# ============================================================
					# GLOBAL CE SETUP: Build mappings for ALL tiers first
					# This enables true global CE (softmax over all 50K clusters)
					# ============================================================
					all_cluster_to_neurons = {}  # cluster_id -> (start, end) in its tier's connectivity
					all_cluster_to_bits = {}     # cluster_id -> bits_per_neuron
					tier_info = []               # [(tier_idx, cluster_ids, tier_connectivity)]

					for tier_idx, tc in enumerate(model.layer.tier_configs):
						tier_clusters = []
						for local_idx, cluster_id in enumerate(range(tc.cluster_start, tc.cluster_end)):
							start_neuron = local_idx * tc.neurons_per_cluster
							end_neuron = start_neuron + tc.neurons_per_cluster
							all_cluster_to_neurons[cluster_id] = (start_neuron, end_neuron)
							all_cluster_to_bits[cluster_id] = tc.bits_per_neuron
							tier_clusters.append(cluster_id)
						tier_connectivity = model.layer.tier_memories[tier_idx].connections.clone()
						tier_info.append((tier_idx, tier_clusters, tier_connectivity, tc))

					# Create ONE Rust optimizer for global CE
					rust_optimizer = None
					if use_rust_pc:
						log(f"  Creating global optimizer for {len(all_cluster_to_neurons)} clusters...")
						rust_optimizer = create_rust_optimizer(
							train_contexts=train_contexts,
							train_targets=train_targets,
							eval_contexts=eval_contexts,
							eval_targets=eval_targets,
							context_bits=total_input_bits,
							cluster_to_neurons=all_cluster_to_neurons,
							cluster_to_bits=all_cluster_to_bits,
							num_clusters=model.vocab_size,
							fitness_mode=get_fitness_mode(config.fitness_mode),
							logger=log,
						)

						# Precompute GLOBAL baseline (votes for ALL clusters)
						# This enables true global CE during optimization
						log(f"  Precomputing global baseline (enables true global CE)...")
						all_connectivities = {}
						for tier_idx, tier_clusters, tier_connectivity, tc in tier_info:
							for local_idx, cluster_id in enumerate(tier_clusters):
								start = local_idx * tc.neurons_per_cluster
								end = start + tc.neurons_per_cluster
								cluster_conn = tier_connectivity[start:end]  # [num_neurons, bits]
								all_connectivities[cluster_id] = cluster_conn.flatten().long().tolist()

						rust_optimizer.precompute_global_baseline(all_connectivities)

					# ============================================================
					# Iterative refinement: multiple passes with grouped optimization
					# ============================================================
					num_passes = config.iterative_passes
					use_grouped = config.group_size > 0

					if num_passes > 1 or use_grouped:
						log(f"")
						log(f"  Iterative refinement: {num_passes} passes, group_size={config.group_size}")

					for pass_idx in range(num_passes):
						if num_passes > 1:
							log(f"")
							log(f"  ===== Pass {pass_idx + 1}/{num_passes} =====")

						# ============================================================
						# Optimize each tier using global CE fitness
						# ============================================================
						for tier_idx, tier_clusters, tier_connectivity, tc in tier_info:
							# Scale effort by tier: tier0=100%, tier1=30%, tier2=10%
							tier_scale = 1.0 / (3 ** tier_idx)
							ga_gens = max(5, int(config.ga_generations * tier_scale))
							ts_iters = max(5, int(config.ts_iterations * tier_scale))

							# For subsequent passes, reduce iterations (refinement)
							if pass_idx > 0:
								ga_gens = max(3, ga_gens // 2)
								ts_iters = max(3, ts_iters // 2)

							log(f"")
							log(f"  Tier {tier_idx}: {tc.cluster_count} clusters × {tc.neurons_per_cluster} neurons × {tc.bits_per_neuron} bits")
							log(f"    Budget: GA {ga_gens} gens, TS {ts_iters} iters")
							log(f"    Connectivity shape: {tier_connectivity.shape}")

							# Build tier-local mappings (neuron indices are LOCAL to tier's connectivity)
							cluster_to_neurons = {}
							cluster_to_bits = {}
							for local_idx, cluster_id in enumerate(tier_clusters):
								start_neuron = local_idx * tc.neurons_per_cluster
								end_neuron = start_neuron + tc.neurons_per_cluster
								cluster_to_neurons[cluster_id] = (start_neuron, end_neuron)
								cluster_to_bits[cluster_id] = tc.bits_per_neuron

							# Create tier config
							tier_opt_config = TierOptConfig(
								tier=tier_idx,
								ga_gens=ga_gens,
								ga_population=config.ga_population,
								ts_iters=ts_iters,
								ts_neighbors=config.ts_neighbors,
								mutation_rate=0.01,
								enabled=True,
							)

							if use_rust_pc and rust_optimizer is not None:
								# Use Rust-accelerated per-cluster optimization with GLOBAL CE
								# Decide whether to use grouped or individual optimization
								# Use grouped for tier 0/1 if group_size > 0 and cluster count is reasonable
								use_grouped_for_tier = (
									use_grouped and
									tier_idx <= 1 and  # Only for tier 0 and 1
									tc.cluster_count <= 500  # Don't group if too many clusters
								)

								if use_grouped_for_tier:
									# Group size scales with tier (tier 0: full group, tier 1: half)
									effective_group_size = config.group_size if tier_idx == 0 else max(2, config.group_size // 2)
									tier_results = rust_optimizer.optimize_tier_grouped(
										tier=tier_idx,
										cluster_ids=tier_clusters,
										current_connectivity=tier_connectivity,
										config=tier_opt_config,
										group_size=effective_group_size,
										seed=42 + tier_idx + pass_idx * 100,
									)
								else:
									# Individual optimization (for tier 2 or when grouping disabled)
									tier_results = rust_optimizer.optimize_tier(
										tier=tier_idx,
										cluster_ids=tier_clusters,
										current_connectivity=tier_connectivity,
										config=tier_opt_config,
										seed=42 + tier_idx + pass_idx * 100,
									)

								# Update tier's connectivity
								tier_connectivity = rust_optimizer.update_connectivity(
									tier_connectivity, tier_results
								)

								# Update global baseline with optimized connectivities
								for result in tier_results:
									rust_optimizer.update_global_baseline(
										result.cluster_id,
										result.final_connectivity.flatten().long().tolist()
									)

								total_clusters_optimized += len(tier_results)

								# Build summary (for final pass only to avoid double-counting)
								if pass_idx == num_passes - 1 and tier_results:
									avg_improvement = sum(r.improvement_pct for r in tier_results) / len(tier_results)
									all_tier_summaries[tier_idx] = {
										"clusters": len(tier_results),
										"avg_improvement": avg_improvement,
									}
							else:
								# Use Python fallback
								evaluator = IncrementalEvaluator(
									train_contexts=train_contexts,
									train_targets=train_targets,
									eval_contexts=eval_contexts,
									eval_targets=eval_targets,
									context_bits=total_input_bits,
									cluster_to_neurons=cluster_to_neurons,
									cluster_to_bits=cluster_to_bits,
									num_clusters=model.vocab_size,
									logger=log,
								)

								pc_optimizer = PerClusterOptimizer(
									tier_configs=[tier_opt_config],
									evaluator=evaluator,
									fitness_mode=get_fitness_mode(config.fitness_mode),
									cluster_order="random",
									seed=42 + tier_idx + pass_idx * 100,
									logger=log,
								)

								# Run optimization
								pc_result = pc_optimizer.optimize_all_tiers(
									initial_connectivity=tier_connectivity,
									tier_to_clusters={tier_idx: tier_clusters},
								)

								# Update tier's connectivity in place
								for cr in pc_result.cluster_results:
									start, end = cluster_to_neurons[cr.cluster_id]
									tier_connectivity[start:end] = cr.final_connectivity

								total_clusters_optimized += pc_result.total_clusters_optimized
								if pass_idx == num_passes - 1 and pc_result.tier_summaries:
									all_tier_summaries[tier_idx] = pc_result.tier_summaries.get(tier_idx, {})

							# Update model connectivity for next tier in this pass
							model.layer.tier_memories[tier_idx].connections = tier_connectivity

							# Also update tier_info for next pass
							tier_info[tier_idx] = (tier_idx, tier_clusters, tier_connectivity, tc)

					# Update current_connections from optimized tier_memories
					current_connections = get_combined_connections()

				else:
					# Uniform architecture: single tier with all clusters
					log("  Processing uniform architecture...")

					cluster_to_neurons = {}
					cluster_to_bits = {}
					for cluster_id in range(model.vocab_size):
						start_neuron = cluster_id * model.layer.neurons_per_cluster
						end_neuron = start_neuron + model.layer.neurons_per_cluster
						cluster_to_neurons[cluster_id] = (start_neuron, end_neuron)
						cluster_to_bits[cluster_id] = model.layer.bits_per_neuron

					log(f"    Clusters: {len(cluster_to_neurons):,}")

					tier_opt_config = TierOptConfig(
						tier=0,
						ga_gens=config.ga_generations,
						ga_population=config.ga_population,
						ts_iters=config.ts_iterations,
						ts_neighbors=config.ts_neighbors,
						mutation_rate=0.01,
						enabled=True,
					)

					all_clusters = list(range(model.vocab_size))

					if use_rust_pc:
						# Use Rust-accelerated per-cluster optimization
						rust_optimizer = create_rust_optimizer(
							train_contexts=train_contexts,
							train_targets=train_targets,
							eval_contexts=eval_contexts,
							eval_targets=eval_targets,
							context_bits=total_input_bits,
							cluster_to_neurons=cluster_to_neurons,
							cluster_to_bits=cluster_to_bits,
							num_clusters=model.vocab_size,
							fitness_mode=get_fitness_mode(config.fitness_mode),
							logger=log,
						)

						# Precompute GLOBAL baseline (votes for ALL clusters)
						# This enables true global CE during optimization
						log(f"  Precomputing global baseline for {len(all_clusters)} clusters...")
						all_connectivities = {}
						for cluster_id in all_clusters:
							start, end = cluster_to_neurons[cluster_id]
							cluster_conn = current_connections[start:end]  # [num_neurons, bits]
							all_connectivities[cluster_id] = cluster_conn.flatten().long().tolist()
						rust_optimizer.precompute_global_baseline(all_connectivities)

						# Run optimization via Rust
						tier_results = rust_optimizer.optimize_tier(
							tier=0,
							cluster_ids=all_clusters,
							current_connectivity=current_connections,
							config=tier_opt_config,
							seed=42,
						)

						# Update connectivity
						current_connections = rust_optimizer.update_connectivity(
							current_connections, tier_results
						)
						total_clusters_optimized = len(tier_results)

						# Build summary
						if tier_results:
							avg_improvement = sum(r.improvement_pct for r in tier_results) / len(tier_results)
							all_tier_summaries = {0: {
								"clusters": len(tier_results),
								"avg_improvement": avg_improvement,
							}}
					else:
						# Use Python fallback
						evaluator = IncrementalEvaluator(
							train_contexts=train_contexts,
							train_targets=train_targets,
							eval_contexts=eval_contexts,
							eval_targets=eval_targets,
							context_bits=total_input_bits,
							cluster_to_neurons=cluster_to_neurons,
							cluster_to_bits=cluster_to_bits,
							num_clusters=model.vocab_size,
							logger=log,
						)

						pc_optimizer = PerClusterOptimizer(
							tier_configs=[tier_opt_config],
							evaluator=evaluator,
							fitness_mode=get_fitness_mode(config.fitness_mode),
							cluster_order="random",
							seed=42,
							logger=log,
						)

						pc_result = pc_optimizer.optimize_all_tiers(
							initial_connectivity=current_connections,
							tier_to_clusters={0: all_clusters},
						)

						# Update connectivity
						for cr in pc_result.cluster_results:
							start, end = cluster_to_neurons[cr.cluster_id]
							current_connections[start:end] = cr.final_connectivity

						total_clusters_optimized = pc_result.total_clusters_optimized
						all_tier_summaries = pc_result.tier_summaries

				result = PCResultAdapter(
					optimized_connections=current_connections.clone(),
					initial_error=0.0,  # Will be computed from CE below
					final_error=0.0,
					iterations_run=total_clusters_optimized,
					early_stopped_overfitting=False,
				)

				log(f"")
				log(f"  Per-cluster optimization complete:")
				log(f"    Total clusters optimized: {total_clusters_optimized}")
				for tier_idx, summary in all_tier_summaries.items():
					if summary:
						log(f"    Tier {tier_idx}: {summary.get('clusters', 0)} clusters, avg improvement: {summary.get('avg_improvement', 0):.2f}%")

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

			# Save final error for next strategy (e.g., GA → TS chaining)
			prev_optimizer_final_error = result.final_error

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

	# Per-tier breakdown for validation (if enabled and tiered)
	if config.per_tier and 'by_tier' in final_val_stats:
		log()
		TierResultsTable.from_stats("Validation", final_val_stats).print(log)

	# Per-tier breakdown for test (if enabled and tiered)
	if config.per_tier and 'by_tier' in test_stats:
		log()
		TierResultsTable.from_stats("Test", test_stats).print(log)

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


# ============================================================================
# Sweep Mode Functions
# ============================================================================

def parse_sweep_output(output: str) -> tuple[dict, list[SweepTierResult], dict]:
	"""Parse benchmark output to extract per-tier results and optimization data."""
	import re
	lines = output.split('\n')

	overall = {'ppl': 0.0, 'accuracy': 0.0}
	tier_results = []
	opt_data = {
		'initial_val_ppl': None,
		'final_val_ppl': None,
		'optimizations': [],
		'total_ce_improvement': None,
		'val_ppl_improvement': None,
	}

	in_initial_eval = False
	in_final_eval = False

	for line in lines:
		# Track sections
		if 'Initial Evaluation (Validation Set)' in line:
			in_initial_eval = True
			in_final_eval = False
		elif 'Final Evaluation (Validation Set)' in line:
			in_initial_eval = False
			in_final_eval = True
		elif 'Final Evaluation (Test Set)' in line:
			in_initial_eval = False
			in_final_eval = False

		# Parse PPL based on section
		ppl_match = re.search(r'Perplexity:\s*([\d.]+)', line)
		if ppl_match:
			ppl_val = float(ppl_match.group(1))
			if in_initial_eval and opt_data['initial_val_ppl'] is None:
				opt_data['initial_val_ppl'] = ppl_val
			elif in_final_eval and opt_data['final_val_ppl'] is None:
				opt_data['final_val_ppl'] = ppl_val

		# Test PPL for overall results
		test_ppl_match = re.search(r'Test PPL:\s*([\d.]+)', line)
		if test_ppl_match:
			overall['ppl'] = float(test_ppl_match.group(1))

		acc_match = re.search(r'Test Acc:\s*([\d.]+)%', line)
		if acc_match:
			overall['accuracy'] = float(acc_match.group(1)) / 100

	# Parse optimization results
	opt_pattern = re.compile(r'\[(GA|TS)\] (?:Final|Results).*?(?:CE|error).*?([\d.]+)')
	init_pattern = re.compile(r'\[(GA|TS)\] Initial.*?(?:CE|error).*?([\d.]+)')

	init_ces = {}
	final_ces = {}

	for line in lines:
		init_match = init_pattern.search(line)
		if init_match:
			strategy = init_match.group(1)
			init_ces[strategy] = float(init_match.group(2))

		# Look for improvement percentages
		imp_match = re.search(r'\[(GA|TS)\].*Improvement:\s*([\d.]+)%', line)
		if imp_match:
			strategy = imp_match.group(1)
			improvement = float(imp_match.group(2))
			opt_data['optimizations'].append(SweepOptResult(
				strategy=strategy,
				initial_ce=init_ces.get(strategy, 0.0),
				final_ce=0.0,
				improvement_pct=improvement,
				iterations=0,
			))

	# Calculate total CE improvement
	if opt_data['optimizations']:
		total_imp = sum(o.improvement_pct for o in opt_data['optimizations'])
		opt_data['total_ce_improvement'] = total_imp

	# Calculate validation PPL improvement
	if opt_data['initial_val_ppl'] and opt_data['final_val_ppl']:
		init_ppl = opt_data['initial_val_ppl']
		final_ppl = opt_data['final_val_ppl']
		if init_ppl > 0:
			opt_data['val_ppl_improvement'] = ((init_ppl - final_ppl) / init_ppl) * 100

	# Parse tier results (format: Tier N  clusters  neurons  bits  data%  ppl  accuracy%)
	tier_pattern = re.compile(
		r'Tier\s+(\d+)\s+'
		r'([\d,]+)\s+'
		r'(\d+)\s+'
		r'(\d+)\s+'
		r'([\d.]+)%\s+'
		r'([\d.]+)\s+'
		r'([\d.]+)%'
	)

	in_tier_section = False
	for line in lines:
		if 'Per-Tier Test Results' in line:
			in_tier_section = True
			continue
		if in_tier_section and 'TOTAL' in line:
			in_tier_section = False
			continue

		if in_tier_section:
			match = tier_pattern.search(line)
			if match:
				tier_results.append(SweepTierResult(
					tier=int(match.group(1)),
					name=f"tier_{match.group(1)}",
					clusters=int(match.group(2).replace(',', '')),
					neurons=int(match.group(3)),
					bits=int(match.group(4)),
					data_pct=float(match.group(5)),
					ppl=float(match.group(6)),
					accuracy=float(match.group(7)) / 100,
				))

	return overall, tier_results, opt_data


def estimate_sweep_runtime(experiments: list[SweepExperiment], optimize: bool) -> float:
	"""Estimate total runtime in hours."""
	total_minutes = 0
	for exp in experiments:
		base = 45  # minutes
		parts = exp.tiered.split(';')
		max_bits = max(int(p.split(',')[2]) for p in parts)
		bits_adj = max(0, (max_bits - 10) // 2) * 5
		ctx_adj = max(0, exp.context - 4) * 10
		opt_adj = 75 if optimize else 0
		total_minutes += base + bits_adj + ctx_adj + opt_adj
	return total_minutes / 60


def load_sweep_completed(output_path: Path) -> set[str]:
	"""Load names of already-completed experiments."""
	if not output_path.exists():
		return set()
	try:
		with open(output_path) as f:
			data = json.load(f)
		return {r['name'] for r in data}
	except (json.JSONDecodeError, KeyError):
		return set()


def load_sweep_results(output_path: Path) -> list[dict]:
	"""Load existing results for merging."""
	if not output_path.exists():
		return []
	try:
		with open(output_path) as f:
			return json.load(f)
	except json.JSONDecodeError:
		return []


def print_sweep_summary(results: list[SweepResult]):
	"""Print a summary table of sweep results."""
	has_opt = any(r.optimization_results for r in results)

	print("\n" + "=" * 150)
	print("SWEEP RESULTS")
	print("=" * 150)

	if has_opt:
		print(f"\n{'Experiment':<20} {'Config':<26} {'PPL':>7} {'Acc':>5} | {'GA%':>5} {'TS%':>5} {'Val%':>5} | {'T0 PPL':>7} {'T0 Ac':>5} {'T1 PPL':>7} {'T1 Ac':>5} {'T2 PPL':>7} {'T2 Ac':>5}")
		print("-" * 150)
	else:
		print(f"\n{'Experiment':<20} {'Config':<30} {'PPL':>8} {'Acc':>6} | {'T0 PPL':>7} {'T0 Ac':>5} {'T1 PPL':>7} {'T1 Ac':>5} {'T2 PPL':>7} {'T2 Ac':>5}")
		print("-" * 130)

	for r in sorted(results, key=lambda x: x.overall_ppl):
		t0 = next((t for t in r.tier_results if t.tier == 0), None)
		t1 = next((t for t in r.tier_results if t.tier == 1), None)
		t2 = next((t for t in r.tier_results if t.tier == 2), None)

		t0_ppl = f"{t0.ppl:.0f}" if t0 else "-"
		t0_acc = f"{t0.accuracy:.1%}" if t0 else "-"
		t1_ppl = f"{t1.ppl:.0f}" if t1 else "-"
		t1_acc = f"{t1.accuracy:.1%}" if t1 else "-"
		t2_ppl = f"{t2.ppl:.0f}" if t2 else "-"
		t2_acc = f"{t2.accuracy:.1%}" if t2 else "-"

		if has_opt:
			ga_imp = ts_imp = val_imp = "-"
			if r.optimization_results:
				for opt in r.optimization_results:
					if opt.strategy == "GA":
						ga_imp = f"{opt.improvement_pct:.1f}"
					elif opt.strategy == "TS":
						ts_imp = f"{opt.improvement_pct:.1f}"
			if r.val_ppl_improvement_pct is not None:
				val_imp = f"{r.val_ppl_improvement_pct:.1f}"

			print(f"{r.name:<20} {r.config:<26} {r.overall_ppl:>7.0f} {r.overall_accuracy:>5.1%} | {ga_imp:>5} {ts_imp:>5} {val_imp:>5} | {t0_ppl:>7} {t0_acc:>5} {t1_ppl:>7} {t1_acc:>5} {t2_ppl:>7} {t2_acc:>5}")
		else:
			print(f"{r.name:<20} {r.config:<30} {r.overall_ppl:>8.0f} {r.overall_accuracy:>6.1%} | {t0_ppl:>7} {t0_acc:>5} {t1_ppl:>7} {t1_acc:>5} {t2_ppl:>7} {t2_acc:>5}")

	print("-" * (150 if has_opt else 130))

	best = min(results, key=lambda x: x.overall_ppl)
	print(f"\nBest PPL: {best.name} with {best.overall_ppl:.1f}")


def run_single_experiment(exp: SweepExperiment, optimize: bool, strategy: str) -> Optional[SweepResult]:
	"""Run a single sweep experiment."""
	print(f"\n{'=' * 70}")
	print(f"Running: {exp.name}")
	print(f"Config: {exp.tiered}")
	print(f"Context: {exp.context}")
	print(f"Optimize: {optimize} ({strategy})" if optimize else "Optimize: False")
	print(f"Description: {exp.description}")
	print(f"{'=' * 70}\n")

	cmd = [
		sys.executable, __file__,
		"--mode", "full",
		"--tiered", exp.tiered,
		"--context", str(exp.context),
		"--per-tier",
		"--full-data",
	]

	if optimize:
		cmd.append("--optimize")
		cmd.extend(["--strategy", strategy])

	start_time = time.perf_counter()
	timeout = 14400 if optimize else 7200  # 4 hours vs 2 hours

	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout,
		)
		elapsed = time.perf_counter() - start_time

		if result.returncode != 0:
			print(f"ERROR: Experiment failed with return code {result.returncode}")
			print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
			return None

		output = result.stdout + result.stderr
		overall, tier_results, opt_data = parse_sweep_output(output)

		print(f"\nCompleted in {elapsed / 60:.1f} minutes")
		print(f"Overall PPL: {overall['ppl']:.1f}, Accuracy: {overall['accuracy']:.2%}")

		return SweepResult(
			name=exp.name,
			config=exp.tiered,
			context=exp.context,
			overall_ppl=overall['ppl'],
			overall_accuracy=overall['accuracy'],
			tier_results=tier_results,
			train_time=elapsed,
			eval_time=0.0,
			timestamp=datetime.now().isoformat(),
			initial_val_ppl=opt_data['initial_val_ppl'],
			final_val_ppl=opt_data['final_val_ppl'],
			val_ppl_improvement_pct=opt_data['val_ppl_improvement'],
			optimization_results=opt_data['optimizations'] if opt_data['optimizations'] else None,
			total_ce_improvement_pct=opt_data['total_ce_improvement'],
		)

	except subprocess.TimeoutExpired:
		print(f"ERROR: Experiment timed out after {timeout / 3600:.0f} hours")
		return None
	except Exception as e:
		print(f"ERROR: {e}")
		return None


def run_sweep(args) -> int:
	"""Run multiple experiments in sweep mode."""
	all_experiments = define_sweep_experiments()

	# Determine output path
	output_file = args.output or "sweep_results.json"
	output_path = Path("experiments") / output_file

	# Filter by set or specific experiments
	if args.experiments:
		names = [n.strip() for n in args.experiments.split(',')]
		experiments = [e for e in all_experiments if e.name in names]
		if not experiments:
			print(f"ERROR: No experiments matched: {names}")
			print(f"Available: {[e.name for e in all_experiments]}")
			return 1
	else:
		max_priority = {"quick": 1, "standard": 2, "extended": 3}[args.set]
		experiments = [e for e in all_experiments if e.priority <= max_priority]

	# Skip completed experiments (unless --force-rerun)
	completed = set()
	if not args.force_rerun:
		completed = load_sweep_completed(output_path)
		if completed:
			original_count = len(experiments)
			experiments = [e for e in experiments if e.name not in completed]
			skipped = original_count - len(experiments)
			if skipped > 0:
				print(f"\nSkipping {skipped} already-completed experiments: {sorted(completed)}")

	if not experiments:
		print("\nAll experiments already completed! Use --force-rerun to re-run them.")
		return 0

	# Optimization settings (GA,TS enabled by default in sweep mode)
	optimize = not args.no_optimize
	strategy = args.strategy

	# Estimate runtime
	est_hours = estimate_sweep_runtime(experiments, optimize)

	print(f"\n{'#' * 70}")
	print(f"# RAMLM Sweep Mode")
	print(f"# Set: {args.set} ({len(experiments)} new + {len(completed)} completed)")
	if optimize:
		print(f"# Optimization: {strategy} (enabled by default)")
	else:
		print(f"# Optimization: disabled (use without --no-optimize to enable)")
	print(f"# Estimated runtime: {est_hours:.1f} hours")
	print(f"# Output: {output_path}")
	print(f"{'#' * 70}")

	print(f"\nExperiments to run:")
	for i, exp in enumerate(experiments, 1):
		print(f"  {i}. {exp.name}: {exp.description}")

	print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

	# Load existing results for merging
	existing_results = load_sweep_results(output_path)
	results = []

	for i, exp in enumerate(experiments):
		print(f"\n[{i + 1}/{len(experiments)}] {exp.name}")
		result = run_single_experiment(exp, optimize, strategy)
		if result:
			results.append(result)
			# Merge with existing and save
			merged = existing_results + [asdict(r) for r in results]
			output_path.parent.mkdir(exist_ok=True)
			with open(output_path, 'w') as f:
				json.dump(merged, f, indent=2, default=str)
			print(f"Saved to {output_path}")

	# Print summary
	if results:
		print_sweep_summary(results)
		print(f"\nResults saved to: {output_path}")

	return 0


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
	parser.add_argument("--strategy", type=str, default="PC",
		help="Optimization strategy: PC (per-cluster), GA, TS, SA, or comma-separated like GA,TS (default: PC)")
	parser.add_argument("--fitness", type=str, default="SIMPLE",
		choices=["SIMPLE", "CE", "PENALIZE", "ACCURACY"],
		help="Fitness function: SIMPLE (+vote correct, -vote wrong), CE (cross-entropy), PENALIZE (penalize high votes), ACCURACY (count wins when should win). Default: SIMPLE")

	# Optimization parameters
	parser.add_argument("--ts-iters", type=int, default=None,
		help="Tabu Search iterations (default: 5 fast, 10 full, 20 overnight)")
	parser.add_argument("--ts-neighbors", type=int, default=30,
		help="Tabu Search neighbors per iteration (default: 30)")
	parser.add_argument("--ga-pop", type=int, default=None,
		help="GA population size (default: 10 fast, 20 full, 30 overnight)")
	parser.add_argument("--ga-gens", type=int, default=None,
		help="GA generations (default: 20 fast, 50 full, 100 overnight)")
	parser.add_argument("--patience", type=int, default=None,
		help="Early stop patience (checks every 5 gens/iters). Default: 1. Weekend mode: 20+")

	# Iterative refinement options
	parser.add_argument("--iterative-passes", type=int, default=1,
		help="Number of optimization passes for iterative refinement (default: 1)")
	parser.add_argument("--group-size", type=int, default=10,
		help="Group size for joint optimization of tier 0/1 (default: 10). 0 disables grouping.")

	# Optimization sample size
	parser.add_argument("--opt-train-windows", type=int, default=None,
		help="Training windows for optimization (default: 50). Each window is 200 tokens.")
	parser.add_argument("--opt-eval-windows", type=int, default=None,
		help="Eval windows for optimization (default: 20). Each window is 200 tokens.")

	# Evaluation options
	parser.add_argument("--per-tier", action="store_true",
		help="Track per-tier PPL and accuracy (only for tiered architectures)")

	# Acceleration options
	parser.add_argument("--cpu-only", action="store_true",
		help="Force CPU-only evaluation (disable hybrid CPU+GPU). Default uses hybrid if available.")

	# Sweep mode (multi-experiment)
	parser.add_argument("--sweep", action="store_true",
		help="Run multiple experiments from a predefined set (GA,TS optimization enabled by default)")
	parser.add_argument("--set", type=str, default="quick",
		choices=["quick", "standard", "extended"],
		help="Experiment set for sweep mode: quick (4), standard (6), extended (10)")
	parser.add_argument("--experiments", type=str, default=None,
		help="Comma-separated list of specific experiments to run in sweep mode")
	parser.add_argument("--output", type=str, default=None,
		help="Output JSON file for sweep results (default: sweep_results.json)")
	parser.add_argument("--force-rerun", action="store_true",
		help="Re-run completed experiments in sweep mode")
	parser.add_argument("--no-optimize", action="store_true",
		help="Disable optimization in sweep mode (optimization is ON by default in sweep)")

	args = parser.parse_args()

	# Handle sweep mode separately (no logging setup needed - each experiment has its own)
	if args.sweep:
		sys.exit(run_sweep(args))

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
	config.fitness_mode = args.fitness
	config.windowed_pretrain = args.windowed_pretrain
	config.per_tier = args.per_tier
	config.cpu_only = args.cpu_only
	config.iterative_passes = args.iterative_passes
	config.group_size = args.group_size
	if args.opt_train_windows:
		config.opt_train_windows = args.opt_train_windows
	if args.opt_eval_windows:
		config.opt_eval_windows = args.opt_eval_windows
		config.opt_dev_windows = args.opt_eval_windows  # Keep dev same as eval

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
	if args.patience is not None:
		config.early_stop_patience = args.patience

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
