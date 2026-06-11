"""
Adaptive Cluster Architecture Search — Per-Neuron Bits

Each neuron owns its synapse count (bit count) and evolves independently.
The GA mutates both connectivity AND structure (bits per neuron, neurons per cluster).

Key data structures:
- bits_per_neuron: [total_neurons] — each neuron's synapse count
- neurons_per_cluster: [num_clusters] — structural grouping
- connections: flat list, sum(bits_per_neuron) entries

Key insight: Frequent tokens need different architectures than rare tokens.
Let the data decide rather than hand-tuning tier boundaries.
"""

from __future__ import annotations

import gzip
import json
import math
import random

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch import Tensor

from wnn.progress import ProgressTracker
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger
from wnn.ram.strategies.factory import (
	OptimizerStrategyFactory,
	OptimizerStrategyType,
)
from wnn.ram.strategies.connectivity.generic_strategies import (
	GAConfig,
	TSConfig,
	OptimizerResult,
)

if TYPE_CHECKING:
	pass

# Try to import Rust accelerator for fast connection generation
try:
	import ram_accelerator as _accel
	_HAS_RUST = True
except ImportError:
	_HAS_RUST = False

# ClusterGenome + its config/helpers moved to wnn.ram.genome (D6c, 11/06/2026)
# — core/ may import them without reaching into strategies/. Re-exported here
# so existing imports keep working.
from wnn.ram.genome import (  # noqa: F401
	ClusterGenome,
	AdaptiveClusterConfig,
	GenomeInitStrategy,
	generate_connections,
	enforce_unique_connections,
	OptimizationDimension,
	PhaseType,
)

# =============================================================================
# Helper Functions (module-level, internal)
# =============================================================================

def _frequency_scaled_init(
	num_clusters: int,
	token_frequencies: list[int],
	config: AdaptiveClusterConfig,
	for_bits: bool = True,
) -> list[int]:
	"""
	Initialize bits or neurons scaled by token frequency.

	More frequent tokens get more bits/neurons (larger capacity).
	Rare tokens get fewer bits/neurons (small capacity sufficient).

	Uses log-scale mapping from frequency to value.
	"""
	if len(token_frequencies) != num_clusters:
		raise ValueError(
			f"token_frequencies length ({len(token_frequencies)}) != "
			f"num_clusters ({num_clusters})"
		)

	# Find frequency range (avoid log(0))
	freqs = [max(1, f) for f in token_frequencies]
	max_freq = max(freqs)
	min_freq = min(freqs)

	# Log-scale mapping: high freq -> high value, low freq -> low value
	log_max = math.log(max_freq)
	log_min = math.log(min_freq)
	log_range = log_max - log_min if log_max > log_min else 1.0

	if for_bits:
		min_val, max_val = config.min_bits, config.max_bits
	else:
		min_val, max_val = config.min_neurons, config.max_neurons

	val_range = max_val - min_val

	values = []
	for freq in freqs:
		# Normalize log frequency to [0, 1]
		log_freq = math.log(freq)
		normalized = (log_freq - log_min) / log_range

		# Map to value range
		val = min_val + int(normalized * val_range)
		val = max(min_val, min(max_val, val))
		values.append(val)

	return values


# =============================================================================
# Rust Parallel Evaluator
# =============================================================================

class RustParallelEvaluator:
	"""
	Rust-accelerated parallel genome evaluation using rayon.

	Evaluates multiple genomes concurrently in Rust threads.
	Much faster than Python multiprocessing - no process spawn or pickle overhead.

	Usage:
		evaluator = RustParallelEvaluator(config)
		fitness_list = evaluator.evaluate_batch(genomes)
	"""

	def __init__(self, config: 'EvaluatorConfig'):
		"""
		Initialize Rust parallel evaluator.

		Args:
			config: Evaluation configuration with pre-computed training data
		"""
		self.config = config
		self._prepared = False
		self._train_data = None
		self._eval_data = None

	def _prepare_data(self):
		"""Pre-encode training and evaluation data for Rust (vectorized)."""
		if self._prepared:
			return

		import numpy as np
		from collections import Counter
		from wnn.ram.core import bits_needed

		cfg = self.config
		bits_per_token = bits_needed(cfg.vocab_size)
		total_input_bits = cfg.context_size * bits_per_token

		# Build cluster map once (if needed)
		cluster_map = None
		if cfg.cluster_order is not None:
			cluster_map = np.zeros(cfg.vocab_size, dtype=np.int64)
			for idx, tid in enumerate(cfg.cluster_order):
				if tid < cfg.vocab_size:
					cluster_map[tid] = idx

		# Convert tokens to numpy array for vectorized ops
		train_tokens = np.array(cfg.train_tokens, dtype=np.int64)
		eval_tokens = np.array(cfg.eval_tokens, dtype=np.int64)

		# === TRAINING DATA (vectorized) ===
		n_train = len(train_tokens) - cfg.context_size

		# Build context windows: [n_train, context_size]
		train_contexts = np.lib.stride_tricks.sliding_window_view(
			train_tokens[:n_train + cfg.context_size - 1], cfg.context_size
		)[:n_train]

		# Encode contexts to bits using vectorized operations
		# Shape: [n_train, context_size, bits_per_token]
		bit_shifts = np.arange(bits_per_token - 1, -1, -1, dtype=np.int64)
		train_bits_3d = ((train_contexts[:, :, np.newaxis] >> bit_shifts) & 1).astype(np.uint8)
		train_input_bits = train_bits_3d.reshape(-1)  # Flatten to 1D

		# Targets
		train_targets_raw = train_tokens[cfg.context_size:cfg.context_size + n_train]
		if cluster_map is not None:
			train_targets = cluster_map[train_targets_raw]
		else:
			train_targets = train_targets_raw

		# === NEGATIVE SAMPLES (vectorized) ===
		counts = Counter(cfg.train_tokens)
		# Cap to actual unique tokens in case vocab is smaller than global_top_k
		actual_top_k = min(cfg.global_top_k, len(counts))
		top_k_tokens = np.array([t for t, _ in counts.most_common(actual_top_k)], dtype=np.int64)
		num_negatives = min(5, actual_top_k)

		rng = np.random.RandomState(42)
		neg_indices = rng.randint(0, actual_top_k, (n_train, num_negatives))
		neg_tokens = top_k_tokens[neg_indices]  # [n_train, num_negatives]

		if cluster_map is not None:
			train_negatives = cluster_map[neg_tokens].reshape(-1)
		else:
			train_negatives = neg_tokens.reshape(-1)

		# === EVALUATION DATA (vectorized) ===
		n_eval = len(eval_tokens) - cfg.context_size

		eval_contexts = np.lib.stride_tricks.sliding_window_view(
			eval_tokens[:n_eval + cfg.context_size - 1], cfg.context_size
		)[:n_eval]

		eval_bits_3d = ((eval_contexts[:, :, np.newaxis] >> bit_shifts) & 1).astype(np.uint8)
		eval_input_bits = eval_bits_3d.reshape(-1)

		eval_targets_raw = eval_tokens[cfg.context_size:cfg.context_size + n_eval]
		if cluster_map is not None:
			eval_targets = cluster_map[eval_targets_raw]
		else:
			eval_targets = eval_targets_raw

		self._train_data = {
			'input_bits': train_input_bits,
			'targets': train_targets.astype(np.int64),
			'negatives': train_negatives.astype(np.int64),
			'num_examples': n_train,
			'num_negatives': num_negatives,
		}
		self._eval_data = {
			'input_bits': eval_input_bits,
			'targets': eval_targets.astype(np.int64),
			'num_examples': n_eval,
		}
		self._total_input_bits = total_input_bits
		self._prepared = True

	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
		batch_size: int = 1,  # Sequential: each genome gets full thread pool for token parallelism
		generation: Optional[int] = None,  # Current generation for logging
		total_generations: Optional[int] = None,  # Total generations for logging
		min_accuracy: Optional[float] = None,  # Threshold for log level selection
	) -> list[tuple[float, float]]:
		"""
		Evaluate multiple genomes using Rust/rayon.

		Rust evaluates genomes SEQUENTIALLY (memory-safe) while each genome's
		training/eval uses full CPU parallelism. This avoids the memory explosion
		that occurred with parallel genome evaluation.

		Args:
			genomes: List of genomes to evaluate
			logger: Optional logging function
			batch_size: Genomes per Rust call (1 = per-genome logging, >1 = batch logging)
			generation: Current generation number for logging (None = initial population)
			total_generations: Total number of generations for logging context
			min_accuracy: If provided, genomes below this threshold log at TRACE level

		Returns:
			List of (cross-entropy, accuracy) tuples for each genome
		"""
		import ram_accelerator
		import time

		# Use OptimizationLogger for leveled logging, or fallback to callable
		if isinstance(logger, OptimizationLogger):
			log_debug = logger.debug
			log_trace = logger.trace
		elif logger is not None:
			log_debug = logger
			log_trace = logger  # Fallback: no level distinction
		else:
			log_debug = lambda x: None
			log_trace = lambda x: None

		# Prepare data on first call
		self._prepare_data()

		all_fitness = []
		total_genomes = len(genomes)
		genome_width = len(str(total_genomes))  # For zero-padded logging
		start_time = time.time()

		# Generation prefix for logs
		if generation is not None:
			if total_generations is not None:
				gen_width = len(str(total_generations))
				gen_prefix = f"[Gen {generation + 1:0{gen_width}d}/{total_generations}]"
			else:
				gen_prefix = f"[Gen {generation + 1}]"
		else:
			gen_prefix = "[Init]"

		# Process in batches to limit memory usage
		for batch_start in range(0, total_genomes, batch_size):
			batch_end = min(batch_start + batch_size, total_genomes)
			batch_genomes = genomes[batch_start:batch_end]

			# Flatten genome configurations for this batch (canonical marshaller)
			from wnn.accel import flatten_genomes
			genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat = flatten_genomes(batch_genomes)

			# Call Rust parallel evaluator for this batch.
			# Returns list of (CE, accuracy) tuples.
			#
			# CANCEL GUARD (F1=0.49 bug, 01/06/2026): if the process-wide cancel
			# flag is set DURING training, every remaining training example is a
			# no-op (marker_train + dense fallback both poll it), leaving genomes
			# UNTRAINED. The Rust eval then silently scores untrained memory as a
			# trivial predict-majority genome (CE~0.877 / F1~0.49) and that poisons
			# the whole flow. So: detect a cancel that was active across this batch
			# and RETRY from a clean flag instead of accepting untrained results.
			# A genuine shutdown is honored separately via the strategy's
			# should_stop()/_stop_current_flow path (it unwinds the GA loop at the
			# next generation boundary), so resetting the Rust flag here to retrain
			# does NOT defeat a real stop — it only prevents untrained genomes from
			# being scored. After _CANCEL_RETRIES consecutive cancels we abort
			# LOUDLY (raise) rather than ever return a trivial result.
			_CANCEL_RETRIES = 3
			_cancel_attempt = 0
			while True:
				batch_results = ram_accelerator.evaluate_genomes_parallel(
					genomes_bits_flat,
					genomes_neurons_flat,
					genomes_connections_flat,  # Pass connections (empty = random fallback)
					len(batch_genomes),
					self.config.vocab_size,
					self._train_data['input_bits'],
					self._train_data['targets'],
					self._train_data['negatives'],
					self._train_data['num_examples'],
					self._train_data['num_negatives'],
					self._eval_data['input_bits'],
					self._eval_data['targets'],
					self._eval_data['num_examples'],
					self._total_input_bits,
					self.config.empty_value,
					1.0,  # neuron_sample_rate: full sampling (LM path has no sampling knob)
					0,    # rng_seed (unused at sample_rate=1.0)
				)
				try:
					_cancelled = ram_accelerator.is_cancelled()
				except Exception:
					_cancelled = False
				if not _cancelled:
					break  # clean eval — results are trustworthy
				_cancel_attempt += 1
				# Instrumentation: this is the ONLY place we learn the trigger.
				log_debug(
					f"{gen_prefix} [CANCEL-GUARD] cancel flag was SET during eval "
					f"batch [{batch_start}:{batch_end}] — genomes are UNTRAINED and "
					f"would score trivial (the F1=0.49 bug). Discarding results, "
					f"resetting flag, retrying (attempt {_cancel_attempt}/{_CANCEL_RETRIES})."
				)
				try:
					ram_accelerator.reset_cancel_flag()
				except Exception as _e:
					log_debug(f"{gen_prefix} [CANCEL-GUARD] reset_cancel_flag failed: {_e}")
				if _cancel_attempt >= _CANCEL_RETRIES:
					raise RuntimeError(
						f"{gen_prefix} evaluation cancelled {_CANCEL_RETRIES}x consecutively "
						f"(flag re-set each time) — refusing to return UNTRAINED genomes as "
						f"trivial results. Aborting the flow loudly instead of silently "
						f"producing F1=0.49. (If this is a genuine shutdown, the flow will "
						f"stop via should_stop.)"
					)

			# batch_results is already list[(CE, accuracy)]
			all_fitness.extend(batch_results)

			elapsed = time.time() - start_time
			# Log each genome in the batch (with batch timing for efficiency)
			for i, (ce, acc) in enumerate(batch_results):
				genome_idx = batch_start + i + 1
				if batch_size == 1:
					msg = f"{gen_prefix} Genome {genome_idx:0{genome_width}d}/{total_genomes}: CE={ce:.4f}, Acc={acc:.2%} in {elapsed:.1f}s"
				else:
					# Parallel batch: show genome results without individual timing
					msg = f"{gen_prefix} Genome {genome_idx:0{genome_width}d}/{total_genomes}: CE={ce:.4f}, Acc={acc:.2%}"
				# Use TRACE for filtered (below threshold), DEBUG for passed
				if min_accuracy is not None and acc < min_accuracy:
					log_trace(msg)  # Filtered candidate
				else:
					log_debug(msg)  # Passed candidate
			# Log batch timing summary for parallel batches
			if batch_size > 1:
				log_debug(f"{gen_prefix} Batch {batch_start//batch_size + 1}: {len(batch_results)} genomes in {elapsed:.1f}s")

		return all_fitness

	def evaluate_single(self, genome: ClusterGenome) -> float:
		"""Evaluate a single genome, returning CE only."""
		ce, _ = self.evaluate_batch([genome])[0]
		return ce

	def evaluate_single_with_accuracy(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		return self.evaluate_batch([genome])[0]


# =============================================================================
# Genome Evaluation Wrapper
# =============================================================================

@dataclass
class EvaluatorConfig:
	"""Configuration for genome evaluation."""

	# Data
	train_tokens: list[int] = None
	eval_tokens: list[int] = None
	vocab_size: int = 50257
	context_size: int = 4

	# Training
	batch_size: int = 500
	global_top_k: int = 100
	empty_value: float = 0.0

	# Evaluation
	eval_batch_size: int = 1000

	# Token ordering (for cluster assignment)
	cluster_order: Optional[list[int]] = None  # sorted by frequency

	# Random seed for connectivity (None = truly random, no seeding)
	# NOTE: Architecture optimization should NOT depend on specific connectivity
	# patterns, so we default to None for unbiased evaluation.
	rng: Optional[int] = None


class AdaptiveRAMLMWrapper:
	"""
	Lightweight RAMLM-like wrapper using AdaptiveClusteredRAM.

	This class provides train/evaluate functionality similar to RAMLM
	but uses AdaptiveClusteredRAM for per-cluster architecture.

	Used for genome fitness evaluation during architecture search.
	"""

	def __init__(
		self,
		genome: ClusterGenome,
		config: EvaluatorConfig,
	):
		"""
		Initialize wrapper from genome.

		Args:
			genome: ClusterGenome defining per-cluster architecture
			config: Evaluation configuration
		"""
		from wnn.ram.core import AdaptiveClusteredRAM, bits_needed
		from torch import arange, long

		self.genome = genome
		self.config = config
		self.vocab_size = config.vocab_size
		self.context_size = config.context_size
		self.bits_per_token = bits_needed(config.vocab_size)
		self.total_input_bits = config.context_size * self.bits_per_token

		# Create the adaptive layer
		self.layer = AdaptiveClusteredRAM.from_genome(
			genome=genome,
			total_input_bits=self.total_input_bits,
			empty_value=config.empty_value,
			rng=config.rng,
		)

		# Cluster order for training (maps token IDs to cluster IDs)
		self._cluster_order = config.cluster_order
		if self._cluster_order is not None:
			# Build reverse mapping: token_id -> logical_cluster_idx
			self._token_to_cluster = {tid: idx for idx, tid in enumerate(self._cluster_order)}
		else:
			self._token_to_cluster = None

		# Bit positions for encoding
		self._bit_positions = arange(self.bits_per_token - 1, -1, -1, dtype=long)

	def _encode_tokens(self, tokens: list[int]) -> Tensor:
		"""Encode tokens to binary input bits."""
		from torch import tensor, zeros, bool as torch_bool

		n = len(tokens)
		bits = zeros(n, self.bits_per_token, dtype=torch_bool)
		tokens_t = tensor(tokens, dtype=self._bit_positions.dtype)

		for i in range(self.bits_per_token):
			bits[:, i] = ((tokens_t >> self._bit_positions[i]) & 1).bool()

		return bits

	def _encode_context(self, context_tokens: list[int]) -> Tensor:
		"""Encode context tokens to flat input bits."""
		bits = self._encode_tokens(context_tokens)
		return bits.flatten()

	def train_epoch(
		self,
		tokens: list[int],
		global_top_k: int = 100,
		batch_size: int = 500,
		verbose: bool = False,
	) -> Dict:
		"""
		Train on token sequence.

		Args:
			tokens: Training token sequence
			global_top_k: Number of top tokens to use as negatives
			batch_size: Batch size for training
			verbose: Print progress

		Returns:
			Dict with training statistics
		"""
		from collections import Counter
		import time

		from torch import long, randint, stack, tensor, zeros

		start = time.time()

		# Compute global top-k tokens
		counts = Counter(tokens)
		top_k_tokens = [t for t, _ in counts.most_common(global_top_k)]

		# Prepare examples
		n_examples = len(tokens) - self.context_size
		contexts = []
		targets = []

		for i in range(n_examples):
			context = tokens[i:i + self.context_size]
			target = tokens[i + self.context_size]
			contexts.append(context)
			targets.append(target)

		# Encode all contexts
		all_input_bits = []
		for ctx in contexts:
			all_input_bits.append(self._encode_context(ctx))
		input_bits = stack(all_input_bits)  # [n_examples, total_input_bits]

		# Convert targets to cluster indices
		if self._token_to_cluster is not None:
			true_clusters = tensor([self._token_to_cluster.get(t, t) for t in targets], dtype=long)
		else:
			true_clusters = tensor(targets, dtype=long)

		# Generate negative samples (global top-k)
		num_negatives = min(5, global_top_k)
		false_clusters = zeros(n_examples, num_negatives, dtype=long)
		top_k_tensor = tensor(top_k_tokens, dtype=long)

		for i in range(n_examples):
			# Sample from top-k, excluding true target
			neg_indices = randint(0, global_top_k, (num_negatives,))
			neg_tokens = top_k_tensor[neg_indices]
			if self._token_to_cluster is not None:
				neg_clusters = tensor([self._token_to_cluster.get(int(t), int(t)) for t in neg_tokens], dtype=long)
			else:
				neg_clusters = neg_tokens
			false_clusters[i] = neg_clusters

		# Train in batches
		modified = 0
		for start_idx in range(0, n_examples, batch_size):
			end_idx = min(start_idx + batch_size, n_examples)
			batch_input = input_bits[start_idx:end_idx]
			batch_true = true_clusters[start_idx:end_idx]
			batch_false = false_clusters[start_idx:end_idx]

			modified += self.layer.train_batch(
				batch_input, batch_true, batch_false, allow_override=False
			)

			if verbose and (start_idx // batch_size) % 10 == 0:
				pct = start_idx / n_examples * 100
				print(f"  Training: {pct:.1f}%")

		elapsed = time.time() - start

		return {
			'modified': modified,
			'examples': n_examples,
			'time': elapsed,
		}

	def evaluate(
		self,
		tokens: list[int],
		batch_size: int = 1000,
		verbose: bool = False,
	) -> Dict:
		"""
		Evaluate on token sequence.

		Args:
			tokens: Evaluation token sequence
			batch_size: Batch size for evaluation
			verbose: Print progress

		Returns:
			Dict with cross_entropy, perplexity, accuracy
		"""
		import time

		from torch import clamp, log, long, stack, tensor
		from torch.nn.functional import softmax

		start = time.time()

		# Prepare examples
		n_examples = len(tokens) - self.context_size
		contexts = []
		targets = []

		for i in range(n_examples):
			context = tokens[i:i + self.context_size]
			target = tokens[i + self.context_size]
			contexts.append(context)
			targets.append(target)

		# Encode all contexts
		all_input_bits = []
		for ctx in contexts:
			all_input_bits.append(self._encode_context(ctx))
		input_bits = stack(all_input_bits)

		# Convert targets to cluster indices for accuracy
		if self._token_to_cluster is not None:
			target_clusters = tensor([self._token_to_cluster.get(t, t) for t in targets], dtype=long)
		else:
			target_clusters = tensor(targets, dtype=long)

		# Evaluate in batches
		total_ce = 0.0
		total_correct = 0

		for start_idx in range(0, n_examples, batch_size):
			end_idx = min(start_idx + batch_size, n_examples)
			batch_input = input_bits[start_idx:end_idx]
			batch_targets = target_clusters[start_idx:end_idx]

			# Forward pass
			probs = self.layer.forward(batch_input)  # [batch, vocab_size]

			# Softmax over vocabulary
			probs_softmax = softmax(probs, dim=-1)

			# Cross-entropy: -log(p[target])
			target_probs = probs_softmax.gather(1, batch_targets.unsqueeze(1)).squeeze(1)
			target_probs = clamp(target_probs, min=1e-10)
			batch_ce = -log(target_probs).sum().item()
			total_ce += batch_ce

			# Accuracy
			predictions = probs_softmax.argmax(dim=-1)
			total_correct += (predictions == batch_targets).sum().item()

			if verbose and (start_idx // batch_size) % 10 == 0:
				pct = start_idx / n_examples * 100
				print(f"  Evaluating: {pct:.1f}%")

		elapsed = time.time() - start

		avg_ce = total_ce / n_examples
		perplexity = 2 ** (avg_ce / 0.693147)  # Convert to base-2 perplexity
		accuracy = total_correct / n_examples

		return {
			'cross_entropy': avg_ce,
			'perplexity': perplexity,
			'accuracy': accuracy,
			'examples': n_examples,
			'time': elapsed,
		}


def create_genome_evaluator(
	config: EvaluatorConfig,
	verbose: bool = False,
) -> Callable[[ClusterGenome], float]:
	"""
	Create an evaluation function for genome fitness.

	The returned function:
	1. Takes a ClusterGenome
	2. Builds an AdaptiveRAMLMWrapper
	3. Trains on config.train_tokens
	4. Evaluates on config.eval_tokens
	5. Returns cross-entropy (lower is better)

	Args:
		config: Evaluation configuration with tokens and parameters
		verbose: Print progress during train/eval

	Returns:
		Function mapping ClusterGenome -> float (cross-entropy)

	Example:
		config = EvaluatorConfig(
			train_tokens=train_tokens[:100000],
			eval_tokens=val_tokens[:10000],
			vocab_size=50257,
			context_size=4,
		)
		evaluate_fn = create_genome_evaluator(config)

		# Use with optimizer
		optimizer = AdaptiveClusterOptimizer(
			config=opt_config,
			evaluate_fn=evaluate_fn,
			num_clusters=50257,
		)
		result = optimizer.optimize()
	"""
	def evaluate_genome(genome: ClusterGenome) -> float:
		"""Evaluate a genome and return fitness (cross-entropy)."""
		# Build wrapper from genome
		wrapper = AdaptiveRAMLMWrapper(genome, config)

		# Train
		wrapper.train_epoch(
			config.train_tokens,
			global_top_k=config.global_top_k,
			batch_size=config.batch_size,
			verbose=verbose,
		)

		# Evaluate
		stats = wrapper.evaluate(
			config.eval_tokens,
			batch_size=config.eval_batch_size,
			verbose=verbose,
		)

		return stats['cross_entropy']

	return evaluate_genome


def evaluate_genome_with_accuracy(
	genome: ClusterGenome,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
	global_top_k: int = 1000,
	logger: Optional[Callable[[str], None]] = None,
) -> tuple[float, float]:
	"""
	Evaluate a genome and return (cross_entropy, accuracy).

	This is for checkpoint evaluation where we need accuracy in addition to CE.
	Slower than Rust evaluation but provides full metrics.

	Args:
		genome: ClusterGenome to evaluate
		train_tokens: Training token sequence
		eval_tokens: Evaluation token sequence
		vocab_size: Vocabulary size
		context_size: Context window size
		cluster_order: Token-to-cluster mapping order
		global_top_k: Top-k tokens for clustering
		logger: Optional logging function

	Returns:
		Tuple of (cross_entropy, accuracy)
	"""
	# Build config (EvaluatorConfig is defined in this file)
	config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		cluster_order=cluster_order,
		cluster_config=genome.cluster_config if genome.cluster_config else AdaptiveClusterConfig(),
		global_top_k=global_top_k,
	)

	# Create wrapper and train
	wrapper = AdaptiveRAMLMWrapper(genome, config)
	wrapper.train_epoch(train_tokens, global_top_k=global_top_k)

	# Evaluate with full metrics
	stats = wrapper.evaluate(eval_tokens)

	if logger:
		# Use DEBUG level if OptimizationLogger, otherwise call directly
		if isinstance(logger, OptimizationLogger):
			logger.debug(f"  Checkpoint eval: CE={stats['cross_entropy']:.4f}, Acc={stats['accuracy']:.2%}")
		else:
			logger(f"  Checkpoint eval: CE={stats['cross_entropy']:.4f}, Acc={stats['accuracy']:.2%}")

	return stats['cross_entropy'], stats['accuracy']


# =============================================================================
# High-Level API Functions
# =============================================================================

def run_architecture_tabu_search(
	initial_genome: ClusterGenome,
	initial_fitness: float,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
	# TS parameters
	iterations: int = 100,
	neighbors_per_iter: int = 20,
	patience: int = 10,
	# Architecture bounds
	min_bits: int = 4,
	max_bits: int = 20,
	min_neurons: int = 1,
	max_neurons: int = 15,
	phase: int = 2,
	# Other
	empty_value: float = 0.0,
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from previous phase
	initial_neighbors: Optional[list[ClusterGenome]] = None,
) -> OptimizerResult['ClusterGenome']:
	"""
	Run Tabu Search to refine architecture from a GA solution.

	Phase 1b: Takes the best genome from GA and applies local search
	to potentially find better nearby solutions.

	Args:
		initial_genome: Best genome from Phase 1a (GA)
		initial_fitness: Fitness of initial genome
		train_tokens: Training data
		eval_tokens: Evaluation data
		vocab_size: Vocabulary size
		context_size: Context window
		cluster_order: Token ordering by frequency
		iterations: Number of TS iterations
		neighbors_per_iter: Neighbors to evaluate per iteration
		patience: Early stop patience
		min_bits, max_bits: Bits bounds
		min_neurons, max_neurons: Neurons bounds
		phase: Optimization phase
		empty_value: EMPTY cell value
		seed: Random seed
		logger: Logging function
		initial_neighbors: Optional seed neighbors from Phase 1a population

	Returns:
		OptimizerResult with refined genome
	"""
	log = logger or print

	log()
	log("=" * 60)
	log("  Phase 1b: Architecture Tabu Search (Refinement)")
	log("=" * 60)
	log(f"  Initial fitness: {initial_fitness:.4f}")
	log(f"  Iterations: {iterations}")
	log(f"  Neighbors/iter: {neighbors_per_iter}")
	log()

	# Create evaluator config (no rng = truly random connectivity)
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
		# rng=None by default: architecture search should not depend on specific connectivity
	)

	# Create evaluation function
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[ArchitectureTS] Using Rust parallel evaluator")
	except Exception as e:
		log(f"[ArchitectureTS] Using Python evaluator ({e})")

	# Compute total input bits for connection preservation
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[ArchitectureTS] Connection-preserving search enabled ({total_input_bits} input bits)")

	# Create TS strategy using factory
	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_TS,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# TS parameters
		iterations=iterations,
		neighbors_per_iter=neighbors_per_iter,
		patience=patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Run optimization
	result = strategy.optimize(
		initial_genome=initial_genome,
		initial_fitness=initial_fitness,
		evaluate_fn=evaluate_fn,
		initial_neighbors=initial_neighbors,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	# Log results
	log()
	log("=" * 60)
	log("  Phase 1b Complete")
	log("=" * 60)
	stats = result.best_genome.stats()
	log(f"  Initial CE: {result.initial_fitness:.4f}")
	log(f"  Final CE: {result.final_fitness:.4f}")
	improvement = (1 - result.final_fitness / result.initial_fitness) * 100
	log(f"  Improvement: {improvement:.2f}%")
	log(f"  Iterations: {result.iterations_run}")
	log()
	log("  Refined genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")

	return result


def run_architecture_search(
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	token_frequencies: Optional[list[int]] = None,
	cluster_order: Optional[list[int]] = None,
	# GA parameters
	population_size: int = 10,
	generations: int = 20,
	patience: int = 5,
	# Architecture bounds
	min_bits: int = 4,
	max_bits: int = 20,
	min_neurons: int = 1,
	max_neurons: int = 15,
	phase: int = 2,
	# Other
	init_strategy: GenomeInitStrategy = GenomeInitStrategy.FREQUENCY_SCALED,
	empty_value: float = 0.0,
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from previous phase
	initial_population: Optional[list[ClusterGenome]] = None,
) -> OptimizerResult['ClusterGenome']:
	"""
	Run complete architecture search for adaptive cluster configuration.

	This is the main entry point for discovering optimal per-cluster
	architectures using genetic algorithm optimization.

	Args:
		train_tokens: Training token sequence
		eval_tokens: Evaluation token sequence
		vocab_size: Vocabulary size
		context_size: Context window size
		token_frequencies: Token occurrence counts (for FREQUENCY_SCALED init)
		cluster_order: Token IDs sorted by frequency (for tier assignment)

		population_size: GA population size
		generations: Maximum generations
		patience: Early stop patience

		min_bits, max_bits: Bits per neuron bounds
		min_neurons, max_neurons: Neurons per cluster bounds
		phase: 1 = bits only, 2 = bits + neurons

		init_strategy: How to initialize genomes
		empty_value: Value for EMPTY cells (0.0 recommended)
		seed: Random seed
		logger: Logging function

	Returns:
		OptimizerResult with best genome and optimization history

	Example:
		from collections import Counter

		# Compute token frequencies
		counts = Counter(train_tokens)
		token_frequencies = [counts.get(i, 0) for i in range(vocab_size)]
		cluster_order = sorted(range(vocab_size), key=lambda t: -counts.get(t, 0))

		result = run_architecture_search(
			train_tokens=train_tokens[:500000],
			eval_tokens=val_tokens[:50000],
			vocab_size=50257,
			token_frequencies=token_frequencies,
			cluster_order=cluster_order,
			population_size=10,
			generations=50,
		)

		print(f"Best cross-entropy: {result.final_fitness:.4f}")
		print(f"Improvement: {(1 - result.final_fitness/result.initial_fitness)*100:.1f}%")
	"""
	log = logger or print

	log("=" * 60)
	log("  Adaptive Architecture Search")
	log("=" * 60)
	log(f"  Train tokens: {len(train_tokens):,}")
	log(f"  Eval tokens: {len(eval_tokens):,}")
	log(f"  Vocab size: {vocab_size:,}")
	log(f"  Context size: {context_size}")
	log(f"  Population: {population_size}")
	log(f"  Generations: {generations}")
	log(f"  Phase: {phase} ({'bits only' if phase == 1 else 'bits + neurons'})")
	log(f"  Init strategy: {init_strategy.name}")
	log()

	# Create evaluator config (no rng = truly random connectivity)
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
		# rng=None by default: architecture search should not depend on specific connectivity
	)

	# Create evaluation function (fallback for single genome evaluation)
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator for batch evaluation (hybrid dense/sparse memory)
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[ArchitectureGA] Using Rust parallel evaluator (hybrid dense/sparse)")
	except ImportError:
		log("[ArchitectureGA] Rust accelerator not available, using Python sequential")
	except Exception as e:
		log(f"[ArchitectureGA] Warning: Rust evaluator init failed ({e}), using Python")

	# Compute total input bits for connection initialization
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[ArchitectureGA] Connection-preserving search enabled ({total_input_bits} input bits)")

	# Create GA strategy using factory
	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# GA parameters
		population_size=population_size,
		generations=generations,
		patience=patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Run optimization
	log()
	result = strategy.optimize(
		evaluate_fn=evaluate_fn,
		initial_population=initial_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	# Log final results
	log()
	log("=" * 60)
	log("  Architecture Search Complete")
	log("=" * 60)
	stats = result.best_genome.stats()
	log(f"  Initial CE: {result.initial_fitness:.4f}")
	log(f"  Final CE: {result.final_fitness:.4f}")
	improvement = (1 - result.final_fitness / result.initial_fitness) * 100
	log(f"  Improvement: {improvement:.1f}%")
	log(f"  Generations: {result.iterations_run}")
	log(f"  Early stopped: {result.early_stopped}")
	log()
	log("  Best genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")
	log(f"    Total memory: {stats['total_memory_cells']:,} cells")

	return result


# =============================================================================
# Connectivity Optimization (Phase 2)
# =============================================================================

@dataclass
class ConnectivityOptResult:
	"""Result from connectivity optimization (Phase 2 GA→TS)."""

	initial_fitness: float  # Phase 1 baseline
	phase2_baseline: float  # Fitness after Phase 2a GA
	final_fitness: float    # Fitness after Phase 2b TS
	ga_improvement_pct: float  # Improvement from Phase 2a GA
	ts_improvement_pct: float  # Improvement from Phase 2b TS
	total_improvement_pct: float  # Total improvement vs Phase 1 baseline
	ga_iterations: int
	ts_iterations: int
	early_stopped: bool
	initial_accuracy: Optional[float] = None   # Accuracy at Phase 2 start
	ga_final_accuracy: Optional[float] = None  # Accuracy after Phase 2a GA
	final_accuracy: Optional[float] = None     # Accuracy after Phase 2b TS
	# Population seeding for potential future phases
	final_population: Optional[list[ClusterGenome]] = None  # From Phase 2 TS


def run_connectivity_optimization(
	genome: ClusterGenome,
	genome_fitness: float,
	train_tokens: list[int],
	eval_tokens: list[int],
	vocab_size: int = 50257,
	context_size: int = 4,
	cluster_order: Optional[list[int]] = None,
	token_frequencies: Optional[list[int]] = None,
	# GA parameters
	ga_population: int = 20,
	ga_generations: int = 30,
	ga_patience: int = 5,
	# TS parameters
	ts_iterations: int = 50,
	ts_neighbors: int = 30,
	ts_patience: int = 5,
	# Architecture bounds (same as Phase 1)
	min_bits: int = 8,
	max_bits: int = 25,
	min_neurons: int = 3,
	max_neurons: int = 33,
	phase: int = 2,
	# Other
	empty_value: float = 0.0,
	seed: Optional[int] = None,  # None = time-based
	logger: Optional[Callable[[str], None]] = None,
	# Population seeding from Phase 1b
	initial_population: Optional[list[ClusterGenome]] = None,
) -> ConnectivityOptResult:
	"""
	Run Phase 2: Continue architecture optimization with GA→TS.

	Continues optimizing architecture from Phase 1b using GA followed by TS.
	Each evaluation uses random connectivity, so this effectively finds
	architectures robust to connectivity variations.

	The pipeline:
	1. Phase 2a (GA): Evolve architecture with initial_population from Phase 1b
	2. Phase 2b (TS): Refine best GA solution with GA's population as neighbors

	Args:
		genome: Best architecture from Phase 1b (used if no initial_population)
		genome_fitness: Baseline fitness from Phase 1b
		train_tokens: Training data
		eval_tokens: Evaluation data
		vocab_size: Vocabulary size
		context_size: Context window
		cluster_order: Token ordering by frequency
		token_frequencies: Token occurrence counts (for genome generation)
		ga_population: GA population size
		ga_generations: GA generations
		ga_patience: GA early stop patience
		ts_iterations: TS iterations
		ts_neighbors: TS neighbors per iteration
		ts_patience: TS early stop patience
		min_bits, max_bits: Bits bounds
		min_neurons, max_neurons: Neurons bounds
		phase: Optimization phase (1=bits only, 2=bits+neurons)
		empty_value: EMPTY cell value
		seed: Random seed
		logger: Logging function
		initial_population: Seed population from Phase 1b's final_neighbors

	Returns:
		ConnectivityOptResult with optimization statistics and final_population
	"""
	log = logger or print

	log()
	log("=" * 60)
	log("  Phase 2: Architecture Refinement (GA→TS)")
	log("=" * 60)
	log(f"  Phase 1b fitness: {genome_fitness:.4f}")
	log(f"  Phase 2a (GA): pop={ga_population}, gens={ga_generations}")
	log(f"  Phase 2b (TS): iters={ts_iterations}, neighbors={ts_neighbors}")
	if initial_population:
		log(f"  Seed population: {len(initial_population)} genomes from Phase 1b")
	log()

	# Create evaluator config
	eval_config = EvaluatorConfig(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		batch_size=500,
		global_top_k=100,
		empty_value=empty_value,
		eval_batch_size=1000,
		cluster_order=cluster_order,
	)

	# Create evaluation function
	evaluate_fn = create_genome_evaluator(eval_config, verbose=False)

	# Create Rust parallel evaluator
	batch_evaluator = None
	try:
		batch_evaluator = RustParallelEvaluator(eval_config)
		log("[Phase2] Using Rust parallel evaluator")
	except Exception as e:
		log(f"[Phase2] Using Python evaluator ({e})")

	# Create batch evaluation function using Rust evaluator (returns list[(CE, accuracy)])
	batch_evaluate_fn = None
	if batch_evaluator is not None:
		batch_evaluate_fn = lambda genomes, min_accuracy=None: batch_evaluator.evaluate_batch(genomes, logger=log, min_accuracy=min_accuracy)

	# Compute total input bits for connection preservation
	from wnn.ram.core import bits_needed
	bits_per_token = bits_needed(vocab_size)
	total_input_bits = context_size * bits_per_token
	log(f"[Phase2] Connection-preserving search enabled ({total_input_bits} input bits)")

	# =========================================================================
	# Phase 2a: GA
	# =========================================================================
	log()
	log("-" * 40)
	log("  Phase 2a: GA Architecture Refinement")
	log("-" * 40)

	# Create GA strategy using factory
	ga_strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		token_frequencies=token_frequencies,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# GA parameters
		population_size=ga_population,
		generations=ga_generations,
		patience=ga_patience,
		seed=seed,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Run GA with seeded population from Phase 1b
	ga_result = ga_strategy.optimize(
		evaluate_fn=evaluate_fn,
		initial_population=initial_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	log()
	log(f"[Phase2a] GA complete: {ga_result.final_fitness:.4f} "
		f"({(1 - ga_result.final_fitness / genome_fitness) * 100:.2f}% vs Phase 1b)")

	# =========================================================================
	# Phase 2b: TS
	# =========================================================================
	log()
	log("-" * 40)
	log("  Phase 2b: TS Architecture Refinement")
	log("-" * 40)

	# Create TS strategy using factory
	ts_strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_TS,
		num_clusters=vocab_size,
		min_bits=min_bits,
		max_bits=max_bits,
		min_neurons=min_neurons,
		max_neurons=max_neurons,
		phase=phase,
		total_input_bits=total_input_bits,  # Enable connection preservation
		# TS parameters
		iterations=ts_iterations,
		neighbors_per_iter=ts_neighbors,
		patience=ts_patience,
		seed=(seed + 500) if seed is not None else None,
		logger=log,
		batch_evaluator=batch_evaluator,
	)

	# Run TS with GA's population as initial neighbors
	ts_result = ts_strategy.optimize(
		initial_genome=ga_result.best_genome,
		initial_fitness=ga_result.final_fitness,
		evaluate_fn=evaluate_fn,
		initial_neighbors=ga_result.final_population,
		batch_evaluate_fn=batch_evaluate_fn,
	)

	log()
	log(f"[Phase2b] TS complete: {ts_result.final_fitness:.4f} "
		f"({(1 - ts_result.final_fitness / ga_result.final_fitness) * 100:.2f}% vs Phase 2a)")

	# =========================================================================
	# Summary
	# =========================================================================
	ga_improvement = (genome_fitness - ga_result.final_fitness) / genome_fitness * 100 if genome_fitness > 0 else 0
	ts_improvement = (ga_result.final_fitness - ts_result.final_fitness) / ga_result.final_fitness * 100 if ga_result.final_fitness > 0 else 0
	total_improvement = (genome_fitness - ts_result.final_fitness) / genome_fitness * 100 if genome_fitness > 0 else 0

	log()
	log("=" * 60)
	log("  Phase 2 Complete")
	log("=" * 60)
	log(f"  Phase 1b baseline: {genome_fitness:.4f}")
	log(f"  After Phase 2a (GA): {ga_result.final_fitness:.4f} ({ga_improvement:.2f}% improvement)")
	log(f"  After Phase 2b (TS): {ts_result.final_fitness:.4f} ({ts_improvement:.2f}% improvement)")
	log(f"  Total Phase 2 improvement: {total_improvement:.2f}%")
	log()
	stats = ts_result.best_genome.stats()
	log("  Best genome:")
	log(f"    Bits: [{stats['min_bits']}, {stats['max_bits']}], mean: {stats['mean_bits']:.1f}")
	log(f"    Neurons: [{stats['min_neurons']}, {stats['max_neurons']}], mean: {stats['mean_neurons']:.1f}")

	return ConnectivityOptResult(
		initial_fitness=genome_fitness,
		phase2_baseline=ga_result.final_fitness,
		final_fitness=ts_result.final_fitness,
		ga_improvement_pct=ga_improvement,
		ts_improvement_pct=ts_improvement,
		total_improvement_pct=total_improvement,
		ga_iterations=ga_result.iterations_run,
		ts_iterations=ts_result.iterations_run,
		early_stopped=ga_result.early_stopped or ts_result.early_stopped,
		initial_accuracy=ga_result.initial_accuracy,
		ga_final_accuracy=ga_result.final_accuracy,  # After Phase 2a GA
		final_accuracy=ts_result.final_accuracy,     # After Phase 2b TS
		final_population=ts_result.final_population,  # For potential Phase 3
	)
