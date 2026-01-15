"""
Accelerated connectivity optimization with Rust backend.

This module provides:
1. EvaluationContext - holds all data needed for batch evaluation
2. AcceleratedOptimizer - wraps strategies with automatic Rust acceleration
3. OptimizerConfig - configuration for optimization strategies

The optimizer handles everything transparently - callers just provide
initial connectivity and the optimizer returns optimized connectivity.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from enum import Enum, auto

import torch
from torch import Tensor

from wnn.ram.core import AccelerationMode
from wnn.ram.strategies.connectivity.base import OptimizerResult, OptimizerStrategyBase
from wnn.ram.strategies.connectivity.genetic_algorithm import (
	GeneticAlgorithmStrategy,
	GeneticAlgorithmConfig,
)
from wnn.ram.strategies.connectivity.tabu_search import (
	TabuSearchStrategy,
	TabuSearchConfig,
)


# ============================================================================
# RUST ACCELERATOR DETECTION
# ============================================================================
try:
	import ram_accelerator
	RUST_AVAILABLE = True
	RUST_CPU_CORES = ram_accelerator.cpu_cores()
	METAL_AVAILABLE = ram_accelerator.metal_available()
	METAL_DEVICE = ram_accelerator.metal_device_info() if METAL_AVAILABLE else None
	# GPU cores estimated from Metal device (M4 Max = 40, M4 Pro = 20, etc.)
	# Metal doesn't expose core count directly, but we can estimate from device name
	METAL_GPU_CORES = 40 if METAL_AVAILABLE else 0  # Conservative default for Apple Silicon
except ImportError:
	RUST_AVAILABLE = False
	RUST_CPU_CORES = 0
	METAL_AVAILABLE = False
	METAL_DEVICE = None
	METAL_GPU_CORES = 0


def get_effective_cores(mode: 'AccelerationMode') -> int:
	"""Get effective parallel worker count for the acceleration mode."""
	from wnn.ram.core import AccelerationMode
	match mode:
		case AccelerationMode.CPU:
			return RUST_CPU_CORES
		case AccelerationMode.METAL:
			return METAL_GPU_CORES if METAL_AVAILABLE else RUST_CPU_CORES
		case AccelerationMode.HYBRID:
			return (RUST_CPU_CORES + METAL_GPU_CORES) if METAL_AVAILABLE else RUST_CPU_CORES
		case AccelerationMode.AUTO:
			# AUTO uses HYBRID if GPU available, else CPU
			if METAL_AVAILABLE:
				return RUST_CPU_CORES + METAL_GPU_CORES
			return RUST_CPU_CORES
		case _:
			return RUST_CPU_CORES


def resolve_acceleration_mode(mode: 'AccelerationMode') -> 'AccelerationMode':
	"""Resolve AUTO mode to the actual backend that will be used."""
	from wnn.ram.core import AccelerationMode
	if mode == AccelerationMode.AUTO:
		if METAL_AVAILABLE:
			return AccelerationMode.HYBRID
		elif RUST_AVAILABLE:
			return AccelerationMode.CPU
		else:
			return AccelerationMode.PYTORCH
	return mode


class OptimizationStrategy(Enum):
	"""Available optimization strategies."""
	GENETIC_ALGORITHM = auto()
	TABU_SEARCH = auto()
	HYBRID_GA_TS = auto()  # GA first, then TS refinement
	HYBRID_TS_GA = auto()  # TS first, then GA crossover


@dataclass
class OptimizerConfig:
	"""
	Configuration for connectivity optimization.

	Attributes:
		strategy: Which optimization strategy to use
		acceleration_mode: Hardware acceleration (AUTO, CPU, METAL, HYBRID)
		ga_population: GA population size
		ga_generations: GA number of generations
		ga_mutation_rate: GA per-connection mutation probability
		ga_crossover_rate: GA crossover probability
		ts_iterations: TS number of iterations
		ts_neighbors: TS neighbors per iteration
		ts_tabu_size: TS tabu list size
		ts_mutation_rate: TS per-connection mutation probability
		timeout_seconds: Maximum time for optimization
		seed: Random seed for reproducibility
		verbose: Print progress during optimization
	"""
	strategy: OptimizationStrategy = OptimizationStrategy.HYBRID_GA_TS

	# Hardware acceleration (AUTO uses Metal if available, best for unified memory)
	acceleration_mode: AccelerationMode = AccelerationMode.AUTO

	# Genetic Algorithm parameters (Garcia 2003 defaults)
	ga_population: int = 30
	ga_generations: int = 50
	ga_mutation_rate: float = 0.01
	ga_crossover_rate: float = 0.7
	ga_elitism: int = 2

	# Tabu Search parameters (Garcia 2003 defaults)
	ts_iterations: int = 5
	ts_neighbors: int = 30
	ts_tabu_size: int = 5
	ts_mutation_rate: float = 0.01

	# General parameters
	timeout_seconds: Optional[int] = None
	seed: Optional[int] = 42
	verbose: bool = True


@dataclass
class EvaluationContext:
	"""
	Holds all data needed for evaluating connectivity patterns.

	This encapsulates:
	- Word encoding (word_to_bits mapping)
	- Training/test token sequences
	- Evaluation parameters

	The context is used internally by AcceleratedOptimizer to create
	batch evaluation functions that leverage the Rust accelerator.
	"""
	word_to_bits: dict[str, int]  # word -> 12-bit encoding
	train_tokens: list[str]
	test_tokens: list[str]
	bits_per_neuron: int
	eval_subset: int = 3000
	n_context_words: int = 4  # n-gram size

	@classmethod
	def from_word_clusters(
		cls,
		word_to_cluster: dict[str, int],
		train_tokens: list[str],
		test_tokens: list[str],
		bits_per_neuron: int,
		eval_subset: int = 3000,
		n_context_words: int = 4,
	) -> "EvaluationContext":
		"""
		Create context from word clusters (computes word_to_bits automatically).

		This matches the encoding used in GeneralizedNGramRAM:
		- 7 bits from cluster ID
		- 5 bits from word hash
		"""
		n_clusters = 256
		word_to_bits = {}

		def compute_word_bits(word: str, cluster: int) -> int:
			"""Compute 12-bit encoding: 7 cluster bits + 5 hash bits"""
			h = hash(word)
			bits = 0
			# First 7 bits from cluster
			for i in range(7):
				if (cluster >> i) & 1:
					bits |= 1 << i
			# Next 5 bits from hash
			for i in range(5):
				if (h >> i) & 1:
					bits |= 1 << (7 + i)
			return bits

		# Encode words from cluster mapping
		for word, cluster in word_to_cluster.items():
			word_to_bits[word] = compute_word_bits(word, cluster)

		# Add missing words from tokens (use hash-based cluster)
		for word in set(train_tokens) | set(test_tokens):
			if word not in word_to_bits:
				cluster = hash(word) % n_clusters
				word_to_bits[word] = compute_word_bits(word, cluster)

		return cls(
			word_to_bits=word_to_bits,
			train_tokens=train_tokens,
			test_tokens=test_tokens,
			bits_per_neuron=bits_per_neuron,
			eval_subset=eval_subset,
			n_context_words=n_context_words,
		)


class AcceleratedOptimizer:
	"""
	Connectivity optimizer with automatic Rust acceleration.

	This class wraps the optimization strategies and provides:
	1. Automatic Rust accelerator detection and usage
	2. Batch evaluation for population-based algorithms
	3. Transparent hybrid strategy support
	4. Progress logging

	Usage:
		context = EvaluationContext.from_word_clusters(...)
		optimizer = AcceleratedOptimizer(config, context)
		result = optimizer.optimize(initial_connectivity)
	"""

	def __init__(
		self,
		config: OptimizerConfig,
		context: EvaluationContext,
		log_fn: Optional[Callable[[str], None]] = None,
	):
		self.config = config
		self.context = context
		self.log_fn = log_fn or print

		# Create evaluation function (uses Rust if available)
		self._batch_evaluate_fn = self._create_batch_evaluate_fn()
		self._single_evaluate_fn = self._create_single_evaluate_fn()

		# Cache for evaluated patterns
		self._eval_cache: dict[tuple, float] = {}
		self._cache_hits = 0
		self._cache_misses = 0

	def _log(self, msg: str):
		"""Log message if verbose."""
		if self.config.verbose:
			self.log_fn(msg)

	def _create_batch_evaluate_fn(self) -> Callable[[list, int], list[float]]:
		"""Create batch evaluation function based on acceleration mode."""
		ctx = self.context
		resolved_mode = resolve_acceleration_mode(self.config.acceleration_mode)

		if not RUST_AVAILABLE:
			# Fallback to Python (much slower)
			return self._batch_evaluate_python

		# Select evaluation function based on resolved mode
		# With Apple Silicon unified memory, Metal has zero copy overhead
		use_metal = METAL_AVAILABLE and resolved_mode in (
			AccelerationMode.METAL,
			AccelerationMode.HYBRID,
		)

		if use_metal:
			def batch_eval_metal(connectivities: list, total_pop: int = 0) -> list[float]:
				conn_list = [
					conn.tolist() if hasattr(conn, 'tolist') else conn
					for conn in connectivities
				]
				return ram_accelerator.evaluate_batch_metal(
					conn_list,
					ctx.word_to_bits,
					ctx.train_tokens,
					ctx.test_tokens,
					ctx.bits_per_neuron,
					ctx.eval_subset,
				)
			return batch_eval_metal
		else:
			def batch_eval_cpu(connectivities: list, total_pop: int = 0) -> list[float]:
				conn_list = [
					conn.tolist() if hasattr(conn, 'tolist') else conn
					for conn in connectivities
				]
				return ram_accelerator.evaluate_batch_cpu(
					conn_list,
					ctx.word_to_bits,
					ctx.train_tokens,
					ctx.test_tokens,
					ctx.bits_per_neuron,
					ctx.eval_subset,
				)
			return batch_eval_cpu

	def _batch_evaluate_python(self, connectivities: list) -> list[float]:
		"""Python fallback for batch evaluation using joblib for parallelism."""
		try:
			from joblib import Parallel, delayed
			import multiprocessing
			n_workers = max(1, multiprocessing.cpu_count() - 2)

			if len(connectivities) <= 1:
				return [self._evaluate_single_python(conn) for conn in connectivities]

			# Parallel evaluation with joblib
			results = Parallel(n_jobs=n_workers, prefer="processes")(
				delayed(self._evaluate_single_python)(conn)
				for conn in connectivities
			)
			return results
		except ImportError:
			# Fallback to serial if joblib not available
			return [self._evaluate_single_python(conn) for conn in connectivities]

	def _evaluate_single_python(self, connectivity) -> float:
		"""Evaluate single connectivity pattern in Python."""
		from collections import defaultdict, Counter

		ctx = self.context

		# Convert to list if tensor
		if hasattr(connectivity, 'tolist'):
			conn_list = connectivity.tolist()
		else:
			conn_list = connectivity

		# Build RAM neurons
		neurons = []
		for connected_bits in conn_list:
			neurons.append({
				'connected_bits': connected_bits,
				'ram': defaultdict(Counter)
			})

		# Encode word to bits
		def word_to_bits_tuple(word: str) -> tuple:
			bits_val = ctx.word_to_bits.get(word, 0)
			return tuple((bits_val >> i) & 1 for i in range(12))

		# Train
		n = ctx.n_context_words
		for i in range(len(ctx.train_tokens) - n):
			context_words = ctx.train_tokens[i:i + n]
			target = ctx.train_tokens[i + n]

			# Build full bit vector
			full_bits = []
			for w in context_words:
				full_bits.extend(word_to_bits_tuple(w))
			full_bits = tuple(full_bits)

			# Train each neuron
			for neuron in neurons:
				addr = tuple(full_bits[b] for b in neuron['connected_bits'] if b < len(full_bits))
				if addr:
					neuron['ram'][addr][target] += 1

		# Test
		correct = 0
		total = min(ctx.eval_subset, len(ctx.test_tokens) - n)

		for i in range(total):
			context_words = ctx.test_tokens[i:i + n]
			target = ctx.test_tokens[i + n]

			full_bits = []
			for w in context_words:
				full_bits.extend(word_to_bits_tuple(w))
			full_bits = tuple(full_bits)

			# Voting prediction
			votes = Counter()
			for neuron in neurons:
				addr = tuple(full_bits[b] for b in neuron['connected_bits'] if b < len(full_bits))
				if addr and addr in neuron['ram']:
					counts = neuron['ram'][addr]
					total_count = sum(counts.values())
					best, count = counts.most_common(1)[0]
					votes[best] += count / total_count

			if votes:
				pred, _ = votes.most_common(1)[0]
				if pred == target:
					correct += 1

		return 1.0 - (correct / total) if total > 0 else 1.0

	def _create_single_evaluate_fn(self) -> Callable[[Tensor], float]:
		"""Create single-pattern evaluation function."""
		batch_fn = self._batch_evaluate_fn

		def single_eval(connectivity: Tensor) -> float:
			# Check cache
			if hasattr(connectivity, 'tolist'):
				cache_key = tuple(tuple(row) for row in connectivity.tolist())
			else:
				cache_key = tuple(tuple(row) for row in connectivity)

			if cache_key in self._eval_cache:
				self._cache_hits += 1
				return self._eval_cache[cache_key]

			self._cache_misses += 1
			result = batch_fn([connectivity])[0]
			self._eval_cache[cache_key] = result
			return result

		return single_eval

	def optimize(
		self,
		initial_connectivity: Tensor,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""
		Optimize connectivity pattern.

		Args:
			initial_connectivity: Starting connectivity [num_neurons, n_bits_per_neuron]
			total_input_bits: Total bits in input vector
			num_neurons: Number of RAM neurons
			n_bits_per_neuron: Fan-in per neuron

		Returns:
			OptimizerResult with optimized connectivity and statistics
		"""
		self._eval_cache.clear()
		self._cache_hits = 0
		self._cache_misses = 0

		strategy = self.config.strategy
		resolved_mode = resolve_acceleration_mode(self.config.acceleration_mode)
		effective_cores = get_effective_cores(resolved_mode)
		self._log(f"Starting optimization with {strategy.name}")

		# Log acceleration backend with resolved mode and core count
		if not RUST_AVAILABLE:
			self._log("Accelerator: Python (slow)")
		else:
			match resolved_mode:
				case AccelerationMode.HYBRID:
					self._log(f"Accelerator: Hybrid CPU+GPU ({RUST_CPU_CORES}+{METAL_GPU_CORES}={effective_cores} cores)")
				case AccelerationMode.METAL:
					self._log(f"Accelerator: Metal GPU ({METAL_GPU_CORES} cores)")
				case AccelerationMode.CPU:
					self._log(f"Accelerator: Rust CPU ({RUST_CPU_CORES} cores)")
				case _:
					self._log(f"Accelerator: {resolved_mode.name} ({effective_cores} cores)")

		if strategy == OptimizationStrategy.GENETIC_ALGORITHM:
			return self._run_ga(initial_connectivity, total_input_bits, num_neurons, n_bits_per_neuron)

		elif strategy == OptimizationStrategy.TABU_SEARCH:
			return self._run_ts(initial_connectivity, total_input_bits, num_neurons, n_bits_per_neuron)

		elif strategy == OptimizationStrategy.HYBRID_GA_TS:
			return self._run_hybrid_ga_ts(initial_connectivity, total_input_bits, num_neurons, n_bits_per_neuron)

		elif strategy == OptimizationStrategy.HYBRID_TS_GA:
			return self._run_hybrid_ts_ga(initial_connectivity, total_input_bits, num_neurons, n_bits_per_neuron)

		else:
			raise ValueError(f"Unknown strategy: {strategy}")

	def _run_ga(
		self,
		connections: Tensor,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""Run Genetic Algorithm optimization."""
		cfg = self.config

		ga_config = GeneticAlgorithmConfig(
			population_size=cfg.ga_population,
			generations=cfg.ga_generations,
			mutation_rate=cfg.ga_mutation_rate,
			crossover_rate=cfg.ga_crossover_rate,
			elitism=cfg.ga_elitism,
		)
		strategy = GeneticAlgorithmStrategy(
			config=ga_config,
			seed=cfg.seed,
			verbose=cfg.verbose,
		)

		return strategy.optimize(
			connections=connections,
			evaluate_fn=self._single_evaluate_fn,
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			batch_evaluate_fn=self._batch_evaluate_fn,  # Use batch evaluation!
		)

	def _run_ts(
		self,
		connections: Tensor,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""Run Tabu Search optimization."""
		cfg = self.config

		ts_config = TabuSearchConfig(
			iterations=cfg.ts_iterations,
			neighbors_per_iter=cfg.ts_neighbors,
			tabu_size=cfg.ts_tabu_size,
			mutation_rate=cfg.ts_mutation_rate,
		)
		strategy = TabuSearchStrategy(
			config=ts_config,
			seed=cfg.seed,
			verbose=cfg.verbose,
		)

		return strategy.optimize(
			connections=connections,
			evaluate_fn=self._single_evaluate_fn,
			total_input_bits=total_input_bits,
			num_neurons=num_neurons,
			n_bits_per_neuron=n_bits_per_neuron,
			batch_evaluate_fn=self._batch_evaluate_fn,  # Use batch evaluation!
		)

	def _run_hybrid_ga_ts(
		self,
		connections: Tensor,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""Run hybrid GA→TS optimization."""
		cfg = self.config
		initial_error = self._single_evaluate_fn(connections)

		self._log(f"Phase 1: GA ({cfg.ga_population} pop × {cfg.ga_generations} gens)")

		# Phase 1: GA exploration
		ga_result = self._run_ga(
			connections, total_input_bits, num_neurons, n_bits_per_neuron
		)

		self._log(f"  GA: {(1-initial_error)*100:.2f}% → {(1-ga_result.final_fitness)*100:.2f}% acc×cov")
		self._log(f"Phase 2: TS ({cfg.ts_neighbors} neighbors × {cfg.ts_iterations} iters)")

		# Phase 2: TS refinement from GA's best
		ts_result = self._run_ts(
			ga_result.best_genome,
			total_input_bits, num_neurons, n_bits_per_neuron
		)

		self._log(f"  TS: {(1-ga_result.final_fitness)*100:.2f}% → {(1-ts_result.final_fitness)*100:.2f}% acc×cov")

		# Combine results
		final_error = ts_result.final_fitness
		improvement = ((initial_error - final_error) / initial_error * 100) if final_error < initial_error else 0.0

		self._log(f"Hybrid complete: {(1-initial_error)*100:.2f}% → {(1-final_error)*100:.2f}% ({improvement:.1f}% improvement)")
		self._log(f"Cache: {self._cache_hits} hits / {self._cache_hits + self._cache_misses} total")

		return OptimizerResult(
			initial_genome=connections,
			best_genome=ts_result.best_genome,
			initial_fitness=initial_error,
			final_fitness=final_error,
			improvement_percent=improvement,
			iterations_run=cfg.ga_generations + cfg.ts_iterations,
			method_name="Hybrid_GA_TS",
			history=ga_result.history + [(g + cfg.ga_generations, e) for g, e in ts_result.history],
		)

	def _run_hybrid_ts_ga(
		self,
		connections: Tensor,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""Run hybrid TS→GA optimization."""
		cfg = self.config
		initial_error = self._single_evaluate_fn(connections)

		self._log(f"Phase 1: TS ({cfg.ts_neighbors} neighbors × {cfg.ts_iterations} iters)")

		# Phase 1: TS exploration
		ts_result = self._run_ts(
			connections, total_input_bits, num_neurons, n_bits_per_neuron
		)

		self._log(f"  TS: {(1-initial_error)*100:.2f}% → {(1-ts_result.final_fitness)*100:.2f}% acc×cov")
		self._log(f"Phase 2: GA ({cfg.ga_population} pop × {cfg.ga_generations} gens)")

		# Phase 2: GA crossover from TS's best
		ga_result = self._run_ga(
			ts_result.best_genome,
			total_input_bits, num_neurons, n_bits_per_neuron
		)

		self._log(f"  GA: {(1-ts_result.final_fitness)*100:.2f}% → {(1-ga_result.final_fitness)*100:.2f}% acc×cov")

		# Combine results
		final_error = ga_result.final_fitness
		improvement = ((initial_error - final_error) / initial_error * 100) if final_error < initial_error else 0.0

		self._log(f"Hybrid complete: {(1-initial_error)*100:.2f}% → {(1-final_error)*100:.2f}% ({improvement:.1f}% improvement)")
		self._log(f"Cache: {self._cache_hits} hits / {self._cache_hits + self._cache_misses} total")

		return OptimizerResult(
			initial_genome=connections,
			best_genome=ga_result.best_genome,
			initial_fitness=initial_error,
			final_fitness=final_error,
			improvement_percent=improvement,
			iterations_run=cfg.ts_iterations + cfg.ga_generations,
			method_name="Hybrid_TS_GA",
			history=ts_result.history + [(i + cfg.ts_iterations, e) for i, e in ga_result.history],
		)


def create_optimizer(
	word_to_cluster: dict[str, int],
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int = 3000,
	n_context_words: int = 4,
	config: Optional[OptimizerConfig] = None,
	log_fn: Optional[Callable[[str], None]] = None,
) -> AcceleratedOptimizer:
	"""
	Factory function to create an AcceleratedOptimizer.

	This is the recommended way to create an optimizer - it handles
	all the setup including Rust accelerator detection.

	Args:
		word_to_cluster: Mapping from words to cluster IDs
		train_tokens: Training token sequence
		test_tokens: Test token sequence
		bits_per_neuron: Fan-in per neuron
		eval_subset: Number of test samples for evaluation
		n_context_words: N-gram context size
		config: Optimizer configuration (default: hybrid GA→TS)
		log_fn: Logging function (default: print)

	Returns:
		AcceleratedOptimizer ready to use
	"""
	context = EvaluationContext.from_word_clusters(
		word_to_cluster=word_to_cluster,
		train_tokens=train_tokens,
		test_tokens=test_tokens,
		bits_per_neuron=bits_per_neuron,
		eval_subset=eval_subset,
		n_context_words=n_context_words,
	)

	if config is None:
		config = OptimizerConfig()

	return AcceleratedOptimizer(config, context, log_fn)
