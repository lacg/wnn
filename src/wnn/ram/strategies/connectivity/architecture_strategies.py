"""
Architecture optimization strategies using generic GA/TS base classes.

These implement ClusterGenome-specific operations while reusing the core
GA/TS algorithms from generic_strategies.py.

Features:
- Rust/Metal batch evaluation support for parallel genome evaluation
- ProgressTracker integration for consistent logging
- Population seeding between phases (GA → TS → GA → ...)
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

from wnn.progress import ProgressTracker
from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy,
	GenericTSStrategy,
	GAConfig,
	TSConfig,
	GenericOptResult,
)

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		RustParallelEvaluator,
		AdaptiveClusterConfig,
	)


@dataclass
class ArchitectureConfig:
	"""Configuration for architecture space bounds."""
	num_clusters: int
	min_bits: int = 4
	max_bits: int = 20
	min_neurons: int = 1
	max_neurons: int = 15
	# Phase: 1 = bits only, 2 = bits + neurons
	phase: int = 2
	# Token frequencies for frequency-scaled initialization
	token_frequencies: Optional[List[int]] = None


class ArchitectureGAStrategy(GenericGAStrategy['ClusterGenome']):
	"""
	Genetic Algorithm for architecture (bits, neurons per cluster) optimization.

	Inherits core GA loop from GenericGAStrategy, implements ClusterGenome operations.

	Features:
	- Rust/Metal batch evaluation (default when available)
	- ProgressTracker for consistent logging with accuracy
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
	):
		super().__init__(config=ga_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		self._tracker: Optional[ProgressTracker] = None

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""Mutate architecture by adjusting bits/neurons for random clusters."""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		for i in range(cfg.num_clusters):
			if self._rng.random() < mutation_rate:
				# Mutate bits
				delta = self._rng.choice([-2, -1, 1, 2])
				new_bits = mutant.bits_per_cluster[i] + delta
				mutant.bits_per_cluster[i] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				# Mutate neurons (if phase 2)
				if cfg.phase >= 2:
					delta = self._rng.choice([-2, -1, 1, 2])
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		return mutant

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""Single-point crossover at cluster boundary."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		n = len(parent1.bits_per_cluster)
		crossover_point = self._rng.randint(1, n - 1)

		return ClusterGenome(
			bits_per_cluster=parent1.bits_per_cluster[:crossover_point] + parent2.bits_per_cluster[crossover_point:],
			neurons_per_cluster=parent1.neurons_per_cluster[:crossover_point] + parent2.neurons_per_cluster[crossover_point:],
		)

	def create_random_genome(self) -> 'ClusterGenome':
		"""Create a random genome (frequency-scaled if frequencies available)."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		cfg = self._arch_config

		if cfg.token_frequencies is not None:
			return self._create_frequency_scaled_genome()

		# Uniform random
		bits = [self._rng.randint(cfg.min_bits, cfg.max_bits) for _ in range(cfg.num_clusters)]
		neurons = [self._rng.randint(cfg.min_neurons, cfg.max_neurons) for _ in range(cfg.num_clusters)]
		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons)

	def _create_frequency_scaled_genome(self) -> 'ClusterGenome':
		"""Create genome with bits/neurons scaled by token frequency."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._arch_config
		freqs = cfg.token_frequencies

		# Normalize frequencies to [0, 1]
		max_freq = max(freqs) if freqs else 1
		norm_freqs = [f / max_freq if max_freq > 0 else 0 for f in freqs]

		bits = []
		neurons = []
		for nf in norm_freqs:
			# Higher frequency -> more bits (can fill larger address space)
			b = int(cfg.min_bits + nf * (cfg.max_bits - cfg.min_bits))
			# Higher frequency -> more neurons (more capacity)
			n = int(cfg.min_neurons + nf * (cfg.max_neurons - cfg.min_neurons))
			bits.append(max(cfg.min_bits, min(cfg.max_bits, b)))
			neurons.append(max(cfg.min_neurons, min(cfg.max_neurons, n)))

		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons)

	def optimize(
		self,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[List['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[List['ClusterGenome']], List[float]]] = None,
		accuracy_fn: Optional[Callable[['ClusterGenome'], float]] = None,
	) -> GenericOptResult['ClusterGenome']:
		"""
		Run GA with Rust batch evaluation and ProgressTracker logging.

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		"""
		# Use Rust batch evaluator if available
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log,
			)

		# Create progress tracker
		self._tracker = ProgressTracker(
			logger=self._log,
			minimize=True,
			prefix=f"[{self.name}]",
			total_generations=self._config.generations,
		)

		return super().optimize(
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_population=initial_population,
			batch_evaluate_fn=batch_evaluate_fn,
			accuracy_fn=accuracy_fn,
		)


class ArchitectureTSStrategy(GenericTSStrategy['ClusterGenome']):
	"""
	Tabu Search for architecture (bits, neurons per cluster) optimization.

	Inherits core TS loop from GenericTSStrategy, implements ClusterGenome operations.

	Features:
	- Rust/Metal batch evaluation (default when available)
	- ProgressTracker for consistent logging with accuracy
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
	):
		super().__init__(config=ts_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		self._tracker: Optional[ProgressTracker] = None

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> Tuple['ClusterGenome', Any]:
		"""
		Generate a neighbor by modifying one cluster's configuration.

		Returns (new_genome, move_info) where move_info = (cluster_idx, field, old_val, new_val)
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Pick a random cluster to modify
		cluster_idx = self._rng.randint(0, cfg.num_clusters - 1)

		# Pick what to modify (bits or neurons)
		if cfg.phase == 1 or self._rng.random() < 0.5:
			# Modify bits
			old_val = mutant.bits_per_cluster[cluster_idx]
			delta = self._rng.choice([-2, -1, 1, 2])
			new_val = max(cfg.min_bits, min(cfg.max_bits, old_val + delta))
			mutant.bits_per_cluster[cluster_idx] = new_val
			move = (cluster_idx, 'bits', old_val, new_val)
		else:
			# Modify neurons
			old_val = mutant.neurons_per_cluster[cluster_idx]
			delta = self._rng.choice([-2, -1, 1, 2])
			new_val = max(cfg.min_neurons, min(cfg.max_neurons, old_val + delta))
			mutant.neurons_per_cluster[cluster_idx] = new_val
			move = (cluster_idx, 'neurons', old_val, new_val)

		return mutant, move

	def is_tabu_move(self, move: Any, tabu_list: List[Any]) -> bool:
		"""Check if move reverses a recent tabu move."""
		if move is None:
			return False

		cluster_idx, field, old_val, new_val = move

		# A move is tabu if it reverses a previous move
		for tabu_move in tabu_list:
			t_cluster, t_field, t_old, t_new = tabu_move
			if cluster_idx == t_cluster and field == t_field:
				if new_val == t_old and old_val == t_new:
					return True

		return False

	def optimize(
		self,
		initial_genome: 'ClusterGenome',
		initial_fitness: float,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_neighbors: Optional[List['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[List['ClusterGenome']], List[float]]] = None,
		accuracy_fn: Optional[Callable[['ClusterGenome'], float]] = None,
	) -> GenericOptResult['ClusterGenome']:
		"""
		Run TS with Rust batch evaluation and ProgressTracker logging.

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		"""
		# Use Rust batch evaluator if available
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log,
			)

		# Create progress tracker
		self._tracker = ProgressTracker(
			logger=self._log,
			minimize=True,
			prefix=f"[{self.name}]",
			total_generations=self._config.iterations,
		)

		return super().optimize(
			initial_genome=initial_genome,
			initial_fitness=initial_fitness,
			evaluate_fn=evaluate_fn,
			initial_neighbors=initial_neighbors,
			batch_evaluate_fn=batch_evaluate_fn,
			accuracy_fn=accuracy_fn,
		)
