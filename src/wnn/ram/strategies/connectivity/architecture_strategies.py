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
	OptimizerResult,
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
	# Total input bits for connection initialization/mutation
	total_input_bits: Optional[int] = None


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
		"""
		Mutate architecture by adjusting bits/neurons for random clusters.

		Also adjusts connections when architecture changes:
		- Bits increase: Add random new connections
		- Bits decrease: Remove last connections
		- Neurons increase: Copy and mutate connections from existing neuron
		- Neurons decrease: Remove last neuron's connections
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Track old architecture for connection adjustment
		old_bits = genome.bits_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

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

		# Adjust connections if they exist
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation(
				genome, mutant, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant

	def _adjust_connections_for_mutation(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits: List[int],
		old_neurons: List[int],
		total_input_bits: int,
	) -> List[int]:
		"""Adjust connections when architecture changes during mutation."""
		result = []
		old_idx = 0

		for cluster_idx in range(len(new_genome.bits_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			o_bits = old_bits[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			n_bits = new_genome.bits_per_cluster[cluster_idx]

			for neuron_idx in range(n_neurons):
				if neuron_idx < o_neurons:
					# Existing neuron - copy and adjust connections
					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							# Copy existing connection, with small random mutation
							conn_idx = old_idx + neuron_idx * o_bits + bit_idx
							old_conn = old_genome.connections[conn_idx]
							# 10% chance of small perturbation
							if self._rng.random() < 0.1:
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
							else:
								new_conn = old_conn
							result.append(new_conn)
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy connections from random existing neuron with mutations
					if o_neurons > 0:
						template_neuron = self._rng.randint(0, o_neurons - 1)
						for bit_idx in range(n_bits):
							if bit_idx < o_bits:
								# Copy from template with mutation
								conn_idx = old_idx + template_neuron * o_bits + bit_idx
								old_conn = old_genome.connections[conn_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						# No existing neurons to copy from - fully random
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

			# Update old_idx for next cluster
			old_idx += o_neurons * o_bits

		return result

	def crossover_genomes(self, parent1: 'ClusterGenome', parent2: 'ClusterGenome') -> 'ClusterGenome':
		"""
		Single-point crossover at cluster boundary.

		Connections are inherited from the parent whose cluster config is taken.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		n = len(parent1.bits_per_cluster)
		crossover_point = self._rng.randint(1, n - 1)

		# Build child architecture
		child_bits = parent1.bits_per_cluster[:crossover_point] + parent2.bits_per_cluster[crossover_point:]
		child_neurons = parent1.neurons_per_cluster[:crossover_point] + parent2.neurons_per_cluster[crossover_point:]

		# Build child connections if parents have them
		child_connections = None
		if parent1.connections is not None and parent2.connections is not None:
			child_connections = []
			p1_idx = 0
			p2_idx = 0
			for i in range(n):
				p1_conn_size = parent1.neurons_per_cluster[i] * parent1.bits_per_cluster[i]
				p2_conn_size = parent2.neurons_per_cluster[i] * parent2.bits_per_cluster[i]

				if i < crossover_point:
					# Take from parent1
					child_connections.extend(parent1.connections[p1_idx:p1_idx + p1_conn_size])
				else:
					# Take from parent2
					child_connections.extend(parent2.connections[p2_idx:p2_idx + p2_conn_size])

				p1_idx += p1_conn_size
				p2_idx += p2_conn_size

		return ClusterGenome(
			bits_per_cluster=child_bits,
			neurons_per_cluster=child_neurons,
			connections=child_connections,
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

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			connections = []
			for i in range(cfg.num_clusters):
				for _ in range(neurons[i]):
					for _ in range(bits[i]):
						connections.append(self._rng.randint(0, cfg.total_input_bits - 1))

		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons, connections=connections)

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

		# Initialize connections if total_input_bits available
		connections = None
		if cfg.total_input_bits is not None:
			connections = []
			for i in range(cfg.num_clusters):
				for _ in range(neurons[i]):
					for _ in range(bits[i]):
						connections.append(self._rng.randint(0, cfg.total_input_bits - 1))

		return ClusterGenome(bits_per_cluster=bits, neurons_per_cluster=neurons, connections=connections)

	def optimize(
		self,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[List['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[List['ClusterGenome']], List[Tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA with Rust batch evaluation and ProgressTracker logging.

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return List[(CE, accuracy)] tuples.
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

		Also adjusts connections when architecture changes:
		- Bits change: adjust connection count for that cluster
		- Neurons change: adjust neuron count for that cluster

		Returns (new_genome, move_info) where move_info = (cluster_idx, field, old_val, new_val)
		Note: mutation_rate is ignored for TS (always mutates one cluster)
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Pick a random cluster to modify
		cluster_idx = self._rng.randint(0, cfg.num_clusters - 1)

		# Track old values for connection adjustment
		old_bits = genome.bits_per_cluster[cluster_idx]
		old_neurons = genome.neurons_per_cluster[cluster_idx]

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

		# Adjust connections if they exist
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_cluster_connections(
				genome, mutant, cluster_idx, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant, move

	def _adjust_cluster_connections(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		cluster_idx: int,
		old_bits: int,
		old_neurons: int,
		total_input_bits: int,
	) -> List[int]:
		"""Adjust connections for a single cluster that changed."""
		# Copy all connections, then adjust the changed cluster
		result = old_genome.connections.copy()

		n_neurons = new_genome.neurons_per_cluster[cluster_idx]
		n_bits = new_genome.bits_per_cluster[cluster_idx]

		# Calculate offset into connections for this cluster
		conn_offset = 0
		for i in range(cluster_idx):
			conn_offset += old_genome.neurons_per_cluster[i] * old_genome.bits_per_cluster[i]

		old_conn_size = old_neurons * old_bits
		new_conn_size = n_neurons * n_bits

		# Build new connections for this cluster
		new_cluster_conns = []
		old_cluster_conns = result[conn_offset:conn_offset + old_conn_size]

		for neuron_idx in range(n_neurons):
			if neuron_idx < old_neurons:
				# Existing neuron - copy and adjust
				for bit_idx in range(n_bits):
					if bit_idx < old_bits:
						# Copy existing connection with small mutation
						old_conn = old_cluster_conns[neuron_idx * old_bits + bit_idx]
						if self._rng.random() < 0.1:
							delta = self._rng.choice([-2, -1, 1, 2])
							new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
						else:
							new_conn = old_conn
						new_cluster_conns.append(new_conn)
					else:
						# New bit position
						new_cluster_conns.append(self._rng.randint(0, total_input_bits - 1))
			else:
				# New neuron - copy from random existing with mutation
				if old_neurons > 0:
					template = self._rng.randint(0, old_neurons - 1)
					for bit_idx in range(n_bits):
						if bit_idx < old_bits:
							old_conn = old_cluster_conns[template * old_bits + bit_idx]
							delta = self._rng.choice([-2, -1, 1, 2])
							new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
							new_cluster_conns.append(new_conn)
						else:
							new_cluster_conns.append(self._rng.randint(0, total_input_bits - 1))
				else:
					for _ in range(n_bits):
						new_cluster_conns.append(self._rng.randint(0, total_input_bits - 1))

		# Replace the cluster's connections in the result
		result = result[:conn_offset] + new_cluster_conns + result[conn_offset + old_conn_size:]
		return result

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
		batch_evaluate_fn: Optional[Callable[[List['ClusterGenome']], List[Tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS with Rust batch evaluation and ProgressTracker logging.

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return List[(CE, accuracy)] tuples.
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
		)
