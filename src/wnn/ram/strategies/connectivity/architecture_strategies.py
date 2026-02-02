"""
Architecture optimization strategies using generic GA/TS base classes.

These implement ClusterGenome-specific operations while reusing the core
GA/TS algorithms from generic_strategies.py.

Features:
- Rust/Metal batch evaluation support for parallel genome evaluation
- Population seeding between phases (GA → TS → GA → ...)
- Checkpoint/resume support for long optimization runs
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from wnn.ram.strategies.connectivity.generic_strategies import (
	GenericGAStrategy,
	GenericTSStrategy,
	GAConfig,
	TSConfig,
	OptimizerResult,
	StopReason,
)
from wnn.ram.fitness import FitnessCalculatorFactory, FitnessCalculatorType
from wnn.ram.architecture.genome_log import (
	GenomeLogType,
	format_genome_log,
	format_gen_prefix,
)
from wnn.ram.strategies.filters import PercentileFilter, FilterMode

# Optional tracker integration for genome tracking
try:
	from wnn.ram.experiments.tracker import TierConfig, GenomeConfig, GenomeRole
	HAS_GENOME_TRACKING = True
except ImportError:
	HAS_GENOME_TRACKING = False
	TierConfig = None
	GenomeConfig = None
	GenomeRole = None

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		RustParallelEvaluator,
		AdaptiveClusterConfig,
	)


# =============================================================================
# Shared Mixin for Architecture Strategies
# =============================================================================

class ArchitectureStrategyMixin:
	"""
	Mixin providing common functionality for GA and TS architecture strategies.

	Reduces code duplication by extracting:
	- Metal cleanup logic
	- Shutdown checking
	- genome_to_config conversion
	- Result building with stop_reason
	"""

	_shutdown_check: Optional[Callable[[], bool]]
	_log: Any  # Logger

	def _cleanup_metal(self, iteration: int, log_interval: int = 10) -> None:
		"""Run GC and reset Metal evaluators to prevent buffer accumulation."""
		import gc
		gc.collect()
		try:
			import ram_accelerator
			ram_accelerator.reset_metal_evaluators()
			if iteration % log_interval == 0:
				self._log.info(f"[{self.name}] GC + Metal reset at iteration {iteration}")
		except Exception:
			pass  # Ignore if accelerator not available

	def _check_and_set_shutdown(self, shutdown_flag: list[bool]) -> bool:
		"""
		Check if shutdown is requested and update the shutdown flag.

		Args:
			shutdown_flag: List with single bool element (mutable reference)

		Returns:
			True if shutdown was requested
		"""
		if self._shutdown_check and self._shutdown_check():
			shutdown_flag[0] = True
			return True
		return False

	def _genome_to_config_impl(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""
		Convert a ClusterGenome to a GenomeConfig for tracking.

		Finds contiguous runs of clusters with the same (neurons, bits) config.
		This enables proper tracking of which cluster indices belong to which tier.

		The tier index is assigned based on the order of first appearance
		(earliest cluster index = tier 0), preserving the original tier config order.
		"""
		if not HAS_GENOME_TRACKING or GenomeConfig is None or TierConfig is None:
			return None

		# Find contiguous runs of clusters with same (neurons, bits)
		runs: list[tuple[int, int, int, int]] = []  # (start, end, neurons, bits)
		if len(genome.neurons_per_cluster) == 0:
			return GenomeConfig(tiers=[])

		current_neurons = genome.neurons_per_cluster[0]
		current_bits = genome.bits_per_cluster[0]
		run_start = 0

		for i in range(1, len(genome.neurons_per_cluster)):
			neurons = genome.neurons_per_cluster[i]
			bits = genome.bits_per_cluster[i]
			if neurons != current_neurons or bits != current_bits:
				# End current run, start new one
				runs.append((run_start, i, current_neurons, current_bits))
				current_neurons = neurons
				current_bits = bits
				run_start = i

		# Don't forget the last run
		runs.append((run_start, len(genome.neurons_per_cluster), current_neurons, current_bits))

		# Assign tier indices based on first appearance of each (neurons, bits) config
		config_to_tier: dict[tuple[int, int], int] = {}
		next_tier = 0
		for _, _, neurons, bits in runs:
			key = (neurons, bits)
			if key not in config_to_tier:
				config_to_tier[key] = next_tier
				next_tier += 1

		# Create TierConfig for each contiguous run
		tiers = []
		for start, end, neurons, bits in runs:
			tier_idx = config_to_tier[(neurons, bits)]
			tiers.append(TierConfig(
				tier=tier_idx,
				clusters=end - start,
				neurons=neurons,
				bits=bits,
				start_cluster=start,
				end_cluster=end,
			))

		return GenomeConfig(tiers=tiers)

	def _determine_stop_reason(
		self,
		shutdown_requested: bool,
		early_stopper: Any,
	) -> Optional[StopReason]:
		"""Determine the stop reason based on shutdown flag and early stopper state."""
		if shutdown_requested:
			return StopReason.SHUTDOWN
		elif hasattr(early_stopper, 'patience_exhausted') and early_stopper.patience_exhausted:
			return StopReason.CONVERGENCE
		return None


# =============================================================================
# Checkpoint System for Resume Support
# =============================================================================

@dataclass
class CheckpointConfig:
	"""Configuration for checkpoint saving."""
	enabled: bool = True
	interval: int = 50                       # Save every N iterations
	checkpoint_dir: Optional[Path] = None    # Directory for checkpoint files
	filename_prefix: str = "checkpoint"      # Prefix for checkpoint filenames


class CheckpointManager:
	"""
	Reusable checkpoint manager for optimization runs.

	Usage:
		# Create manager
		manager = CheckpointManager(
			config=CheckpointConfig(checkpoint_dir=Path("checkpoints")),
			phase_name="Phase 1a: GA Neurons",
			optimizer_type="GA",
			total_iterations=1000,
			logger=print,
		)

		# In optimization loop:
		for iteration in range(1000):
			# ... do optimization ...

			# Save checkpoint every N iterations
			manager.maybe_save(
				iteration=iteration,
				population=population,
				best_genome=best_genome,
				best_fitness=(ce, acc),
				current_threshold=threshold,
				extra_state={"patience": patience_counter},
			)

		# To resume:
		if manager.has_checkpoint():
			state = manager.load()
			start_iteration = state['current_iteration'] + 1
			population = state['population']
	"""

	def __init__(
		self,
		config: CheckpointConfig,
		phase_name: str,
		optimizer_type: str,
		total_iterations: int,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._config = config
		self._phase_name = phase_name
		self._optimizer_type = optimizer_type
		self._total_iterations = total_iterations
		self._logger = logger or (lambda x: None)

		# Create checkpoint directory if needed
		if config.enabled and config.checkpoint_dir:
			config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	@property
	def checkpoint_path(self) -> Optional[Path]:
		"""Path to the checkpoint file."""
		if not self._config.enabled or not self._config.checkpoint_dir:
			return None
		return self._config.checkpoint_dir / f"{self._config.filename_prefix}_{self._optimizer_type.lower()}.json"

	def has_checkpoint(self) -> bool:
		"""Check if a checkpoint file exists."""
		path = self.checkpoint_path
		return path is not None and path.exists()

	def should_save(self, iteration: int) -> bool:
		"""Check if we should save at this iteration."""
		if not self._config.enabled:
			return False
		# Save at interval (1-indexed), but also at iteration 0 for safety
		return iteration > 0 and (iteration + 1) % self._config.interval == 0

	def maybe_save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> bool:
		"""
		Save checkpoint if at the right interval.

		Args:
			iteration: Current iteration (0-indexed)
			population: List of (genome, ce_fitness) tuples
			best_genome: Best genome found so far
			best_fitness: (CE, accuracy) of best genome
			current_threshold: Current threshold value
			config_dict: Optional config as dict
			extra_state: Optional extra state to save (patience, baseline, etc.)

		Returns:
			True if checkpoint was saved, False otherwise
		"""
		if not self.should_save(iteration):
			return False

		self.save(
			iteration=iteration,
			population=population,
			best_genome=best_genome,
			best_fitness=best_fitness,
			current_threshold=current_threshold,
			config_dict=config_dict,
			extra_state=extra_state,
		)
		return True

	def save(
		self,
		iteration: int,
		population: list[tuple['ClusterGenome', float]],
		best_genome: 'ClusterGenome',
		best_fitness: tuple[float, float],
		current_threshold: float,
		config_dict: Optional[dict] = None,
		extra_state: Optional[dict] = None,
	) -> None:
		"""Save checkpoint now (regardless of interval)."""
		import datetime

		path = self.checkpoint_path
		if path is None:
			return

		# Serialize population
		pop_data = []
		for genome, ce in population:
			gd = self._genome_to_dict(genome)
			# Try to get accuracy from cached fitness
			if hasattr(genome, '_cached_fitness') and genome._cached_fitness:
				gd['fitness'] = list(genome._cached_fitness)
			else:
				gd['fitness'] = [ce, 0.0]
			pop_data.append(gd)

		# Build checkpoint data
		data = {
			'phase_name': self._phase_name,
			'optimizer_type': self._optimizer_type,
			'current_iteration': iteration,
			'total_iterations': self._total_iterations,
			'population': pop_data,
			'best_genome': self._genome_to_dict(best_genome),
			'best_fitness': list(best_fitness),
			'current_threshold': current_threshold,
			'config': config_dict or {},
			'extra_state': extra_state or {},
			'saved_at': datetime.datetime.now().isoformat(),
		}

		# Write atomically (temp file + rename)
		temp_path = path.with_suffix('.tmp')
		with open(temp_path, 'w') as f:
			json.dump(data, f, indent=2)
		temp_path.rename(path)

		self._logger(f"[Checkpoint] Saved at iteration {iteration + 1}/{self._total_iterations}")

	def load(self, genome_class: type) -> dict:
		"""
		Load checkpoint from file.

		Args:
			genome_class: The ClusterGenome class to use for reconstruction

		Returns:
			Dict with:
				- current_iteration: int
				- population: list of (genome, ce) tuples
				- best_genome: ClusterGenome
				- best_fitness: (CE, accuracy)
				- current_threshold: float
				- config: dict
				- extra_state: dict
		"""
		path = self.checkpoint_path
		if path is None or not path.exists():
			raise FileNotFoundError(f"No checkpoint found at {path}")

		with open(path, 'r') as f:
			data = json.load(f)

		# Reconstruct population
		population = []
		for gd in data['population']:
			genome = self._dict_to_genome(gd, genome_class)
			ce = gd['fitness'][0] if gd.get('fitness') else 0.0
			# Restore cached fitness if available
			if gd.get('fitness'):
				genome._cached_fitness = tuple(gd['fitness'])
			population.append((genome, ce))

		# Reconstruct best genome
		best_genome = self._dict_to_genome(data['best_genome'], genome_class)

		self._logger(f"[Checkpoint] Loaded from iteration {data['current_iteration'] + 1}")

		return {
			'current_iteration': data['current_iteration'],
			'population': population,
			'best_genome': best_genome,
			'best_fitness': tuple(data['best_fitness']),
			'current_threshold': data['current_threshold'],
			'config': data.get('config', {}),
			'extra_state': data.get('extra_state', {}),
			'saved_at': data.get('saved_at', ''),
		}

	@staticmethod
	def _genome_to_dict(genome: 'ClusterGenome') -> dict:
		"""Convert a ClusterGenome to a serializable dict."""
		return {
			'bits_per_cluster': list(genome.bits_per_cluster),
			'neurons_per_cluster': list(genome.neurons_per_cluster),
			'connections': list(genome.connections) if genome.connections else None,
		}

	@staticmethod
	def _dict_to_genome(d: dict, genome_class: type) -> 'ClusterGenome':
		"""Convert a dict back to a ClusterGenome."""
		return genome_class(
			bits_per_cluster=d['bits_per_cluster'],
			neurons_per_cluster=d['neurons_per_cluster'],
			connections=d.get('connections'),
		)


@dataclass
class ArchitectureConfig:
	"""
	Configuration for architecture optimization.

	Controls both the search space bounds and what gets optimized.
	The optimizer is phase-agnostic - callers control what to optimize
	by setting the optimize_* flags.

	Example usage:
		# Phase 1: Optimize neurons only (bits fixed at default_bits)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=True,
			default_bits=8,  # All genomes start with 8 bits
		)

		# Phase 2: Optimize bits only (pass seed genome from Phase 1)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=True,
			optimize_neurons=False,
		)

		# Phase 3: Optimize connections only (pass seed genome from Phase 2)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
		)
	"""
	num_clusters: int
	min_bits: int = 4
	max_bits: int = 24
	min_neurons: int = 3
	max_neurons: int = 30
	# Explicit control over what gets optimized (no magic phase numbers)
	optimize_bits: bool = True
	optimize_neurons: bool = True
	optimize_connections: bool = False
	# Default values for dimensions not being optimized (used in random genome init)
	default_bits: int = 8
	default_neurons: int = 5
	# Token frequencies for frequency-scaled initialization
	token_frequencies: Optional[list[int]] = None
	# Total input bits for connection initialization/mutation
	total_input_bits: Optional[int] = None
	# Tier0-only: only mutate first N clusters (None = all clusters mutable)
	mutable_clusters: Optional[int] = None


class ArchitectureGAStrategy(ArchitectureStrategyMixin, GenericGAStrategy['ClusterGenome']):
	"""
	Genetic Algorithm for architecture (bits, neurons per cluster) optimization.

	Inherits core GA loop from GenericGAStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based offspring search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	- Checkpoint/resume support for long runs
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ga_config: Optional[GAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_offspring
		checkpoint_config: Optional[CheckpointConfig] = None,  # Checkpoint configuration
		phase_name: str = "GA Optimization",  # Phase name for checkpoints
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ga_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_offspring
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_offspring'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._checkpoint_config = checkpoint_config
		self._phase_name = phase_name
		self._shutdown_check = shutdown_check

	@property
	def name(self) -> str:
		return "ArchitectureGA"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> 'ClusterGenome':
		"""
		Mutate genome based on optimize_* flags in config.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token (stays close to token boundaries)

		Also adjusts connections when architecture changes.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))

		# Track old architecture for connection adjustment
		old_bits = genome.bits_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# If only optimizing connections, skip architecture mutation
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutant.connections = self._mutate_connections_only(
					genome.connections.copy(), cfg.total_input_bits, mutation_rate
				)
			return mutant

		# Mutate architecture (bits and/or neurons)
		# If mutable_clusters is set, only mutate first N clusters (tier0-only mode)
		max_mutable = cfg.mutable_clusters if cfg.mutable_clusters is not None else cfg.num_clusters
		for i in range(max_mutable):
			if self._rng.random() < mutation_rate:
				if cfg.optimize_bits:
					# Random delta in [-bits_delta_max, +bits_delta_max]
					delta = self._rng.randint(-bits_delta_max, bits_delta_max)
					new_bits = mutant.bits_per_cluster[i] + delta
					mutant.bits_per_cluster[i] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					# Random delta in [-neurons_delta_max, +neurons_delta_max]
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation(
				genome, mutant, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant

	def _mutate_connections_only(
		self,
		connections: list[int],
		total_input_bits: int,
		mutation_rate: float,
	) -> list[int]:
		"""
		Mutate connections without changing architecture.

		Delta range is 10% of bits_per_token (stays close to token boundaries).
		Assumes context × bits_per_token = total_input_bits.
		"""
		# Estimate bits_per_token: typically 16 for GPT-2 vocab
		# Use sqrt heuristic: bits_per_token ≈ log2(vocab) ≈ 16
		bits_per_token = 16  # Could be passed in, but 16 is reasonable default
		conn_delta_max = max(1, round(0.1 * bits_per_token))  # 10% of 16 = 2

		result = connections.copy()
		for i in range(len(result)):
			if self._rng.random() < mutation_rate:
				delta = self._rng.randint(-conn_delta_max, conn_delta_max)
				result[i] = max(0, min(total_input_bits - 1, result[i] + delta))
		return result

	def _adjust_connections_for_mutation(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
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
		"""
		Create a random genome based on optimize_* flags.

		- If optimize_bits=True: random bits in [min_bits, max_bits]
		- If optimize_bits=False: use default_bits for all clusters
		- Same logic for neurons

		When optimizing connections only, both bits and neurons use defaults.
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		self._ensure_rng()
		cfg = self._arch_config

		if cfg.token_frequencies is not None:
			return self._create_frequency_scaled_genome()

		# Initialize bits: random if optimizing, default otherwise
		if cfg.optimize_bits:
			bits = [self._rng.randint(cfg.min_bits, cfg.max_bits) for _ in range(cfg.num_clusters)]
		else:
			bits = [cfg.default_bits] * cfg.num_clusters

		# Initialize neurons: random if optimizing, default otherwise
		if cfg.optimize_neurons:
			neurons = [self._rng.randint(cfg.min_neurons, cfg.max_neurons) for _ in range(cfg.num_clusters)]
		else:
			neurons = [cfg.default_neurons] * cfg.num_clusters

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
		"""
		Create genome with bits/neurons scaled by token frequency.

		- If optimize_bits=True: scale bits by frequency
		- If optimize_bits=False: use default_bits
		- Same logic for neurons
		"""
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._arch_config
		freqs = cfg.token_frequencies

		# Normalize frequencies to [0, 1]
		max_freq = max(freqs) if freqs else 1
		norm_freqs = [f / max_freq if max_freq > 0 else 0 for f in freqs]

		bits = []
		neurons = []
		for nf in norm_freqs:
			# Bits: scaled if optimizing, default otherwise
			if cfg.optimize_bits:
				b = int(cfg.min_bits + nf * (cfg.max_bits - cfg.min_bits))
			else:
				b = cfg.default_bits

			# Neurons: scaled if optimizing, default otherwise
			if cfg.optimize_neurons:
				n = int(cfg.min_neurons + nf * (cfg.max_neurons - cfg.min_neurons))
			else:
				n = cfg.default_neurons

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
		initial_population: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA with Rust batch evaluation.

		If cached_evaluator was provided at init, uses Rust search_offspring for
		offspring generation (eliminates Python↔Rust round trips).

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return list[(CE, accuracy)] tuples.
		"""
		# If we have a cached_evaluator, use Rust-based offspring search
		if self._cached_evaluator is not None:
			return self._optimize_with_rust_search(
				initial_population=initial_population,
			)

		# Fall back to original behavior
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		return super().optimize(
			evaluate_fn=evaluate_fn,
			initial_genome=initial_genome,
			initial_population=initial_population,
			batch_evaluate_fn=batch_evaluate_fn,
		)

	def _optimize_with_rust_search(
		self,
		initial_population: Optional[list['ClusterGenome']] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run GA using Rust search_offspring for offspring generation.

		This eliminates Python↔Rust round trips by doing tournament selection,
		crossover, mutation, evaluation, and filtering entirely in Rust.
		"""
		import time
		from wnn.ram.strategies.connectivity.generic_strategies import (
			OptimizerResult, EarlyStoppingConfig, EarlyStoppingTracker
		)
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._config
		arch_cfg = self._arch_config
		evaluator = self._cached_evaluator

		# Initialize checkpoint manager if configured
		checkpoint_mgr: Optional[CheckpointManager] = None
		resume_state: Optional[dict] = None
		if self._checkpoint_config and self._checkpoint_config.enabled:
			checkpoint_mgr = CheckpointManager(
				config=self._checkpoint_config,
				phase_name=self._phase_name,
				optimizer_type="GA",
				total_iterations=cfg.generations,
				logger=self._log.info,
			)
			# Check for existing checkpoint to resume from
			if checkpoint_mgr.has_checkpoint():
				from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
				resume_state = checkpoint_mgr.load(ClusterGenome)
				self._log.info(f"[{self.name}] Resuming from checkpoint at generation {resume_state['current_iteration'] + 1}")

		# Create fitness calculator for ranking
		fitness_calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
			min_accuracy_floor=cfg.min_accuracy_floor if cfg.min_accuracy_floor > 0 else None,
		)
		self._log.info(f"[{self.name}] Fitness calculator: {fitness_calculator.name}")

		# Threshold continuity
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		# End threshold depends on actual generations vs reference (constant rate)
		actual_progress = min(1.0, cfg.generations / cfg.threshold_reference)
		end_threshold = start_threshold + actual_progress * cfg.threshold_delta

		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.4%} → {end_threshold:.4%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/gen)")
		self._log.info(f"[{self.name}] Using Rust search_offspring (single-call offspring search)")

		# Initialize population (show ALL genomes - no threshold filtering for init)
		# If fresh_population=True, ignore seeds and generate all random genomes
		if cfg.fresh_population:
			self._log.info(f"[{self.name}] Generating {cfg.population_size} fresh random genomes (fresh_population=True)")
			initial_population = None  # Force random generation below
		# If we have seed genomes but fewer than population_size, expand with mutations (unless seed_only)
		elif initial_population and len(initial_population) > 0:
			# CRITICAL: Ensure all seed genomes have connections (fixes reproducibility bug)
			# Without connections, Rust generates random ones each evaluation → inconsistent results
			for g in initial_population:
				if not g.has_connections():
					g.initialize_connections(evaluator.total_input_bits)
			seed_count = len(initial_population)
			need_count = cfg.population_size - seed_count
			if need_count > 0 and not cfg.seed_only:
				self._log.info(f"[{self.name}] Seeding from {seed_count} genomes, generating {need_count} mutations to fill population of {cfg.population_size}")
				# Generate mutations of seed genomes to fill population
				from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
				arch_cfg = self._arch_config
				# Create mutation config with phase-aware rates (only mutate what we're optimizing)
				# BUG FIX: Previously used hardcoded 0.3 for both, which corrupted tiered configs
				# during neurons-only phase by mutating bits too
				mutation_config = AdaptiveClusterConfig(
					min_bits=arch_cfg.min_bits,
					max_bits=arch_cfg.max_bits,
					min_neurons=arch_cfg.min_neurons,
					max_neurons=arch_cfg.max_neurons,
					bits_mutation_rate=0.3 if arch_cfg.optimize_bits else 0.0,
					neurons_mutation_rate=0.3 if arch_cfg.optimize_neurons else 0.0,
				)
				expanded_population = list(initial_population)
				for i in range(need_count):
					# Pick a seed genome to mutate (round-robin)
					seed = initial_population[i % seed_count]
					mutated = seed.mutate(
						config=mutation_config,
						total_input_bits=evaluator.total_input_bits,
					)
					expanded_population.append(mutated)
				initial_population = expanded_population
			else:
				self._log.info(f"[{self.name}] Seeding from {seed_count} genomes (using as-is, no mutation expansion)")
		else:
			self._log.info(f"[{self.name}] Initializing with {cfg.population_size} random genomes")

		# Get train subset for this PHASE (fixed for ALL evaluations: initial, generations, and elites)
		# This ensures all genomes are evaluated on the same data, making fitness values directly comparable.
		phase_train_idx = evaluator.next_train_idx()

		if initial_population and len(initial_population) > 0:
			# Evaluate initial population with streaming (Rust logs per-genome to WNN_LOG_PATH)
			import os
			stream_batch_size = int(os.environ.get('WNN_STREAM_BATCH_SIZE', '15'))
			results = evaluator.evaluate_batch(
				initial_population,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				logger=self._log,
				generation=0,
				total_generations=cfg.generations,
				min_accuracy=None,  # Show all for initial population
				streaming=True,
				stream_batch_size=stream_batch_size,
			)
			# Store cached fitness on each genome for elite logging
			population = []
			for g, (ce, acc) in zip(initial_population, results):
				g._cached_fitness = (ce, acc)
				population.append((g, ce))
		else:
			# Generate random initial population
			random_genomes = [self.create_random_genome() for _ in range(cfg.population_size)]
			results = evaluator.evaluate_batch(
				random_genomes,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				logger=self._log,
				generation=0,
				total_generations=cfg.generations,
				min_accuracy=None,  # Show all for initial population
				streaming=True,  # Rust logs per-genome to WNN_LOG_PATH
			)
			# Store cached fitness on each genome for elite logging
			population = []
			for g, (ce, acc) in zip(random_genomes, results):
				g._cached_fitness = (ce, acc)
				population.append((g, ce))

		# Rank population using fitness calculator
		# Convert to (genome, ce, acc) format for ranking
		pop_for_ranking = [(g, ce, g._cached_fitness[1]) for g, ce in population]
		ranked = fitness_calculator.rank(pop_for_ranking)
		# Convert back to (genome, ce) format
		population = [(g, g._cached_fitness[0]) for g, _ in ranked]

		# Track best (CE and accuracy)
		best_genome, best_fitness = population[0]
		best_acc = best_genome._cached_fitness[1] if hasattr(best_genome, '_cached_fitness') and best_genome._cached_fitness else 0.0
		initial_genome = best_genome.clone()
		initial_fitness = best_fitness

		# Early stopping with baseline-based overfitting detection
		# Compares top-K elites on FULL data vs baseline (init full evaluation)
		from wnn.ram.strategies.connectivity.generic_strategies import EarlyStoppingConfig, AdaptiveScaler
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stop = EarlyStoppingTracker(early_stop_config, self._log.debug, self.name)

		# Adaptive scaling for parameter adjustment based on health level
		adaptive_scaler = AdaptiveScaler(
			base_population=cfg.population_size,
			base_mutation=cfg.mutation_rate,
			name=self.name,
		)

		# Evaluate top-K elites on FULL data for baseline (used by OverfitDetector)
		elite_count_for_overfit = max(1, int(cfg.elitism_pct * 2 * cfg.population_size))
		elite_genomes_for_baseline = [g for g, _ in ranked[:elite_count_for_overfit]]
		baseline_results = evaluator.evaluate_batch_full(elite_genomes_for_baseline)
		baseline_fitness = [ce for ce, _ in baseline_results]
		early_stop.reset_baseline(baseline_fitness)
		self._log.info(f"[{self.name}] Baseline mean (top-{elite_count_for_overfit} on full data): {sum(baseline_fitness)/len(baseline_fitness):.4f}")

		# Track threshold for logging
		prev_threshold: Optional[float] = None
		seed_offset = int(time.time() * 1000) % (2**16)

		# Resume from checkpoint if available
		start_generation = 0
		if resume_state is not None:
			# Restore state from checkpoint
			population = resume_state['population']
			best_genome = resume_state['best_genome']
			best_fitness = resume_state['best_fitness'][0]  # CE
			best_acc = resume_state['best_fitness'][1]  # Accuracy
			start_generation = resume_state['current_iteration'] + 1
			prev_threshold = resume_state['current_threshold']

			# Restore early stopping state if available
			extra = resume_state.get('extra_state', {})
			if extra.get('baseline_mean') is not None:
				early_stop._overfit_detector.baseline_mean = extra['baseline_mean']
			if extra.get('patience_counter') is not None:
				early_stop._patience_counter = extra['patience_counter']

			self._log.info(f"[{self.name}] Resumed: best CE={best_fitness:.4f}, acc={best_acc:.4%}, population={len(population)}")

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		shutdown_requested = False  # Track shutdown for stop_reason
		for generation in range(start_generation, cfg.generations):
			# Check for shutdown request at start of each generation
			if self._shutdown_check and self._shutdown_check():
				self._log.info(f"[{self.name}] Shutdown requested at generation {generation}, stopping...")
				# Save checkpoint before exiting
				if checkpoint_mgr and checkpoint_mgr.config.enabled:
					checkpoint_mgr.maybe_save(
						iteration=generation,
						population=population,
						best_genome=best,
						best_fitness=(best_fitness, best_acc),
						current_threshold=current_threshold,
						extra_state={"patience_counter": early_stop._patience_counter},
						force=True,
					)
				shutdown_requested = True
				break

			gen_start = time.time()

			# Cleanup to prevent Metal buffer accumulation
			if generation > 0:
				self._cleanup_metal(generation, log_interval=10)

			# Save checkpoint every 50 generations
			if generation > 0 and generation % 50 == 0 and checkpoint_mgr is not None:
				checkpoint_mgr.maybe_save(
					iteration=generation,
					population=population,
					best_genome=best_genome,
					best_fitness=(best_fitness, best_acc),
					current_threshold=current_threshold if prev_threshold else start_threshold,
					extra_state={
						'patience_counter': early_stop._patience_counter if hasattr(early_stop, '_patience_counter') else 0,
						'baseline_mean': early_stop._overfit_detector.baseline_mean if hasattr(early_stop, '_overfit_detector') and early_stop._overfit_detector else None,
					},
				)

			current_threshold = get_threshold(generation / cfg.threshold_reference)
			# Only log if formatted values differ
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# Elite selection: Always use top-k by fitness (unified approach)
			# For CE mode: fitness = CE, so top-k = best CE
			# For weighted modes: fitness = combined score (lower is better)
			gen_prefix = format_gen_prefix(generation + 1, cfg.generations)

			# Single elite pool by fitness score (20% total)
			elite_count = max(1, int(cfg.elitism_pct * 2 * cfg.population_size))

			# Population is already ranked by fitness, take top N
			elites = population[:elite_count]

			# Log elites
			self._log.info("=" * 60)
			self._log.info(f"{gen_prefix} Elites: {elite_count} by {fitness_calculator.name}")
			self._log.info("=" * 60)
			for i, (g, ce) in enumerate(elites):
				acc = g._cached_fitness[1] if hasattr(g, '_cached_fitness') and g._cached_fitness else 0.0
				self._log.info(format_genome_log(
					generation + 1, cfg.generations, GenomeLogType.ELITE_CE,
					i + 1, elite_count, ce, acc
				))

			# Generate offspring using Rust (return_best_n=True for soft threshold)
			# Phase-aware mutation rates: only mutate the dimension being optimized
			# During neurons phase: bits_mutation_rate=0.0, neurons_mutation_rate=cfg.mutation_rate
			# During bits phase: bits_mutation_rate=cfg.mutation_rate, neurons_mutation_rate=0.0
			bits_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_bits else 0.0
			neurons_mutation_rate = cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0

			needed_offspring = cfg.population_size - elite_count
			search_result = evaluator.search_offspring(
				population=population,
				target_count=needed_offspring,
				max_attempts=needed_offspring * 5,  # 5x cap
				accuracy_threshold=current_threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=bits_mutation_rate,
				neurons_mutation_rate=neurons_mutation_rate,
				crossover_rate=cfg.crossover_rate,
				tournament_size=cfg.tournament_size,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				seed=seed_offset + generation,
				generation=generation,
				total_generations=cfg.generations,
				return_best_n=True,  # Soft threshold: return top N by CE if not enough pass
				mutable_clusters=arch_cfg.mutable_clusters,  # Tier0-only: only mutate first N clusters
			)
			# search_result contains: genomes, evaluated (total tested), viable (passed threshold)

			# Convert offspring to (genome, ce) tuples for filtering
			offspring_with_ce = []
			for g in search_result.genomes:
				if hasattr(g, '_cached_fitness'):
					ce, acc = g._cached_fitness
					offspring_with_ce.append((g, ce))

			# Apply percentile filter if configured (uses fitness calculator for ranking)
			# Note: 0 means disabled, same as None
			if cfg.fitness_percentile and cfg.fitness_percentile > 0 and offspring_with_ce:
				if cfg.fitness_calculator_type == FitnessCalculatorType.CE:
					# CE mode: Filter by CE only
					ce_filter = PercentileFilter(
						percentile=cfg.fitness_percentile,
						mode=FilterMode.LOWER_IS_BETTER,
						metric_name="CE",
					)
					filter_result = ce_filter.apply(offspring_with_ce, key=lambda g, f: f)
					offspring_with_ce = filter_result.kept
				else:
					# HARMONIC_RANK or NORMALIZED: Filter by fitness calculator score
					# This ensures low-accuracy genomes are filtered even if they have good CE
					pop_for_filter = [(g, ce, g._cached_fitness[1]) for g, ce in offspring_with_ce]
					fitness_scores = fitness_calculator.fitness(pop_for_filter)
					offspring_with_fitness = [(g, ce, score) for (g, ce), score in zip(offspring_with_ce, fitness_scores)]

					fitness_filter = PercentileFilter(
						percentile=cfg.fitness_percentile,
						mode=FilterMode.LOWER_IS_BETTER,
						metric_name=fitness_calculator.name,
					)
					# Filter expects 2-tuples: (genome, metric_value)
					# We keep a mapping to recover CE after filtering
					genome_ce_map = {id(g): ce for g, ce, _ in offspring_with_fitness}
					filter_population = [(g, score) for g, _, score in offspring_with_fitness]
					filter_result = fitness_filter.apply(filter_population, key=lambda g, f: f)
					offspring_with_ce = [(g, genome_ce_map[id(g)]) for g, _ in filter_result.kept]

				if filter_result.filtered:
					self._log.debug(
						f"[{self.name}] {filter_result.metric_name} filter: kept {filter_result.kept_count}/{filter_result.total_count} "
						f"(threshold={filter_result.threshold_value:.4f})"
					)

			# Build new population: elites + filtered offspring
			new_population = list(elites)
			new_population.extend(offspring_with_ce)

			# Rank using fitness calculator, then truncate
			pop_for_ranking = [(g, ce, g._cached_fitness[1]) for g, ce in new_population]
			ranked = fitness_calculator.rank(pop_for_ranking)
			population = [(g, g._cached_fitness[0]) for g, _ in ranked[:cfg.population_size]]

			# Update best (best by fitness ranking, get CE from cached fitness)
			best_genome_candidate = population[0][0]
			best_ce_candidate = best_genome_candidate._cached_fitness[0]
			if best_fitness is None or best_ce_candidate < best_fitness:
				best_genome = best_genome_candidate
				best_fitness = best_ce_candidate
				best_acc = best_genome._cached_fitness[1] if hasattr(best_genome, '_cached_fitness') and best_genome._cached_fitness else 0.0

			# Log generation summary with duration
			# Note: best/avg are from subset evaluation, not full validation
			gen_elapsed = time.time() - gen_start

			# V2 tracking: record iteration data (uses base class _tracker set via set_tracker)
			# All genomes use the same phase_train_idx, so their _cached_fitness values are comparable
			valid_data = [(g._cached_fitness[0], g._cached_fitness[1])
			              for g, _ in population
			              if hasattr(g, '_cached_fitness') and g._cached_fitness]
			if valid_data:
				avg_fitness = sum(ce for ce, _ in valid_data) / len(valid_data)
				avg_acc = sum(acc for _, acc in valid_data) / len(valid_data)
				actual_best_ce = min(ce for ce, _ in valid_data)
				actual_best_acc = max(acc for _, acc in valid_data)
			else:
				# Fallback: use CE from population tuple if no cached fitness
				avg_fitness = sum(ce for _, ce in population) / len(population)
				avg_acc = None
				actual_best_ce = best_fitness
				actual_best_acc = best_acc

			self._log.info(f"[{self.name}] Gen {generation+1:03d}/{cfg.generations}: "
						   f"best={best_fitness:.4f}, avg={avg_fitness:.4f} (subset) ({gen_elapsed:.1f}s)")

			if self._tracker and self._tracker_phase_id:
				try:

					# Get baseline and patience info for dashboard
					baseline_ce = None
					if hasattr(early_stop, '_overfit_detector') and early_stop._overfit_detector:
						baseline_ce = early_stop._overfit_detector.baseline_mean
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stop._patience_counter if hasattr(early_stop, '_patience_counter') else 0
					# Use actual counts from Rust: evaluated = total tested, viable = passed threshold
					candidates_total = search_result.evaluated

					iteration_id = self._tracker.record_iteration(
						phase_id=self._tracker_phase_id,
						iteration_num=generation + 1,
						best_ce=actual_best_ce,  # Use actual min CE, not fitness-ranked best
						best_accuracy=actual_best_acc,
						avg_ce=avg_fitness,
						avg_accuracy=avg_acc,
						elite_count=elite_count,
						offspring_count=search_result.evaluated,  # Total candidates tested
						offspring_viable=search_result.viable,    # Passed accuracy threshold
						fitness_threshold=current_threshold,
						elapsed_secs=gen_elapsed,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=candidates_total,
					)

					# Record genome evaluations (if genome_to_config is implemented)
					self._log.info(f"[{self.name}] Genome tracking: iter_id={iteration_id}, exp_id={self._tracker_experiment_id}, HAS={HAS_GENOME_TRACKING}")
					if iteration_id and self._tracker_experiment_id and HAS_GENOME_TRACKING and GenomeRole is not None:
						evaluations = []
						for pos, (genome, ce) in enumerate(population):
							config = self.genome_to_config(genome)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								acc = genome._cached_fitness[1] if hasattr(genome, '_cached_fitness') and genome._cached_fitness else 0.0
								# Role: first elite_count are elites, rest are offspring
								role = GenomeRole.ELITE if pos < elite_count else GenomeRole.OFFSPRING
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos,
									"role": role,
									"ce": ce,
									"accuracy": acc,
									"elite_rank": pos if pos < elite_count else None,
								})
						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					# Don't fail optimization on tracking errors
					self._log.debug(f"[{self.name}] V2 tracking error: {e}")

			# Health check: evaluate top-K elites on FULL data vs baseline + stagnation
			# Only at check_interval to avoid expensive full evaluations every generation
			if (generation + 1) % cfg.check_interval == 0:
				elite_genomes_for_check = [g for g, _ in ranked[:elite_count_for_overfit]]
				full_results = evaluator.evaluate_batch_full(elite_genomes_for_check)
				current_fitness = [ce for ce, _ in full_results]

				if early_stop.check_health(generation, current_fitness):
					self._log.info(f"[{self.name}] Early stopping at generation {generation + 1}")
					break

				# Adaptive scaling based on health level
				adaptive_scaler.update(early_stop.current_level)
				if adaptive_scaler.level_changed:
					adaptive_scaler.log_transition(self._log.info)
					cfg.population_size = adaptive_scaler.population
					cfg.mutation_rate = adaptive_scaler.mutation_rate

			# Update previous best for next iteration's delta computation
			prev_best_fitness = best_fitness

		# === VALIDATION SUMMARY: Evaluate top genomes on FULL validation data ===
		self._log.info("")
		self._log.info("=" * 60)
		self._log.info(f"[{self.name}] VALIDATION SUMMARY (Full Dataset)")
		self._log.info("=" * 60)

		# Get unique top genomes by different criteria (20% of population)
		top_k = max(1, int(len(population) * 0.2))
		top_genomes_by_ce = [g for g, _ in population[:top_k]]

		# Evaluate on full validation data
		full_results = evaluator.evaluate_batch_full(top_genomes_by_ce)
		full_evals = list(zip(top_genomes_by_ce, full_results))

		# Sort by CE and Acc
		by_ce = sorted(full_evals, key=lambda x: x[1][0])  # (genome, (ce, acc))
		by_acc = sorted(full_evals, key=lambda x: -x[1][1])  # descending acc

		# Best by CE
		best_ce_genome, (best_ce_ce, best_ce_acc) = by_ce[0]

		# Best by Acc
		best_acc_genome, (best_acc_ce, best_acc_acc) = by_acc[0]

		# Best by fitness (use the existing best_genome from optimization)
		best_fit_ce, best_fit_acc = evaluator.evaluate_batch_full([best_genome])[0]

		# Log summary
		self._log.info(f"  Best by CE:       CE={best_ce_ce:.4f}, Acc={best_ce_acc:.4%}")
		self._log.info(f"  Best by Accuracy: CE={best_acc_ce:.4f}, Acc={best_acc_acc:.4%}")

		# Check if best_fit genome is the same as best_ce or best_acc
		if best_fit_ce == best_ce_ce and best_fit_acc == best_ce_acc:
			self._log.info(f"  Best by Fitness:  (same as Best by CE)")
		elif best_fit_ce == best_acc_ce and best_fit_acc == best_acc_acc:
			self._log.info(f"  Best by Fitness:  (same as Best by Accuracy)")
		else:
			self._log.info(f"  Best by Fitness:  CE={best_fit_ce:.4f}, Acc={best_fit_acc:.4%}")

		# Top-K mean
		top_k_ce = sum(ce for _, (ce, _) in full_evals) / len(full_evals)
		top_k_acc = sum(acc for _, (_, acc) in full_evals) / len(full_evals)
		self._log.info(f"  Top-{top_k} Mean:    CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
		self._log.info("=" * 60)

		# Record phase results via tracker if available
		if self._tracker and self._tracker_phase_id:
			try:
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="best_ce",
					ce=best_ce_ce,
					accuracy=best_ce_acc,
					improvement_pct=(initial_fitness - best_ce_ce) / initial_fitness * 100 if initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="best_acc",
					ce=best_acc_ce,
					accuracy=best_acc_acc,
					improvement_pct=(initial_fitness - best_acc_ce) / initial_fitness * 100 if initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="top_k_mean",
					ce=top_k_ce,
					accuracy=top_k_acc,
					improvement_pct=(initial_fitness - top_k_ce) / initial_fitness * 100 if initial_fitness else 0.0,
				)
			except Exception as e:
				self._log.debug(f"[{self.name}] Failed to record phase results: {e}")

		# Use best_ce for the final result (most important metric)
		final_ce, final_acc = best_ce_ce, best_ce_acc
		improvement_pct = (initial_fitness - final_ce) / initial_fitness * 100 if initial_fitness != 0 else 0.0
		stop_reason = self._determine_stop_reason(shutdown_requested, early_stop)
		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best_ce_genome,  # Return the genome with best CE on full validation
			initial_fitness=initial_fitness,
			final_fitness=final_ce,
			improvement_percent=improvement_pct,
			iterations_run=generation + 1,
			method_name=self.name,
			final_population=[g for g, _ in population],
			final_threshold=current_threshold,
			final_accuracy=final_acc,
			initial_accuracy=None,
			stop_reason=stop_reason,
		)


class ArchitectureTSStrategy(ArchitectureStrategyMixin, GenericTSStrategy['ClusterGenome']):
	"""
	Tabu Search for architecture (bits, neurons per cluster) optimization.

	Inherits core TS loop from GenericTSStrategy, implements ClusterGenome operations.
	Uses ArchitectureStrategyMixin for shared functionality (Metal cleanup, shutdown, etc.)

	Features:
	- Rust/Metal batch evaluation (default when available)
	- Rust-based neighbor search with threshold (when cached_evaluator provided)
	- Population seeding from previous phases
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		ts_config: Optional[TSConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional['RustParallelEvaluator'] = None,
		cached_evaluator: Optional[Any] = None,  # CachedEvaluator for Rust search_neighbors
		shutdown_check: Optional[Callable[[], bool]] = None,  # Callable returning True if shutdown requested
	):
		super().__init__(config=ts_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		# Use cached_evaluator if provided, or check if batch_evaluator has search_neighbors
		if cached_evaluator is not None:
			self._cached_evaluator = cached_evaluator
		elif batch_evaluator is not None and hasattr(batch_evaluator, 'search_neighbors'):
			self._cached_evaluator = batch_evaluator
		else:
			self._cached_evaluator = None
		self._shutdown_check = shutdown_check

	@property
	def name(self) -> str:
		return "ArchitectureTS"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""
		Generate a neighbor by mutating multiple clusters based on mutation_rate.

		The optimizer is phase-agnostic. Callers control what gets optimized
		by setting optimize_bits, optimize_neurons, optimize_connections.

		Mutation delta ranges are 10% of (min + max):
		- Neurons: 10% × (min_neurons + max_neurons)
		- Bits: 10% × (min_bits + max_bits)
		- Connections: 10% × bits_per_token

		Returns (new_genome, move_info) where move_info is a tuple of mutated cluster indices
		for tabu tracking.
		"""
		self._ensure_rng()
		cfg = self._arch_config
		mutant = genome.clone()

		# Calculate delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (cfg.min_bits + cfg.max_bits)))
		neurons_delta_max = max(1, round(0.1 * (cfg.min_neurons + cfg.max_neurons)))
		bits_per_token = 16  # Reasonable default for GPT-2 vocab
		conn_delta_max = max(1, round(0.1 * bits_per_token))

		# If only optimizing connections, mutate connections based on mutation_rate
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			if genome.connections is not None and cfg.total_input_bits is not None:
				mutated_indices = []
				for i in range(len(genome.connections)):
					if self._rng.random() < mutation_rate:
						old_val = mutant.connections[i]
						delta = self._rng.randint(-conn_delta_max, conn_delta_max)
						new_val = max(0, min(cfg.total_input_bits - 1, old_val + delta))
						mutant.connections[i] = new_val
						mutated_indices.append(i)
				move = tuple(mutated_indices) if mutated_indices else None
			else:
				move = None
			return mutant, move

		# Track old architecture for connection adjustment
		old_bits = genome.bits_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# Mutate multiple clusters based on mutation_rate
		# If mutable_clusters is set, only mutate first N clusters (tier0-only mode)
		max_mutable = cfg.mutable_clusters if cfg.mutable_clusters is not None else cfg.num_clusters
		mutated_clusters = []
		for i in range(max_mutable):
			if self._rng.random() < mutation_rate:
				mutated_clusters.append(i)

				if cfg.optimize_bits:
					delta = self._rng.randint(-bits_delta_max, bits_delta_max)
					new_bits = mutant.bits_per_cluster[i] + delta
					mutant.bits_per_cluster[i] = max(cfg.min_bits, min(cfg.max_bits, new_bits))

				if cfg.optimize_neurons:
					delta = self._rng.randint(-neurons_delta_max, neurons_delta_max)
					new_neurons = mutant.neurons_per_cluster[i] + delta
					mutant.neurons_per_cluster[i] = max(cfg.min_neurons, min(cfg.max_neurons, new_neurons))

		# Move is tuple of mutated cluster indices (for tabu tracking)
		move = tuple(mutated_clusters) if mutated_clusters else None

		# Adjust connections if they exist and architecture changed
		if genome.connections is not None and cfg.total_input_bits is not None:
			mutant.connections = self._adjust_connections_for_mutation_ts(
				genome, mutant, old_bits, old_neurons, cfg.total_input_bits
			)

		return mutant, move

	def _adjust_connections_for_mutation_ts(
		self,
		old_genome: 'ClusterGenome',
		new_genome: 'ClusterGenome',
		old_bits: list[int],
		old_neurons: list[int],
		total_input_bits: int,
	) -> list[int]:
		"""Adjust connections when architecture changes during TS mutation."""
		result = []
		old_idx = 0

		for cluster_idx in range(len(new_genome.bits_per_cluster)):
			o_neurons = old_neurons[cluster_idx]
			o_bits = old_bits[cluster_idx]
			n_neurons = new_genome.neurons_per_cluster[cluster_idx]
			n_bits = new_genome.bits_per_cluster[cluster_idx]

			for neuron_idx in range(n_neurons):
				if neuron_idx < o_neurons:
					# Existing neuron - copy connections
					for bit_idx in range(n_bits):
						if bit_idx < o_bits:
							conn_idx = old_idx + neuron_idx * o_bits + bit_idx
							result.append(old_genome.connections[conn_idx])
						else:
							# New bit position - add random connection
							result.append(self._rng.randint(0, total_input_bits - 1))
				else:
					# New neuron - copy from random existing with slight mutation
					if o_neurons > 0:
						template_neuron = self._rng.randint(0, o_neurons - 1)
						for bit_idx in range(n_bits):
							if bit_idx < o_bits:
								conn_idx = old_idx + template_neuron * o_bits + bit_idx
								old_conn = old_genome.connections[conn_idx]
								delta = self._rng.choice([-2, -1, 1, 2])
								new_conn = max(0, min(total_input_bits - 1, old_conn + delta))
								result.append(new_conn)
							else:
								result.append(self._rng.randint(0, total_input_bits - 1))
					else:
						for _ in range(n_bits):
							result.append(self._rng.randint(0, total_input_bits - 1))

			old_idx += o_neurons * o_bits

		return result

	def is_tabu_move(self, move: Any, tabu_list: list[Any]) -> bool:
		"""
		Check if move overlaps significantly with recent tabu moves.

		Move is now a tuple of mutated cluster indices. A move is tabu if
		it shares more than 50% of clusters with a recent tabu move.
		"""
		if move is None or not move:
			return False

		move_set = set(move)
		for tabu_move in tabu_list:
			if tabu_move is None:
				continue
			tabu_set = set(tabu_move)
			overlap = len(move_set & tabu_set)
			# Tabu if >50% overlap with any recent move
			if overlap > len(move_set) * 0.5:
				return True

		return False

	def optimize(
		self,
		initial_genome: 'ClusterGenome',
		initial_fitness: float,
		evaluate_fn: Callable[['ClusterGenome'], float],
		initial_neighbors: Optional[list['ClusterGenome']] = None,
		batch_evaluate_fn: Optional[Callable[[list['ClusterGenome']], list[tuple[float, float]]]] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS with Rust batch evaluation.

		If cached_evaluator was provided at init, uses Rust search_neighbors for
		neighbor generation (eliminates Python↔Rust round trips).

		If batch_evaluator was provided at init, uses it for parallel evaluation.
		batch_evaluate_fn should return list[(CE, accuracy)] tuples.
		"""
		# If we have a cached_evaluator, use Rust-based neighbor search
		if self._cached_evaluator is not None:
			return self._optimize_with_rust_search(
				initial_genome=initial_genome,
				initial_fitness=initial_fitness,
				initial_neighbors=initial_neighbors,
			)

		# Fall back to original behavior
		if self._batch_evaluator is not None and batch_evaluate_fn is None:
			batch_evaluate_fn = lambda genomes, min_accuracy=None: self._batch_evaluator.evaluate_batch(
				genomes, logger=self._log, min_accuracy=min_accuracy,
			)

		return super().optimize(
			initial_genome=initial_genome,
			initial_fitness=initial_fitness,
			evaluate_fn=evaluate_fn,
			initial_neighbors=initial_neighbors,
			batch_evaluate_fn=batch_evaluate_fn,
		)

	def _optimize_with_rust_search(
		self,
		initial_genome: 'ClusterGenome',
		initial_fitness: float,
		initial_neighbors: Optional[list['ClusterGenome']] = None,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run TS using Rust search_neighbors for neighbor generation.

		Supports two modes based on fitness_calculator_type:
		- HARMONIC_RANK: Single path, 50 neighbors from best by harmonic rank
		- CE: Dual path, 25 neighbors from best_ce + 25 from best_acc
		"""
		import time
		from wnn.ram.strategies.connectivity.generic_strategies import (
			OptimizerResult, EarlyStoppingConfig, EarlyStoppingTracker
		)

		cfg = self._config
		arch_cfg = self._arch_config
		evaluator = self._cached_evaluator

		# Always create fitness calculator for unified ranking approach
		# For CE mode: fitness = CE, ranking by fitness = ranking by CE
		# For weighted modes: fitness = combined score
		fitness_calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weight_ce=cfg.fitness_weight_ce,
			weight_acc=cfg.fitness_weight_acc,
			min_accuracy_floor=cfg.min_accuracy_floor if cfg.min_accuracy_floor > 0 else None,
		)
		self._log.info(f"[{self.name}] Fitness calculator: {fitness_calculator.name}")

		# Threshold continuity
		start_threshold = cfg.initial_threshold if cfg.initial_threshold is not None else cfg.min_accuracy
		# End threshold depends on actual iterations vs reference (constant rate)
		actual_progress = min(1.0, cfg.iterations / cfg.threshold_reference)
		end_threshold = start_threshold + actual_progress * cfg.threshold_delta

		def get_threshold(progress: float = 0.0) -> float:
			if not cfg.progressive_threshold:
				return start_threshold
			progress = max(0.0, min(1.0, progress))
			return start_threshold + progress * cfg.threshold_delta

		self._log.info(f"[{self.name}] Progressive threshold: {start_threshold:.4%} → {end_threshold:.4%} (rate: {cfg.threshold_delta/cfg.threshold_reference:.4%}/iter)")
		self._log.info(f"[{self.name}] Using Rust search_neighbors (single-call neighbor search)")

		# CRITICAL: Ensure initial genome has connections (fixes reproducibility bug)
		if not initial_genome.has_connections():
			initial_genome.initialize_connections(evaluator.total_input_bits)
		if initial_neighbors:
			for g in initial_neighbors:
				if not g.has_connections():
					g.initialize_connections(evaluator.total_input_bits)

		# Best tracking - unified approach using fitness calculator
		# best_ranked_*: best genome by fitness ranking (used for neighbor search)
		# best_ce_*: best by pure CE (for tracking/logging)
		# best_acc_*: best by pure accuracy (for tracking/logging)
		best_ranked_genome = initial_genome.clone()
		best_ranked_ce = initial_fitness
		best_ranked_accuracy: Optional[float] = None

		best_ce_genome = initial_genome.clone()
		best_ce_fitness = initial_fitness
		best_ce_accuracy: Optional[float] = None

		best_acc_genome = initial_genome.clone()
		best_acc_fitness = initial_fitness
		best_acc_accuracy: Optional[float] = None

		# Global best (for early stopping and result)
		best = initial_genome.clone()
		best_fitness = initial_fitness
		best_accuracy: Optional[float] = None
		start_fitness = initial_fitness

		# All neighbors cache
		all_neighbors: list[tuple['ClusterGenome', float, Optional[float]]] = [
			(initial_genome.clone(), initial_fitness, None)
		]

		current_threshold = get_threshold(0.0)

		# Get train subset for this PHASE (fixed for ALL evaluations: initial neighbors and all iterations)
		# This ensures all genomes are evaluated on the same data, making fitness values directly comparable.
		phase_train_idx = evaluator.next_train_idx()

		# Seed with initial neighbors
		if initial_neighbors:
			self._log.info(f"[{self.name}] Seeding from {len(initial_neighbors)} neighbors")
			results = evaluator.evaluate_batch(initial_neighbors, train_subset_idx=phase_train_idx)
			for g, (ce, acc) in zip(initial_neighbors, results):
				g._cached_fitness = (ce, acc)
				all_neighbors.append((g.clone(), ce, acc))

			# Track best by CE and best by Acc for logging
			for g, ce, acc in all_neighbors[1:]:  # Skip initial
				if best_ce_fitness is None or ce < best_ce_fitness:
					best_ce_genome = g.clone()
					best_ce_fitness = ce
					best_ce_accuracy = acc
				if acc is not None and (best_acc_accuracy is None or acc > best_acc_accuracy):
					best_acc_genome = g.clone()
					best_acc_fitness = ce
					best_acc_accuracy = acc

			# Find best by fitness ranking (unified approach for all modes)
			pop_for_ranking = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
			if pop_for_ranking:
				ranked = fitness_calculator.rank(pop_for_ranking)
				best_ranked_genome = ranked[0][0].clone()
				best_ranked_ce = best_ranked_genome._cached_fitness[0]
				best_ranked_accuracy = best_ranked_genome._cached_fitness[1]
				best = best_ranked_genome.clone()
				best_fitness = best_fitness_value
				best_accuracy = best_fitness_accuracy

		history = [(0, best_fitness)]

		# Early stopping with baseline-based overfitting detection
		early_stop_config = EarlyStoppingConfig(
			patience=cfg.patience,
			check_interval=cfg.check_interval,
			min_improvement_pct=cfg.min_improvement_pct,
		)
		early_stopper = EarlyStoppingTracker(early_stop_config, self._log, self.name)

		# Adaptive scaling for parameter adjustment based on health level
		from wnn.ram.strategies.connectivity.generic_strategies import AdaptiveScaler
		adaptive_scaler = AdaptiveScaler(
			base_population=cfg.neighbors_per_iter,  # TS uses neighbors as "population"
			base_mutation=cfg.mutation_rate,
			name=self.name,
		)

		# Evaluate initial genome(s) on FULL data for baseline (used by OverfitDetector)
		elite_count_for_overfit = max(1, min(10, len(all_neighbors)))
		elite_genomes_for_baseline = [g for g, _, _ in all_neighbors[:elite_count_for_overfit]]
		baseline_results = evaluator.evaluate_batch_full(elite_genomes_for_baseline)
		baseline_fitness = [ce for ce, _ in baseline_results]
		early_stopper.reset_baseline(baseline_fitness)
		self._log.info(f"[{self.name}] Baseline mean (top-{elite_count_for_overfit} on full data): {sum(baseline_fitness)/len(baseline_fitness):.4f}")

		# Helper to compute top-K% fitness values for early stopping
		# all_neighbors is capped at 50, so fitness ranks are always on consistent scale
		def get_top_k_fitness(neighbors: list[tuple['ClusterGenome', float, Optional[float]]],
							   k_pct: float = 0.2) -> list[float]:
			"""Get fitness values of top-K% neighbors (unified by fitness calculator)."""
			if not neighbors:
				return []
			# Filter to neighbors with accuracy
			valid = [(g, ce, acc) for g, ce, acc in neighbors if acc is not None]
			if not valid:
				sorted_by_ce = sorted(neighbors, key=lambda x: x[1])
				k_count = max(1, int(len(sorted_by_ce) * k_pct))
				return [ce for _, ce, _ in sorted_by_ce[:k_count]]

			# Rank by fitness (unified approach for all modes)
			ranked = fitness_calculator.rank(valid)
			k_count = max(1, int(len(ranked) * k_pct))
			return [score for _, score in ranked[:k_count]]

		# Initialize early stopper with top-K% CE values
		elite_fitness = get_top_k_fitness(all_neighbors)
		early_stopper.reset_trend(elite_fitness)

		# Log config (unified single-path approach for all modes)
		self._log.info(f"[{self.name}] Config: neighbors={cfg.neighbors_per_iter} (single path by {fitness_calculator.name}), "
					   f"iters={cfg.iterations}")

		prev_threshold: Optional[float] = None
		iteration = 0
		seed_offset = int(time.time() * 1000) % (2**16)

		# Track previous best for delta computation
		prev_best_fitness = best_fitness

		shutdown_requested = False  # Track shutdown for stop_reason
		for iteration in range(cfg.iterations):
			# Check for shutdown request at start of each iteration
			if self._shutdown_check and self._shutdown_check():
				self._log.info(f"[{self.name}] Shutdown requested at iteration {iteration}/{cfg.iterations}, stopping gracefully...")
				self._log.info(f"[{self.name}] Progress will be saved (best CE so far: {best_ce_fitness:.4f})")
				shutdown_requested = True
				break

			iter_start = time.time()

			# Cleanup to prevent Metal buffer accumulation
			if iteration > 0:
				self._cleanup_metal(iteration, log_interval=10)

			current_threshold = get_threshold(iteration / cfg.threshold_reference)
			if prev_threshold is not None and f"{prev_threshold:.4%}" != f"{current_threshold:.4%}":
				self._log.debug(f"[{self.name}] Threshold changed: {prev_threshold:.4%} → {current_threshold:.4%}")
			prev_threshold = current_threshold

			# === Unified single-path search: neighbors from best by fitness ranking ===
			self._log.debug(f"[{self.name}] Searching {cfg.neighbors_per_iter} neighbors from best ranked...")
			neighbors = evaluator.search_neighbors(
				genome=best_ranked_genome,
				target_count=cfg.neighbors_per_iter,
				max_attempts=cfg.neighbors_per_iter * 5,
				accuracy_threshold=current_threshold,
				min_bits=arch_cfg.min_bits,
				max_bits=arch_cfg.max_bits,
				min_neurons=arch_cfg.min_neurons,
				max_neurons=arch_cfg.max_neurons,
				bits_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_bits else 0.0,
				neurons_mutation_rate=cfg.mutation_rate if arch_cfg.optimize_neurons else 0.0,
				train_subset_idx=phase_train_idx,
				eval_subset_idx=0,
				seed=seed_offset + iteration * 1000,
				generation=iteration,
				total_generations=cfg.iterations,
				return_best_n=True,
			)

			# Track per-iteration stats (before and after filter)
			iter_neighbors_generated = len(neighbors)

			# Apply percentile filter by fitness ranking
			# Note: 0 means disabled, same as None
			if cfg.fitness_percentile and cfg.fitness_percentile > 0 and neighbors:
				pop_for_filter = [(g, g._cached_fitness[0], g._cached_fitness[1]) for g in neighbors]
				fitness_scores = fitness_calculator.fitness(pop_for_filter)
				neighbor_with_fitness = [(g, score) for g, score in zip(neighbors, fitness_scores)]

				fitness_filter = PercentileFilter(
					percentile=cfg.fitness_percentile,
					mode=FilterMode.LOWER_IS_BETTER,
					metric_name=fitness_calculator.name,
				)
				filter_result = fitness_filter.apply(neighbor_with_fitness, key=lambda g, f: f)
				neighbors = [g for g, _ in filter_result.kept]

			# Track neighbors after filter (for per-iteration stats)
			iter_neighbors_viable = len(neighbors)

			# Store new neighbors for tracking (before merging into cache)
			new_neighbors_this_iter = [(g.clone(), g._cached_fitness[0], g._cached_fitness[1]) for g in neighbors]

			# Add to all_neighbors and track best_ce/best_acc for logging
			for g in neighbors:
				ce, acc = g._cached_fitness
				all_neighbors.append((g.clone(), ce, acc))
				# Track best by CE
				if best_ce_fitness is None or ce < best_ce_fitness:
					best_ce_genome = g.clone()
					best_ce_fitness = ce
					best_ce_accuracy = acc
				# Track best by Acc
				if acc is not None and (best_acc_accuracy is None or acc > best_acc_accuracy):
					best_acc_genome = g.clone()
					best_acc_fitness = ce
					best_acc_accuracy = acc

			# Cap all_neighbors to best N by fitness ranking
			cache_size = cfg.total_neighbors_size or cfg.neighbors_per_iter
			if len(all_neighbors) > cache_size:
				valid_for_cap = [(g, ce, acc) for g, ce, acc in all_neighbors if acc is not None]
				if valid_for_cap:
					ranked_for_cap = fitness_calculator.rank(valid_for_cap)
					all_neighbors = [(g, g._cached_fitness[0], g._cached_fitness[1]) for g, _ in ranked_for_cap[:cache_size]]

			# Find new best by fitness ranking (include current best)
			pop_for_ranking = [(g, ce, acc or 0.0) for g, ce, acc in all_neighbors if acc is not None]
			if pop_for_ranking:
				ranked = fitness_calculator.rank(pop_for_ranking)
				new_best = ranked[0][0]
				new_ce = new_best._cached_fitness[0]
				new_acc = new_best._cached_fitness[1]

				# Update best_ranked (always, since fitness ranking may change)
				best_ranked_genome = new_best.clone()
				best_ranked_ce = new_ce
				best_ranked_accuracy = new_acc

				# Update global best (by CE for early stopping)
				if best_fitness is None or new_ce < best_fitness:
					best = new_best.clone()
					best_fitness = new_ce
					best_accuracy = new_acc

			# Log iteration summary
			iter_elapsed = time.time() - iter_start
			acc_str = f"{best_acc_accuracy:.4%}" if best_acc_accuracy else "N/A"
			ranked_acc_str = f"{best_ranked_accuracy:.4%}" if best_ranked_accuracy else "N/A"
			self._log.info(f"[{self.name}] Iter {iteration+1:03d}/{cfg.iterations}: "
						   f"best_ranked=(CE={best_ranked_ce:.4f}, Acc={ranked_acc_str}), "
						   f"best_ce={best_ce_fitness:.4f}, best_acc={acc_str} (subset) ({iter_elapsed:.1f}s)")

			history.append((iteration + 1, best_fitness))

			# V2 tracking: record iteration data (uses base class _tracker set via set_tracker)
			# All neighbors use the same phase_train_idx, so their fitness values are comparable
			valid_neighbors = [(g, ce, acc) for g, ce, acc in all_neighbors if acc is not None]
			if valid_neighbors:
				avg_ce = sum(ce for _, ce, _ in valid_neighbors) / len(valid_neighbors)
				avg_acc = sum(acc for _, _, acc in valid_neighbors) / len(valid_neighbors)
				actual_best_ce = min(ce for _, ce, _ in valid_neighbors)
				actual_best_acc = max(acc for _, _, acc in valid_neighbors)
			else:
				# Fallback: use all_neighbors if no valid neighbors with accuracy
				avg_ce = sum(ce for _, ce, _ in all_neighbors) / len(all_neighbors) if all_neighbors else None
				avg_acc = None
				actual_best_ce = min(ce for _, ce, _ in all_neighbors) if all_neighbors else best_fitness
				actual_best_acc = best_accuracy

			if self._tracker and self._tracker_phase_id:
				try:

					# Get baseline and patience info for dashboard
					baseline_ce = None
					if hasattr(early_stopper, '_overfit_detector') and early_stopper._overfit_detector:
						baseline_ce = early_stopper._overfit_detector.baseline_mean
					delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
					delta_previous = best_fitness - prev_best_fitness
					patience_counter = early_stopper._patience_counter if hasattr(early_stopper, '_patience_counter') else 0
					candidates_total = len(all_neighbors)  # Total neighbors in cache (for reference)

					iteration_id = self._tracker.record_iteration(
						phase_id=self._tracker_phase_id,
						iteration_num=iteration + 1,
						best_ce=actual_best_ce,  # Use actual min CE, not fitness-ranked best
						best_accuracy=actual_best_acc,
						avg_ce=avg_ce,
						avg_accuracy=avg_acc,
						elite_count=1,  # TS has one "current" genome
						offspring_count=iter_neighbors_generated,  # Neighbors generated this iteration
						offspring_viable=iter_neighbors_viable,  # Neighbors after percentile filter
						fitness_threshold=current_threshold,
						elapsed_secs=iter_elapsed,
						baseline_ce=baseline_ce,
						delta_baseline=delta_baseline,
						delta_previous=delta_previous,
						patience_counter=patience_counter,
						patience_max=cfg.patience,
						candidates_total=candidates_total,
					)

					# Record genome evaluations for TS
					# - Position 0: CURRENT (best genome used to generate)
					# - Positions 1-N: TOP_K (top 50 cache after merge+cap)
					# - Positions 100+: NEIGHBOR (new neighbors generated this iteration)
					self._log.info(f"[{self.name}] Genome tracking: iter_id={iteration_id}, exp_id={self._tracker_experiment_id}, HAS={HAS_GENOME_TRACKING}")
					if iteration_id and self._tracker_experiment_id and HAS_GENOME_TRACKING and GenomeRole is not None:
						evaluations = []

						# 1. Record current best genome (position 0)
						config = self.genome_to_config(best)
						if config is not None:
							genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, config
							)
							evaluations.append({
								"iteration_id": iteration_id,
								"genome_id": genome_id,
								"position": 0,
								"role": GenomeRole.CURRENT,
								"ce": best_fitness,
								"accuracy": best_accuracy or 0.0,
								"elite_rank": 0,
							})

						# 2. Record top 50 cache (positions 1-50) - the elite pool
						for pos, (g, ce, acc) in enumerate(valid_neighbors[:50]):
							config = self.genome_to_config(g)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": pos + 1,
									"role": GenomeRole.TOP_K,
									"ce": ce,
									"accuracy": acc or 0.0,
									"elite_rank": pos,  # Rank in top 50
								})

						# 3. Record new neighbors generated this iteration (positions 100+)
						for pos, (g, ce, acc) in enumerate(new_neighbors_this_iter[:50]):
							config = self.genome_to_config(g)
							if config is not None:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, config
								)
								evaluations.append({
									"iteration_id": iteration_id,
									"genome_id": genome_id,
									"position": 100 + pos,
									"role": GenomeRole.NEIGHBOR,
									"ce": ce,
									"accuracy": acc or 0.0,
									"elite_rank": None,  # Not ranked yet
								})

						if evaluations:
							self._tracker.record_genome_evaluations_batch(evaluations)
				except Exception as e:
					# Don't fail optimization on tracking errors
					self._log.debug(f"[{self.name}] V2 tracking error: {e}")

			# Health check: evaluate top-K neighbors on FULL data vs baseline + stagnation
			# Only at check_interval to avoid expensive full evaluations every iteration
			if (iteration + 1) % cfg.check_interval == 0:
				top_k_genomes = [g for g, _, _ in all_neighbors[:elite_count_for_overfit]]
				full_results = evaluator.evaluate_batch_full(top_k_genomes)
				current_fitness = [ce for ce, _ in full_results]

				if early_stopper.check_health(iteration, current_fitness):
					self._log.info(f"[{self.name}] Early stopping at iteration {iteration + 1}")
					break

				# Adaptive scaling based on health level
				adaptive_scaler.update(early_stopper.current_level)
				if adaptive_scaler.level_changed:
					adaptive_scaler.log_transition(self._log.info)
					cfg.neighbors_per_iter = adaptive_scaler.population
					cfg.mutation_rate = adaptive_scaler.mutation_rate

			# Update previous best for next iteration's delta computation
			prev_best_fitness = best_fitness

		# Build final population: Always use fitness ranking (unified approach)
		# For CE mode: fitness = CE, so ranking = best CE
		# For weighted modes: fitness = combined score
		cache_size = cfg.total_neighbors_size or cfg.neighbors_per_iter
		valid_neighbors = [n for n in all_neighbors if n[2] is not None]

		def genome_key(g: ClusterGenome) -> int:
			"""Unique key including connections (for connections-only phases)."""
			conn_hash = hash(tuple(g.connections[:1000])) if g.connections is not None else 0
			return hash((tuple(g.bits_per_cluster), tuple(g.neurons_per_cluster), conn_hash))

		# Rank by fitness and take top N unique genomes
		pop_for_ranking = [(g, ce, acc) for g, ce, acc in valid_neighbors]
		ranked = fitness_calculator.rank(pop_for_ranking)

		seen = set()
		final_population = []
		for g, _ in ranked:
			key = genome_key(g)
			if key not in seen:
				seen.add(key)
				final_population.append(g)
				if len(final_population) >= cache_size:
					break

		# === VALIDATION SUMMARY: Evaluate top genomes on FULL validation data ===
		self._log.info("")
		self._log.info("=" * 60)
		self._log.info(f"[{self.name}] VALIDATION SUMMARY (Full Dataset)")
		self._log.info("=" * 60)

		# Get top genomes for full evaluation (20% of neighbors cache)
		top_k = max(1, int(len(final_population) * 0.2))
		top_genomes = final_population[:top_k]

		# Evaluate on full validation data
		full_results = evaluator.evaluate_batch_full(top_genomes)
		full_evals = list(zip(top_genomes, full_results))

		# Sort by CE and Acc
		by_ce = sorted(full_evals, key=lambda x: x[1][0])  # (genome, (ce, acc))
		by_acc = sorted(full_evals, key=lambda x: -x[1][1])  # descending acc

		# Best by CE
		best_ce_genome, (best_ce_ce, best_ce_acc) = by_ce[0]

		# Best by Acc
		best_acc_genome, (best_acc_ce, best_acc_acc) = by_acc[0]

		# Best by fitness (use the existing best genome from optimization)
		best_fit_ce, best_fit_acc = evaluator.evaluate_batch_full([best])[0]

		# Log summary
		self._log.info(f"  Best by CE:       CE={best_ce_ce:.4f}, Acc={best_ce_acc:.4%}")
		self._log.info(f"  Best by Accuracy: CE={best_acc_ce:.4f}, Acc={best_acc_acc:.4%}")

		# Check if best_fit genome is the same as best_ce or best_acc
		if best_fit_ce == best_ce_ce and best_fit_acc == best_ce_acc:
			self._log.info(f"  Best by Fitness:  (same as Best by CE)")
		elif best_fit_ce == best_acc_ce and best_fit_acc == best_acc_acc:
			self._log.info(f"  Best by Fitness:  (same as Best by Accuracy)")
		else:
			self._log.info(f"  Best by Fitness:  CE={best_fit_ce:.4f}, Acc={best_fit_acc:.4%}")

		# Top-K mean
		top_k_ce = sum(ce for _, (ce, _) in full_evals) / len(full_evals)
		top_k_acc = sum(acc for _, (_, acc) in full_evals) / len(full_evals)
		self._log.info(f"  Top-{top_k} Mean:    CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
		self._log.info("=" * 60)

		# Record phase results via tracker if available
		if self._tracker and self._tracker_phase_id:
			try:
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="best_ce",
					ce=best_ce_ce,
					accuracy=best_ce_acc,
					improvement_pct=(start_fitness - best_ce_ce) / start_fitness * 100 if start_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="best_acc",
					ce=best_acc_ce,
					accuracy=best_acc_acc,
					improvement_pct=(start_fitness - best_acc_ce) / start_fitness * 100 if start_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					phase_id=self._tracker_phase_id,
					metric_type="top_k_mean",
					ce=top_k_ce,
					accuracy=top_k_acc,
					improvement_pct=(start_fitness - top_k_ce) / start_fitness * 100 if start_fitness else 0.0,
				)
			except Exception as e:
				self._log.debug(f"[{self.name}] Failed to record phase results: {e}")

		# Use best_ce for the final result (most important metric)
		improvement_pct = (start_fitness - best_ce_ce) / start_fitness * 100 if start_fitness != 0 else 0.0
		stop_reason = self._determine_stop_reason(shutdown_requested, early_stopper)
		return OptimizerResult(
			initial_genome=initial_genome,
			best_genome=best_ce_genome,  # Return the genome with best CE on full validation
			initial_fitness=start_fitness,
			final_fitness=best_ce_ce,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			early_stopped=iteration + 1 < cfg.iterations,
			history=history,
			final_population=final_population,
			final_threshold=current_threshold,
			initial_accuracy=None,
			final_accuracy=best_ce_acc,
			stop_reason=stop_reason,
		)
