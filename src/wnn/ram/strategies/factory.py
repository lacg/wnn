"""
Strategy Factories

Factory classes for creating training, forward, and optimizer strategies.
Uses match-case for clean type dispatching.

Usage:
	from wnn.ram.strategies.factory import TrainStrategyFactory, ForwardStrategyFactory

	# Create training strategy by type
	strategy = TrainStrategyFactory.create(
		TrainStrategyType.CURRICULUM,
		num_stages=5,
		epochs_per_stage=3,
	)

	# Create forward strategy by type
	fwd_strategy = ForwardStrategyFactory.create(
		ForwardStrategyType.AUTOREGRESSIVE,
		use_cache=True,
	)

	# Create optimizer strategy by type
	from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType
	optimizer = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=50257,
		population_size=30,
		generations=50,
	)
"""

from enum import IntEnum, auto
from typing import Any

from wnn.ram.strategies.base import (
	TrainStrategyBase,
	ForwardStrategyBase,
)
from wnn.ram.strategies.config import (
	TrainConfig,
	GreedyTrainConfig,
	IterativeTrainConfig,
	CurriculumTrainConfig,
	ContrastiveTrainConfig,
	ForwardConfig,
	AutoregressiveConfig,
)
from wnn.ram.strategies.train import (
	GreedyTrainStrategy,
	IterativeTrainStrategy,
	CurriculumTrainStrategy,
	ContrastiveTrainStrategy,
)
from wnn.ram.strategies.forward import (
	AutoregressiveForwardStrategy,
	ParallelForwardStrategy,
)


class TrainStrategyType(IntEnum):
	"""Available training strategy types."""
	GREEDY = auto()
	ITERATIVE = auto()
	CURRICULUM = auto()
	CONTRASTIVE = auto()


class ForwardStrategyType(IntEnum):
	"""Available forward strategy types."""
	AUTOREGRESSIVE = auto()
	PARALLEL = auto()


class TrainStrategyFactory:
	"""
	Factory for creating training strategies.

	Provides a single entry point for strategy creation with
	type-safe configuration options.

	Usage:
		# With kwargs (converted to config internally)
		strategy = TrainStrategyFactory.create(
			TrainStrategyType.CURRICULUM,
			num_stages=5,
			epochs=20,
		)

		# With explicit config
		config = CurriculumTrainConfig(num_stages=5)
		strategy = TrainStrategyFactory.create(
			TrainStrategyType.CURRICULUM,
			config=config,
		)
	"""

	@staticmethod
	def create(
		strategy_type: TrainStrategyType,
		config: TrainConfig | None = None,
		**kwargs: Any,
	) -> TrainStrategyBase:
		"""
		Create a training strategy.

		Args:
			strategy_type: Type of strategy to create
			config: Optional pre-built config (takes precedence over kwargs)
			**kwargs: Configuration options (used if config not provided)

		Returns:
			Configured training strategy

		Raises:
			ValueError: If strategy_type is not recognized
		"""
		match strategy_type:
			case TrainStrategyType.GREEDY:
				if config is None:
					config = GreedyTrainConfig(**_filter_kwargs(GreedyTrainConfig, kwargs))
				return GreedyTrainStrategy(config)

			case TrainStrategyType.ITERATIVE:
				if config is None:
					config = IterativeTrainConfig(**_filter_kwargs(IterativeTrainConfig, kwargs))
				return IterativeTrainStrategy(config)

			case TrainStrategyType.CURRICULUM:
				if config is None:
					config = CurriculumTrainConfig(**_filter_kwargs(CurriculumTrainConfig, kwargs))
				return CurriculumTrainStrategy(config)

			case TrainStrategyType.CONTRASTIVE:
				if config is None:
					config = ContrastiveTrainConfig(**_filter_kwargs(ContrastiveTrainConfig, kwargs))
				return ContrastiveTrainStrategy(config)

			case _:
				raise ValueError(f"Unknown training strategy type: {strategy_type}")

	@staticmethod
	def get_config_class(strategy_type: TrainStrategyType) -> type[TrainConfig]:
		"""Get the configuration class for a strategy type."""
		match strategy_type:
			case TrainStrategyType.GREEDY:
				return GreedyTrainConfig
			case TrainStrategyType.ITERATIVE:
				return IterativeTrainConfig
			case TrainStrategyType.CURRICULUM:
				return CurriculumTrainConfig
			case TrainStrategyType.CONTRASTIVE:
				return ContrastiveTrainConfig
			case _:
				return TrainConfig


class ForwardStrategyFactory:
	"""
	Factory for creating forward strategies.

	Provides a single entry point for inference strategy creation.

	Usage:
		strategy = ForwardStrategyFactory.create(
			ForwardStrategyType.AUTOREGRESSIVE,
			use_cache=True,
			max_length=100,
		)
	"""

	@staticmethod
	def create(
		strategy_type: ForwardStrategyType,
		config: ForwardConfig | None = None,
		**kwargs: Any,
	) -> ForwardStrategyBase:
		"""
		Create a forward strategy.

		Args:
			strategy_type: Type of strategy to create
			config: Optional pre-built config (takes precedence over kwargs)
			**kwargs: Configuration options (used if config not provided)

		Returns:
			Configured forward strategy

		Raises:
			ValueError: If strategy_type is not recognized
		"""
		match strategy_type:
			case ForwardStrategyType.AUTOREGRESSIVE:
				if config is None:
					config = AutoregressiveConfig(**_filter_kwargs(AutoregressiveConfig, kwargs))
				return AutoregressiveForwardStrategy(config)

			case ForwardStrategyType.PARALLEL:
				if config is None:
					config = ForwardConfig(**_filter_kwargs(ForwardConfig, kwargs))
				return ParallelForwardStrategy(config)

			case _:
				raise ValueError(f"Unknown forward strategy type: {strategy_type}")

	@staticmethod
	def get_config_class(strategy_type: ForwardStrategyType) -> type[ForwardConfig]:
		"""Get the configuration class for a strategy type."""
		match strategy_type:
			case ForwardStrategyType.AUTOREGRESSIVE:
				return AutoregressiveConfig
			case ForwardStrategyType.PARALLEL:
				return ForwardConfig
			case _:
				return ForwardConfig


def _filter_kwargs(config_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
	"""
	Filter kwargs to only include fields that exist in the config class.

	This allows passing extra kwargs that get ignored, enabling
	more flexible factory usage.
	"""
	import dataclasses
	if dataclasses.is_dataclass(config_class):
		valid_fields = {f.name for f in dataclasses.fields(config_class)}
		# Also include parent class fields
		for base in config_class.__mro__:
			if dataclasses.is_dataclass(base):
				valid_fields.update(f.name for f in dataclasses.fields(base))
		return {k: v for k, v in kwargs.items() if k in valid_fields}
	return kwargs


# =============================================================================
# Optimizer Strategy Factory (GA/TS for connectivity and architecture)
# =============================================================================

class OptimizerStrategyType(IntEnum):
	"""
	Available optimizer strategy types for connectivity and architecture search.

	Two domains:
	- CONNECTIVITY_*: Optimize which input bits each neuron observes (tiered)
	- ARCHITECTURE_*: Optimize bits/neurons per cluster (adaptive)

	Three algorithms:
	- *_GA: Genetic Algorithm (global search, population-based)
	- *_TS: Tabu Search (local search from initial solution)
	- *_SA: Simulated Annealing (probabilistic acceptance for escaping local minima)
	"""
	# Connectivity optimization (tiered clustered RAM)
	CONNECTIVITY_GA = auto()
	CONNECTIVITY_TS = auto()
	CONNECTIVITY_SA = auto()
	# Architecture optimization (adaptive clustered RAM)
	ARCHITECTURE_GA = auto()
	ARCHITECTURE_TS = auto()


class OptimizerStrategyFactory:
	"""
	Factory for creating optimizer strategies (GA/TS).

	Supports both connectivity optimization (tiered) and architecture
	optimization (adaptive) strategies.

	Usage:
		# Architecture GA (adaptive cluster)
		strategy = OptimizerStrategyFactory.create(
			OptimizerStrategyType.ARCHITECTURE_GA,
			num_clusters=50257,
			population_size=30,
			generations=50,
		)

		# Connectivity GA (tiered cluster)
		strategy = OptimizerStrategyFactory.create(
			OptimizerStrategyType.CONNECTIVITY_GA,
			num_clusters=100,
			bits_per_cluster=16,
		)
	"""

	@staticmethod
	def create(
		strategy_type: OptimizerStrategyType,
		# Architecture params
		num_clusters: int | None = None,
		min_bits: int = 8,
		max_bits: int = 25,
		min_neurons: int = 3,
		max_neurons: int = 33,
		# Explicit control over what gets optimized (no magic phase numbers)
		optimize_bits: bool = True,
		optimize_neurons: bool = True,
		optimize_connections: bool = False,
		# Default values for dimensions not being optimized
		default_bits: int = 8,
		default_neurons: int = 5,
		token_frequencies: list[int] | None = None,
		total_input_bits: int | None = None,  # For connection optimization
		# GA params
		population_size: int = 30,
		generations: int = 50,
		mutation_rate: float = 0.1,
		crossover_rate: float = 0.7,
		elitism: int = 2,
		# TS params
		iterations: int = 100,
		neighbors_per_iter: int = 20,
		tabu_size: int = 10,
		# SA params
		initial_temp: float = 1.0,
		cooling_rate: float = 0.95,
		# Early stopping (GA: 0.05%, TS: 0.5%)
		patience: int = 5,
		check_interval: int = 10,
		min_improvement_pct: float | None = None,  # None = use strategy default
		# Threshold continuity (replaces phase_index)
		initial_threshold: float | None = None,  # Start threshold from previous phase (None = first phase)
		# TS-specific: cache size for final population diversity
		total_neighbors_size: int | None = None,  # None = use neighbors_per_iter
		# Fitness percentile filter (None = disabled, 0.75 = keep top 75% by fitness)
		fitness_percentile: float | None = None,
		# Tier0-only: only mutate first N clusters (None = all clusters)
		mutable_clusters: int | None = None,
		# Common
		seed: int | None = None,  # None = time-based
		verbose: bool = False,
		logger: Any = None,
		batch_evaluator: Any = None,
		# GA-specific: generate fresh random population instead of seeding from given genomes
		fresh_population: bool = False,
		# Checkpoint configuration for resumable optimization
		checkpoint_config: Any = None,
		phase_name: str = "Optimization",
	):
		"""
		Create an optimizer strategy.

		Args:
			strategy_type: Type of optimizer to create (OptimizerStrategyType enum)

			Architecture params (ARCHITECTURE_* types):
				num_clusters: Number of clusters (required for ARCHITECTURE_*)
				min_bits: Minimum bits per cluster
				max_bits: Maximum bits per cluster
				min_neurons: Minimum neurons per cluster
				max_neurons: Maximum neurons per cluster
				optimize_bits: Whether to optimize bits per cluster (default: True)
				optimize_neurons: Whether to optimize neurons per cluster (default: True)
				optimize_connections: Whether to optimize connectivity (default: False)
				default_bits: Default bits when not optimizing (default: 8)
				default_neurons: Default neurons when not optimizing (default: 5)
				token_frequencies: Token frequency list for initialization

			GA params (for *_GA types):
				population_size: GA population size (default: 30)
				generations: Number of GA generations (default: 50)
				mutation_rate: GA mutation rate (default: 0.1 for arch, 0.01 for conn)
				crossover_rate: Crossover probability (default: 0.7)
				elitism: Number of elite individuals preserved (default: 2)

			TS params (for *_TS types):
				iterations: Number of TS iterations (default: 100)
				neighbors_per_iter: Neighbors to evaluate per iteration (default: 20)
				tabu_size: Size of tabu list (default: 10)

			SA params (for *_SA types):
				iterations: Number of SA iterations (default: 600)
				initial_temp: Initial temperature (default: 1.0)
				cooling_rate: Temperature decay rate (default: 0.95)

			Early stopping:
				patience: Checks without improvement before stopping (default: 5)
				check_interval: Check every N generations/iterations (default: 10)
				min_improvement_pct: Minimum improvement % (default: GA=0.05%, TS=0.5%)

			Common:
				seed: Random seed (default: 42)
				verbose: Print progress during optimization (default: False)
				logger: Logger function (default: print)
				batch_evaluator: RustParallelEvaluator for batch evaluation (ARCHITECTURE_* only)

		Returns:
			Configured optimizer strategy

		Example:
			# Architecture GA with Rust batch evaluation
			strategy = OptimizerStrategyFactory.create(
				OptimizerStrategyType.ARCHITECTURE_GA,
				num_clusters=50257,
				generations=100,
				batch_evaluator=rust_evaluator,
			)

			# Connectivity GA for tiered RAM
			strategy = OptimizerStrategyFactory.create(
				OptimizerStrategyType.CONNECTIVITY_GA,
				generations=50,
				verbose=True,
			)
		"""
		# Time-based seed if not specified
		if seed is None:
			import time
			seed = int(time.time() * 1000) % (2**32)

		match strategy_type:
			case OptimizerStrategyType.ARCHITECTURE_GA:
				return OptimizerStrategyFactory._create_architecture_ga(
					num_clusters=num_clusters,
					min_bits=min_bits,
					max_bits=max_bits,
					min_neurons=min_neurons,
					max_neurons=max_neurons,
					optimize_bits=optimize_bits,
					optimize_neurons=optimize_neurons,
					optimize_connections=optimize_connections,
					default_bits=default_bits,
					default_neurons=default_neurons,
					token_frequencies=token_frequencies,
					total_input_bits=total_input_bits,
					mutable_clusters=mutable_clusters,
					population_size=population_size,
					generations=generations,
					mutation_rate=mutation_rate,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.05,
					initial_threshold=initial_threshold,
					fitness_percentile=fitness_percentile,
					seed=seed,
					logger=logger,
					batch_evaluator=batch_evaluator,
					fresh_population=fresh_population,
					checkpoint_config=checkpoint_config,
					phase_name=phase_name,
				)

			case OptimizerStrategyType.ARCHITECTURE_TS:
				return OptimizerStrategyFactory._create_architecture_ts(
					num_clusters=num_clusters,
					min_bits=min_bits,
					max_bits=max_bits,
					min_neurons=min_neurons,
					max_neurons=max_neurons,
					optimize_bits=optimize_bits,
					optimize_neurons=optimize_neurons,
					optimize_connections=optimize_connections,
					default_bits=default_bits,
					default_neurons=default_neurons,
					token_frequencies=token_frequencies,
					total_input_bits=total_input_bits,
					mutable_clusters=mutable_clusters,
					iterations=iterations,
					neighbors_per_iter=neighbors_per_iter,
					total_neighbors_size=total_neighbors_size,
					tabu_size=tabu_size,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.5,
					initial_threshold=initial_threshold,
					fitness_percentile=fitness_percentile,
					seed=seed,
					logger=logger,
					batch_evaluator=batch_evaluator,
				)

			case OptimizerStrategyType.CONNECTIVITY_GA:
				return OptimizerStrategyFactory._create_connectivity_ga(
					population_size=population_size,
					generations=generations,
					mutation_rate=mutation_rate,
					crossover_rate=crossover_rate,
					elitism=elitism,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.05,
					seed=seed,
					verbose=verbose,
					logger=logger,
				)

			case OptimizerStrategyType.CONNECTIVITY_TS:
				return OptimizerStrategyFactory._create_connectivity_ts(
					iterations=iterations,
					neighbors_per_iter=neighbors_per_iter,
					tabu_size=tabu_size,
					mutation_rate=mutation_rate,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.5,
					seed=seed,
					verbose=verbose,
					logger=logger,
				)

			case OptimizerStrategyType.CONNECTIVITY_SA:
				return OptimizerStrategyFactory._create_connectivity_sa(
					iterations=iterations,
					initial_temp=initial_temp,
					cooling_rate=cooling_rate,
					mutation_rate=mutation_rate,
					seed=seed,
					verbose=verbose,
				)

			case _:
				raise ValueError(f"Unknown optimizer strategy type: {strategy_type}")

	# =========================================================================
	# Internal factory methods
	# =========================================================================

	@staticmethod
	def _create_architecture_ga(
		num_clusters: int,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		optimize_bits: bool,
		optimize_neurons: bool,
		optimize_connections: bool,
		default_bits: int,
		default_neurons: int,
		token_frequencies: list[int] | None,
		total_input_bits: int | None,
		mutable_clusters: int | None,
		population_size: int,
		generations: int,
		mutation_rate: float,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		initial_threshold: float | None,
		fitness_percentile: float | None,
		seed: int,
		logger: Any,
		batch_evaluator: Any,
		fresh_population: bool = False,
		checkpoint_config: Any = None,
		phase_name: str = "GA Optimization",
	):
		"""Create an ArchitectureGAStrategy."""
		from wnn.ram.strategies.connectivity.architecture_strategies import (
			ArchitectureGAStrategy,
			ArchitectureConfig,
		)
		from wnn.ram.strategies.connectivity.generic_strategies import GAConfig

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			optimize_bits=optimize_bits,
			optimize_neurons=optimize_neurons,
			optimize_connections=optimize_connections,
			default_bits=default_bits,
			default_neurons=default_neurons,
			token_frequencies=token_frequencies,
			total_input_bits=total_input_bits,
			mutable_clusters=mutable_clusters,
		)
		ga_config = GAConfig(
			population_size=population_size,
			generations=generations,
			patience=patience,
			check_interval=check_interval,
			min_improvement_pct=min_improvement_pct,
			mutation_rate=mutation_rate,
			initial_threshold=initial_threshold,
			fitness_percentile=fitness_percentile,
			fresh_population=fresh_population,
		)
		# Pass batch_evaluator as cached_evaluator if it supports search_offspring
		cached_evaluator = batch_evaluator if batch_evaluator and hasattr(batch_evaluator, 'search_offspring') else None
		return ArchitectureGAStrategy(
			arch_config, ga_config, seed, logger, batch_evaluator, cached_evaluator,
			checkpoint_config=checkpoint_config, phase_name=phase_name
		)

	@staticmethod
	def _create_architecture_ts(
		num_clusters: int,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		optimize_bits: bool,
		optimize_neurons: bool,
		optimize_connections: bool,
		default_bits: int,
		default_neurons: int,
		token_frequencies: list[int] | None,
		total_input_bits: int | None,
		mutable_clusters: int | None,
		iterations: int,
		neighbors_per_iter: int,
		total_neighbors_size: int | None,
		tabu_size: int,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		initial_threshold: float | None,
		fitness_percentile: float | None,
		seed: int,
		logger: Any,
		batch_evaluator: Any,
	):
		"""Create an ArchitectureTSStrategy."""
		from wnn.ram.strategies.connectivity.architecture_strategies import (
			ArchitectureTSStrategy,
			ArchitectureConfig,
		)
		from wnn.ram.strategies.connectivity.generic_strategies import TSConfig

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			optimize_bits=optimize_bits,
			optimize_neurons=optimize_neurons,
			optimize_connections=optimize_connections,
			default_bits=default_bits,
			default_neurons=default_neurons,
			token_frequencies=token_frequencies,
			total_input_bits=total_input_bits,
			mutable_clusters=mutable_clusters,
		)
		ts_config = TSConfig(
			iterations=iterations,
			neighbors_per_iter=neighbors_per_iter,
			total_neighbors_size=total_neighbors_size,
			patience=patience,
			check_interval=check_interval,
			min_improvement_pct=min_improvement_pct,
			tabu_size=tabu_size,
			initial_threshold=initial_threshold,
			fitness_percentile=fitness_percentile,
		)
		return ArchitectureTSStrategy(arch_config, ts_config, seed, logger, batch_evaluator)

	@staticmethod
	def _create_connectivity_ga(
		population_size: int,
		generations: int,
		mutation_rate: float,
		crossover_rate: float,
		elitism: int,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		seed: int,
		verbose: bool,
		logger: Any,
	):
		"""Create a GeneticAlgorithmStrategy for connectivity optimization."""
		from wnn.ram.strategies.connectivity.genetic_algorithm import (
			GeneticAlgorithmStrategy,
			GeneticAlgorithmConfig,
		)
		config = GeneticAlgorithmConfig(
			population_size=population_size,
			generations=generations,
			mutation_rate=mutation_rate,
			crossover_rate=crossover_rate,
			elitism=elitism,
			early_stop_patience=patience,
			early_stop_threshold_pct=min_improvement_pct,
		)
		return GeneticAlgorithmStrategy(config=config, seed=seed, verbose=verbose, logger=logger)

	@staticmethod
	def _create_connectivity_ts(
		iterations: int,
		neighbors_per_iter: int,
		tabu_size: int,
		mutation_rate: float,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		seed: int,
		verbose: bool,
		logger: Any,
	):
		"""Create a TabuSearchStrategy for connectivity optimization."""
		from wnn.ram.strategies.connectivity.tabu_search import (
			TabuSearchStrategy,
			TabuSearchConfig,
		)
		config = TabuSearchConfig(
			iterations=iterations,
			neighbors_per_iter=neighbors_per_iter,
			tabu_size=tabu_size,
			mutation_rate=mutation_rate,
			early_stop_patience=patience,
			early_stop_threshold_pct=min_improvement_pct,
		)
		return TabuSearchStrategy(config=config, seed=seed, verbose=verbose, logger=logger)

	@staticmethod
	def _create_connectivity_sa(
		iterations: int,
		initial_temp: float,
		cooling_rate: float,
		mutation_rate: float,
		seed: int,
		verbose: bool,
	):
		"""Create a SimulatedAnnealingStrategy for connectivity optimization."""
		from wnn.ram.strategies.connectivity.simulated_annealing import (
			SimulatedAnnealingStrategy,
			SimulatedAnnealingConfig,
		)
		config = SimulatedAnnealingConfig(
			iterations=iterations,
			initial_temp=initial_temp,
			cooling_rate=cooling_rate,
			mutation_rate=mutation_rate,
		)
		return SimulatedAnnealingStrategy(config=config, seed=seed, verbose=verbose)
