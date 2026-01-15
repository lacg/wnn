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

	Two algorithms:
	- *_GA: Genetic Algorithm (global search, population-based)
	- *_TS: Tabu Search (local search from initial solution)
	"""
	# Connectivity optimization (tiered clustered RAM)
	CONNECTIVITY_GA = auto()
	CONNECTIVITY_TS = auto()
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
		phase: int = 2,
		token_frequencies: list[int] | None = None,
		# GA params
		population_size: int = 30,
		generations: int = 50,
		mutation_rate: float = 0.1,
		# TS params
		iterations: int = 100,
		neighbors_per_iter: int = 20,
		tabu_size: int = 10,
		# Early stopping (GA: 0.05%, TS: 0.5%)
		patience: int = 5,
		check_interval: int = 5,
		min_improvement_pct: float | None = None,  # None = use strategy default
		# Common
		seed: int = 42,
		logger: Any = None,
		batch_evaluator: Any = None,
	):
		"""
		Create an optimizer strategy.

		Args:
			strategy_type: Type of optimizer to create

			Architecture params:
				num_clusters: Number of clusters (required for ARCHITECTURE_*)
				min_bits: Minimum bits per cluster
				max_bits: Maximum bits per cluster
				min_neurons: Minimum neurons per cluster
				max_neurons: Maximum neurons per cluster
				phase: Optimization phase (1=bits only, 2=bits+neurons)
				token_frequencies: Token frequency list for initialization

			GA params:
				population_size: GA population size
				generations: Number of GA generations
				mutation_rate: GA mutation rate

			TS params:
				iterations: Number of TS iterations
				neighbors_per_iter: Neighbors to evaluate per iteration
				tabu_size: Size of tabu list

			Early stopping:
				patience: Checks without improvement before stopping
				check_interval: Check every N generations/iterations
				min_improvement_pct: Minimum improvement % (default: GA=0.05%, TS=0.5%)

			Common:
				seed: Random seed
				logger: Logger function
				batch_evaluator: RustParallelEvaluator for batch evaluation

		Returns:
			Configured optimizer strategy
		"""
		match strategy_type:
			case OptimizerStrategyType.ARCHITECTURE_GA:
				return OptimizerStrategyFactory._create_architecture_ga(
					num_clusters=num_clusters,
					min_bits=min_bits,
					max_bits=max_bits,
					min_neurons=min_neurons,
					max_neurons=max_neurons,
					phase=phase,
					token_frequencies=token_frequencies,
					population_size=population_size,
					generations=generations,
					mutation_rate=mutation_rate,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.05,
					seed=seed,
					logger=logger,
					batch_evaluator=batch_evaluator,
				)

			case OptimizerStrategyType.ARCHITECTURE_TS:
				return OptimizerStrategyFactory._create_architecture_ts(
					num_clusters=num_clusters,
					min_bits=min_bits,
					max_bits=max_bits,
					min_neurons=min_neurons,
					max_neurons=max_neurons,
					phase=phase,
					token_frequencies=token_frequencies,
					iterations=iterations,
					neighbors_per_iter=neighbors_per_iter,
					tabu_size=tabu_size,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.5,
					seed=seed,
					logger=logger,
					batch_evaluator=batch_evaluator,
				)

			case OptimizerStrategyType.CONNECTIVITY_GA:
				return OptimizerStrategyFactory._create_connectivity_ga(
					population_size=population_size,
					generations=generations,
					mutation_rate=mutation_rate,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.05,
					seed=seed,
					logger=logger,
				)

			case OptimizerStrategyType.CONNECTIVITY_TS:
				return OptimizerStrategyFactory._create_connectivity_ts(
					iterations=iterations,
					neighbors_per_iter=neighbors_per_iter,
					tabu_size=tabu_size,
					patience=patience,
					check_interval=check_interval,
					min_improvement_pct=min_improvement_pct if min_improvement_pct is not None else 0.5,
					seed=seed,
					logger=logger,
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
		phase: int,
		token_frequencies: list[int] | None,
		population_size: int,
		generations: int,
		mutation_rate: float,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		seed: int,
		logger: Any,
		batch_evaluator: Any,
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
			phase=phase,
			token_frequencies=token_frequencies,
		)
		ga_config = GAConfig(
			population_size=population_size,
			generations=generations,
			patience=patience,
			check_interval=check_interval,
			min_improvement_pct=min_improvement_pct,
			mutation_rate=mutation_rate,
		)
		return ArchitectureGAStrategy(arch_config, ga_config, seed, logger, batch_evaluator)

	@staticmethod
	def _create_architecture_ts(
		num_clusters: int,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		phase: int,
		token_frequencies: list[int] | None,
		iterations: int,
		neighbors_per_iter: int,
		tabu_size: int,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
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
			phase=phase,
			token_frequencies=token_frequencies,
		)
		ts_config = TSConfig(
			iterations=iterations,
			neighbors_per_iter=neighbors_per_iter,
			patience=patience,
			check_interval=check_interval,
			min_improvement_pct=min_improvement_pct,
			tabu_size=tabu_size,
		)
		return ArchitectureTSStrategy(arch_config, ts_config, seed, logger, batch_evaluator)

	@staticmethod
	def _create_connectivity_ga(
		population_size: int,
		generations: int,
		mutation_rate: float,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		seed: int,
		logger: Any,
	):
		"""Create a GeneticAlgorithmStrategy for connectivity optimization."""
		from wnn.ram.strategies.connectivity.genetic_algorithm import (
			GeneticAlgorithmStrategy,
		)
		# TODO: Update GeneticAlgorithmStrategy to accept these params
		return GeneticAlgorithmStrategy(seed=seed, logger=logger)

	@staticmethod
	def _create_connectivity_ts(
		iterations: int,
		neighbors_per_iter: int,
		tabu_size: int,
		patience: int,
		check_interval: int,
		min_improvement_pct: float,
		seed: int,
		logger: Any,
	):
		"""Create a TabuSearchStrategy for connectivity optimization."""
		from wnn.ram.strategies.connectivity.tabu_search import (
			TabuSearchStrategy,
		)
		# TODO: Update TabuSearchStrategy to accept these params
		return TabuSearchStrategy(seed=seed, logger=logger)
