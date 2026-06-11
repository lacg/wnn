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
	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		GAConfig(population_size=30, generations=50),
		arch_config=ArchitectureConfig(num_clusters=2, total_input_bits=336),
		batch_evaluator=evaluator,
	)
"""

from enum import IntEnum, auto
from typing import Any, Optional

from wnn.ram.metrics import FitnessWeights

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
	Available optimizer strategy types for architecture search.

	Three algorithms (all operate on ClusterGenome architecture + connectivity):
	- ARCHITECTURE_GA: Genetic Algorithm (global search, population-based)
	- ARCHITECTURE_TS: Tabu Search (local refinement, μ+λ population)
	- ARCHITECTURE_SA: Simulated Annealing (Metropolis chains, Garcia 2003)

	The LM-era CONNECTIVITY_* stack (raw connection tensors, fixed architecture)
	was removed 10/06/2026 — see docs/ARCHITECTURE_REVIEW_2026-06.md §2.3.
	"""
	# Architecture optimization (adaptive clustered RAM)
	ARCHITECTURE_GA = auto()
	ARCHITECTURE_TS = auto()
	ARCHITECTURE_SA = auto()
	ARCHITECTURE_GRID_SEARCH = auto()
	# Stats-guided adaptation (DEPRECATED: one type per mode — back-compat only).
	ARCHITECTURE_NEUROGENESIS = auto()
	ARCHITECTURE_SYNAPTOGENESIS = auto()
	ARCHITECTURE_AXONOGENESIS = auto()
	# Unified Lamarckian strategy — genesis_mode param picks neuro/synapto/axono
	# (mirrors GA/TS being ONE type with a dimension param).
	ARCHITECTURE_LAMARCKIAN = auto()


class OptimizerStrategyFactory:
	"""
	Factory for creating optimizer strategies (GA/TS).

	Supports both connectivity optimization (tiered) and architecture
	optimization (adaptive) strategies.

	Usage:
		# Architecture GA (adaptive cluster)
		strategy = OptimizerStrategyFactory.create(
			OptimizerStrategyType.ARCHITECTURE_GA,
			GAConfig(population_size=30, generations=50),
			arch_config=ArchitectureConfig(num_clusters=50257),
		)

	"""

	@staticmethod
	def create(
		strategy_type: OptimizerStrategyType,
		opt_config: Any,
		arch_config: Optional[Any] = None,
		seed: Optional[int] = None,
		logger: Optional[Any] = None,
		batch_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Any] = None,
		checkpoint_config: Optional[Any] = None,
		phase_name: Optional[str] = None,
	):
		"""Create an optimizer strategy from TYPED configs (no kwargs — D6d).

		Args:
			strategy_type: Which strategy to build.
			opt_config: The algorithm config — GAConfig, TSConfig, SAConfig,
				GridSearchConfig, or AdaptationConfig. Callers construct it
				explicitly; every knob is a visible, typed field.
			arch_config: ArchitectureConfig (bounds, optimize flags, tier info).
				Required for GA/TS/SA; unused for grid search and adaptation
				(their configs carry the architecture bounds).
			seed: RNG seed (None = time-based).
			logger: Logging callable.
			batch_evaluator: Rust-backed batch evaluator (also used as
				cached_evaluator where it supports Rust neighbor search).
			shutdown_check: Callable returning True to request a graceful stop.
			checkpoint_config: CheckpointConfig (GA only).
			phase_name: Display/checkpoint name for the phase.
		"""
		from wnn.ram.strategies.connectivity.framework import GAConfig, TSConfig, SAConfig
		from wnn.ram.strategies.connectivity.architecture_strategies import (
			ArchitectureConfig,
			ArchitectureGAStrategy,
			ArchitectureTSStrategy,
			ArchitectureSAStrategy,
			GridSearchConfig,
			GridSearchStrategy,
			AdaptationConfig,
			AdaptationStrategy,
		)

		if seed is None:
			import time
			seed = int(time.time() * 1000) % (2**32)

		def _expect(config: Any, expected: type) -> None:
			if not isinstance(config, expected):
				raise TypeError(
					f"{strategy_type.name} requires opt_config of type "
					f"{expected.__name__}, got {type(config).__name__}"
				)

		def _expect_arch() -> None:
			if not isinstance(arch_config, ArchitectureConfig):
				raise TypeError(
					f"{strategy_type.name} requires an ArchitectureConfig, "
					f"got {type(arch_config).__name__}"
				)

		match strategy_type:
			case OptimizerStrategyType.ARCHITECTURE_GA:
				_expect(opt_config, GAConfig)
				_expect_arch()
				cached_evaluator = (
					batch_evaluator
					if batch_evaluator is not None and hasattr(batch_evaluator, 'search_offspring')
					else None
				)
				return ArchitectureGAStrategy(
					arch_config, opt_config, seed, logger, batch_evaluator, cached_evaluator,
					checkpoint_config=checkpoint_config,
					phase_name=phase_name or "GA Optimization",
					shutdown_check=shutdown_check,
				)

			case OptimizerStrategyType.ARCHITECTURE_TS:
				_expect(opt_config, TSConfig)
				_expect_arch()
				return ArchitectureTSStrategy(
					arch_config, opt_config, seed, logger, batch_evaluator,
					shutdown_check=shutdown_check,
				)

			case OptimizerStrategyType.ARCHITECTURE_SA:
				_expect(opt_config, SAConfig)
				_expect_arch()
				return ArchitectureSAStrategy(
					arch_config, opt_config, seed, logger, batch_evaluator,
					shutdown_check=shutdown_check,
				)

			case OptimizerStrategyType.ARCHITECTURE_GRID_SEARCH:
				_expect(opt_config, GridSearchConfig)
				return GridSearchStrategy(
					config=opt_config,
					batch_evaluator=batch_evaluator,
					seed=seed,
					logger=logger,
					shutdown_check=shutdown_check,
				)

			case (OptimizerStrategyType.ARCHITECTURE_LAMARCKIAN
				| OptimizerStrategyType.ARCHITECTURE_NEUROGENESIS
				| OptimizerStrategyType.ARCHITECTURE_SYNAPTOGENESIS
				| OptimizerStrategyType.ARCHITECTURE_AXONOGENESIS):
				_expect(opt_config, AdaptationConfig)
				# Deprecated per-mode types override the config's adaptation_mode.
				_alias_mode = {
					OptimizerStrategyType.ARCHITECTURE_NEUROGENESIS: "neurogenesis",
					OptimizerStrategyType.ARCHITECTURE_SYNAPTOGENESIS: "synaptogenesis",
					OptimizerStrategyType.ARCHITECTURE_AXONOGENESIS: "axonogenesis",
				}
				if strategy_type in _alias_mode:
					opt_config.adaptation_mode = _alias_mode[strategy_type]
				return AdaptationStrategy(
					config=opt_config,
					seed=seed,
					logger=logger,
					cached_evaluator=batch_evaluator,
					phase_name=phase_name,
					shutdown_check=shutdown_check,
				)

			case _:
				raise ValueError(f"Unknown strategy type: {strategy_type}")
