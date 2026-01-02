"""
Strategy Factories

Factory classes for creating training and forward strategies.
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
