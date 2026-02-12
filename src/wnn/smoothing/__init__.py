"""
Smoothing module for n-gram language models.

Provides probability smoothing strategies that handle unseen n-grams,
converting raw counts into proper probability distributions.

Available strategies:
- KneserNeySmoothing: Gold standard, uses continuation probability
- SimpleBackoffSmoothing: Katz-style backoff (simpler, faster)
- AddKSmoothing: Laplace smoothing (baseline)

Usage:
	from wnn.smoothing import KneserNeySmoothing

	# Create and train
	kn = KneserNeySmoothing(max_order=5)
	kn.train(tokens)

	# Get probability
	prob = kn.probability("fox", ("the", "quick", "brown"))

	# Evaluate perplexity
	ppl = kn.perplexity(test_tokens)
"""

from enum import IntEnum
from typing import Type

from wnn.smoothing.base import SmoothingStrategy
from wnn.smoothing.kneser_ney import (
	KneserNeySmoothing,
	SimpleBackoffSmoothing,
	AddKSmoothing,
)
from wnn.smoothing.ram_integration import (
	SmoothedRAMPredictor,
	create_smoothed_evaluator,
)


class SmoothingType(IntEnum):
	"""Smoothing strategy types."""
	NONE = 0           # No smoothing (raw counts, 1/vocab for unseen)
	ADD_K = 1          # Add-k (Laplace) smoothing
	SIMPLE_BACKOFF = 2 # Katz-style backoff
	KNESER_NEY = 3     # Interpolated Modified Kneser-Ney


class SmoothingFactory:
	"""Factory for creating smoothing strategies."""

	_TYPE_TO_CLASS: dict[SmoothingType, Type[SmoothingStrategy]] = {
		SmoothingType.ADD_K: AddKSmoothing,
		SmoothingType.SIMPLE_BACKOFF: SimpleBackoffSmoothing,
		SmoothingType.KNESER_NEY: KneserNeySmoothing,
	}

	@classmethod
	def create(cls, smoothing_type: SmoothingType, **kwargs) -> SmoothingStrategy:
		"""
		Create a smoothing strategy.

		Args:
			smoothing_type: Type of smoothing
			**kwargs: Strategy-specific parameters

		Returns:
			SmoothingStrategy instance
		"""
		if smoothing_type == SmoothingType.NONE:
			raise ValueError("NONE smoothing type not supported - use raw counts")

		strategy_class = cls._TYPE_TO_CLASS.get(smoothing_type)
		if strategy_class is None:
			raise ValueError(f"Unknown smoothing type: {smoothing_type}")

		return strategy_class(**kwargs)


def create_smoothing(
	smoothing_type: SmoothingType = SmoothingType.KNESER_NEY,
	**kwargs,
) -> SmoothingStrategy:
	"""Convenience function to create a smoothing strategy."""
	return SmoothingFactory.create(smoothing_type, **kwargs)


__all__ = [
	# Base
	"SmoothingStrategy",
	# Implementations
	"KneserNeySmoothing",
	"SimpleBackoffSmoothing",
	"AddKSmoothing",
	# RAM Integration
	"SmoothedRAMPredictor",
	"create_smoothed_evaluator",
	# Factory
	"SmoothingFactory",
	"SmoothingType",
	"create_smoothing",
]
