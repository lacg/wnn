"""
Factory for creating fitness calculators.
"""

from typing import TypeVar

from . import FitnessCalculatorType
from .FitnessCalculator import FitnessCalculator
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank

G = TypeVar('G')


class FitnessCalculatorFactory:
	"""Factory for creating fitness calculator instances."""

	@staticmethod
	def create(
		mode: FitnessCalculatorType,
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
	) -> FitnessCalculator:
		"""
		Create a fitness calculator of the specified type.

		Args:
			mode: Type of fitness calculator to create
			weight_ce: Weight for CE rank (only used for HARMONIC_RANK)
			weight_acc: Weight for accuracy rank (only used for HARMONIC_RANK)

		Returns:
			Configured FitnessCalculator instance

		Raises:
			ValueError: If mode is not recognized
		"""
		match mode:
			case FitnessCalculatorType.CE:
				return FitnessCalculatorCE()
			case FitnessCalculatorType.HARMONIC_RANK:
				return FitnessCalculatorHarmonicRank(weight_ce=weight_ce, weight_acc=weight_acc)
			case _:
				raise ValueError(f"Unsupported FitnessCalculatorType: {mode}")
