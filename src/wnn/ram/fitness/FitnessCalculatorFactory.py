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
	def create(mode: FitnessCalculatorType) -> FitnessCalculator:
		"""
		Create a fitness calculator of the specified type.

		Args:
			mode: Type of fitness calculator to create

		Returns:
			Configured FitnessCalculator instance

		Raises:
			ValueError: If mode is not recognized
		"""
		match mode:
			case FitnessCalculatorType.CE:
				return FitnessCalculatorCE()
			case FitnessCalculatorType.HARMONIC_RANK:
				return FitnessCalculatorHarmonicRank()
			case _:
				raise ValueError(f"Unsupported FitnessCalculatorType: {mode}")
