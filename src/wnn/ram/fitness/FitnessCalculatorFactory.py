"""
Factory for creating fitness calculators.
"""

from typing import Optional, TypeVar

from . import FitnessCalculatorType
from .FitnessCalculator import FitnessCalculator
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank
from .FitnessCalculatorNormalized import FitnessCalculatorNormalized
from .FitnessCalculatorNormalizedHarmonic import FitnessCalculatorNormalizedHarmonic
from .FitnessCalculatorWithAccuracyFloor import FitnessCalculatorWithAccuracyFloor

G = TypeVar('G')


class FitnessCalculatorFactory:
	"""Factory for creating fitness calculator instances."""

	@staticmethod
	def create(
		mode: FitnessCalculatorType,
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
		min_accuracy_floor: Optional[float] = None,
	) -> FitnessCalculator:
		"""
		Create a fitness calculator of the specified type.

		Args:
			mode: Type of fitness calculator to create
			weight_ce: Weight for CE (used for HARMONIC_RANK and NORMALIZED)
			weight_acc: Weight for accuracy (used for HARMONIC_RANK and NORMALIZED)
			min_accuracy_floor: If set (> 0), wrap calculator with accuracy floor.
				Genomes below this accuracy get fitness = infinity.

		Returns:
			Configured FitnessCalculator instance

		Raises:
			ValueError: If mode is not recognized
		"""
		# Create base calculator
		match mode:
			case FitnessCalculatorType.CE:
				base = FitnessCalculatorCE()
			case FitnessCalculatorType.HARMONIC_RANK:
				base = FitnessCalculatorHarmonicRank(weight_ce=weight_ce, weight_acc=weight_acc)
			case FitnessCalculatorType.NORMALIZED:
				base = FitnessCalculatorNormalized(weight_ce=weight_ce, weight_acc=weight_acc)
			case FitnessCalculatorType.NORMALIZED_HARMONIC:
				base = FitnessCalculatorNormalizedHarmonic(weight_ce=weight_ce, weight_acc=weight_acc)
			case _:
				raise ValueError(f"Unsupported FitnessCalculatorType: {mode}")

		# Wrap with accuracy floor if specified
		if min_accuracy_floor is not None and min_accuracy_floor > 0:
			return FitnessCalculatorWithAccuracyFloor(base, min_accuracy=min_accuracy_floor)

		return base
