"""
Fitness calculators for GA/TS genome ranking.

This module provides different strategies for combining cross-entropy (CE)
and accuracy into a single fitness score for ranking genomes.

Available calculators:
- CE: Pure CE ranking (current behavior) - ignores accuracy in ranking
- HARMONIC_RANK: Harmonic mean of CE rank and accuracy rank - balances both
"""

from enum import IntEnum


class FitnessCalculatorType(IntEnum):
	"""Type of fitness calculator for genome ranking."""
	CE = 0              # Pure CE ranking (lower CE = better)
	HARMONIC_RANK = 1   # Harmonic mean of CE and accuracy ranks
	NORMALIZED = 2      # Normalized [0,1] scale weighted sum


from .FitnessCalculator import FitnessCalculator
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank
from .FitnessCalculatorNormalized import FitnessCalculatorNormalized
from .FitnessCalculatorFactory import FitnessCalculatorFactory


__all__ = [
	# Enum
	"FitnessCalculatorType",
	# Base class
	"FitnessCalculator",
	# Implementations
	"FitnessCalculatorCE",
	"FitnessCalculatorHarmonicRank",
	"FitnessCalculatorNormalized",
	# Factory
	"FitnessCalculatorFactory",
]
