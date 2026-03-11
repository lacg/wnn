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
	NORMALIZED = 2      # Normalized [0,1] scale weighted sum (arithmetic mean)
	NORMALIZED_HARMONIC = 3  # Normalized [0,1] scale weighted harmonic mean
	IDS_SECURITY = 4    # F1 × (1 - FPR)^2 — penalizes false positives for IDS
	IDS_RECALL = 5      # F1 × (1 - FPR)^1 — recall-biased, tolerates higher FPR
	# Note: Accuracy floor wrapping is handled separately via min_accuracy_floor parameter


from .FitnessCalculator import FitnessCalculator, GenomeBest, PopulationBests
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank
from .FitnessCalculatorNormalized import FitnessCalculatorNormalized
from .FitnessCalculatorNormalizedHarmonic import FitnessCalculatorNormalizedHarmonic
from .FitnessCalculatorWithAccuracyFloor import FitnessCalculatorWithAccuracyFloor
from .FitnessCalculatorIDSSecurity import FitnessCalculatorIDSSecurity
from .FitnessCalculatorIDSRecall import FitnessCalculatorIDSRecall
from .FitnessCalculatorFactory import FitnessCalculatorFactory


__all__ = [
	# Enum
	"FitnessCalculatorType",
	# Base class + result types
	"FitnessCalculator",
	"GenomeBest",
	"PopulationBests",
	# Implementations
	"FitnessCalculatorCE",
	"FitnessCalculatorHarmonicRank",
	"FitnessCalculatorNormalized",
	"FitnessCalculatorNormalizedHarmonic",
	"FitnessCalculatorWithAccuracyFloor",
	"FitnessCalculatorIDSSecurity",
	"FitnessCalculatorIDSRecall",
	# Factory
	"FitnessCalculatorFactory",
]
