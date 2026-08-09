"""
Fitness calculators for GA/TS genome ranking.

All calculators accept list[Metrics] and return list[float] fitness scores.
"""

from enum import IntEnum


class FitnessCalculatorType(IntEnum):
	"""Type of fitness calculator for genome ranking."""
	CE = 0
	HARMONIC_RANK = 1
	NORMALIZED = 2
	NORMALIZED_HARMONIC = 3
	IDS_SECURITY = 4
	IDS_RECALL = 5
	CONTROLLER = 6
	CONTROLLER_HARMONIC = 7


from .FitnessCalculator import FitnessCalculator, GenomeBest, PopulationBests, compute_ranks
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank
from .FitnessCalculatorNormalized import FitnessCalculatorNormalized
from .FitnessCalculatorNormalizedHarmonic import FitnessCalculatorNormalizedHarmonic
from .FitnessCalculatorWithAccuracyFloor import FitnessCalculatorWithAccuracyFloor
from .FitnessCalculatorIDSSecurity import FitnessCalculatorIDSSecurity
from .FitnessCalculatorIDSRecall import FitnessCalculatorIDSRecall
from .FitnessCalculatorController import FitnessCalculatorController
from .FitnessCalculatorControllerHarmonic import FitnessCalculatorControllerHarmonic
from .FitnessCalculatorFactory import FitnessCalculatorFactory


__all__ = [
	"FitnessCalculatorType",
	"FitnessCalculator", "GenomeBest", "PopulationBests",
	"FitnessCalculatorCE", "FitnessCalculatorHarmonicRank",
	"FitnessCalculatorNormalized", "FitnessCalculatorNormalizedHarmonic",
	"FitnessCalculatorWithAccuracyFloor",
	"FitnessCalculatorIDSSecurity", "FitnessCalculatorIDSRecall",
	"FitnessCalculatorController", "FitnessCalculatorControllerHarmonic",
	"FitnessCalculatorFactory",
]
