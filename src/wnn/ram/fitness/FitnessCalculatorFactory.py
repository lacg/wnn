"""
Factory for creating fitness calculators.
"""

from typing import Optional, Union

from wnn.ram.metrics import FitnessWeights
from . import FitnessCalculatorType
from .FitnessCalculator import FitnessCalculator
from .FitnessCalculatorCE import FitnessCalculatorCE
from .FitnessCalculatorHarmonicRank import FitnessCalculatorHarmonicRank
from .FitnessCalculatorNormalized import FitnessCalculatorNormalized
from .FitnessCalculatorNormalizedHarmonic import FitnessCalculatorNormalizedHarmonic
from .FitnessCalculatorWithAccuracyFloor import FitnessCalculatorWithAccuracyFloor
from .FitnessCalculatorIDSSecurity import FitnessCalculatorIDSSecurity
from .FitnessCalculatorIDSRecall import FitnessCalculatorIDSRecall
from .FitnessCalculatorController import FitnessCalculatorController
from .FitnessCalculatorControllerHarmonic import FitnessCalculatorControllerHarmonic


class FitnessCalculatorFactory:
	"""Factory for creating fitness calculator instances."""

	@staticmethod
	def create(
		mode: FitnessCalculatorType,
		weights: Optional[FitnessWeights] = None,
		min_accuracy_floor: Optional[float] = None,
		# Legacy individual weight params (used if weights is None)
		weight_ce: float = 1.0,
		weight_acc: float = 1.0,
		weight_f1: float = 0.0,
		weight_fpr: float = 0.0,
		# Controller-harmonic weights (only used when mode=CONTROLLER_HARMONIC)
		weight_err_sq: float = 1.0,
		weight_stable: float = 0.0,
		weight_jerk:   float = 0.0,
		weight_mono:   float = 0.0,
		weight_steady: float = 0.0,
		weight_effort: float = 0.0,
		# Scale-free RANK dimensions for the translation channels (17/08/2026).
		# These MUST be forwarded like every other controller weight: they were
		# added to the calculator without being added here, which made
		# --fit-weight-alt a silent no-op in the GA/TS search for two runs.
		weight_alt:    float = 0.0,
		weight_pos:    float = 0.0,
	) -> FitnessCalculator:
		"""
		Create a fitness calculator.

		Prefer passing `weights=FitnessWeights(...)` over individual weight params.
		"""
		# Build FitnessWeights if not provided
		if weights is None:
			weights = FitnessWeights(ce=weight_ce, acc=weight_acc, f1=weight_f1, fpr=weight_fpr)

		match mode:
			case FitnessCalculatorType.CE:
				base = FitnessCalculatorCE()
			case FitnessCalculatorType.HARMONIC_RANK:
				base = FitnessCalculatorHarmonicRank(
					weight_ce=weights.ce, weight_acc=weights.acc,
					weight_f1=weights.f1, weight_fpr=weights.fpr,
				)
			case FitnessCalculatorType.NORMALIZED:
				base = FitnessCalculatorNormalized(weight_ce=weights.ce, weight_acc=weights.acc)
			case FitnessCalculatorType.NORMALIZED_HARMONIC:
				base = FitnessCalculatorNormalizedHarmonic(weight_ce=weights.ce, weight_acc=weights.acc)
			case FitnessCalculatorType.IDS_SECURITY:
				base = FitnessCalculatorIDSSecurity()
			case FitnessCalculatorType.IDS_RECALL:
				base = FitnessCalculatorIDSRecall()
			case FitnessCalculatorType.CONTROLLER:
				base = FitnessCalculatorController()
			case FitnessCalculatorType.CONTROLLER_HARMONIC:
				base = FitnessCalculatorControllerHarmonic(
					weight_err_sq=weight_err_sq, weight_stable=weight_stable,
					weight_jerk=weight_jerk,     weight_mono=weight_mono,
					weight_steady=weight_steady, weight_effort=weight_effort,
					weight_alt=weight_alt,       weight_pos=weight_pos,
				)
			case _:
				raise ValueError(f"Unsupported FitnessCalculatorType: {mode}")

		if min_accuracy_floor is not None and min_accuracy_floor > 0:
			return FitnessCalculatorWithAccuracyFloor(base, min_accuracy=min_accuracy_floor)

		return base
