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
		# Rank-combine aggregation + z clamp (19/08/2026). Forwarded like every
		# other controller knob — the 17/08 alt/pos lesson applies verbatim: a
		# field the calculator understands but this factory does not forward is
		# a silent no-op for every caller that builds through here.
		aggregation:   str   = "harmonic",
		zrank_clamp:   float = 3.0,
		# Viability gate (21/08/2026): forwarded like every other controller
		# knob — the 17/08 --fit-weight-alt lesson applies verbatim. None = off.
		gate_stable_min: Optional[float] = None,
		gate_err_max:    Optional[float] = None,
		# Desirability CE half-anchor (26/08/2026). ABSOLUTE, already normalised
		# against the task's base-rate entropy by the caller. Forwarded here for
		# the SAME reason as aggregation/zrank_clamp above: a field the
		# calculator understands but this factory drops is a silent no-op.
		ce_anchor:       Optional[float] = None,
		f1_anchor:       Optional[float] = None,
		acc_anchor:      Optional[float] = None,
		jerk_anchor:     Optional[float] = None,
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
				# aggregation/zrank_clamp forwarded here for the SAME reason they
				# are forwarded to CONTROLLER_HARMONIC: a field the calculator
				# understands but this factory drops is a silent no-op for every
				# caller that builds through here — the 17/08 --fit-weight-alt bug.
				base = FitnessCalculatorHarmonicRank(
					weight_ce=weights.ce, weight_acc=weights.acc,
					weight_f1=weights.f1, weight_fpr=weights.fpr,
					aggregation=aggregation, zrank_clamp=zrank_clamp,
					ce_anchor=ce_anchor, f1_anchor=f1_anchor, acc_anchor=acc_anchor,
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
					aggregation=aggregation,     zrank_clamp=zrank_clamp,
					gate_stable_min=gate_stable_min, gate_err_max=gate_err_max,
					jerk_anchor=jerk_anchor,
				)
			case _:
				raise ValueError(f"Unsupported FitnessCalculatorType: {mode}")

		if min_accuracy_floor is not None and min_accuracy_floor > 0:
			return FitnessCalculatorWithAccuracyFloor(base, min_accuracy=min_accuracy_floor)

		return base
