"""compute_ranks: fractional tie-aware ranking shared by the IDS and controller
harmonic calculators.

Regression cover for the 09/08/2026 defect: both calculators carried identical
positional-rank copies that gave TIED values distinct ranks by list position —
order-dependent, and biased toward incumbent elites because populations arrive
sorted by the previous generation's fitness. Measured live: 42% of recent IDS
populations carry an fpr tie (worst 20 genomes sharing best-fpr, ranked 1..20);
controller populations tie massively on stable_rate=100%.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

from wnn.ram.fitness import (FitnessCalculatorControllerHarmonic,
                             FitnessCalculatorHarmonicRank, compute_ranks)


# ---- the helper itself -------------------------------------------------------

def test_distinct_values_get_classic_ranks():
	assert compute_ranks([0.3, 0.1, 0.2]) == [3.0, 1.0, 2.0]
	assert compute_ranks([0.3, 0.1, 0.2], ascending=False) == [1.0, 3.0, 2.0]


def test_ties_share_the_average_rank():
	# positions 1..3 tied at the top -> (1+2+3)/3 = 2.0 each
	assert compute_ranks([5.0, 5.0, 5.0, 9.0]) == [2.0, 2.0, 2.0, 4.0]
	# classic "1 2 2 4" case becomes 1, 2.5, 2.5, 4
	assert compute_ranks([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]


def test_rank_sum_is_preserved():
	"""Σranks must equal n(n+1)/2 regardless of ties, so WHM weights keep meaning."""
	for vals in ([1, 1, 1, 1], [1, 2, 2, 3, 3, 3], [7], [2, 2]):
		ranks = compute_ranks([float(v) for v in vals])
		assert sum(ranks) == len(vals) * (len(vals) + 1) / 2


def test_order_independence():
	"""The defect: identical values used to rank by position. Now a permutation of
	the input must permute the output identically."""
	vals = [2.0, 1.0, 2.0, 3.0, 1.0]
	base = compute_ranks(vals)
	rot = vals[2:] + vals[:2]
	rotated = compute_ranks(rot)
	assert rotated == base[2:] + base[:2]


# ---- through the calculators -------------------------------------------------

def _cm(reward, stable, jerk=1.0, mono=0.0):
	return SimpleNamespace(reward=reward, stable_rate=stable, acc=stable,
	                       mean_attitude_error_deg=-reward, mean_steady_error_deg=None,
	                       motor_jerk_mean=jerk, mono_violations_total=mono,
	                       mean_effort=None)


def test_controller_dominant_genome_wins_from_any_position():
	"""Before the fix a dominant candidate at index 5 LOST to a tied one at index 0."""
	calc = FitnessCalculatorControllerHarmonic(weight_err_sq=0.4, weight_stable=0.3,
	                                           weight_jerk=0.2, weight_mono=0.1)
	tied = [_cm(-2.0, 1.0) for _ in range(5)]
	dominant = _cm(-0.5, 1.0, jerk=0.1, mono=0.0)
	for pos in (0, 3, 5):
		ms = list(tied)
		ms.insert(pos, dominant)
		v = calc.fitness(ms)
		assert min(range(len(v)), key=lambda i: v[i]) == pos, \
			f"dominant genome must win from position {pos}"


def test_controller_identical_metrics_identical_fitness():
	calc = FitnessCalculatorControllerHarmonic(weight_err_sq=0.4, weight_stable=0.3,
	                                           weight_jerk=0.2, weight_mono=0.1)
	v = calc.fitness([_cm(-2.0, 1.0) for _ in range(5)])
	assert len(set(v)) == 1, f"identical genomes must tie exactly, got {v}"


def _im(ce, acc, f1=None, fpr=None):
	return SimpleNamespace(ce=ce, acc=acc, f1=f1, fpr=fpr)


def test_ids_fpr_tie_group_shares_rank():
	"""The live-cohort case: many genomes at the best fpr must not be ranked 1..k
	by position. Equal (ce, f1, fpr) with distinct acc must give the tied-fpr
	genomes fitness that depends only on their own values."""
	calc = FitnessCalculatorHarmonicRank(weight_ce=0.0, weight_acc=0.0,
	                                     weight_f1=0.35, weight_fpr=0.35)
	ms = [_im(0.5, 0.9, f1=0.90, fpr=0.05) for _ in range(4)] + [_im(0.5, 0.9, f1=0.80, fpr=0.10)]
	v = calc.fitness(ms)
	assert len(set(v[:4])) == 1, f"tied genomes diverge: {v[:4]}"
	assert v[4] > v[0], "the strictly-worse genome must rank behind the tie group"
