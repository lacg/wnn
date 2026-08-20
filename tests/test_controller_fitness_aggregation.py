"""Aggregation semantics of FitnessCalculatorControllerHarmonic.

The harmonic combine (WHM = Σw / Σ(w/rank)) is dominated by a genome's BEST
weighted rank — rank 1 at weight .35 contributes .350 to the denominator while
rank 9 at weight .15 contributes .017 — so it selects specialists and forgives
catastrophic ranks. Arm 9 of the alt-weight sweep made that concrete: its
headline won on steady rank-1 ALONE, dead last on jerk at nearly zero cost,
while a candidate that was top-2 on THREE metrics lost.

The arithmetic combine (Σ(w·rank)/Σw) makes every rank hurt in proportion to
its weight, which is what the weights read as meaning. Stage-select uses it
since 19/08/2026 (Luiz). Both are pinned here, on arm 9's real numbers.
"""
from types import SimpleNamespace

import pytest

from wnn.ram.fitness import FitnessCalculatorControllerHarmonic
from wnn.ram.fitness.FitnessCalculator import compute_ranks


# Arm 9 (AW_REF_lam16) stage-select candidates, verbatim from the marker:
# (key, stable%, err deg, steady deg, reward, jerk, mono)
ARM9 = [
	("GRID#0",         0.0, 68.13, 88.62, -446218.31, 0.0343, 0.7),
	("GRID#1",         0.0, 64.91, 89.71, -454539.90, 0.0381, 0.6),
	("GRID#2",         5.2, 34.56, 38.93, -266045.09, 0.0445, 0.5),
	("CONNECTIONS#0", 22.4, 22.03, 39.48,  -42195.02, 0.0479, 0.3),
	("CONNECTIONS#1",  0.4, 36.10, 47.63, -162713.12, 0.0413, 0.6),
	("CONNECTIONS#2",  6.0, 26.77, 35.12, -205989.16, 0.0472, 0.4),
	("MEMORY#0",       8.4, 21.20, 25.83, -226324.99, 0.0506, 0.4),
	("MEMORY#1",      16.4, 22.86, 40.70,  -39921.60, 0.0457, 0.3),
	("MEMORY#2",       0.0, 86.33, 87.88, -158382.17, 0.0058, 0.0),
]

S16 = dict(weight_err_sq=0.25, weight_stable=0.20, weight_jerk=0.15,
           weight_mono=0.05, weight_steady=0.35)


def _metrics():
	return [SimpleNamespace(stable_rate=s / 100, mean_attitude_error_deg=e,
	                        mean_steady_error_deg=d, reward=r, motor_jerk_mean=j,
	                        mono_violations_total=mo, mean_effort=None,
	                        mean_altitude_error_m=None, mean_position_error_m=None)
	        for _k, s, e, d, r, j, mo in ARM9]


def _winner(scores):
	return ARM9[scores.index(min(scores))][0]


def test_harmonic_reproduces_arm9_headline():
	"""The WHM path is unchanged: it still picks the steady-specialist."""
	calc = FitnessCalculatorControllerHarmonic(**S16, aggregation="harmonic")
	assert _winner(calc.fitness(_metrics())) == "MEMORY#0"


def test_arithmetic_flips_arm9_to_the_all_rounder():
	"""Four weighted losses outweigh one weighted win under arithmetic."""
	calc = FitnessCalculatorControllerHarmonic(**S16, aggregation="arithmetic")
	assert _winner(calc.fitness(_metrics())) == "CONNECTIONS#0"


def test_arithmetic_is_the_weighted_mean_of_ranks():
	"""Pin the formula itself, not just the argmin."""
	ms = _metrics()
	calc = FitnessCalculatorControllerHarmonic(**S16, aggregation="arithmetic")
	got = calc.fitness(ms)
	ranks = {
		0.25: compute_ranks([-m.reward for m in ms], ascending=True),
		0.20: compute_ranks([m.stable_rate for m in ms], ascending=False),
		0.15: compute_ranks([m.motor_jerk_mean for m in ms], ascending=True),
		0.05: compute_ranks([m.mono_violations_total for m in ms], ascending=True),
		0.35: compute_ranks([m.mean_steady_error_deg for m in ms], ascending=True),
	}
	for i in range(len(ms)):
		expected = sum(w * r[i] for w, r in ranks.items()) / sum(ranks)
		assert got[i] == pytest.approx(expected)


def test_specialist_forgiveness_is_harmonic_only():
	"""The defining difference, isolated: rank-1-somewhere + dead-last-elsewhere
	beats balanced under harmonic, loses under arithmetic. Two equally weighted
	metrics, three genomes; the specialist is best on A and worst on B."""
	genomes = [
		SimpleNamespace(reward=-1.0, stable_rate=0.00),   # specialist: best reward, worst stable
		SimpleNamespace(reward=-2.0, stable_rate=0.50),   # balanced
		SimpleNamespace(reward=-3.0, stable_rate=0.99),   # specialist the other way
	]
	kw = dict(weight_err_sq=0.5, weight_stable=0.5)
	harm = FitnessCalculatorControllerHarmonic(**kw, aggregation="harmonic").fitness(genomes)
	arit = FitnessCalculatorControllerHarmonic(**kw, aggregation="arithmetic").fitness(genomes)
	# harmonic: the balanced genome (ranks 2,2) scores WORSE than both specialists (1,3)
	assert harm[1] > harm[0] and harm[1] > harm[2]
	# arithmetic: all three tie at 2.0 — a rank-1 no longer buys forgiveness
	assert arit[0] == arit[1] == arit[2] == pytest.approx(2.0)


def test_zscore_picks_the_all_rounder_outright():
	"""End-to-end through the wheel: the magnitude-aware combine both flips the
	headline AND breaks the CONN#0/MEM#1 tie that pure ranks could not."""
	calc = FitnessCalculatorControllerHarmonic(**S16, aggregation="zscore")
	scores = calc.fitness(_metrics())
	assert _winner(scores) == "CONNECTIONS#0"
	conn0 = scores[[k for k, *_ in ARM9].index("CONNECTIONS#0")]
	mem1 = scores[[k for k, *_ in ARM9].index("MEMORY#1")]
	assert conn0 < mem1                        # separated, not tied
	assert conn0 == pytest.approx(-0.880, abs=5e-3)   # pinned to the 19/08 session


def test_zscore_name_is_zrank():
	calc = FitnessCalculatorControllerHarmonic(**S16, aggregation="zscore")
	assert calc.name.startswith("ZRank(")


def test_invalid_clamp_refused():
	with pytest.raises(ValueError):
		FitnessCalculatorControllerHarmonic(aggregation="zscore", zrank_clamp=0.0)


def test_default_stays_harmonic():
	"""The in-stage GA builds the calculator without the argument until the
	sweep's round 2 lands; the default must not move under it."""
	assert FitnessCalculatorControllerHarmonic().aggregation == "harmonic"


def test_name_reveals_the_aggregation():
	"""Two runs with identical weights but different combine steps select
	different genomes — the fitness label must not let them look identical."""
	h = FitnessCalculatorControllerHarmonic(**S16, aggregation="harmonic")
	a = FitnessCalculatorControllerHarmonic(**S16, aggregation="arithmetic")
	assert h.name.startswith("ControllerHarmonic(")
	assert a.name.startswith("ControllerArithRank(")
	assert h.name.split("(", 1)[1] == a.name.split("(", 1)[1]   # same weights printed


def test_unknown_aggregation_refused():
	with pytest.raises(ValueError):
		FitnessCalculatorControllerHarmonic(aggregation="geometric")
