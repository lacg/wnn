"""Every controller fitness weight must REACH the calculator (18/08/2026).

This guards a bug class, not a bug. It has now shipped twice:

  01/06/2026  weight_jerk / weight_mono were accepted and ignored, because the
              METRIC arrived None (see test_jerk_mono_plumbing.py).
  17/08/2026  weight_alt / weight_pos were accepted and ignored, because the
              WEIGHT never arrived: FitnessCalculatorControllerHarmonic gained
              the two rank dimensions, but the three config layers that build it
              (StrategyConfig -> FitnessCalculatorFactory, and the GA/TS builders
              that populate StrategyConfig) were never extended.

The second one cost two runs of the alt-weight sweep — ~5.5 h — and was almost
invisible: only the stage-select site passes the weight directly, so only IT
warned. The GA and TS search paths dropped it in total silence, and the arm came
out bit-identical to its own alt=0.00 control.

So the tests below are deliberately TABLE-DRIVEN over the full weight list
rather than spot-checks of alt/pos. A weight added to the calculator without
being plumbed fails here on the day it is added.
"""

import math

import pytest

from wnn.control.ga_strategy import default_controller_ga_config
from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.fitness.FitnessCalculatorControllerHarmonic import (
	FitnessCalculatorControllerHarmonic,
)


# (builder kwarg / config suffix / calculator attribute) — all three share the
# stem, which is the naming contract this test also pins.
CONTROLLER_WEIGHTS = ["err_sq", "stable", "jerk", "mono", "steady", "effort", "alt", "pos"]


def test_calculator_exposes_every_weight():
	"""The stem list above IS the calculator's surface — no more, no less."""
	calc = FitnessCalculatorControllerHarmonic()
	for stem in CONTROLLER_WEIGHTS:
		assert hasattr(calc, f"weight_{stem}"), f"calculator lost weight_{stem}"


@pytest.mark.parametrize("stem", CONTROLLER_WEIGHTS)
def test_ga_builder_forwards_weight_to_calculator(stem):
	"""default_controller_ga_config -> GAConfig -> factory -> calculator.

	The value 0.37 is arbitrary but non-default and non-zero: a weight that is
	silently dropped comes back 0.0 (or 1.0 for err_sq), never 0.37.
	"""
	# `stable` rides along so the harmonic calculator is always the one built:
	# err_sq on its own stays single-objective BY DESIGN (see the test below),
	# and FitnessCalculatorController carries no weight attributes to check.
	weights = {"weight_stable": 0.11, f"weight_{stem}": 0.37}
	cfg = default_controller_ga_config(**weights)
	assert getattr(cfg, f"fitness_weight_{stem}") == pytest.approx(0.37), \
		f"builder did not set fitness_weight_{stem} on the config"
	calc = cfg.create_fitness_calculator()
	assert getattr(calc, f"weight_{stem}") == pytest.approx(0.37), \
		f"config did not forward weight_{stem} through the factory"


@pytest.mark.parametrize("stem", [s for s in CONTROLLER_WEIGHTS if s != "err_sq"])
def test_any_nonzero_weight_selects_the_harmonic_calculator(stem):
	"""multi_obj gating must know about every dimension.

	err_sq is excluded: it is the single-objective default, so on its own it
	must NOT flip the calculator. Every OTHER weight is a rank dimension that
	only the harmonic calculator can honour — if the gate does not list it, the
	run silently ranks on reward alone and the weight does nothing.
	"""
	cfg = default_controller_ga_config(**{f"weight_{stem}": 0.37})
	assert cfg.fitness_calculator_type == FitnessCalculatorType.CONTROLLER_HARMONIC, \
		f"weight_{stem} > 0 did not switch to CONTROLLER_HARMONIC"


def test_err_sq_alone_stays_single_objective():
	cfg = default_controller_ga_config(weight_err_sq=1.0)
	assert cfg.fitness_calculator_type == FitnessCalculatorType.CONTROLLER


def test_ts_builder_forwards_every_weight():
	"""_build_ts_config mirrors the GA builder — same weights, same config."""
	from types import SimpleNamespace

	from wnn.control.phased_ga import _build_ts_config

	args = SimpleNamespace(
		pop=6, check_interval=2, magnitude_aware_patience=False,
		**{f"fit_weight_{s}": 0.37 for s in CONTROLLER_WEIGHTS})
	tscfg = _build_ts_config(args, gens=3, patience=2)
	for stem in CONTROLLER_WEIGHTS:
		assert getattr(tscfg, f"fitness_weight_{stem}") == pytest.approx(0.37), \
			f"_build_ts_config dropped fit_weight_{stem}"
	assert tscfg.fitness_calculator_type == FitnessCalculatorType.CONTROLLER_HARMONIC


def test_label_names_alt_even_at_zero():
	"""The fitness label is how an arm is audited after the fact.

	`alt` prints unconditionally, including 0.00, because for THIS dimension zero
	and absent do not mean the same thing: a 0.00 arm is a deliberate control in
	the alt-weight sweep, and hiding the zero makes it indistinguishable from a
	run whose alt weight never reached the calculator — the exact 18/08 failure
	this label should have exposed and did not.
	"""
	control = FitnessCalculatorControllerHarmonic(weight_err_sq=0.4, weight_stable=0.3,
	                                              weight_alt=0.0)
	weighted = FitnessCalculatorControllerHarmonic(weight_err_sq=0.4, weight_stable=0.3,
	                                               weight_alt=0.1)
	assert "alt=0.0" in control.name, f"control must show its zero: {control.name}"
	assert "alt=0.1" in weighted.name, f"weighted arm must show its weight: {weighted.name}"
	assert control.name != weighted.name, \
		"an alt=0.00 control and an alt=0.10 arm must not print the same fitness"


def test_label_hides_pos_while_it_is_inert():
	"""pos stays conditional until --xy-offset > 0 makes it a live dimension."""
	calc = FitnessCalculatorControllerHarmonic(weight_err_sq=1.0, weight_pos=0.0)
	assert "pos=" not in calc.name
	assert "pos=0.2" in FitnessCalculatorControllerHarmonic(weight_err_sq=1.0,
	                                                        weight_pos=0.2).name


class _M:
	"""Minimal metrics stand-in carrying only what the calculator ranks."""

	def __init__(self, err, alt):
		self.reward = -err
		self.mean_attitude_error_deg = err
		self.mean_steady_error_deg = err
		self.acc = 0.0
		self.stable_rate = 0.0
		self.motor_jerk_mean = 0.0
		self.mono_violations_total = 0.0
		self.mean_effort = 0.0
		self.mean_altitude_error_m = alt
		self.mean_position_error_m = alt


def test_alt_weight_actually_changes_the_ranking():
	"""Plumbing is necessary but not sufficient — the weight must MOVE the order.

	Two genomes that disagree on both axes: A flies well but drifts vertically,
	B holds altitude while tumbling. With no altitude weight A wins on attitude;
	with altitude dominant the order must invert. If it does not, the weight is
	arriving but not being ranked.
	"""
	a, b = _M(err=11.0, alt=4.2), _M(err=52.0, alt=0.2)

	attitude_only = FitnessCalculatorControllerHarmonic(weight_err_sq=1.0)
	fa, fb = attitude_only.fitness([a, b])
	assert fa < fb, "without an altitude weight the better-flying genome must win"

	altitude_heavy = FitnessCalculatorControllerHarmonic(weight_err_sq=0.05, weight_alt=0.95)
	fa2, fb2 = altitude_heavy.fitness([a, b])
	assert fb2 < fa2, "a dominant altitude weight must invert the ranking"


def test_alt_weight_is_ignored_loudly_when_the_metric_is_missing():
	"""The guard that DID fire in the sweep — keep it firing.

	A None altitude must warn rather than crash or silently rank, because a
	missing metric means the run cannot answer the question it was launched to
	answer.
	"""
	a, b = _M(err=11.0, alt=None), _M(err=52.0, alt=None)
	calc = FitnessCalculatorControllerHarmonic(weight_err_sq=1.0, weight_alt=0.95)
	with pytest.warns(RuntimeWarning, match="weight_alt > 0"):
		fa, fb = calc.fitness([a, b])
	assert fa < fb, "with altitude dropped the ranking must fall back to attitude"
