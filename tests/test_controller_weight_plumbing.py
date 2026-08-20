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


def _args_with_all_weights(**extra):
	"""An argparse-ish namespace carrying every controller weight at 0.37."""
	from types import SimpleNamespace
	base = dict(pop=6, check_interval=2, magnitude_aware_patience=False,
	            elitism=0.2, crossover_rate=0.7, immigrants=0.0)
	base.update({f"fit_weight_{s}": 0.37 for s in CONTROLLER_WEIGHTS})
	base.update(extra)
	return SimpleNamespace(**base)


@pytest.mark.parametrize("stem", CONTROLLER_WEIGHTS)
def test_ga_BUILDER_in_phased_ga_forwards_weight(stem):
	"""_build_ga_config is the path the run ACTUALLY takes — pin it directly.

	This is the test that was missing on 18/08. The library function
	default_controller_ga_config was verified by calling it directly with
	weight_alt=0.10, which proves the FUNCTION forwards the weight and says
	nothing about whether phased_ga passes it. It did not, so the GA stages —
	which are the search, since --strategy ts is not the default — stayed blind
	through an entire re-flown arm that came back bit-identical to the void run
	it was meant to replace, cell tallies and all.

	Assert on the calculator, not the config: the config field existing is not
	the same as the calculator receiving it.
	"""
	from wnn.control.phased_ga import _build_ga_config

	cfg = _build_ga_config(_args_with_all_weights(), gens=3, patience=2)
	calc = cfg.create_fitness_calculator()
	assert getattr(calc, f"weight_{stem}") == pytest.approx(0.37), \
		f"_build_ga_config dropped fit_weight_{stem} before the calculator"


@pytest.mark.parametrize("stem", CONTROLLER_WEIGHTS)
def test_grid_stage_ranks_on_every_weight(stem):
	"""The GRID stage builds its own calculator and was blind to alt/pos too.

	Its label is what prints on the "GRID WINNER (by ControllerHarmonic(...))"
	line, so a weight missing here is both a wrong ranking AND a log that
	truthfully reports the wrong ranking — which is exactly how it was misread
	as cosmetic on 18/08.
	"""
	from wnn.control.ga_strategy import default_controller_ga_config

	args = _args_with_all_weights()
	# Mirror controller_grid_search.py's construction exactly.
	calc = default_controller_ga_config(
		population_size=args.pop,
		weight_err_sq=args.fit_weight_err_sq, weight_stable=args.fit_weight_stable,
		weight_jerk=args.fit_weight_jerk, weight_mono=args.fit_weight_mono,
		weight_steady=args.fit_weight_steady, weight_effort=args.fit_weight_effort,
		weight_alt=args.fit_weight_alt, weight_pos=args.fit_weight_pos,
	).create_fitness_calculator()
	assert getattr(calc, f"weight_{stem}") == pytest.approx(0.37)


def test_every_args_weight_reaches_every_builder():
	"""Belt and braces: no builder may silently know fewer weights than argparse.

	Catches a NEW weight added to the CLI and wired into only some of the three
	stage builders — the shape of every occurrence of this bug so far.
	"""
	from wnn.control.phased_ga import _build_ga_config, _build_ts_config

	args = _args_with_all_weights()
	for name, cfg in (("GA", _build_ga_config(args, gens=3, patience=2)),
	                  ("TS", _build_ts_config(args, gens=3, patience=2))):
		calc = cfg.create_fitness_calculator()
		missing = [s for s in CONTROLLER_WEIGHTS
		           if getattr(calc, f"weight_{s}", None) != pytest.approx(0.37)]
		assert not missing, f"{name} builder dropped: {missing}"


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


# ---------------------------------------------------------------------------
# --fit-aggregation plumbing (19/08/2026). Same bug class, different knob: an
# aggregation the calculator understands but a builder does not forward is a
# silent no-op — and identical weights under different combines select
# DIFFERENT genomes (arm 9: MEMORY#0 vs CONNECTIONS#0), so a dropped flag
# would not merely mistune a run, it would run the WRONG EXPERIMENT while
# printing the right weights.
# ---------------------------------------------------------------------------

AGGREGATION_SITES = ["ga", "ts", "grid_config"]


@pytest.mark.parametrize("agg", ["harmonic", "arithmetic", "zscore"])
def test_ga_builder_forwards_aggregation_to_calculator(agg):
	from wnn.control.phased_ga import _build_ga_config
	cfg = _build_ga_config(_args_with_all_weights(fit_aggregation=agg, zrank_clamp=2.5),
	                       gens=3, patience=2)
	calc = cfg.create_fitness_calculator()
	assert calc.aggregation == agg, "_build_ga_config dropped fit_aggregation"
	assert calc.zrank_clamp == pytest.approx(2.5), "_build_ga_config dropped zrank_clamp"


@pytest.mark.parametrize("agg", ["harmonic", "arithmetic", "zscore"])
def test_ts_builder_forwards_aggregation_to_calculator(agg):
	from wnn.control.phased_ga import _build_ts_config
	tscfg = _build_ts_config(_args_with_all_weights(fit_aggregation=agg, zrank_clamp=2.5),
	                         gens=3, patience=2)
	calc = tscfg.create_fitness_calculator()
	assert calc.aggregation == agg, "_build_ts_config dropped fit_aggregation"
	assert calc.zrank_clamp == pytest.approx(2.5), "_build_ts_config dropped zrank_clamp"


def test_unset_flag_means_legacy_split():
	"""No --fit-aggregation = the banked behavior: harmonic IN-SEARCH (so the
	alt-weight round 2 replicates round 1 with only the seed changed) and
	arithmetic STAGE-SELECT (ratified 19/08 after the arm-9 specialist win)."""
	from wnn.control.ga_strategy import search_aggregation, select_aggregation
	from wnn.control.phased_ga import _build_ga_config
	args = _args_with_all_weights(fit_aggregation=None)
	assert search_aggregation(args) == "harmonic"
	assert select_aggregation(args) == "arithmetic"
	assert _build_ga_config(args, gens=3, patience=2).create_fitness_calculator() \
		.aggregation == "harmonic"


def test_set_flag_is_coherent_everywhere():
	"""--fit-aggregation set = ONE mode end-to-end (the fitness A/B contract)."""
	from wnn.control.ga_strategy import search_aggregation, select_aggregation
	args = _args_with_all_weights(fit_aggregation="zscore")
	assert search_aggregation(args) == select_aggregation(args) == "zscore"


def test_zscore_forces_multi_objective_even_err_only():
	"""The single-objective CONTROLLER type has no aggregation knob; selecting
	it under a non-default aggregation would silently ignore the flag."""
	cfg = default_controller_ga_config(weight_err_sq=1.0, aggregation="zscore")
	assert cfg.fitness_calculator_type == FitnessCalculatorType.CONTROLLER_HARMONIC
	assert cfg.create_fitness_calculator().aggregation == "zscore"


def test_cli_flag_reaches_the_builders():
	"""END-TO-END through argparse — the path the run takes starts at the CLI.
	SimpleNamespace tests above prove the builders forward the field; this one
	proves the field EXISTS on parsed args with the right name and choices."""
	from wnn.control.phased_ga import build_arg_parser
	args = build_arg_parser().parse_args(
		["--fit-aggregation", "zscore", "--zrank-clamp", "2.5"])
	assert args.fit_aggregation == "zscore"
	assert args.zrank_clamp == pytest.approx(2.5)
	default = build_arg_parser().parse_args([])
	assert default.fit_aggregation is None
	assert default.zrank_clamp == pytest.approx(3.0)
