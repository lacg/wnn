"""ABI 60 (Luiz 19/11/2026): the mid-window scorer row / --refine-w-mid
(docs/ctrl58_a2_prereg.md, Amendment 19/11) — the Python side.

Pinned: default 0 = the pre-amendment token, payload and calculator; > 0 adds the cost column
(changes the ranking when only mean_mid_error_deg differs), refuses a missing value, stays out
of the stage-select calculator, prints on the report row only when present, and is in the
params registry; the facade expects ABI 60.
Needs the installed ram_controller at ABI 60. No GPU, no sim.
"""
from dataclasses import asdict, replace

import pytest

from wnn.control.refine_config import (RefineConfig, mid_armed_from_args, refine_token,
                                      refine_weights_record, validate_refine_args)
from wnn.control.refine_fitness import FitnessCalculatorControllerRefine
from wnn.ram.metrics import ControllerMetrics


def _args(*argv):
	from wnn.control.phased_ga import build_arg_parser
	return build_arg_parser().parse_args(list(argv))


def test_params_registry_and_facade_abi():
	from wnn.control import _accel
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert "controller_refine_w_mid" in KNOWN_PARAMS
	assert _accel.EXPECTED_ABI == 60 and _accel.ABI_VERSION == 60


# ---- --refine-w-mid ---------------------------------------------------------------------

def _m(steady=0.5, mid=None, err=1.0, pos=0.3, reward=-1.0):
	return ControllerMetrics(reward=reward, stable_rate=1.0, mean_attitude_error_deg=err,
	                         mean_steady_error_deg=steady, mean_position_error_m=pos,
	                         mean_mid_error_deg=mid)


def _calc(cfg, with_tail=True):
	return FitnessCalculatorControllerRefine(cfg, zrank_clamp=3.0, zrank_mad_floor=False,
	                                         zrank_pos_floor=0.0, pos_metric="pos", with_tail=with_tail)


def test_w_mid_default_is_the_pre_amendment_run():
	a = _args("--refine-fitness", "--refine-w-tail", "0")
	assert a.refine_w_mid == 0.0 and not mid_armed_from_args(a) and not mid_armed_from_args(_args())
	tok = refine_token(a)
	assert "mid" not in tok and tok.endswith(":pfloor=0.15") and ",pos0.1:" in tok
	rec = refine_weights_record(a)
	assert "w_mid" not in rec and rec == {k: v for k, v in asdict(RefineConfig(w_tail=0.0, search=True)).items() if k != "w_mid"}
	cfg = RefineConfig(w_tail=0.0)
	pool = [_m(0.5), _m(0.6, err=1.2, reward=-1.2), _m(0.7, err=0.9, reward=-0.9)]
	calc = _calc(cfg)
	assert calc.weight_mid == 0.0 and "mid=" not in calc.name
	# The column is absent: values in mean_mid_error_deg cannot move the fitness.
	with_mid = [replace(m, mean_mid_error_deg=0.1 * (i + 1)) for i, m in enumerate(pool)]
	assert calc.fitness(pool) == calc.fitness(with_mid)


def test_w_mid_token_record_and_arming():
	a = _args("--refine-fitness", "--refine-w-steady", "0.40", "--refine-w-mid", "0.25", "--refine-w-reward", "0.10")
	validate_refine_args(a)
	assert mid_armed_from_args(a)
	assert ":w=steady0.4,rw0.1,tail0.15,pos0.1,mid0.25:" in refine_token(a)
	assert refine_weights_record(a)["w_mid"] == 0.25
	# --refine-select alone never arms the search column.
	assert not mid_armed_from_args(_args("--refine-select", "--select-tail", "p99", "--refine-w-mid", "0.25"))
	with pytest.raises(SystemExit, match="refine-w"):
		validate_refine_args(_args("--refine-fitness", "--refine-w-mid", "-0.1"))


def test_w_mid_changes_the_ranking_when_only_mid_differs():
	cfg = RefineConfig(w_tail=0.0, w_steady=0.40, w_reward=0.10, w_mid=0.25)
	same = dict(steady=0.5, err=1.0, pos=0.3, reward=-1.0)
	pool = [_m(mid=0.9, **same), _m(mid=0.5, **same), _m(mid=0.7, **same)]
	f = _calc(cfg).fitness(pool)
	order = sorted(range(3), key=lambda i: f[i])
	rev = sorted(range(3), key=lambda i: -f[i])
	by_mid = [1, 2, 0]                                        # lowest mid first
	assert by_mid in (order, rev), f"fitness {f} does not order the pool by mid"
	assert len(set(f)) == 3
	# With the weight at 0 the same pool is a three-way tie.
	assert len(set(_calc(replace(cfg, w_mid=0.0)).fitness(pool))) == 1
	assert "mid=0.25" in _calc(cfg).name


def test_w_mid_refuses_a_missing_value_and_stays_out_of_stage_select():
	cfg = RefineConfig(w_tail=0.0, w_mid=0.25)
	with pytest.raises(ValueError, match="mean_mid_error_deg"):
		_calc(cfg).fitness([_m(mid=0.5), _m(mid=None, err=1.2)])
	with pytest.raises(ValueError, match="mean_mid_error_deg"):
		_calc(cfg).fitness([_m(mid=0.5), _m(mid=float("nan"), err=1.2)])
	select = _calc(cfg, with_tail=False)
	assert select.weight_mid == 0.0
	select.fitness([_m(mid=None), _m(mid=None, err=1.2, steady=0.6)])   # no mid needed, no refusal


def test_heldout_row_prints_mid_only_when_present():
	from wnn.control.phased_ga import _heldout_row_str
	off = _heldout_row_str(_m(steady=0.5))
	on = _heldout_row_str(_m(steady=0.5, mid=0.62))
	assert "mid=" not in off
	assert on == off.replace("  steady=0.50°", "  steady=0.50°  mid=0.62°") and "mid=0.62°" in on
