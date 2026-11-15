"""CTRL-48 block 10b (Luiz 15/11/2026): --fit-pos-metric {pos,hold} and the pos-only floor.

Pinned: the default ranks exactly as before (bit-identical fitness); `hold` ranks the
hold-window column with the same weight; a candidate with NO hold value (NaN: every
episode diverged before the window, or None) ranks LAST on the column and never reaches
the combine as a non-finite; a column absent from every metric is dropped as before;
`--zrank-pos-floor auto` follows the ranked column; the CLI and provenance carry it.
Needs the installed ram_controller (fitness_combine); no GPU.
"""
import math
from types import SimpleNamespace

import pytest

from wnn.ram.fitness import FitnessCalculatorControllerHarmonic as FC
from wnn.control.pos_floor import (POS_HOLD_SCALE_FLOOR_M, POS_SCALE_FLOOR_M,
                                   fit_pos_metric_from_args, zrank_pos_floor_from_args)

W = dict(weight_err_sq=0.2557, weight_stable=0.2045, weight_steady=0.358, weight_pos=0.1818,
         aggregation="zscore", zrank_clamp=3.0)


def _m(pos, hold, steady=0.5, reward=-2.0, stable=1.0):
	return SimpleNamespace(reward=reward, stable_rate=stable, mean_steady_error_deg=steady,
	                       mean_position_error_m=pos, mean_position_hold_m=hold, mean_attitude_error_deg=1.0)


POOL = [_m(0.340, 0.270, 0.45, -1.9), _m(0.335, 0.300, 0.50, -2.0), _m(0.345, 0.250, 0.55, -2.1),
        _m(0.338, 0.290, 0.48, -1.8), _m(0.342, 0.260, 0.52, -2.2)]


def test_default_is_the_legacy_ranking():
	legacy = FC(**W).fitness(POOL)
	assert FC(**W, pos_metric="pos").fitness(POOL) == legacy
	# and pos_metric=pos ignores the hold column entirely
	poisoned = [_m(m.mean_position_error_m, float("nan"), m.mean_steady_error_deg, m.reward) for m in POOL]
	assert FC(**W, pos_metric="pos").fitness(poisoned) == legacy


def test_hold_ranks_the_hold_column():
	hold = FC(**W, pos_metric="hold").fitness(POOL)
	# Rebuild the same pool with the hold values in the pos slot: identical ranking.
	swapped = [_m(m.mean_position_hold_m, None, m.mean_steady_error_deg, m.reward) for m in POOL]
	assert hold == FC(**W, pos_metric="pos").fitness(swapped)
	assert hold != FC(**W).fitness(POOL)
	assert "pos_hold=0.1818" in FC(**W, pos_metric="hold").name


@pytest.mark.parametrize("missing", [float("nan"), None])
def test_missing_hold_ranks_last_and_stays_finite(missing):
	pool = POOL[:4] + [_m(0.320, missing, 0.45, -1.9)]
	f = FC(**W, pos_metric="hold").fitness(pool)
	assert all(math.isfinite(x) for x in f)
	# Its pos term is the column's worst: identical to giving it a hold far beyond the pool.
	far = POOL[:4] + [_m(0.320, 0.300 + FC.HOLD_MISSING_PENALTY_M, 0.45, -1.9)]
	assert f == FC(**W, pos_metric="hold").fitness(far)
	# ...and NOT its whole-episode pos (0.320 would be the best pos in the pool).
	best = POOL[:4] + [_m(0.320, 0.240, 0.45, -1.9)]
	assert f[-1] > FC(**W, pos_metric="hold").fitness(best)[-1]


def test_all_missing_is_finite_and_neutral():
	pool = [_m(0.34, float("nan"), s, r) for s, r in ((0.4, -1.9), (0.5, -2.0), (0.6, -2.1))]
	f = FC(**W, pos_metric="hold").fitness(pool)
	assert all(math.isfinite(x) for x in f)
	no_pos = FC(**{**W, "weight_pos": 0.0}).fitness(pool)
	order = lambda v: sorted(range(len(v)), key=lambda i: v[i])
	assert order(f) == order(no_pos), "an all-missing column must not move the order"


def test_hold_column_absent_everywhere_is_dropped_with_a_warning():
	pool = [_m(0.34, None, s, r) for s, r in ((0.4, -1.9), (0.5, -2.0), (0.6, -2.1))]
	with pytest.warns(RuntimeWarning):
		f = FC(**W, pos_metric="hold").fitness(pool)
	assert f == FC(**{**W, "weight_pos": 0.0}).fitness(pool)


def test_floor_auto_follows_the_column():
	args = lambda **k: SimpleNamespace(**k)
	assert zrank_pos_floor_from_args(args()) == 0.0
	assert zrank_pos_floor_from_args(args(zrank_pos_floor="auto")) == POS_SCALE_FLOOR_M
	assert zrank_pos_floor_from_args(args(zrank_pos_floor="auto", fit_pos_metric="hold")) == POS_HOLD_SCALE_FLOOR_M
	assert zrank_pos_floor_from_args(args(zrank_pos_floor="0.002", fit_pos_metric="hold")) == 0.002
	assert fit_pos_metric_from_args(args()) == "pos"
	with pytest.raises(ValueError):
		FC(**W, pos_metric="whole")


def test_cli_and_provenance():
	from wnn.control.phased_ga import block10_tokens, build_arg_parser
	ap = build_arg_parser()
	off = ap.parse_args([])
	assert off.fit_pos_metric == "pos" and block10_tokens(off)["pos_metric"] == ""
	on = ap.parse_args(["--fit-pos-metric", "hold", "--zrank-pos-floor", "auto"])
	assert block10_tokens(on)["pos_metric"] == "hold"
	assert zrank_pos_floor_from_args(on) == POS_HOLD_SCALE_FLOOR_M
	from wnn.control.provenance import collect_provenance
	assert "pos_metric=hold" in collect_provenance("x", pos_metric="hold").line()
	assert "pos_metric=" not in collect_provenance("x").line()
