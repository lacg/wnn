"""Pre-registered tail-aware headline selection (docs/ctrl_select_tail_prereg.md).

  --val-units N        stage-select val units (default 5 = the pre-flag list)
  --select-tail p99    tail = p99 (numpy inverted_cdf) of pooled per-episode err, diverged = +inf

Unit tests only (no wheel, no GPU). The end-to-end check is a --recalc-headline on a
saved checkpoint (see the commit message for the command).

Run:  PYTHONPATH=src/wnn:src python tests/controller_select_tail_p99.py
"""

from __future__ import annotations

import math
from types import SimpleNamespace as ns

import numpy as np

R = math.radians
# The a45 rf_c200_dres s1 run (logs/controller/ctrl48/...dres_full_s1.out, [seeds] line).
A45 = ns(train=4196112846, test=3412990376, val=163226054)
REPORT = list(range(99990101, 99990136))


def _rows(errs_deg: list, diverged: tuple = (), genome: int = 0) -> list:
	"""score_with_episodes rows: [genome, episode, stable, diverged, mean_err_rad]."""
	return [[genome, i, 1.0, 1.0 if i in diverged else 0.0, R(e)] for i, e in enumerate(errs_deg)]


def _refuses(fn, *a) -> bool:
	try:
		fn(*a)
	except SystemExit:
		return True
	return False


# ---------------------------------------------------------------- p99
def test_p99_known_arrays():
	from wnn.control.select_tail import p99_deg
	# inverted_cdf = sorted[ceil(0.99 n) - 1]: n=100 -> 99th value, n=500 -> 495th, n=2000 -> 1980th.
	assert math.isclose(p99_deg(_rows(list(range(1, 101)))), 99.0)
	assert math.isclose(p99_deg(_rows(list(range(1, 501)))), 495.0)
	assert math.isclose(p99_deg(_rows(list(range(1, 2001)))), 1980.0)
	assert math.isclose(p99_deg(_rows([float(x) for x in range(2000, 0, -1)])), 1980.0)  # order-free
	# Every value is an episode actually flown (no interpolation).
	e = list(np.random.default_rng(0).uniform(0.5, 3.0, 2000))
	assert any(math.isclose(p99_deg(_rows(e)), x, rel_tol=1e-12) for x in e)
	print("✓ P1 p99 = inverted_cdf order statistic on known arrays (100 / 500 / 2000 episodes)")


def test_p99_divergence_is_inf():
	from wnn.control.select_tail import episode_errors_deg, p99_deg
	base = [1.0] * 2000
	# 20 diverged of 2000: the 1980th value is still finite; 21 -> +inf.
	assert math.isclose(p99_deg(_rows(base, diverged=tuple(range(20)))), 1.0)
	assert p99_deg(_rows(base, diverged=tuple(range(21)))) == math.inf
	# A diverged row with a small err is +inf regardless of its err field.
	assert episode_errors_deg(_rows([0.1], diverged=(0,)))[0] == math.inf
	# A non-finite err on a non-diverged row is +inf, never NaN.
	rows = _rows([1.0] * 99) + [[0, 99, 1.0, 0.0, float("nan")]]
	errs = episode_errors_deg(rows)
	assert errs[-1] == math.inf and not np.isnan(errs).any()
	# All diverged: +inf (numpy linear would give NaN here — the reason for inverted_cdf).
	assert p99_deg(_rows([1.0] * 100, diverged=tuple(range(100)))) == math.inf
	print("✓ P2 diverged / non-finite -> +inf; p99 turns +inf exactly at > 1% diverged; no NaN")


def test_p99_pools_units_and_combines_like_near_fail():
	from wnn.control.select_tail import (genome0_rows, tail_cell, tail_rank_winner, tail_value,
	                                     tail_count, near_fail_count)
	args = ns(select_tail="p99", select_tail_deg=4.0)
	# Pooled, not per unit: two units of 100, the bad episodes split 2 + 1.
	u1 = [{"episodes": _rows([1.0] * 98 + [9.0, 7.0])}]
	u2 = [{"episodes": _rows([1.0] * 99 + [8.0]) + _rows([50.0] * 100, genome=1)}]
	pool = genome0_rows(u1, "u1") + genome0_rows(u2, "u2")
	assert len(pool) == 200                               # genome 1 dropped
	assert math.isclose(tail_value(args, pool), 7.0)      # sorted[ceil(198) - 1] of 200
	assert tail_cell(args, 7.0) == "p99=7.00°"
	# near_fail on the pool == the per-unit sum it always was.
	nf = ns(select_tail="near_fail", select_tail_deg=4.0)
	assert tail_value(nf, pool) == tail_count(u1, 4.0, "u1") + tail_count(u2, 4.0, "u2") == 3
	assert near_fail_count(pool, 4.0) == 3 and tail_cell(nf, 3) == "tail=3"
	# Combination: the SAME rank sum with float tails (+inf ties with +inf).
	labels = ["G#0", "N#0", "N#1", "M#0"]
	fits = {"G#0": 1.0, "N#0": 1.1, "N#1": 1.2, "M#0": 3.0}
	same = {l: 2.5 for l in labels}
	assert tail_rank_winner(labels, fits, same)[0] == "G#0"          # equal tails: today's pick
	p99 = {"G#0": 4.8, "N#0": 2.1, "N#1": 2.0, "M#0": math.inf}
	win, comb = tail_rank_winner(labels, fits, p99)
	assert comb == {"G#0": 1 + 3, "N#0": 2 + 2, "N#1": 3 + 1, "M#0": 4 + 4}, comb
	assert win == "G#0"                                              # 3-way tie -> lowest fit
	# Two diverged-tail candidates tie at +inf (rank 3.5 each); the clean one flips the pick.
	p99b = {"G#0": math.inf, "N#0": math.inf, "N#1": 1.0, "M#0": 2.0}
	win, comb = tail_rank_winner(labels, fits, p99b)
	assert win == "N#1" and comb == {"G#0": 4.5, "N#0": 5.5, "N#1": 4.0, "M#0": 6.0}, comb
	print("✓ P3 p99 pools all units (genome 0), near_fail == per-unit sum, rank sum + ties")


def test_tokens_and_validation():
	from wnn.control.select_tail import select_tail_token, validate_select_tail
	from wnn.control.val_units import validate_val_units
	assert select_tail_token(ns(select_tail="off", select_tail_deg=4.0)) == "off"
	assert select_tail_token(ns(select_tail="off", select_tail_deg=4.0, val_units=5)) == "off"
	assert select_tail_token(ns(select_tail="near_fail", select_tail_deg=3.5)) == "near_fail:3.5"
	assert select_tail_token(ns(select_tail="p99", select_tail_deg=4.0, val_units=20)) == "p99,val_units=20"
	assert select_tail_token(ns(select_tail="off", select_tail_deg=4.0, val_units=20)) == "off,val_units=20"
	assert not _refuses(validate_select_tail, ns(select_tail="p99", select_tail_deg=4.0))
	assert _refuses(validate_select_tail, ns(select_tail="p95", select_tail_deg=4.0))
	assert _refuses(validate_val_units, ns(val_units=0))
	assert not _refuses(validate_val_units, ns(val_units=20))
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert {"controller_val_units", "controller_select_tail"} <= set(KNOWN_PARAMS)
	print("✓ T  provenance tokens (default line unchanged), validation, KNOWN_PARAMS")


# ---------------------------------------------------------------- val units
def test_val_units_default_is_todays_list():
	from wnn.control.val_units import checked_val_unit_seeds
	todays = [int(A45.val) + i for i in range(5)]          # the pre-flag expression, verbatim
	args = ns(val_units=5, report_seeds=REPORT, num_eval_folds=5)
	assert checked_val_unit_seeds(args, A45) == todays == list(range(163226054, 163226059))
	assert checked_val_unit_seeds(ns(report_seeds=REPORT, num_eval_folds=5), A45) == todays
	print(f"✓ V1 --val-units 5 (and absent) == today's list {todays[0]}..{todays[-1]}")


def test_val_units_20_distinct_and_disjoint():
	from wnn.control.evaluator import fold_pool_seed
	from wnn.control.val_units import checked_val_unit_seeds
	args = ns(val_units=20, report_seeds=REPORT, num_eval_folds=5)
	u = checked_val_unit_seeds(args, A45)
	assert len(u) == len(set(u)) == 20 and u[:5] == [int(A45.val) + i for i in range(5)]
	assert not set(u) & (set(REPORT) | {A45.train, A45.test})
	pools = lambda ss: {fold_pool_seed(s, k) for s in ss for k in range(5)} | set(ss)
	assert not pools(u) & (pools(REPORT) | pools([A45.train, A45.test]))
	print(f"✓ V2 --val-units 20: 20 distinct {u[0]}..{u[-1]}, disjoint from 35 report units + "
	      "train/test (raw + K=5 pool seeds), first 5 = today's")


def test_val_units_overlap_refused():
	from wnn.control.val_units import checked_val_unit_seeds
	clash_report = ns(train=1, test=2, val=99990090)       # +11 reaches 99990101
	assert _refuses(checked_val_unit_seeds, ns(val_units=20, report_seeds=REPORT,
	                                           num_eval_folds=5), clash_report)
	assert not _refuses(checked_val_unit_seeds, ns(val_units=5, report_seeds=REPORT,
	                                               num_eval_folds=5), clash_report)
	clash_train = ns(train=500, test=2, val=490)
	assert _refuses(checked_val_unit_seeds, ns(val_units=20, report_seeds=None, report_seed=None,
	                                           num_eval_folds=5), clash_train)
	clash_single = ns(train=1, test=2, val=700)
	assert _refuses(checked_val_unit_seeds, ns(val_units=5, report_seeds=None, report_seed=702,
	                                           num_eval_folds=5), clash_single)
	print("✓ V3 overlap with report units / train / a single --report-seed is REFUSED")


if __name__ == "__main__":
	test_p99_known_arrays()
	test_p99_divergence_is_inf()
	test_p99_pools_units_and_combines_like_near_fail()
	test_tokens_and_validation()
	test_val_units_default_is_todays_list()
	test_val_units_20_distinct_and_disjoint()
	test_val_units_overlap_refused()
	print("ALL PASS")
