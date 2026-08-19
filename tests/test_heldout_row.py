"""The held-out report must carry EVERY metric the scorer measured.

This is a regression guard for a bug that has now happened four times: a metric is
computed by the Rust scorer, carried on Metrics, and never printed on the held-out
line, because each print site kept its own hand-written list of fields. steady was
lost until 05/08/2026, pos and alt until 14-15/08 (which left the alt-weight sweep's
pre-registered "rank by held-out altitude error" with nothing to rank), and jerk
until 19/08 — jerk being 20% of the C10 fitness and the term that won C10 its sweep.

The guard is deliberately aimed at the SHAPE of the failure, not at one field name:
every attribute in _HELDOUT_ROW must reach every held-out surface, and the aggregate
must return them all rather than a subset.
"""
import math
from types import SimpleNamespace

import pytest

from wnn.control.phased_ga import (
	_HELDOUT_ROW, _heldout_row_agg_str, _heldout_row_stats, _heldout_row_str,
)


def _full_metrics():
	"""A metrics object carrying every row field, with distinguishable values."""
	m = SimpleNamespace(acc=0.552, mean_attitude_error_deg=13.92, fitness=-764.59)
	for i, (attr, _label, _unit, _dp) in enumerate(_HELDOUT_ROW):
		setattr(m, attr, 1.0 + i)
	return m


def test_every_row_field_reaches_the_per_seed_line():
	"""No field may be silently dropped — this is the bug, stated directly."""
	line = _heldout_row_str(_full_metrics())
	for _attr, label, _unit, _dp in _HELDOUT_ROW:
		assert f"{label}=" in line, f"{label} measured but absent from the held-out line"


def test_jerk_specifically_is_reported():
	"""Named explicitly: jerk is 20% of C10 and was the field missing on 19/08."""
	assert "motor_jerk_mean" in [attr for attr, _l, _u, _d in _HELDOUT_ROW]
	assert "jerk=" in _heldout_row_str(_full_metrics())


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_unmeasured_metrics_are_omitted_not_zeroed(missing):
	"""A zero altitude reads as a genome holding height perfectly — the opposite
	of 'not measured'. Absent metrics must vanish from the line, never print 0."""
	m = _full_metrics()
	m.mean_altitude_error_m = missing
	line = _heldout_row_str(m)
	assert "alt=" not in line
	assert "alt=0" not in line
	assert "steady=" in line          # the others still report


def test_aggregate_line_carries_every_field():
	results = [_full_metrics(), _full_metrics()]
	agg = _heldout_row_agg_str(_heldout_row_stats(results))
	for _attr, label, _unit, _dp in _HELDOUT_ROW:
		assert f"{label}=" in agg, f"{label} absent from the MULTI-SEED line"


def test_aggregate_stats_mean_and_spread():
	a, b = _full_metrics(), _full_metrics()
	a.mean_steady_error_deg, b.mean_steady_error_deg = 15.0, 17.0
	stats = _heldout_row_stats([a, b])
	mean_v, sd_v = stats["mean_steady_error_deg"]
	assert mean_v == pytest.approx(16.0)
	assert sd_v == pytest.approx(1.0)          # population SD, matching the report


def test_single_result_has_zero_spread():
	stats = _heldout_row_stats([_full_metrics()])
	assert stats["mean_steady_error_deg"][1] == 0.0


def test_partially_measured_metric_uses_only_measured_seeds():
	"""One seed missing a field must not poison the other seeds' mean."""
	a, b = _full_metrics(), _full_metrics()
	a.motor_jerk_mean, b.motor_jerk_mean = 0.05, None
	stats = _heldout_row_stats([a, b])
	assert stats["motor_jerk_mean"][0] == pytest.approx(0.05)


def test_absent_everywhere_means_absent_from_stats():
	a, b = _full_metrics(), _full_metrics()
	a.mean_effort = b.mean_effort = None
	stats = _heldout_row_stats([a, b])
	assert "mean_effort" not in stats
	assert "effort=" not in _heldout_row_agg_str(stats)


def test_row_declaration_is_well_formed():
	"""Guards the declaration itself: unique attrs, unique labels, sane decimals."""
	attrs = [a for a, _l, _u, _d in _HELDOUT_ROW]
	labels = [l for _a, l, _u, _d in _HELDOUT_ROW]
	assert len(set(attrs)) == len(attrs)
	assert len(set(labels)) == len(labels)
	assert all(isinstance(d, int) and d >= 0 for _a, _l, _u, d in _HELDOUT_ROW)


def test_downstream_regexes_still_match():
	"""The round-2 ranking parses these lines with regexes; adding a field to the
	middle of the row must not break the ones already in use."""
	import re
	m = _full_metrics()
	head = f"stable={m.acc*100:.1f}% err={m.mean_attitude_error_deg:.2f}°{_heldout_row_str(m)}"
	for pattern in (r"stable=([0-9.]+)", r"err=([0-9.]+)", r"steady=([0-9.]+)", r"alt=([0-9.]+)"):
		assert re.search(pattern, head), f"{pattern} no longer matches the headline row"
