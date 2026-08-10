"""Thermometer coverage: fit the ladder on what the STUDENT visits, not only what
the teacher demonstrates.

Two mismatches were measured on 10/08/2026 between the calibration distribution
and the operating one:

  PLANT   the fitter ran PID on a clean AttitudeSim() while every run flies L4C.
          Fixed (8077d176) — arming L4C widens the ladder 1.86x at 5deg ICs.
  POLICY  the fitter rolls out PID, a BETTER controller than the student, so the
          ladder under-covers the student's own excursions. DAgger covariate
          shift, living in the INPUT REPRESENTATION where training cannot fix it.
          That is what `extra_samples` (A) and `outer_quantile` (C) address.

Evidence these exist to prevent: calibrating at the flown 5deg tilt — which
removes the hardcoded 30deg's accidental coverage margin — made hold 2.9x WORSE
on s31337002 and 1.8x worse on s31337003, degrading even the GRID stage (1.39 ->
5.34), i.e. before any GA search runs.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control.evaluator import (ControllerSpec, EpisodeConfig,
                                   fit_thresholds_from_pid_rollouts)


def _spec(bits=8):
	return ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=bits,
	                      input_window_k=1, state_neurons=0, state_bits_per_neuron=0,
	                      output_bits_per_neuron=24, memory_mode="BINARY")


def _cfg(tilt=5.0, steps=300):
	return EpisodeConfig(dt=0.001, steps_per_episode=steps,
	                     max_initial_tilt_rad=math.radians(tilt),
	                     max_initial_yaw_rad=math.radians(tilt),
	                     max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)


def _fit(**kw):
	return fit_thresholds_from_pid_rollouts(_spec(), num_episodes=2, seed=3,
	                                        episode_config=_cfg(), **kw)


def _spans(t, bits=8):
	nf = len(t) // bits
	return [max(t[f * bits:(f + 1) * bits]) - min(t[f * bits:(f + 1) * bits])
	        for f in range(nf)]


# ---- C: coverage margin -------------------------------------------------------

def test_outer_quantile_widens_the_ladder():
	base, wide = _fit(), _fit(outer_quantile=0.02)
	sb, sw = sum(_spans(base)), sum(_spans(wide))
	assert sw > sb * 1.1, f"0.02 must reach further into the tails: {sb:.3f} -> {sw:.3f}"


def test_outer_quantile_is_monotone_in_coverage():
	"""Smaller outer quantile = more coverage. A non-monotone knob would be a trap."""
	spans = [sum(_spans(_fit(outer_quantile=q))) for q in (0.20, 0.10, 0.02)]
	assert spans[0] < spans[1] < spans[2], f"not monotone: {spans}"


def test_outer_quantile_none_is_a_no_op():
	assert _fit() == _fit(outer_quantile=None)


def test_thresholds_stay_sorted_under_coverage():
	"""A thermometer whose thresholds are not ascending encodes nonsense."""
	t = _fit(outer_quantile=0.02)
	b = _spec().bits_per_feature
	for f in range(len(t) // b):
		row = t[f * b:(f + 1) * b]
		assert row == sorted(row), f"feature {f} not ascending: {row}"


# ---- A: student states --------------------------------------------------------

def test_extra_samples_none_is_a_no_op():
	assert _fit() == _fit(extra_samples=None)


def test_extra_samples_must_be_a_real_fraction_to_matter():
	"""THE trap this test exists for: a handful of student samples is swamped by
	the teacher's thousands, so `extra_samples` silently does nothing. Measured —
	2 extreme values per feature moved the total span by 1.00x. The collector must
	contribute samples on the same ORDER as the teacher pool, or A is a placebo."""
	nf = len(_fit()) // _spec().bits_per_feature
	base = _fit()
	tiny = _fit(extra_samples=[[-5.0, 5.0] for _ in range(nf)])
	assert sum(_spans(tiny)) < sum(_spans(base)) * 1.02, \
		"a 2-sample injection should NOT move a quantile fit — if it does, the " \
		"pool is far smaller than assumed and the fit is fragile"
	# ...whereas a comparable-sized injection must move it.
	big = _fit(extra_samples=[[-5.0, 5.0] * 2000 for _ in range(nf)])
	assert sum(_spans(big)) > sum(_spans(base)) * 1.5, \
		f"a teacher-sized injection must widen the ladder: {sum(_spans(base)):.2f} " \
		f"-> {sum(_spans(big)):.2f}"


def test_extra_samples_shorter_than_features_is_tolerated():
	"""A collector that yields fewer feature rows (e.g. an aborted rollout) must
	degrade to the teacher-only fit for the missing ones, not crash mid-cohort."""
	t = _fit(extra_samples=[[-5.0, 5.0]])          # 1 row for 9+ features
	assert len(t) == len(_fit())


def test_a_and_c_compose():
	"""They target the same failure from different sides and must be usable together."""
	nf = len(_fit()) // _spec().bits_per_feature
	both = _fit(outer_quantile=0.02, extra_samples=[[-5.0, 5.0] * 2000 for _ in range(nf)])
	assert sum(_spans(both)) > sum(_spans(_fit(outer_quantile=0.02)))
