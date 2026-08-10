"""Thermometer thresholds must calibrate on the regime the controller FLIES.

Regression cover for the 09/08/2026 finding: fit_thresholds_from_pid_rollouts
hardcoded a 30-degree initial-tilt rollout config while every production recipe
flies --tilt 5.0, so the quantile-fitted thermometer spent resolution on states
the controller never visits and coarsened the near-zero region where the hold
floor lives. The fix threads the run's real EpisodeConfig through; None keeps the
legacy config so the ~60 other call sites are unchanged.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

from wnn.control.evaluator import (ControllerSpec, EpisodeConfig,
                                   fit_thresholds_from_pid_rollouts)


def _spec():
	return ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
	                      input_window_k=1, state_neurons=0, state_bits_per_neuron=0,
	                      output_bits_per_neuron=24, memory_mode="BINARY")


def _cfg(tilt_deg):
	return EpisodeConfig(dt=0.001, steps_per_episode=400,
	                     max_initial_tilt_rad=math.radians(tilt_deg),
	                     max_initial_yaw_rad=math.radians(tilt_deg),
	                     max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)


def test_operating_regime_changes_the_thresholds():
	"""A 5-degree fit must not equal the legacy 30-degree fit — otherwise the
	parameter is inert and the whole finding is unfalsifiable."""
	spec = _spec()
	legacy = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=11)
	tilt5 = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=11,
	                                         episode_config=_cfg(5.0))
	assert len(legacy) == len(tilt5)
	assert legacy != tilt5, "episode_config had no effect on the fitted thresholds"


def test_narrow_regime_gives_finer_near_zero_resolution():
	"""The point of the fix: calibrating on the flown regime must CONCENTRATE
	thresholds near zero, not merely change them. Measured per feature as the
	spread of its threshold ladder."""
	spec = _spec()
	wide = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=11,
	                                        episode_config=_cfg(30.0))
	narrow = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=11,
	                                          episode_config=_cfg(5.0))
	b = spec.bits_per_feature
	nfeat = len(wide) // b
	tighter = 0
	for f in range(nfeat):
		w = wide[f * b:(f + 1) * b]
		n = narrow[f * b:(f + 1) * b]
		if (max(n) - min(n)) < (max(w) - min(w)):
			tighter += 1
	assert tighter > nfeat // 2, (
		f"only {tighter}/{nfeat} features tightened their ladder on the narrow "
		f"regime — the calibration is not tracking the operating distribution")


def test_none_keeps_the_legacy_config():
	"""The ~60 legacy call sites must be bit-identical to before the change."""
	spec = _spec()
	a = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=5)
	legacy_equivalent = EpisodeConfig(
		dt=0.001, steps_per_episode=2000,          # the hardcoded default's LENGTH too —
		max_initial_tilt_rad=math.radians(30.0),   # episode length changes the settled
		max_initial_yaw_rad=math.radians(30.0),    # fraction, hence the quantiles
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3)
	b = fit_thresholds_from_pid_rollouts(spec, num_episodes=3, seed=5,
	                                     episode_config=legacy_equivalent)
	assert a == b, "the None default no longer reproduces the legacy 30-degree fit"
