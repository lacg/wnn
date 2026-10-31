"""ABI 32 (31/10/2026): the real-life plant flags of phased_ga.

Luiz: with --translation the specific-force IMU, rotor drag and the estimator-fed
attitude features are the DEFAULTS; the legacy flags reproduce ABI 31; a
specific-force run without the estimator is refused unless forced. Every new
EpisodeConfig kwarg helper is EMPTY when off (byte-identical Rust calls).

Run: PYTHONPATH=src python -m pytest tests/test_abi32_plant_flags.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control import phased_ga as P
from wnn.control.training import EpisodeConfig

BASE = ["--airframe", "cf21_brushless", "--obs-peraxis-p"]


def _args(extra):
	return P.build_arg_parser().parse_args([*BASE, *extra])


def test_translation_defaults_to_the_real_life_plant():
	a = _args(["--translation"])
	P.resolve_abi32_plant(a)
	assert (a.imu_specific_force, a.rotor_drag, a.obs_att_estimator) == (True, P.DEFAULT_ROTOR_DRAG, True)
	P.resolve_abi32_plant(a)  # idempotent
	assert (a.imu_specific_force, a.rotor_drag, a.obs_att_estimator) == (True, P.DEFAULT_ROTOR_DRAG, True)


def test_legacy_flags_reproduce_abi31_and_attitude_only_is_untouched():
	a = _args(["--translation", "--imu-inclinometer", "--no-rotor-drag"])
	P.resolve_abi32_plant(a)
	assert (a.imu_specific_force, a.rotor_drag, a.obs_att_estimator) == (False, 0.0, False)
	b = _args([])
	P.resolve_abi32_plant(b)
	assert (b.imu_specific_force, b.rotor_drag, b.obs_att_estimator) == (False, 0.0, False)
	ec = P.episode_config_from_args(b)
	assert ec.imu_kwargs() == {} and ec.att_ref_kwargs() == {}


def test_refusals():
	with pytest.raises(SystemExit, match="blind|cannot see"):
		P.resolve_abi32_plant(_args(["--translation", "--no-obs-att-estimator"]))
	a = _args(["--translation", "--no-obs-att-estimator", "--force-blind-attitude"])
	P.resolve_abi32_plant(a)
	assert a.obs_att_estimator is False
	with pytest.raises(SystemExit, match="require --translation"):
		P.resolve_abi32_plant(_args(["--imu-specific-force"]))
	with pytest.raises(SystemExit, match="require --translation"):
		P.resolve_abi32_plant(_args(["--rotor-drag", "0.3"]))
	with pytest.raises(SystemExit, match="mutually exclusive"):
		P.resolve_abi32_plant(_args(["--translation", "--imu-specific-force", "--imu-inclinometer"]))


def test_episode_config_kwargs():
	ec = EpisodeConfig(translation=True, imu_specific_force=True, rotor_drag=0.3,
	                   max_initial_xy_offset_m=0.5, att_err_ref="tilt")
	assert ec.imu_kwargs() == {"imu_specific_force": True, "rotor_drag": 0.3}
	assert ec.att_ref_kwargs() == {"att_err_tilt_ref": True}
	with pytest.raises(ValueError):
		EpisodeConfig(translation=False, rotor_drag=0.3).imu_kwargs()
	with pytest.raises(ValueError):
		EpisodeConfig(translation=True, att_err_ref="tilt").att_ref_kwargs()
