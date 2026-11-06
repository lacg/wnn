"""ABI 35 item I (Luiz 05/11/2026): the AIRFRAME-BOUND motor-lag default.

Under --translation (the real-life plant) a run without --motor-lag-s flies its
airframe's measured lag (cf21_brushless 0.20 s, Gräfe 2026; cf2x_firmware 0.26 s,
Förster 2015); any other airframe is refused unless the lag is given; --motor-lag-s 0
and --no-motor-lag opt out; without --translation nothing changes. Every entry point
(phased_ga, the rival/rescore scripts through episode_config_from_args,
compute_baselines) goes through the ONE resolver.

Run: PYTHONPATH=src python -m pytest tests/test_motor_lag_default.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.control import phased_ga as P

BASE = ["--obs-peraxis-p"]


def _ec(extra):
	args = P.build_arg_parser().parse_args([*BASE, *extra])
	return args, P.episode_config_from_args(args)


def test_airframe_defaults_under_translation():
	a, ec = _ec(["--airframe", "cf21_brushless", "--translation"])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.20, "airframe:grafe2026")
	a, ec = _ec(["--airframe", "cf2x_firmware", "--translation"])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.26, "airframe:forster2015")
	assert ec.motor_lag_kwargs() == {"motor_lag_s": 0.26}


def test_explicit_override_and_opt_outs():
	a, ec = _ec(["--airframe", "cf21_brushless", "--translation", "--motor-lag-s", "0.30"])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.30, "explicit")
	for flag in (["--motor-lag-s", "0"], ["--no-motor-lag"], ["--no-motor-lag", "--motor-lag-s", "0"]):
		a, ec = _ec(["--airframe", "cf21_brushless", "--translation", *flag])
		assert (ec.motor_lag_s, a.motor_lag_source) == (0.0, "opt-out")
		assert ec.motor_lag_kwargs() == {}  # byte-identical Rust calls (lag off)
	with pytest.raises(SystemExit, match="contradicts"):
		_ec(["--airframe", "cf21_brushless", "--translation", "--no-motor-lag", "--motor-lag-s", "0.2"])
	with pytest.raises(SystemExit, match=">= 0"):
		_ec(["--airframe", "cf21_brushless", "--translation", "--motor-lag-s", "-0.1"])


def test_unknown_airframe_refused_unless_given():
	assert "mystery_quad" not in P.AIRFRAME_MOTOR_LAG_S
	with pytest.raises(SystemExit, match="no measured lag"):
		P.resolve_motor_lag(True, "mystery_quad", None, False)
	assert P.resolve_motor_lag(True, "mystery_quad", 0.15, False) == (0.15, "explicit")
	assert P.resolve_motor_lag(True, "mystery_quad", None, True) == (0.0, "opt-out")


def test_non_translation_unchanged():
	a, ec = _ec([])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.0, "off")
	a, ec = _ec(["--airframe", "cf21_brushless"])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.0, "off")
	a, ec = _ec(["--motor-lag-s", "0.15"])
	assert (ec.motor_lag_s, a.motor_lag_source) == (0.15, "explicit")


def test_idempotent_and_header():
	a, ec = _ec(["--airframe", "cf21_brushless", "--translation"])
	ec2 = P.episode_config_from_args(a)  # second resolve keeps the source
	assert (ec2.motor_lag_s, a.motor_lag_source) == (0.20, "airframe:grafe2026")
	assert "motor_lag=0.2s(airframe:grafe2026)" in P._plant_header(a, ec2)
