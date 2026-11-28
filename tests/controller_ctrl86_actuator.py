"""CTRL-86 S0-S2 (ABI 65): the x500_sitl airframe entry and the real-life plant's actuator
extras (asymmetric motor lag, armed-idle floor), Python side.

  * S0: x500_sitl carries the plan's §1a numbers (PX4 v1.16.0 / gz models e05f4312) and is
    NOT flyable yet — every flight path refuses it (no inner loop until S3/S4);
  * S1/S2 resolution: under --translation x500 resolves lag up 0.05 / down 0.10 s and idle
    0.15 from the airframe; cf21 resolves down 0 / idle 0 (bit-identical args); explicit
    flags win; refusals (down without a lag, either without translation, bad ranges);
  * EpisodeConfig.imu_kwargs() carries the extras only when set, so cf21 calls are
    byte-identical; the Python per-step loops refuse them; provenance actuator= is "" for
    cf21; the registry knows the keys and the flow adapter refuses them;
  * the ABI-65 wheel accepts the kwargs on RewardGatedConfigPacked and refuses bad values.

Run: PYTHONPATH=/tmp/ctrl86_site:src python tests/controller_ctrl86_actuator.py
"""

from __future__ import annotations

import math


def _close(a: float, b: float, tol: float, what: str):
	assert abs(a - b) <= tol, f"{what}: {a} vs {b} (tol {tol})"


def _raises(fn, exc, text: str, what: str):
	try:
		fn()
	except exc as e:
		assert text in str(e), f"{what}: wrong refusal: {e}"
		return
	raise AssertionError(f"{what}: not refused")


def test_x500_airframe_values():
	from wnn.control.airframe import Airframe
	a = Airframe.preset("x500_sitl")
	# The plan's §1a table (values printed there to their stated digits).
	_close(a.mass, 2.06431, 1e-5, "mass")
	assert a.inertia == (0.023839, 0.023942, 0.044000), a.inertia
	_close(a.arm_length, 0.348, 1e-6, "arm")
	_close(a.k_thrust, 8.54858, 1e-6, "k_thrust")
	_close(a.k_drag, 0.016, 1e-12, "k_drag")
	_close(a.thrust_to_weight, 1.6885, 1e-4, "T/W")
	_close(a.hover_pwm, 0.76956, 1e-5, "hover")
	_close(a.angular_authority, 124.8, 0.05, "authority")
	for text in ("v1.16.0", "e05f4312", "motorConstant", "momentConstant"):
		assert text in a.source, f"source misses {text!r}"
	assert "DERIVED" in a.inertia_source and "parallel-axis" in a.inertia_source
	# cf21 untouched.
	cf = Airframe.preset("cf21_brushless")
	assert (cf.mass, cf.k_thrust) == (0.0393, 0.2)
	print(f"✓ x500_sitl: m {a.mass:.6f} kg, I {a.inertia}, arm {a.arm_length:.6f} m, k {a.k_thrust:.5f} N, "
	      f"T/W {a.thrust_to_weight:.4f}, hover {a.hover_pwm:.5f}, authority {a.angular_authority:.2f} rad/s^2")


def test_x500_not_flyable():
	from wnn.control.airframe import Airframe, require_inner_loop
	from wnn.control.training import EpisodeConfig
	x5 = Airframe.preset("x500_sitl")
	_raises(lambda: require_inner_loop(x5), ValueError, "no inner loop", "require_inner_loop")
	ec = EpisodeConfig(airframe=x5)
	for name in ("airframe_kwargs", "sim_kwargs", "cascade_kwargs"):
		_raises(getattr(ec, name), ValueError, "cannot be flown yet", name)
	require_inner_loop(Airframe.preset("cf21_brushless"))
	assert EpisodeConfig(airframe=Airframe.preset("cf21_brushless")).airframe_kwargs()
	print("✓ x500_sitl refused on every flight path (airframe_kwargs / sim_kwargs / cascade_kwargs)")


def _resolved(airframe: str, extra: list) -> object:
	from wnn.control import phased_ga as pg
	args = pg.build_arg_parser().parse_args(["--airframe", airframe, "--translation", *extra])
	pg._resolve_motor_lag_args(args)
	return args


def test_resolution():
	from wnn.control import phased_ga as pg
	x = _resolved("x500_sitl", [])
	assert (x.motor_lag_s, x.motor_lag_down_s, x.pwm_min) == (0.05, 0.10, 0.15), vars(x)
	assert x.motor_lag_source == "airframe:px4_v1.16.0_gz_x500"
	assert x.pwm_min_source == "airframe:px4_v1.16.0_SIM_GZ_EC_MIN"
	assert pg.actuator_token(x) == "down=0.1s,idle=0.15"
	c = _resolved("cf21_brushless", [])
	assert (c.motor_lag_s, c.motor_lag_down_s, c.pwm_min) == (0.20, 0.0, 0.0)
	assert pg.actuator_token(c) == ""
	# Explicit flags win; an explicit up time never inherits the airframe's down time.
	assert _resolved("x500_sitl", ["--motor-lag-down-s", "0.2"]).motor_lag_down_s == 0.2
	assert _resolved("x500_sitl", ["--motor-lag-s", "0.08"]).motor_lag_down_s == 0.0
	assert _resolved("x500_sitl", ["--pwm-min", "0"]).pwm_min == 0.0
	assert _resolved("cf21_brushless", ["--motor-lag-down-s", "0.4", "--pwm-min", "0.1"]).pwm_min == 0.1
	# The legacy pair-returning resolver keeps its contract (scripts/compute_baselines.py).
	assert pg.resolve_motor_lag(True, "x500_sitl", None, False) == (0.05, "airframe:px4_v1.16.0_gz_x500")
	assert pg.resolve_motor_lag(True, "cf21_brushless", None, False) == (0.20, "airframe:grafe2026")
	# Refusals.
	_raises(lambda: _resolved("x500_sitl", ["--no-motor-lag", "--motor-lag-down-s", "0.1"]), SystemExit, "needs a motor lag", "down without lag")
	_raises(lambda: _resolved("x500_sitl", ["--pwm-min", "1.0"]), SystemExit, "[0, 1)", "floor 1")
	_raises(lambda: _resolved("x500_sitl", ["--motor-lag-down-s", "nan"]), SystemExit, "finite", "down nan")
	_raises(lambda: pg.resolve_pwm_min(False, "cf21_brushless", 0.1), SystemExit, "--translation", "floor w/o translation")
	_raises(lambda: pg.resolve_motor_lag_down(False, "cf21_brushless", 0.2, "explicit", 0.4), SystemExit, "--translation", "down w/o translation")
	# Without --translation nothing is airframe-bound.
	assert pg.resolve_pwm_min(False, "x500_sitl", None) == (0.0, "off")
	print("✓ resolution: x500 0.05/0.10 s + idle 0.15 (airframe), cf21 0/0, explicit flags win, refusals")


def test_episode_config_kwargs():
	from wnn.control.training import EpisodeConfig, refuse_python_abi32_plant
	assert EpisodeConfig(translation=True, rotor_drag=0.78, motor_lag_s=0.2).imu_kwargs() == {"rotor_drag": 0.78}
	ec = EpisodeConfig(translation=True, rotor_drag=0.12, motor_lag_s=0.05, motor_lag_down_s=0.10, pwm_min=0.15)
	assert ec.imu_kwargs() == {"rotor_drag": 0.12, "motor_lag_down_s": 0.10, "pwm_min": 0.15}
	_raises(EpisodeConfig(translation=False, motor_lag_s=0.05, pwm_min=0.15).imu_kwargs, ValueError, "translation", "no translation")
	_raises(EpisodeConfig(translation=True, motor_lag_down_s=0.1).imu_kwargs, ValueError, "motor_lag_s > 0", "down w/o lag")
	_raises(EpisodeConfig(translation=True, pwm_min=1.0).imu_kwargs, ValueError, "[0, 1)", "floor 1")
	_raises(lambda: refuse_python_abi32_plant(EpisodeConfig(pwm_min=0.15), "t"), NotImplementedError, "pwm_min", "python loop")
	refuse_python_abi32_plant(EpisodeConfig(), "t")
	print("✓ EpisodeConfig.imu_kwargs carries the extras only when set; Python loops refuse them")


def test_provenance_registry_adapter():
	from wnn.control.provenance import collect_provenance
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.control.flow_adapter import controller_spec_from_params
	line = collect_provenance("x", actuator="down=0.1s,idle=0.15").line()
	assert "actuator=down=0.1s,idle=0.15 " in line
	assert "actuator" not in collect_provenance("x").line()
	assert {"controller_motor_lag_down_s", "controller_pwm_min"} <= KNOWN_PARAMS
	for key in ("controller_motor_lag_down_s", "controller_pwm_min"):
		_raises(lambda: controller_spec_from_params({key: 0.1}), ValueError, "not built", key)
	print("✓ provenance actuator= (absent when off), KNOWN_PARAMS, flow adapter refuses the keys")


def test_wheel_accepts_and_refuses():
	import wnn.control._accel as ra
	assert int(ra.EXPECTED_ABI) == 65
	base = dict(translation=True, af_mass=2.064308, af_k_thrust=8.54858, motor_lag_s=0.05)
	cfg = ra.RewardGatedConfigPacked(**base, motor_lag_down_s=0.10, pwm_min=0.15)
	_close(cfg.motor_lag_down_s, 0.10, 1e-7, "getter down")
	_close(cfg.pwm_min, 0.15, 1e-7, "getter floor")
	assert ra.RewardGatedConfigPacked().motor_lag_down_s == 0.0
	_raises(lambda: ra.RewardGatedConfigPacked(**{**base, "motor_lag_s": 0.0}, motor_lag_down_s=0.1), ValueError, "motor_lag_down_s", "rust down w/o lag")
	_raises(lambda: ra.RewardGatedConfigPacked(pwm_min=0.15), ValueError, "translation", "rust floor w/o translation")
	_raises(lambda: ra.RewardGatedConfigPacked(**base, pwm_min=1.0), ValueError, "pwm_min", "rust floor 1")
	assert not math.isnan(cfg.pwm_min)
	print("✓ ABI-65 wheel: RewardGatedConfigPacked(motor_lag_down_s=, pwm_min=) + Rust refusals")


if __name__ == "__main__":
	test_x500_airframe_values()
	test_x500_not_flyable()
	test_resolution()
	test_episode_config_kwargs()
	test_provenance_registry_adapter()
	test_wheel_accepts_and_refuses()
	print("\nCTRL-86 S0-S2 Python side: all passed.")
