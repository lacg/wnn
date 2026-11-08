"""Rust <-> Python parity of the controller's input-feature table (CTRL-45).

feature_layout.py is the ONE Python name/family table; Rust
WnnController::compute_features is the ORDER authority. This holds them equal:

  1. COUNT — every combination of the 14 motor-level obs flags, plus the CTRL-38
     cascade blocks: WnnController.num_features() == len(feature_layout(spec)).
  2. POSITION — with every block on, one step on sentinel inputs, and the value
     read back from get_last_feature_vector() at the index the layout names must
     be the sentinel (base sensors, per-axis errors and integrals, the yaw anchor,
     the vertical and horizontal channels). The blocks whose first-step value is
     not a sentinel (obs_pwm, L1 d-hat, the cascade blocks) sit between / after
     pinned blocks, so their positions and sizes are pinned by the neighbours and
     the total count.

Run: PYTHONPATH=src python tests/controller_feature_layout.py
"""

from __future__ import annotations

import itertools

import numpy as np

from wnn.control.evaluator import (
	ControllerSpec, arch_shape_from_spec, build_controller, controller_genome_from_arch, ctbr_spec_dict,
)
from wnn.control.feature_layout import feature_layout, feature_names
from wnn.control.recurrent_genome import RecurrentArchGenome

FLAGS = ("obs_tilt_p", "obs_tilt_i", "obs_peraxis_p", "obs_peraxis_i", "obs_peraxis_yaw",
         "obs_pwm", "obs_yaw_err", "obs_yaw_err_i", "dhat", "obs_collective_cmd",
         "obs_alt_err", "obs_vz", "obs_pos_err_xy", "obs_vel_xy")
DHAT_B = (40.0, 40.0, 10.0)
TARGET = (0.31, 0.47, 0.59)
VERT = (0.71, 0.83, 0.97)
HORIZ = (1.13, 1.27, 1.39, 1.51)


def _spec(on: dict, ctbr: "dict | None" = None) -> ControllerSpec:
	kw = {f: bool(on.get(f, False)) for f in FLAGS if f != "dhat"}
	return ControllerSpec(num_motors=4, levels_per_motor=2, bits_per_feature=2, input_window_k=1,
	                      state_neurons=0, state_bits_per_neuron=0, output_bits_per_neuron=4,
	                      dhat_b=DHAT_B if on.get("dhat") else None, ctbr=ctbr, **kw)


def _controller(spec: ControllerSpec):
	g = RecurrentArchGenome.random(arch_shape_from_spec(spec), 0, 8, 0, 4, np.random.default_rng(0))
	th = list(np.linspace(-1.0, 1.0, spec.num_features() * spec.bits_per_feature))
	return build_controller(controller_genome_from_arch(g, spec, th))


def _cascade_dicts() -> list[dict]:
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	ec = episode_config_from_args(build_arg_parser().parse_args(
		["--airframe", "cf21_brushless", "--translation", "--output-mode", "ctbr",
		 "--imu-inclinometer", "--no-rotor-drag"]))
	return [ctbr_spec_dict(ec, obs_rate_i=r, obs_cascade_dhat=d)
	        for r, d in itertools.product((False, True), repeat=2)]


def test_count_every_flag_combination():
	n = 0
	for bits in itertools.product((False, True), repeat=len(FLAGS)):
		spec = _spec(dict(zip(FLAGS, bits)))
		assert _controller(spec).num_features() == len(feature_layout(spec)), dict(zip(FLAGS, bits))
		n += 1
	for ctbr in _cascade_dicts():
		spec = _spec(dict(obs_peraxis_p=True, obs_vel_xy=True), ctbr=ctbr)
		assert _controller(spec).num_features() == len(feature_layout(spec)), ctbr
	print(f"✓ feature count: {n} motor-level combinations + 4 cascade variants agree with Rust")


def _expected(spec: ControllerSpec) -> dict[str, float]:
	"""First-step value of every SENTINEL-pinned feature (level accel, zero gyro-z)."""
	r, p, y = TARGET
	exp = {"gyro_x": 0.011, "gyro_y": 0.022, "gyro_z": 0.0, "acc_x": 0.0, "acc_y": 0.0,
	       "acc_z": 9.81, "tgt_r": r, "tgt_p": p, "tgt_y": y, "tilt": 0.0, "tilt_i": 0.0,
	       "roll_err": r, "pitch_err": p, "yaw_err_pa": y, "yaw_err": y,
	       "coll_cmd": VERT[0], "alt_err": VERT[1], "vz": VERT[2],
	       "e_x": HORIZ[0], "e_y": HORIZ[1], "v_x": HORIZ[2], "v_y": HORIZ[3]}
	s = spec.integral_scale
	exp.update({"roll_i": r * s, "pitch_i": p * s, "yaw_i_pa": y * s, "yaw_err_i": y * s})
	return exp


def _assert_positions(spec: ControllerSpec) -> int:
	c = _controller(spec)
	c.reset()
	c.set_vertical_obs(*VERT)
	c.set_horizontal_obs(*HORIZ)
	c.step([0.011, 0.022, 0.0], [0.0, 0.0, 9.81], list(TARGET))
	vec = [float(v) for v in c.get_last_feature_vector()]
	names = feature_names(spec)
	assert len(vec) == len(names), (len(vec), len(names))
	exp = _expected(spec)
	pinned = [(i, n) for i, n in enumerate(names) if n in exp]
	for i, n in pinned:
		assert abs(vec[i] - exp[n]) < 1e-5, f"{n} expected at index {i} = {exp[n]}, Rust has {vec[i]}"
	return len(pinned)


def test_positions_with_every_block_on():
	every = {f: True for f in FLAGS}
	k = _assert_positions(_spec(every))
	k2 = _assert_positions(_spec({**every, "obs_peraxis_yaw": False}))
	for ctbr in _cascade_dicts()[1:]:
		_assert_positions(_spec({**every, "obs_pwm": False, "dhat": False}, ctbr=ctbr))
	print(f"✓ feature positions: {k} / {k2} sentinel-pinned features at the layout's indices "
	      f"(all blocks on; yaw-dropped per-axis; + cascade tails)")


def test_recipe_family_sizes():
	"""The CTRL-37/38 posx2 recipe layout (22 features) has the design's family sizes."""
	from wnn.control.feature_layout import BANK_FAMILIES
	spec = _spec(dict(obs_peraxis_p=True, obs_peraxis_i=True, obs_yaw_err=True, obs_yaw_err_i=True,
	                  obs_collective_cmd=True, obs_alt_err=True, obs_vz=True,
	                  obs_pos_err_xy=True, obs_vel_xy=True))
	layout = feature_layout(spec)
	sizes = [sum(1 for d in layout if d.family & f) for f in BANK_FAMILIES]
	assert (len(layout), sizes) == (22, [4, 7, 7, 4]), (len(layout), sizes)
	print("✓ recipe: 22 features, T/R/P/Y families 4/7/7/4")


if __name__ == "__main__":
	test_recipe_family_sizes()
	test_positions_with_every_block_on()
	test_count_every_flag_combination()
	print("\nAll feature-layout parity tests passed.")
