"""ABI 70: the golden-cell TRAINING ranges, Python-only (the CTRL-87 --train-motor-asym-mag pattern).

  --train-mass-jitter M          mass spread ±M on training + search pools (--mass-jitter keeps the report pools)
  --train-noise-scale S          the preset's gyro_sigma / gyro_bias_walk / accel_sigma x S, training + search
  --train-motor-asym-range LO HI per-motor wear factor U(LO, HI) x the preset base, training + search

  * CLI: all three default None; flag off the EpisodeConfig helpers are the identity
    (search_disturbance() IS the preset object, train_vertical_ec() IS the config);
    refusals (ranges, exclusivity with --train-motor-asym-mag, no disturbance / translation);
  * TRAINING hosts (_dist_packed_fields_ec + _train_asym_kwargs + _stage1_train_kwargs — DAgger,
    per-round eval, calibration sampler; the MEMORY recorder's vertical draws): flag off
    byte-identical, flags on carry the training plant;
  * POOLS: with all three ON a REPORT evaluator (train_spread=False: TEST, VAL stage-select,
    held-out) hands the scorer EXACTLY the flag-off kwargs; a SEARCH evaluator flies the
    training plant on the same seeds;
  * provenance train_plant= (absent off), KNOWN_PARAMS, the flow adapter refuses the keys.

No wheel call is made (the scorer is captured), so this runs on any build whose src tree is this one:
  PYTHONPATH=src python tests/controller_train_ranges.py
"""

from __future__ import annotations

import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl86_flag_off as H  # noqa: E402  the flown l2b-family argv
from controller_ctrl87_train_spread import _Capture, _raises  # noqa: E402

RANGES = ["--train-mass-jitter", "0.25", "--train-noise-scale", "2.0", "--train-motor-asym-range", "0.64", "1.44"]


def _args(extra: list):
	from wnn.control import phased_ga as pg
	args = pg.build_arg_parser().parse_args(shlex.split(H.GT_ARGV) + extra)
	return args, pg.episode_config_from_args(args)


def test_cli_and_refusals():
	a, off = _args([])
	assert (a.train_mass_jitter, a.train_noise_scale, a.train_motor_asym_range) == (None, None, None)
	assert (off.train_mass_jitter, off.train_noise_scale, off.train_motor_asym_range) == (None, None, None)
	assert not off.train_spread_on()
	assert off.search_disturbance() is off.disturbance and off.train_vertical_ec() is off
	assert off.train_mass_jitter_value() == off.mass_jitter == 0.15
	_, on = _args(RANGES)
	assert on.train_spread_on() and on.mass_jitter == 0.15, "--mass-jitter (the report pools) must stay 0.15"
	assert on.train_vertical_ec().mass_jitter == 0.25 and on.train_mass_jitter_value() == 0.25
	d0, d1 = on.disturbance, on.search_disturbance()
	assert d0.gyro_sigma == off.disturbance.gyro_sigma and d0.motor_asym_mag == 0.2, "the preset must not move"
	for f in ("gyro_sigma", "gyro_bias_walk", "accel_sigma"):
		assert abs(getattr(d1, f) - 2.0 * getattr(d0, f)) < 1e-15, f
	# U(0.64, 1.44) spelled base x 1.04 x (1 ± 0.3846): the stress cell's construction.
	assert all(abs(b - 1.04 * b0) < 1e-12 for b, b0 in zip(d1.motor_asym, d0.motor_asym)), d1.motor_asym
	assert abs(d1.motor_asym_mag - 0.80 / 2.08) < 1e-12 and abs(on.train_asym_mag() - d1.motor_asym_mag) < 1e-15
	lo, hi = 1.04 * (1 - d1.motor_asym_mag), 1.04 * (1 + d1.motor_asym_mag)
	assert abs(lo - 0.64) < 1e-12 and abs(hi - 1.44) < 1e-12, (lo, hi)
	for extra, text in [(["--train-mass-jitter", "1.0"], "[0, 1)"),
	                    (["--train-noise-scale", "-1"], ">= 0"),
	                    (["--train-noise-scale", "2", "--disturbance", "OFF"], "disturbance"),
	                    (["--train-motor-asym-range", "1.4", "0.6"], "0 < lo <= hi"),
	                    (["--train-motor-asym-range", "0.6", "1.4", "--train-motor-asym-mag", "0.3"], "exclusive"),
	                    (["--train-motor-asym-range", "0.6", "1.4", "--disturbance", "OFF"], "disturbance")]:
		_raises(lambda: _args(extra), SystemExit, text, " ".join(extra))
	print("✓ CLI: three flags default None (helpers = identity off); on: mass 0.25 / noise x2 / wear U(0.64, 1.44) "
	      "on the training plant, preset + --mass-jitter untouched; refusals")


def test_training_hosts():
	from wnn.control.evaluator import _dist_packed_fields_ec, _stage1_train_kwargs, _train_asym_kwargs, train_plant_token
	_, off = _args([])
	_, on = _args(RANGES)
	f_off, f_on = _dist_packed_fields_ec(off), _dist_packed_fields_ec(on)
	d = off.disturbance
	assert f_off == _dist_packed_fields_ec(H._args([])[1]) and f_off[5] == d.gyro_sigma and f_off[4] == list(d.motor_asym)
	assert (f_on[5], f_on[6], f_on[7]) == (2 * d.gyro_sigma, 2 * d.gyro_bias_walk, 2 * d.accel_sigma), f_on
	assert all(abs(a - 1.04 * b) < 1e-12 for a, b in zip(f_on[4], d.motor_asym))
	assert [f_on[i] for i in (0, 1, 2, 3, 8, 9, 10, 11)] == [f_off[i] for i in (0, 1, 2, 3, 8, 9, 10, 11)]
	assert _train_asym_kwargs(off) == {"dist_motor_asym_mag": 0.2}
	assert abs(_train_asym_kwargs(on)["dist_motor_asym_mag"] - 0.80 / 2.08) < 1e-15
	s_off, s_on = _stage1_train_kwargs(off), _stage1_train_kwargs(on)
	assert s_off["mass_jitter"] == 0.15 and s_on["mass_jitter"] == 0.25
	assert {k: v for k, v in s_on.items() if k != "mass_jitter"} == {k: v for k, v in s_off.items() if k != "mass_jitter"}
	# The MEMORY recorder's vertical draws read the training mass spread.
	import inspect
	import wnn.control.ga_memory as GM
	assert "train_vertical_ec()" in inspect.getsource(GM._recorder_plant_kwargs)
	assert (train_plant_token(off), train_plant_token(on)) == ("", "mass:0.25+noise:x2+asym:0.64,1.44")
	print("✓ training hosts: packed weather noise x2 + wear base x1.04 / mag 0.3846, trainer mass_jitter 0.25, "
	      "recorder reads train_vertical_ec(); flag off identical")


def _pool_kwargs(ec, train_spread: bool) -> dict:
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control import phased_ga as pg
	args, _ = _args([])
	pg._resolve_ctbr(args, ec)
	spec = pg.ControllerSpec(num_motors=4, levels_per_motor=16)
	ev = ControllerEvaluator(spec, num_eval_episodes=6, seed=99990101, episode_config=ec, num_eval_folds=5,
	                         train_spread=train_spread)
	ev._advance_fold()
	cap = _Capture()
	ev._score_population_rust([object()], cap)
	assert len(cap.calls) == 1, getattr(ev, "_rust_score_error", None)
	return cap.calls[0]


def test_report_pools_unchanged_search_pools_train_plant():
	_, off = _args([])
	_, on = _args(RANGES)
	ref = _pool_kwargs(off, train_spread=False)
	# REPORT (TEST units, VAL stage-select, held-out): EXACTLY the flag-off call.
	rep = _pool_kwargs(on, train_spread=False)
	assert repr(sorted(rep.items())) == repr(sorted(ref.items())), "a report pool moved with the ranges on"
	assert repr(sorted(_pool_kwargs(off, train_spread=True).items())) == repr(sorted(ref.items()))
	srch = _pool_kwargs(on, train_spread=True)
	assert srch["dist_seed"] == ref["dist_seed"] and srch["_q0"] == ref["_q0"]
	assert srch["ep_mass"] != ref["ep_mass"], "the search pool did not fly the training mass spread"
	af = float(on.airframe.mass)
	r_dev = [m / af - 1.0 for m in ref["ep_mass"]]
	s_dev = [m / af - 1.0 for m in srch["ep_mass"]]
	assert all(abs(s - r * 0.25 / 0.15) < 1e-9 for r, s in zip(r_dev, s_dev)), (r_dev, s_dev)
	assert abs(srch["dist_gyro_sigma"] - 2.0 * ref["dist_gyro_sigma"]) < 1e-15
	assert srch["dist_motor_asym"] != ref["dist_motor_asym"]
	assert all(0.64 - 1e-9 <= a <= 1.44 + 1e-9 for a in srch["dist_motor_asym"]), srch["dist_motor_asym"]
	moved = {"ep_mass", "dist_gyro_sigma", "dist_gyro_bias_walk", "dist_accel_sigma", "dist_motor_asym"}
	rest = lambda d: {k: v for k, v in d.items() if k not in moved}
	assert repr(sorted(rest(srch).items())) == repr(sorted(rest(ref).items()))
	print(f"✓ pools: report kwargs == flag-off ({len(ref)} kwargs); search = same seeds, mass ±0.25 (same draws "
	      f"x 5/3), gyro_sigma x2, wear {['%.4f' % a for a in srch['dist_motor_asym']]} in [0.64, 1.44]")


def test_provenance_registry_adapter():
	from wnn.control.provenance import collect_provenance
	from wnn.ram.experiments.params import KNOWN_PARAMS
	from wnn.control.flow_adapter import controller_spec_from_params
	assert "train_plant=mass:0.25 " in collect_provenance("x", train_plant="mass:0.25").line()
	assert "train_plant" not in collect_provenance("x").line()
	keys = ("controller_train_mass_jitter", "controller_train_noise_scale", "controller_train_motor_asym_range")
	assert set(keys) <= KNOWN_PARAMS
	for key in keys:
		_raises(lambda: controller_spec_from_params({key: 0.3}), ValueError, "not built", key)
	print("✓ provenance train_plant= (absent off), KNOWN_PARAMS, flow adapter refuses the three keys")


if __name__ == "__main__":
	test_cli_and_refusals()
	test_training_hosts()
	test_report_pools_unchanged_search_pools_train_plant()
	test_provenance_registry_adapter()
	print("\nABI-70 golden-cell training ranges (Python): all passed.")
