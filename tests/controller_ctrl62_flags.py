"""ABI 62 (CTRL-69): the cascade's model-free rate-loop inputs, Python side.

  * the first-arm flags (N0 + --obs-rate-err --obs-rate-err-tau 0.05 --obs-rate-err-axes
    roll,pitch --obs-rate-i --obs-rate-i-axes yaw) build the layout Rust builds, with the
    rate-loop columns where feature_layout says (cascade block, then the CTRL-46 tail);
  * calibration fits those columns on the TRAINING cascade (non-degenerate ladders; under
    --translation every column comes from the Rust training-cascade sampler);
  * provenance names them; the defaults leave every ABI-61 dict / token unchanged;
  * checkpoint round trip (asdict -> ControllerSpec -> CtbrSpec) keeps them;
  * refusals: no ctbr, subset/tau without the block, unknown axis names.

Run: PYTHONPATH=/tmp/ctrl62_site:src python tests/controller_ctrl62_flags.py
"""

from __future__ import annotations

import numpy as np

from wnn.control import phased_ga as pg
from wnn.control.feature_layout import cascade_columns, feature_names
from wnn.control.threshold_calibration import ThresholdCalibration


# The N0 s1 recipe's observation set (model-free: no --obs-cascade-dhat-residual,
# no --yaw-trim-ff), shortened for a probe.
N0 = ["--airframe", "cf21_brushless", "--translation", "--output-mode", "ctbr", "--teacher", "mpcof",
      "--obs-peraxis-p", "--obs-peraxis-i", "--no-obs-peraxis-yaw", "--obs-yaw-err", "--obs-yaw-err-i",
      "--obs-collective-cmd", "--obs-alt-err", "--obs-vz", "--obs-pos-err-xy", "--obs-vel-xy",
      "--xy-offset", "0.5", "--att-err-ref", "tilt", "--memory-mode", "BINARY", "--disturbance", "L4C",
      "--motor-lag-s", "0.20", "--action-repeat", "10", "--ctbr-delta-law", "--tilt", "5.0",
      "--rotor-drag", "0.78", "--grid-state-neurons", "0", "--max-state-neurons", "0",
      "--grid-bits", "24", "--grid-output-neurons", "64", "--max-output-neurons", "64",
      "--obs-att-bias-pred", "--obs-tilt-ref-err", "--obs-yaw-from-att-est",
      "--steps", "2000", "--eval-episodes", "2", "--num-eval-folds", "1", "--pop", "4"]
FIRST_ARM = ["--obs-rate-err", "--obs-rate-err-tau", "0.05", "--obs-rate-err-axes", "roll,pitch",
             "--obs-rate-i", "--obs-rate-i-axes", "yaw"]


def _args(extra: list[str]):
	args = pg.build_arg_parser().parse_args(N0 + extra)
	args.cli_argv = N0 + extra
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg.resolve_ctrl46_args(args)
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return args, ec


def _spec(extra: list[str]):
	from wnn.control.controller_grid_search import ControllerGridSearch
	args, ec = _args(extra)
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	gs._valid_pairs, spec = gs._compute_valid_pairs()
	return args, ec, spec


def _rust_features(spec) -> int:
	from wnn.control.evaluator import arch_shape_from_spec, build_controller, controller_genome_from_arch
	from wnn.control.recurrent_genome import RecurrentArchGenome
	g = RecurrentArchGenome.random(arch_shape_from_spec(spec), 0, 64, 0, 24, np.random.default_rng(0))
	return build_controller(controller_genome_from_arch(g, spec, [0.0] * (spec.num_features() * spec.bits_per_feature))).num_features()


def test_first_arm_layout_and_dict():
	args, ec, spec = _spec(FIRST_ARM)
	c = args._ctbr
	assert (c["obs_rate_i"], c["obs_rate_i_axes"], c["obs_rate_err_tau"], c["obs_rate_err_axes"]) == (True, 4, 0.05, 3), c
	assert "obs_dhat_plant" not in c and "obs_dhat_residual_plant" not in c, "the first arm is model-free"
	names = feature_names(spec)
	cols = cascade_columns(spec)
	assert [names[i] for i in cols] == ["rate_i_z", "rerr_x", "rerr_y"], [names[i] for i in cols]
	assert names[-2:] == ["abp_r", "abp_p"], "the CTRL-46 block stays the tail"
	assert _rust_features(spec) == len(names)
	off, _, off_spec = _spec([])
	assert not any(k in off._ctbr for k in ("obs_rate_i_axes", "obs_rate_err_tau", "obs_rate_err_axes")), off._ctbr
	ri, _ = _args(["--obs-rate-i"])
	assert "obs_rate_i_axes" not in ri._ctbr, "--obs-rate-i alone must stay the ABI-61 dict"
	print(f"✓ first arm: cascade block {[names[i] for i in cols]} at {list(cols)}, abp tail, Rust count {len(names)}; "
	      "defaults leave the ABI-61 dict unchanged")


def test_first_arm_calibration_on_the_training_cascade():
	from wnn.control.evaluator import fit_calibration
	args, ec, spec = _spec(FIRST_ARM)
	fit = fit_calibration(spec, calibration=ThresholdCalibration(episodes=10), seed=3,
	                      episode_config=pg._calib_ec(args, ec))
	names, bpf = feature_names(spec), spec.bits_per_feature
	for n in ("rate_i_z", "rerr_x", "rerr_y"):
		i = names.index(n)
		lad = np.array(fit.thresholds[i * bpf:(i + 1) * bpf])
		print(f"  {n}: ladder [{lad.min():+.4g}, {lad.max():+.4g}]")
		assert np.ptp(lad) > 0, f"{n}: degenerate ladder {lad}"
		assert i not in fit.constant_features
	print("✓ first-arm calibration: rate_i_z / rerr_x / rerr_y fitted on the training cascade, non-degenerate")


def test_provenance_and_round_trip():
	from dataclasses import asdict
	from wnn.control import _accel as ra
	from wnn.control.evaluator import ControllerSpec, ctbr_arg
	from wnn.control.provenance import arch_token
	args, ec, spec = _spec(FIRST_ARM)
	tok = arch_token(args._ctbr, "mpcof", 10, 0.001)
	assert tok.endswith(":obs=rate_i(axes=z):obs=rate_err(tau=0.05)(axes=xy)"), tok
	r = repr(ctbr_arg(spec))
	assert "obs=rate_i(axes=z) obs=rate_err(tau=0.05)(axes=xy)" in r, r
	back = ControllerSpec(**asdict(spec))
	cs = ctbr_arg(back)
	assert (cs.obs_rate_i, cs.obs_rate_i_axes, cs.obs_rate_err_tau, cs.obs_rate_err_axes) == (True, 4, 0.05, 3)
	assert ra.CtbrSpec(**back.ctbr).rate_err_lpf_consts(float(back.dt)) is not None
	off, _ = _args(["--obs-rate-i"])
	assert arch_token(off._ctbr, "mpcof", 10, 0.001).endswith(":obs=rate_i"), "ABI-61 token changed"
	print(f"✓ provenance {tok.split(':', 6)[-1]} ; checkpoint round trip keeps the rate-loop keys")


def test_refusals():
	bad = [(["--obs-rate-err-axes", "roll"], "axes without the block"),
	       (["--obs-rate-err-tau", "0.1"], "tau without the block"),
	       (["--obs-rate-i-axes", "yaw"], "rate-I axes without --obs-rate-i"),
	       (["--obs-rate-err", "--obs-rate-err-tau", "0"], "tau 0"),
	       (["--obs-rate-err", "--obs-rate-err-axes", "roll,heave"], "unknown axis")]
	for extra, why in bad:
		try:
			_args(extra)
		except (SystemExit, ValueError):
			continue
		raise AssertionError(f"must refuse: {why}")
	args = pg.build_arg_parser().parse_args(["--obs-rate-err"])
	try:
		pg._resolve_ctbr(args, pg.episode_config_from_args(args))
		raise AssertionError("--obs-rate-err must refuse the motor path")
	except SystemExit:
		pass
	print("✓ refusals: subset/tau without the block, tau <= 0, unknown axis, motor path")


if __name__ == "__main__":
	test_first_arm_layout_and_dict()
	test_provenance_and_round_trip()
	test_refusals()
	test_first_arm_calibration_on_the_training_cascade()
	print("\nAll CTRL-62 flag tests passed.")
