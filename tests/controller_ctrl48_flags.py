"""CTRL-48 (ABI 45): one test per flag, each failing without it. All flags OFF is
pinned against ABI 44 separately (controller_ctrl48_abi44_pin.py).

  A --calib-episodes   d-hat ladder seed-stable at 200 episodes (moves << 1 bin), not at 10
    --calib-regime     train_mix widens the ladder to the trainer's 8-30 deg states
    --calib-ladder     centred: symmetric, finer near 0, zero-referenced features only
  B --drop-constant-features   decided by MEASURED zero spread (a target fed real values is kept)
  C --drop-features    no connection lands on a named feature, under every operator; typos refused
  D --obs-cascade-dhat-residual  layout = Rust, calibrated columns live, provenance names it

Run: PYTHONPATH=/tmp/ctrl48_site:src python tests/controller_ctrl48_flags.py
"""

from __future__ import annotations

import numpy as np

from wnn.control import phased_ga as pg
from wnn.control.arch_strategy import default_controller_arch_config
from wnn.control.controller_grid_search import ControllerGridSearch
from wnn.control.evaluator import fit_calibration
from wnn.control.feature_layout import feature_names
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.threshold_calibration import ThresholdCalibration
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim

RECIPE = ["--airframe", "cf21_brushless", "--translation", "--output-mode", "ctbr", "--teacher", "mpcof",
          "--obs-peraxis-p", "--obs-peraxis-i", "--no-obs-peraxis-yaw", "--obs-yaw-err", "--obs-yaw-err-i",
          "--obs-collective-cmd", "--obs-alt-err", "--obs-vz", "--obs-pos-err-xy", "--obs-vel-xy",
          "--xy-offset", "0.5", "--att-err-ref", "tilt", "--memory-mode", "BINARY", "--disturbance", "L4C",
          "--motor-lag-s", "0.20", "--action-repeat", "10", "--ctbr-delta-law", "--tilt", "5.0",
          "--rotor-drag", "0.78", "--grid-state-neurons", "0", "--max-state-neurons", "0",
          "--grid-bits", "24", "--grid-output-neurons", "64", "--max-output-neurons", "64",
          "--steps", "2000", "--eval-episodes", "2", "--num-eval-folds", "1", "--pop", "4"]


def _args(extra: list[str]):
	args = pg.build_arg_parser().parse_args(RECIPE + extra)
	args.cli_argv = RECIPE + extra
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg._refuse_rf_combos(args)
	pg._refuse_calib_combos(args, ec)
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return args, ec


def _probe(extra: list[str]):
	"""(args, ec, probe spec) without fitting anything."""
	args, ec = _args(extra)
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	gs._valid_pairs, spec = gs._compute_valid_pairs()
	return args, ec, spec, gs


def _ladder(spec, ec, args, cal: ThresholdCalibration, seed: int, names: list[str]) -> dict:
	fit = fit_calibration(spec, calibration=cal, seed=seed, episode_config=pg._calib_ec(args, ec))
	all_names, bpf = feature_names(spec), spec.bits_per_feature
	return {n: np.array(fit.thresholds[all_names.index(n) * bpf:(all_names.index(n) + 1) * bpf]) for n in names}


def _bin_moves(a: dict, b: dict) -> dict:
	"""Mean threshold movement between two ladders per feature, in units of that
	feature's median bin (seed s2 ladder)."""
	return {n: float(np.mean(np.abs(a[n] - b[n])) / max(float(np.median(np.diff(a[n]))), 1e-12)) for n in a}


def test_a_calib_episodes_seed_stable_dhat():
	args, ec, spec, _ = _probe(["--obs-cascade-dhat"])
	dh = ["cdhat_x", "cdhat_y", "cdhat_z"]
	move = {n: _bin_moves(_ladder(spec, ec, args, ThresholdCalibration(episodes=n), 31337002, dh),
	                      _ladder(spec, ec, args, ThresholdCalibration(episodes=n), 31337003, dh))
	        for n in (10, 200)}
	for k in dh:
		print(f"  {k}: mean s2-vs-s3 move, bins — 10 eps {move[10][k]:.2f}, 200 eps {move[200][k]:.2f}")
		assert move[200][k] <= move[10][k] / 3.0, f"{k}: 200 episodes did not stabilise the ladder"
		assert move[200][k] < 1.0, f"{k}: 200-episode ladder still moves {move[200][k]:.2f} bins per seed"
	print("✓ A --calib-episodes: d-hat ladder 3x+ more seed-stable at 200 episodes, < 1 bin")


def test_a_calib_regime_train_mix_widens():
	"""The audit's metric: the share of the TRAINER's 30-deg states outside the ladder."""
	import dataclasses, math
	from wnn.control.evaluator import _training_cascade_samples
	args, ec, spec, _ = _probe([])
	keys = ["roll_err", "pitch_err", "gyro_x", "gyro_y"]
	flown = _ladder(spec, ec, args, ThresholdCalibration(episodes=10), 5, keys)
	mix = _ladder(spec, ec, args, ThresholdCalibration(episodes=10, regime="train_mix"), 5, keys)
	ec30 = dataclasses.replace(pg._calib_ec(args, ec), max_initial_tilt_rad=math.radians(30.0))
	s30 = _training_cascade_samples(spec, ec30, ThresholdCalibration(episodes=5), 99)
	names = feature_names(spec)
	for n in keys:
		x = np.array(s30[names.index(n)])
		sat = lambda t: float(np.mean((x < t.min()) | (x > t.max())))
		print(f"  {n}: 30-deg states outside the ladder — flown {sat(flown[n]):.3f}, train_mix {sat(mix[n]):.3f}")
		if n.endswith("_err"):   # gyro is noise-dominated near hold: reported, not asserted
			assert sat(mix[n]) < sat(flown[n]), f"{n}: train_mix did not cover more of the trainer's states"
	print("✓ A --calib-regime train_mix: fewer of the trainer's 30-deg states saturate the ladder")


def test_a_calib_ladder_centred():
	args, ec, spec, _ = _probe([])
	keys = ["roll_err", "e_x", "acc_z", "coll_cmd"]
	q = _ladder(spec, ec, args, ThresholdCalibration(), 5, keys)
	c = _ladder(spec, ec, args, ThresholdCalibration(ladder="centred"), 5, keys)
	for n in ("roll_err", "e_x"):
		t = c[n]
		assert np.allclose(t, -t[::-1]), f"{n}: centred ladder not symmetric about 0: {t}"
		gaps = np.diff(t)
		assert gaps[len(gaps) // 2 + 1] < gaps[-1], f"{n}: not finer near 0: {gaps}"
		assert not np.array_equal(t, q[n])
	for n in ("acc_z", "coll_cmd"):   # not zero-referenced: quantile kept
		assert np.array_equal(c[n], q[n]), f"{n} must keep the quantile ladder"
	print("✓ A --calib-ladder centred: symmetric + finer near 0 on zero-ref features only")


def test_b_drop_constant_is_measured():
	args, ec, spec, gs = _probe(["--drop-constant-features"])
	gs._fit_thresholds(spec, 31337002)
	assert set(args._drop_auto) == {"tgt_r", "tgt_p", "tgt_y"}, args._drop_auto
	point = gs._enumerate_points()[0]
	g = gs._random_genome(point)
	assert point.spec.drop_features == ("tgt_p", "tgt_r", "tgt_y")
	assert not any(_feature_bits(g, point.spec, n) for n in ("tgt_r", "tgt_p", "tgt_y")), "a target bit is wired"
	# MEASURED, not named: give tgt_r real spread and it is no longer constant.
	names, nf = feature_names(spec), spec.num_features()
	extra = [[] for _ in range(nf)]
	extra[names.index("tgt_r")] = [0.0, 0.01, -0.02]
	fit = fit_calibration(spec, calibration=ThresholdCalibration(), seed=31337002,
	                      episode_config=pg._calib_ec(args, ec), extra_samples=extra)
	dropped = {names[f] for f in fit.constant_features}
	assert dropped == {"tgt_p", "tgt_y"}, dropped
	print("✓ B --drop-constant-features: constant set = measured zero spread (tgt_r with spread kept)")


def _feature_bits(g, spec, name) -> list:
	f, bpf = feature_names(spec).index(name), spec.bits_per_feature
	fb = spec.num_features() * bpf
	return [b for s in g.output_sampled + g.state_sampled for b in s if (b % fb) // bpf == f]


def test_c_drop_features_never_wired():
	args, ec, spec, gs = _probe(["--drop-features", "acc_x,acc_y"])
	point = gs._enumerate_points()[0]
	cfg = default_controller_arch_config(point.spec)
	cfg.max_output_neurons, cfg.output_block_delta = 128, 2
	a, b = gs._random_genome(point), gs._random_genome(point)
	rng = np.random.default_rng(1)
	for step in range(10):
		for dim in (Dim.NEURONS, Dim.BITS, Dim.CONNECTIONS):
			a = a.mutate(dim, 1.0, cfg, rng)
		b = b.mutate(Dim.NEURONS, 1.0, cfg, rng)
		child = RecurrentArchGenome.crossover_average(a, b, rng, cfg)
		for g in (a, child, RecurrentArchGenome.crossover(a, b, rng, cfg)):
			for n in ("acc_x", "acc_y"):
				assert not _feature_bits(g, point.spec, n), f"step {step}: a connection landed on {n}"
	off = gs._random_genome(_probe([])[3]._enumerate_points()[0])
	assert _feature_bits(off, point.spec, "acc_x"), "without the flag acc_x is wired (non-vacuous)"
	try:
		_probe(["--drop-features", "acc_q"])
		raise AssertionError("an unknown --drop-features name must be refused")
	except ValueError:
		pass
	try:
		_args(["--drop-features", "acc_x", "--conn-policy", "min2"])
		raise AssertionError("--drop-features must refuse min-m policies")
	except SystemExit:
		pass
	print("✓ C --drop-features: excluded under random/mutate/crossovers; typos + min-m refused")


def test_d_residual_layout_calibration_provenance():
	from wnn.control.provenance import arch_token
	args, ec, spec, _ = _probe(["--obs-cascade-dhat-residual"])
	names = feature_names(spec)
	assert names[-3:] == ["cresid_x", "cresid_y", "cresid_z"], names[-4:]
	from wnn.control.evaluator import build_controller, controller_genome_from_arch, arch_shape_from_spec
	g = RecurrentArchGenome.random(arch_shape_from_spec(spec), 0, 64, 0, 24, np.random.default_rng(0))
	c = build_controller(controller_genome_from_arch(g, spec, [0.0] * (spec.num_features() * spec.bits_per_feature)))
	assert c.num_features() == len(names)
	r = _ladder(spec, ec, args, ThresholdCalibration(), 3, ["cresid_x", "cresid_z"])
	assert all(np.ptp(v) > 0 for v in r.values()), f"degenerate residual ladder {r}"
	assert ":obs=dhat_resid" in arch_token(args._ctbr, "mpcof", 10, 0.001)
	assert "rate_i" not in arch_token(args._ctbr, "mpcof", 10, 0.001), "a3 alone must not switch a1 on"
	print("✓ D --obs-cascade-dhat-residual: +3 features last, Rust count agrees, live calibration, provenance")


def test_provenance_tokens():
	args, _ = _args(["--calib-episodes", "200", "--calib-ladder", "centred", "--drop-constant-features",
	                 "--drop-features", "acc_y,acc_x"])
	assert pg.calibration_from_args(args).token() == "eps:200,regime:flown,ladder:centred"
	assert pg.drop_token(args) == "constant+names:acc_x+acc_y"
	off, _ = _args([])
	assert (pg.calibration_from_args(off).token(), pg.drop_token(off)) == ("eps:10,regime:flown,ladder:quantile", "off")
	print("✓ provenance calib= / drop= tokens")


if __name__ == "__main__":
	test_provenance_tokens()
	test_d_residual_layout_calibration_provenance()
	test_c_drop_features_never_wired()
	test_b_drop_constant_is_measured()
	test_a_calib_ladder_centred()
	test_a_calib_regime_train_mix_widens()
	test_a_calib_episodes_seed_stable_dhat()
	print("\nAll CTRL-48 flag tests passed.")
