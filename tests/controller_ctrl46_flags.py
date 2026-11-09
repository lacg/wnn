"""CTRL-46 (ABI 46): the three default-OFF controller features, one test group each.
All flags OFF is the ABI-45 recipe bit-identically (Rust pins: abi35/ctrl17/abi38/ctrl38).

  1 --write-rule {last,vote} + --vote-share θ   the IDS ids_write_rule / ids_vote_share
    vocabulary and validators (shared helpers in ids_param_resolution); QUAD refused
  2 --dagger-label-attitude {true,estimate}     needs the estimator, refuses expert_drives
  3 --obs-att-bias-pred                          +2 features LAST, Rust count agrees, live
    calibration, needs --obs-att-estimator and --obs-vel-xy

Run: PYTHONPATH=/tmp/ctrl46_site:src python tests/controller_ctrl46_flags.py
"""

from __future__ import annotations

import numpy as np

from wnn.control import _accel as ra
from wnn.control import phased_ga as pg
from wnn.control.controller_grid_search import ControllerGridSearch
from wnn.control.evaluator import (
	arch_shape_from_spec, build_controller, controller_genome_from_arch, fit_calibration,
)
from wnn.control.feature_layout import AxisFamily, feature_layout, feature_names
from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.threshold_calibration import ThresholdCalibration
from wnn.ram.experiments.ids_param_resolution import resolve_vote_share, resolve_write_rule

RECIPE = ["--airframe", "cf21_brushless", "--translation", "--output-mode", "ctbr", "--teacher", "mpcof",
          "--obs-peraxis-p", "--obs-peraxis-i", "--no-obs-peraxis-yaw", "--obs-yaw-err", "--obs-yaw-err-i",
          "--obs-collective-cmd", "--obs-alt-err", "--obs-vz", "--obs-pos-err-xy", "--obs-vel-xy",
          "--xy-offset", "0.5", "--att-err-ref", "tilt", "--memory-mode", "BINARY", "--disturbance", "L4C",
          "--motor-lag-s", "0.20", "--action-repeat", "10", "--ctbr-delta-law", "--tilt", "5.0",
          "--rotor-drag", "0.78", "--grid-state-neurons", "0", "--max-state-neurons", "0",
          "--grid-bits", "24", "--grid-output-neurons", "64", "--max-output-neurons", "64",
          "--steps", "2000", "--eval-episodes", "2", "--num-eval-folds", "1", "--pop", "4"]


def _args(extra: list[str], base: list[str] = RECIPE):
	args = pg.build_arg_parser().parse_args(base + extra)
	args.cli_argv = base + extra
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return args, ec


def _refused(extra: list[str], needle: str, base: list[str] = RECIPE) -> None:
	try:
		_args(extra, base)
	except SystemExit as e:
		assert needle in str(e), f"{extra}: refused for the wrong reason: {e}"
		return
	raise AssertionError(f"{extra}: accepted, expected a refusal mentioning {needle!r}")


def _swap(flag: str, value: str) -> list[str]:
	base = list(RECIPE)
	base[base.index(flag) + 1] = value
	return base


# ------------------------------------------------------------------ item 1 --

def test_write_rule_defaults_and_vote():
	off, ec = _args([])
	rg = pg._rg_config(off, ec, 1)
	assert (rg.write_vote, rg.vote_share, rg.dagger_label_estimate) == (False, None, False)
	v, ec = _args(["--write-rule", "vote"])
	rg = pg._rg_config(v, ec, 1)
	assert (rg.write_vote, rg.vote_share) == (True, None), "vote without θ = the plain majority"
	v, ec = _args(["--write-rule", "vote", "--vote-share", "0.3"])
	assert pg._rg_config(v, ec, 1).vote_share == 0.3
	for mode in ("TERNARY", "PLN"):
		_args(["--write-rule", "vote"], _swap("--memory-mode", mode))
	print("✓ 1 --write-rule: default last (rg off); vote = majority; θ carried; BINARY/TERNARY/PLN accepted")


def test_write_rule_refusals_share_the_ids_validators():
	for mode in ("QUAD_WEIGHTED", "QUAD_BINARY", "QSR"):
		_refused(["--write-rule", "vote"], "needs memory_mode in", _swap("--memory-mode", mode))
	_refused(["--vote-share", "0.3"], "only valid with --write-rule vote")
	for bad in ("0.123", "0.6", "0"):
		_refused(["--write-rule", "vote", "--vote-share", bad], "--vote-share")
	# The IDS resolvers still say exactly what they said (labels are the IDS keys).
	for params, needle in (({"memory_mode": "QUAD_WEIGHTED", "ids_write_rule": "vote"}, "ids_write_rule='vote' needs"),
	                       ({"memory_mode": "BINARY", "ids_write_rule": "vote", "ids_vote_share": 0.123},
	                        "ids_vote_share=0.123: must be a multiple of 1/100")):
		try:
			resolve_vote_share(params) if "ids_vote_share" in params else resolve_write_rule(params)
		except ValueError as e:
			assert needle in str(e), str(e)
		else:
			raise AssertionError(f"{params} accepted")
	assert resolve_vote_share({"memory_mode": "BINARY", "ids_write_rule": "vote", "ids_vote_share": 0.3}) == 0.3
	print("✓ 1 refusals: QUAD family, θ without vote, off-grid / out-of-range θ; IDS messages unchanged")


def test_packed_config_surface():
	c = ra.RewardGatedConfigPacked()
	assert (c.write_vote, c.vote_share, c.dagger_label_estimate) == (False, None, False)
	c = ra.RewardGatedConfigPacked(write_vote=True, vote_share=0.25, dagger_label_estimate=True)
	assert (c.write_vote, c.vote_share, c.dagger_label_estimate) == (True, 0.25, True)
	for kw in ({"vote_share": 0.3}, {"write_vote": True, "vote_share": 0.123}, {"write_vote": True, "vote_share": 0.7}):
		try:
			ra.RewardGatedConfigPacked(**kw)
		except ValueError:
			continue
		raise AssertionError(f"RewardGatedConfigPacked({kw}) accepted")
	print("✓ 1/2 RewardGatedConfigPacked(write_vote=, vote_share=, dagger_label_estimate=) validated in Rust")


# ------------------------------------------------------------------ item 2 --

def test_label_attitude():
	a, ec = _args(["--dagger-label-attitude", "estimate"])
	assert pg._rg_config(a, ec, 1).dagger_label_estimate
	_refused(["--dagger-label-attitude", "estimate", "--no-obs-att-estimator", "--force-blind-attitude"],
	         "needs --obs-att-estimator")
	_refused(["--dagger-label-attitude", "estimate", "--expert-drives"], "refuses --expert-drives")
	print("✓ 2 --dagger-label-attitude estimate: carried to the trainer; needs the estimator; refuses expert_drives")


# ------------------------------------------------------------------ item 3 --

def _probe(extra: list[str]):
	args, ec = _args(extra)
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	gs._valid_pairs, spec = gs._compute_valid_pairs()
	return args, ec, spec


def test_att_bias_pred_layout_calibration():
	args, ec, spec = _probe(["--obs-att-bias-pred"])
	assert spec.obs_att_bias_pred
	names = feature_names(spec)
	assert names[-2:] == ["abp_r", "abp_p"], names[-4:]
	fam = {d.name: d.family for d in feature_layout(spec)}
	assert (fam["abp_r"], fam["abp_p"]) == (AxisFamily.ROLL, AxisFamily.PITCH)
	g = RecurrentArchGenome.random(arch_shape_from_spec(spec), 0, 64, 0, 24, np.random.default_rng(0))
	c = build_controller(controller_genome_from_arch(g, spec, [0.0] * (spec.num_features() * spec.bits_per_feature)))
	assert c.num_features() == len(names), (c.num_features(), len(names))
	assert c.obs_att_bias_pred
	off_args, off_ec, off_spec = _probe([])
	assert len(feature_names(off_spec)) == len(names) - 2 and not off_spec.obs_att_bias_pred
	fit = fit_calibration(spec, calibration=ThresholdCalibration(), seed=3, episode_config=pg._calib_ec(args, ec))
	bpf = spec.bits_per_feature
	for n in ("abp_r", "abp_p"):
		i = names.index(n)
		col = np.array(fit.thresholds[i * bpf:(i + 1) * bpf])
		assert np.ptp(col) > 0, f"degenerate {n} ladder {col}"
		print(f"  {n} ladder (deg): {np.degrees(col[[0, len(col) // 2, -1]]).round(3).tolist()}")
	print("✓ 3 --obs-att-bias-pred: +2 features LAST (roll/pitch families), Rust count agrees, live calibration")


def test_att_bias_pred_refusals_and_provenance():
	base = [a for a in RECIPE if a != "--obs-vel-xy"]
	_refused(["--obs-att-bias-pred"], "needs --obs-vel-xy", base)
	_refused(["--obs-att-bias-pred", "--no-obs-att-estimator", "--force-blind-attitude"], "needs --obs-att-estimator")
	off, _ = _args([])
	assert pg.ctrl46_tokens(off) == {"write": "last", "label": "true", "abp": "off"}
	on, _ = _args(["--write-rule", "vote", "--vote-share", "0.3", "--dagger-label-attitude", "estimate",
	               "--obs-att-bias-pred"])
	assert pg.ctrl46_tokens(on) == {"write": "vote:0.3", "label": "estimate", "abp": "on"}
	from wnn.control.provenance import collect_provenance
	line = collect_provenance("x", **pg.ctrl46_tokens(on)).line()
	assert "write=vote:0.3 label=estimate abp=on" in line, line
	assert ra.ABI_VERSION == 46
	print("✓ 3 refusals (no vel_xy / no estimator); provenance write= label= abp=; ABI 46")


if __name__ == "__main__":
	test_packed_config_surface()
	test_write_rule_defaults_and_vote()
	test_write_rule_refusals_share_the_ids_validators()
	test_label_attitude()
	test_att_bias_pred_refusals_and_provenance()
	test_att_bias_pred_layout_calibration()
	print("\nAll CTRL-46 flag tests passed.")
