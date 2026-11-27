"""CTRL-80 (ABI 64, --obs-yaw-lead): the yaw LEAD input yaw_lead = yaw_err - tau * gyro_z,
Python side + the flag-off pin vs ABI 63 across the two builds.

  * FLAG OFF == ABI 63, measured across the two builds on the flown G+T recipe (the CTRL-71/72
    G+T s1 argv, save paths dropped, shrunk for a probe by trailing overrides):
      ABI 63: $CTRL80_BASE_PYTHONPATH (the ABI-63 src tree; the ABI-63 wheel = the venv's);
      ABI 64: this tree with the ABI-64 wheel first on PYTHONPATH.
    Every dump is EQUAL: the feature names, the fitted thresholds, the calibration token, the
    resolved ctbr dict, every parsed argument, the provenance tokens, 40 steps of features, the
    Lamarckian write-back digest after one short DAgger and the scorer rows.
  * --obs-yaw-lead adds ONE feature, `yaw_lead`, directly after yaw_err_i; its ladder is exactly
    +-0.40 / 0.80 / 1.35 / 2.25 deg; every OTHER feature's ladder is the flag-off ladder bit for
    bit (no cut is stolen) and the calibration token is unchanged; the wheel's value in that
    slot is yaw_err - tau * gyro_z (f32) and every other input is the flag-off input; it reaches
    the trainer (write-back digest moves) and the scorer (rows move);
  * --obs-yaw-lead-tau alone (flag off) is the flag-off run; a second tau keeps the ladder and
    moves the lead column only;
  * every Python constructor path hands it to the wheel (build_controller getter); provenance
    says ylead=0.1; refusals (no --obs-yaw-err, --bits-per-feature != 8, tau range; CLI and the
    fit itself); the registry knows the keys; the on-board C exporter refuses it.

Run: CTRL80_BASE_PYTHONPATH=/tmp/ctrl80_base/src \
     PYTHONPATH=/tmp/ctrl80_site:src python tests/controller_ctrl80_yaw_lead.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import numpy as np

GT_ARGV = "--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --test-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived --fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1 --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 --receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --drop-constant-features --test-seeds 99990101 99990102 99990103 99990104 99990105 99990106 99990107 99990108 99990109 99990110 99990111 99990112 99990113 99990114 99990115 99990116 99990117 99990118 99990119 99990120 99990121 99990122 99990123 99990124 99990125 99990126 99990127 99990128 99990129 99990130 99990131 99990132 99990133 99990134 99990135 --base-seed 31337001 --obs-att-bias-pred --write-rule vote --vote-share 0.1 --val-units 20 --select-tail p99 --yaw-centred-ladder dense --yaw-integral-leak 0.9995 --obs-tilt-ref-err --zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5 --ctbr-gamma 1 2 2 3 --gyro-z-ladder dense"
# Probe-sized overrides (argparse: the last occurrence wins) — the same on both builds.
SHRINK = ("--grid-output-neurons 64 --max-output-neurons 64 --steps 400 --eval-episodes 4 --num-eval-folds 1 "
          "--rg-rounds 1 --rg-episodes-per-round 2 --rg-eval-episodes 2 --pop 4 --train-workers 2 "
          "--calib-episodes 20")
VARIANTS = {"off": [], "tau_only": ["--obs-yaw-lead-tau", "0.25"], "on": ["--obs-yaw-lead"],
            "on_tau": ["--obs-yaw-lead", "--obs-yaw-lead-tau", "0.25"]}
BEHAVIOUR = ("names", "thresholds", "token", "ctbr", "features", "train_digest", "scorer_rows")
NEW_KEYS = ("obs_yaw_lead", "obs_yaw_lead_tau")
LEAD_CUTS_DEG = (0.40, 0.80, 1.35, 2.25)


def _args(extra: list[str]):
	from wnn.control import phased_ga as pg
	argv = shlex.split(GT_ARGV) + shlex.split(SHRINK) + extra
	args = pg.build_arg_parser().parse_args(argv)
	args.cli_argv = argv
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return args, ec


def _gyro_z(t: int) -> float:
	return 0.003 * t


def _features(g, gs) -> list:
	from wnn.control.evaluator import build_controller, controller_genome_from_arch
	c = build_controller(controller_genome_from_arch(g, gs._probe_spec, gs.thresholds))
	gs._lead_getter = repr(getattr(c, "obs_yaw_lead", "absent"))  # ABI 63 has no getter
	c.reset()
	c.seed_att_estimator([1.0, 0.0, 0.0, 0.0])
	out = []
	for t in range(40):
		c.set_vertical_obs(0.55 + 0.001 * t, 0.02, -0.01)
		c.set_horizontal_obs(0.1, -0.05, 0.02, 0.01)
		c.step([0.02, -0.01, _gyro_z(t)], [0.3, -0.2, 9.7], [0.0, 0.0, 0.0])
		out.append([repr(float(v)) for v in c.get_last_feature_vector()])
	return out


def _metrics(m) -> dict:
	keep = ("reward", "stable_rate", "mean_attitude_error_deg", "mean_steady_error_deg",
	        "mean_altitude_error_m", "mean_position_error_m", "fitness")
	return {k: (None if getattr(m, k, None) is None else repr(float(getattr(m, k)))) for k in keep}


def dump(variant: str) -> dict:
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	from wnn.control.feature_layout import feature_names
	import wnn, wnn.control._accel as ra
	args, ec = _args(VARIANTS[variant])
	gs = ControllerGridSearch(args, ec, 31337002)
	g = gs._random_genome(gs._enumerate_points()[0])
	feats = _features(g, gs)
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	digest = list(g.cells.fingerprint()) if g.cells is not None else None
	rows = [_metrics(m) for m in gs._evaluator.score_genomes([g])]
	public = {k: repr(v) for k, v in sorted(vars(args).items())
	          if not k.startswith("_") and k not in ("cli_argv",) + NEW_KEYS}
	return {"names": list(feature_names(gs._probe_spec)), "bpf": int(gs._probe_spec.bits_per_feature),
	        "thresholds": [repr(float(t)) for t in gs.thresholds], "token": pg.calibration_from_args(args).token(),
	        "ctbr": repr(sorted(args._ctbr.items())), "args": public,
	        "prov": {k: v for k, v in pg.block10_tokens(args).items() if (k, v) != ("ylead", "")},  # "" = not printed
	        "features": feats, "train_digest": digest, "scorer_rows": rows,
	        "_abi": int(ra.EXPECTED_ABI), "_getter": gs._lead_getter, "_src": os.path.dirname(wnn.__file__)}


def _dump_in(pythonpath: str, variant: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def _here() -> str:
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_flag_off_is_abi63():
	base_pp = os.environ.get("CTRL80_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL80_BASE_PYTHONPATH to the ABI-63 src tree (its wheel = the venv's)")
	new_pp = os.environ.get("PYTHONPATH", _here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	assert (old["_abi"], new["_abi"]) == (63, 64), (old["_abi"], new["_abi"])
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI 63 and ABI 64"
	assert "yaw_lead" not in new["names"] and new["_getter"] == "None" and old["_getter"] == "'absent'"
	print(f"✓ flag off == ABI 63: names ({len(new['names'])}), thresholds ({len(new['thresholds'])}), token, ctbr, "
	      f"args ({len(new['args'])}), provenance, features ({len(new['features'])} steps), train digest "
	      f"{new['train_digest']}, scorer rows")
	return new


def _rows(d: dict, name: str) -> list:
	i, b = d["names"].index(name), d["bpf"]
	return d["thresholds"][i * b:(i + 1) * b]


def _check_layout_and_ladder(on: dict, off: dict) -> int:
	i = on["names"].index("yaw_lead")
	assert on["names"][i - 1] == "yaw_err_i" and on["names"][i - 2] == "yaw_err", on["names"]
	assert on["names"][:i] + on["names"][i + 1:] == off["names"], "another feature moved or changed"
	b = on["bpf"]
	assert b == 8 and on["thresholds"][:i * b] + on["thresholds"][(i + 1) * b:] == off["thresholds"], \
		"--obs-yaw-lead moved ANOTHER feature's ladder (a cut was stolen)"
	want = sorted([-float(np.radians(c)) for c in LEAD_CUTS_DEG] + [float(np.radians(c)) for c in LEAD_CUTS_DEG])
	assert [float(t) for t in _rows(on, "yaw_lead")] == want, _rows(on, "yaw_lead")
	assert on["token"] == off["token"] and on["ctbr"] == off["ctbr"]
	return i


def _check_values(on: dict, off: dict, i: int, tau: float):
	f32 = np.float32
	ye = on["names"].index("yaw_err")
	for t, (row, row_off) in enumerate(zip(on["features"], off["features"])):
		assert row[:i] + row[i + 1:] == row_off, f"step {t}: an input other than the lead differs from flag off"
		want = f32(float(row[ye])) - f32(tau) * f32(_gyro_z(t))
		assert f32(float(row[i])) == f32(want), f"step {t}: lead {row[i]} != yaw_err - tau*gyro_z = {want}"
	assert any(r[i] != r[ye] for r in on["features"]), "vacuous: the lead never differs from yaw_err"


def test_on(off: dict):
	pp = os.environ.get("PYTHONPATH", _here())
	tau_only, on, on_tau = (_dump_in(pp, v) for v in ("tau_only", "on", "on_tau"))
	for key in BEHAVIOUR + ("prov",):
		assert tau_only[key] == off[key], f"--obs-yaw-lead-tau without --obs-yaw-lead moved {key}"
	print("✓ --obs-yaw-lead-tau alone (flag off) is the flag-off run on every behaviour key")
	i = _check_layout_and_ladder(on, off)
	_check_values(on, off, i, 0.10)
	assert on["train_digest"] != off["train_digest"], "--obs-yaw-lead never reached the trainer's write-back"
	assert on["scorer_rows"] != off["scorer_rows"], "--obs-yaw-lead never reached the scorer"
	assert on["prov"]["ylead"] == "0.1" and "ylead" not in off["prov"]
	assert abs(float(on["_getter"]) - 0.10) < 1e-7 and off["_getter"] == "None", (on["_getter"], off["_getter"])
	print(f"✓ --obs-yaw-lead: one feature at index {i} (after yaw_err_i), ladder +-{LEAD_CUTS_DEG} deg, every other "
	      f"ladder + the token unchanged, value == yaw_err - 0.10*gyro_z (f32, 40 steps), write-back and scorer moved")
	assert _check_layout_and_ladder(on_tau, off) == i and on_tau["thresholds"] == on["thresholds"]
	_check_values(on_tau, off, i, 0.25)
	assert on_tau["features"] != on["features"] and on_tau["prov"]["ylead"] == "0.25"
	assert abs(float(on_tau["_getter"]) - 0.25) < 1e-7
	print("✓ --obs-yaw-lead-tau 0.25: same ladder, lead column == yaw_err - 0.25*gyro_z, wheel getter 0.25")


def test_plumbing_and_refusals():
	from wnn.control.controller_grid_search import ControllerGridSearch
	from wnn.control.evaluator import ControllerSpec, _thresholds_from_samples
	from wnn.control.feature_layout import FAMILY, ZERO_REFERENCED, AxisFamily, feature_names
	from wnn.control.provenance import collect_provenance
	from wnn.ram.experiments.params import KNOWN_PARAMS
	args, ec = _args(VARIANTS["on"])
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	_, spec = gs._compute_valid_pairs()
	assert spec.obs_yaw_lead == 0.10 and _spec_off().obs_yaw_lead is None
	assert FAMILY["yaw_lead"] == AxisFamily.YAW and "yaw_lead" in ZERO_REFERENCED
	line = collect_provenance("x", ylead="0.1").line()
	assert "ylead=0.1 " in line and "ylead" not in collect_provenance("x").line()
	assert {"controller_obs_yaw_lead", "controller_obs_yaw_lead_tau"} <= KNOWN_PARAMS
	_refused(_without(shlex.split(GT_ARGV), "--obs-yaw-err") + shlex.split(SHRINK) + ["--obs-yaw-lead"], "no yaw_err")
	_refused(shlex.split(GT_ARGV) + shlex.split(SHRINK) + ["--obs-yaw-lead", "--bits-per-feature", "6"], "bpf 6")
	for bad in ("0", "-0.1", "1.5", "nan"):
		_refused(shlex.split(GT_ARGV) + shlex.split(SHRINK) + ["--obs-yaw-lead", "--obs-yaw-lead-tau", bad], f"tau {bad}")
	# The fit itself refuses a ladder that is not exactly the 8 fixed cuts (not only the CLI).
	bad_spec = ControllerSpec(num_motors=4, levels_per_motor=2, bits_per_feature=6, input_window_k=1,
	                          state_neurons=0, state_bits_per_neuron=0, output_bits_per_neuron=4,
	                          obs_yaw_err=True, obs_yaw_lead=0.1)
	try:
		_thresholds_from_samples([[0.0, 1.0]] * len(feature_names(bad_spec)), bad_spec, "quantile", None, None)
	except ValueError as e:
		assert "yaw-lead ladder" in str(e), e
	else:
		raise AssertionError("a 6-bit yaw-lead ladder was fitted")
	_exporter_refuses(spec)
	print("✓ spec plumbing, family YAW / zero-referenced, provenance ylead=, KNOWN_PARAMS, refusals (CLI + fit), "
	      "on-board C exporter refuses")


def _exporter_refuses(spec):
	sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
	import export_controller_c as ex
	assert any("obs_yaw_lead" in w for w in ex._yaw_block_refusals(spec)), ex._yaw_block_refusals(spec)
	assert not any("obs_yaw_lead" in w for w in ex._yaw_block_refusals(_spec_off()))


def _without(argv: list[str], flag: str) -> list[str]:
	return [a for a in argv if a != flag]


def _spec_off():
	from wnn.control.controller_grid_search import ControllerGridSearch
	args, ec = _args([])
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	return gs._compute_valid_pairs()[1]


def _refused(argv: list[str], why: str):
	from wnn.control import phased_ga as pg
	args = pg.build_arg_parser().parse_args(argv)
	try:
		pg.episode_config_from_args(args)
		pg.resolve_yaw_channel_args(args)
	except SystemExit as e:
		assert "--obs-yaw-lead" in str(e), f"{why}: refused for another reason: {e}"
		return
	raise AssertionError(f"{why}: {argv[-3:]} was not refused")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	off = test_flag_off_is_abi63()
	test_on(off)
	test_plumbing_and_refusals()
	print("\nCTRL-80 --obs-yaw-lead: flag-off pin vs ABI 63 + flag tests passed.")
