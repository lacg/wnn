"""CTRL-74 (ABI 63, --delta-leak-yaw): the yaw bank's own output leak, Python side + the
flag-off pin vs ABI 62 across the two builds.

  * FLAG OFF == ABI 62, measured across the two builds on the flown G+T recipe (the CTRL-71/72
    G+T s1 argv, logs/controller/ctrl48/C-S2-d_wposx2_a62_rf_c200_dc_biasin_ms01_P_both_L1_yld_
    n0g3gz_full_s1.out, save paths dropped, shrunk for a probe by trailing overrides):
      ABI 62: $CTRL74_BASE_PYTHONPATH (the ABI-62 src tree; the ABI-62 wheel = the venv's);
      ABI 63: this tree with the ABI-63 wheel first on PYTHONPATH.
    Every dump is EQUAL: the fitted thresholds, the calibration token, the resolved ctbr dict,
    every parsed argument, the provenance tokens, 40 steps of features, the Lamarckian
    write-back digest after one short DAgger and the scorer rows.
  * --delta-leak-yaw EQUAL to --delta-leak (0.7) is the flag-off run on every behaviour key
    (only the argument / provenance record that it was passed differ);
  * --delta-leak-yaw 0.95 keeps the address function (thresholds, token, features) and MOVES
    the DAgger write-back and the scorer rows (it reaches the trainer and the scorer);
  * every Python constructor path hands it to the wheel (build_controller getter); provenance
    says dleak_yaw=0.95; refusals (range, no delta law, no yaw bank); the registry knows the key.

Run: CTRL74_BASE_PYTHONPATH=/tmp/ctrl74_base/src \\
     PYTHONPATH=/tmp/ctrl74_site:src python tests/controller_ctrl74_delta_leak_yaw.py
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
# The label-side pin (decoder baseline == DAgger label baseline, per bank, per step; the leak
# reaches the trainer's writes) is Rust: ctrl74_delta_leak_yaw_tests.rs. The probe's
# train_digest is the write-back ADDRESS fingerprint, which a teacher-driven probe round
# cannot move (neither --delta-leak nor --delta-max-yaw moves it either, measured 25/11).
VARIANTS = {"off": [], "equal": ["--delta-leak-yaw", "0.7"], "on": ["--delta-leak-yaw", "0.95"]}
BEHAVIOUR = ("thresholds", "token", "ctbr", "features", "train_digest", "scorer_rows")
ADDRESS = ("thresholds", "token", "ctbr", "features")


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


def _features(g, gs) -> list:
	from wnn.control.evaluator import build_controller, controller_genome_from_arch
	c = build_controller(controller_genome_from_arch(g, gs._probe_spec, gs.thresholds))
	gs._dly_getter = repr(getattr(c, "delta_leak_yaw", "absent"))  # ABI 62 has no getter
	c.reset()
	c.seed_att_estimator([1.0, 0.0, 0.0, 0.0])
	out = []
	for t in range(40):
		c.set_vertical_obs(0.55 + 0.001 * t, 0.02, -0.01)
		c.set_horizontal_obs(0.1, -0.05, 0.02, 0.01)
		c.step([0.02, -0.01, 0.003 * t], [0.3, -0.2, 9.7], [0.0, 0.0, 0.0])
		out.append([repr(float(v)) for v in c.get_last_feature_vector()])
	return out


def _metrics(m) -> dict:
	keep = ("reward", "stable_rate", "mean_attitude_error_deg", "mean_steady_error_deg",
	        "mean_altitude_error_m", "mean_position_error_m", "fitness")
	return {k: (None if getattr(m, k, None) is None else repr(float(getattr(m, k)))) for k in keep}


def dump(variant: str) -> dict:
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	import wnn, wnn.control._accel as ra
	args, ec = _args(VARIANTS[variant])
	gs = ControllerGridSearch(args, ec, 31337002)
	g = gs._random_genome(gs._enumerate_points()[0])
	feats = _features(g, gs)
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	digest = list(g.cells.fingerprint()) if g.cells is not None else None
	rows = [_metrics(m) for m in gs._evaluator.score_genomes([g])]
	public = {k: repr(v) for k, v in sorted(vars(args).items())
	          if not k.startswith("_") and k not in ("delta_leak_yaw", "cli_argv")}
	return {"thresholds": [repr(float(t)) for t in gs.thresholds], "token": pg.calibration_from_args(args).token(),
	        "ctbr": repr(sorted(args._ctbr.items())), "args": public, "prov": {k: v for k, v in pg.block10_tokens(args).items() if (k, v) != ("dleak_yaw", "")},  # "" = not printed
	        "features": feats, "train_digest": digest, "scorer_rows": rows,
	        "_abi": int(ra.EXPECTED_ABI), "_getter": gs._dly_getter, "_src": os.path.dirname(wnn.__file__)}


def _dump_in(pythonpath: str, variant: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def _here() -> str:
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_flag_off_is_abi62():
	base_pp = os.environ.get("CTRL74_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL74_BASE_PYTHONPATH to the ABI-62 src tree (its wheel = the venv's)")
	new_pp = os.environ.get("PYTHONPATH", _here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	assert (old["_abi"], new["_abi"]) == (62, 63), (old["_abi"], new["_abi"])
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI 62 and ABI 63"
	print(f"✓ flag off == ABI 62: thresholds ({len(new['thresholds'])}), token, ctbr, args ({len(new['args'])}), "
	      f"provenance, features ({len(new['features'])} steps), train digest {new['train_digest']}, scorer rows")
	return new


def test_equal_and_on(off: dict):
	pp = os.environ.get("PYTHONPATH", _here())
	eq, on = _dump_in(pp, "equal"), _dump_in(pp, "on")
	for key in BEHAVIOUR:
		assert eq[key] == off[key], f"--delta-leak-yaw 0.7 (== --delta-leak) moved {key}"
	assert eq["prov"]["dleak_yaw"] == "0.7"
	print("✓ --delta-leak-yaw 0.7 (== --delta-leak) is the flag-off run on every behaviour key")
	for key in ADDRESS:
		assert on[key] == off[key], f"--delta-leak-yaw 0.95 moved the address function ({key})"
	assert on["train_digest"] == off["train_digest"], "the write-back ADDRESS set moved (it is input-only)"
	assert on["scorer_rows"] != off["scorer_rows"], "--delta-leak-yaw 0.95 never reached the scorer"
	assert on["prov"]["dleak_yaw"] == "0.95" and "dleak_yaw" not in off["prov"]
	# build_controller hands the spec field to the wheel (f32 0.95 / None).
	assert abs(float(on["_getter"]) - 0.95) < 1e-7 and off["_getter"] == "None", (on["_getter"], off["_getter"])
	print("✓ --delta-leak-yaw 0.95: address function + write-back address set unchanged; scorer rows moved")


def test_plumbing_and_refusals():
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	from wnn.control.provenance import collect_provenance
	from wnn.ram.experiments.params import KNOWN_PARAMS
	args, ec = _args(VARIANTS["on"])
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	_, spec = gs._compute_valid_pairs()
	assert spec.delta_leak_yaw == 0.95 and spec.delta_leak == 0.7 and spec.delta_max_yaw == 0.5
	assert _spec_off().delta_leak_yaw is None
	line = collect_provenance("x", dleak_yaw="0.95").line()
	assert "dleak_yaw=0.95 " in line and "dleak_yaw" not in collect_provenance("x").line()
	assert "controller_delta_leak_yaw" in KNOWN_PARAMS
	for bad, why in ((["--delta-leak-yaw", "1.5"], "range"), (["--delta-leak-yaw", "0"], "range"),
	                 (["--delta-leak-yaw", "nan"], "range")):
		_refused(bad, why)
	# No delta law: drop --ctbr-delta-law (the CTBR default is absolute).
	argv = [a for a in shlex.split(GT_ARGV) if a != "--ctbr-delta-law"] + shlex.split(SHRINK) + ["--delta-max-yaw", "0.5"]
	argv = _without(argv, "--delta-max-yaw") + ["--delta-leak-yaw", "0.95"]
	_refused_argv(argv, "no delta law")
	print("✓ spec plumbing, provenance dleak_yaw=, KNOWN_PARAMS, refusals")


def _without(argv: list[str], flag: str) -> list[str]:
	out, skip = [], 0
	for a in argv:
		if skip:
			skip -= 1
			continue
		if a == flag:
			skip = 1
			continue
		out.append(a)
	return out


def _spec_off():
	from wnn.control.controller_grid_search import ControllerGridSearch
	args, ec = _args([])
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	return gs._compute_valid_pairs()[1]


def _refused(extra: list[str], why: str):
	_refused_argv(shlex.split(GT_ARGV) + shlex.split(SHRINK) + extra, why)


def _refused_argv(argv: list[str], why: str):
	from wnn.control import phased_ga as pg
	args = pg.build_arg_parser().parse_args(argv)
	try:
		pg.episode_config_from_args(args)
	except SystemExit as e:
		assert "--delta-leak-yaw" in str(e), f"{why}: refused for another reason: {e}"
		return
	raise AssertionError(f"{why}: {argv[-2:]} was not refused")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	off = test_flag_off_is_abi62()
	test_equal_and_on(off)
	test_plumbing_and_refusals()
	print("\nCTRL-74 --delta-leak-yaw: flag-off pin vs ABI 62 + flag tests passed.")
