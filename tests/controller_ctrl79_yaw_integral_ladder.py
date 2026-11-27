"""CTRL-79 (--yaw-integral-ladder {quantile,dense}), Python side (no wheel / ABI change).

  * flag OFF (absent, and explicit `quantile`) is today's G+T, measured across two source trees
    on the SAME wheel: for the flown G+T recipe (the CTRL-71/72 s1 argv below, save paths
    dropped) the fitted thresholds, the calibration token, the resolved ctbr dict and every
    parsed argument are EQUAL between $CTRL79_BASE_PYTHONPATH (the pre-CTRL-79 tree) and this
    tree;
  * --yaw-integral-ladder dense moves the yaw_err_i block ONLY (found BY NAME), to exactly
    -12, -6.5, -3.5, -1.5, +1.5, +3.5, +6.5, +12 (feature units); provenance says yawi:dense;
  * it composes with --gyro-xy-ladder dense as the union of the two moves;
  * refusals: no --obs-yaw-err-i, --bits-per-feature != 8 (CLI and the fit itself); the
    flow-param registry knows the key, the dashboard adapter refuses it.

Run: CTRL79_BASE_PYTHONPATH=<pre-CTRL-79 src> PYTHONPATH=src python tests/controller_ctrl79_yaw_integral_ladder.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import shlex
import subprocess
import sys

import numpy as np

# The flown G+T s1 argv (logs/controller/ctrl48/..._a62_..._L1_yld_n0g3gz_full_s1.out), save paths dropped.
GT_ARGV = "--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --test-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived --fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1 --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 --receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --drop-constant-features --test-seeds 99990101 99990102 99990103 99990104 99990105 99990106 99990107 99990108 99990109 99990110 99990111 99990112 99990113 99990114 99990115 99990116 99990117 99990118 99990119 99990120 99990121 99990122 99990123 99990124 99990125 99990126 99990127 99990128 99990129 99990130 99990131 99990132 99990133 99990134 99990135 --base-seed 31337001 --obs-att-bias-pred --write-rule vote --vote-share 0.1 --val-units 20 --select-tail p99 --yaw-centred-ladder dense --yaw-integral-leak 0.9995 --obs-tilt-ref-err --zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5 --ctbr-gamma 1 2 2 3 --gyro-z-ladder dense"
GT_TOKEN = "eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense"
ON_FLAG = ["--yaw-integral-ladder", "dense"]
OFF_FLAG = ["--yaw-integral-ladder", "quantile"]
XY_FLAG = ["--gyro-xy-ladder", "dense"]
YAWI_CUTS = [-12.0, -6.5, -3.5, -1.5, 1.5, 3.5, 6.5, 12.0]
# Absent from the pre-CTRL-79 tree's namespace (default quantile; this test is its pin).
NEW_KEYS = ("yaw_integral_ladder", "obs_yaw_lead", "obs_yaw_lead_tau")  # + CTRL-80 (ABI 64), absent from older trees


def _probe(extra: list[str]):
	"""(args, ec, spec) resolved the way phased_ga resolves a run before its grid."""
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	args = pg.build_arg_parser().parse_args(shlex.split(GT_ARGV) + extra)
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg.resolve_ctrl46_args(args)
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	gs._valid_pairs, spec = gs._compute_valid_pairs()
	return args, ec, spec


def dump(extra: list[str]) -> dict:
	"""Everything a run's address function and resolution depend on, JSON-exact (floats by repr)."""
	from wnn.control import phased_ga as pg
	from wnn.control.feature_layout import feature_names
	from wnn.seeds import resolve_seed_set
	args, ec, spec = _probe(extra)
	s = resolve_seed_set(base=args.base_seed, run_index=0, train=args.train_seed, test=args.test_seed, val=args.val_seed)
	thr = pg._test_thresholds(args, ec, spec, 0, s.train, use_score=True)
	public = {k: repr(v) for k, v in sorted(vars(args).items()) if not k.startswith("_") and k not in NEW_KEYS}
	return {"thresholds": [repr(float(t)) for t in thr], "names": list(feature_names(spec)),
	        "bpf": int(spec.bits_per_feature), "token": pg.calibration_from_args(args).token(),
	        "ctbr": repr(sorted(args._ctbr.items())), "args": public, "_src": os.path.dirname(pg.__file__)}


def _dump_in(pythonpath: str, extra: list[str]) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", *extra], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"dump @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def _here() -> str:
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _blocks(d: dict) -> dict:
	bpf = d["bpf"]
	return {n: [float(t) for t in d["thresholds"][i * bpf:(i + 1) * bpf]] for i, n in enumerate(d["names"])}


def _moved(off: dict, on: dict) -> list[str]:
	b_off, b_on = _blocks(off), _blocks(on)
	return [n for n in b_off if b_off[n] != b_on[n]]


def test_flag_off_is_gt():
	base_pp = os.environ.get("CTRL79_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL79_BASE_PYTHONPATH to the pre-CTRL-79 src tree")
	old = _dump_in(base_pp, [])
	for label, extra in (("absent", []), ("explicit quantile", OFF_FLAG)):
		new = _dump_in(_here(), extra)
		assert old["_src"] != new["_src"], "both dumps imported the same source tree"
		assert set(old) == set(new)
		for key in old:
			if not key.startswith("_"):
				assert old[key] == new[key], f"flag OFF ({label}): {key} moved between the pre-CTRL-79 tree and this tree"
		assert new["token"] == GT_TOKEN, new["token"]
	print(f"✓ flag OFF (absent and explicit quantile) == G+T: {len(old['thresholds'])} thresholds, calib token "
	      f"{old['token']}, ctbr dict and {len(old['args'])} parsed arguments identical across the two trees "
	      f"({old['_src']} vs {new['_src']})")


def test_flag_on_moves_yaw_err_i_only():
	off, on = _dump_in(_here(), []), _dump_in(_here(), ON_FLAG)
	assert _moved(off, on) == ["yaw_err_i"], _moved(off, on)
	assert on["bpf"] == 8 and _blocks(on)["yaw_err_i"] == YAWI_CUTS, _blocks(on)["yaw_err_i"]
	assert on["token"] == GT_TOKEN + ",yawi:dense", on["token"]
	assert on["args"] == off["args"] and on["ctbr"] == off["ctbr"], "flag ON changed another resolved setting"
	print(f"✓ --yaw-integral-ladder dense: yaw_err_i {_blocks(off)['yaw_err_i']} -> {YAWI_CUTS}; the other "
	      f"{len(_blocks(off)) - 1} ladders byte-identical; token {on['token']}")


def test_composes_with_gyro_xy():
	off, on = _dump_in(_here(), []), _dump_in(_here(), XY_FLAG + ON_FLAG)
	assert _moved(off, on) == ["gyro_x", "gyro_y", "yaw_err_i"], _moved(off, on)
	assert _blocks(on)["yaw_err_i"] == YAWI_CUTS
	assert on["token"] == GT_TOKEN + ",gyroxy:dense,yawi:dense", on["token"]
	print(f"✓ with --gyro-xy-ladder dense: exactly the union of the two moves; token {on['token']}")


def _refuses(fn, exc) -> bool:
	try:
		fn()
	except exc:
		return True
	return False


def test_cli_refusals():
	from wnn.control import phased_ga as pg
	gt = shlex.split(GT_ARGV)
	parse = pg.build_arg_parser().parse_args
	pg.resolve_yaw_integral_ladder_args(parse(gt + ON_FLAG))   # the G+T recipe itself is accepted
	pg.resolve_yaw_integral_ladder_args(parse(gt + ["--bits-per-feature", "10"]))   # flag off: no opinion
	for extra in (["--no-obs-yaw-err-i"], ["--bits-per-feature", "10"], ["--bits-per-feature", "6"]):
		args = parse(gt + ON_FLAG + extra)
		assert _refuses(lambda: pg.resolve_yaw_integral_ladder_args(args), SystemExit), f"must refuse dense with {extra}"
	with contextlib.redirect_stderr(io.StringIO()):
		assert _refuses(lambda: parse(gt + ["--yaw-integral-ladder", "centred"]), SystemExit)
	print("✓ CLI refuses dense without --obs-yaw-err-i, at --bits-per-feature 6 / 10, and an unknown choice")


def test_fit_refusals_and_calibration():
	from wnn.control.evaluator import DENSE_YAW_INTEGRAL_CUTS, _check_yaw_integral_dense, _dense_inner_for
	from wnn.control.threshold_calibration import ThresholdCalibration
	assert [-c for c in reversed(DENSE_YAW_INTEGRAL_CUTS)] + list(DENSE_YAW_INTEGRAL_CUTS) == YAWI_CUTS
	with contextlib.redirect_stdout(io.StringIO()):
		_, _, spec = _probe([])
	_check_yaw_integral_dense(spec)
	for bad in (dataclasses.replace(spec, bits_per_feature=10), dataclasses.replace(spec, obs_yaw_err_i=False)):
		assert _refuses(lambda: _check_yaw_integral_dense(bad), ValueError), "the fit must refuse this spec"
	assert _dense_inner_for("yaw_err_i", spec, True, True, yaw_integral_dense=True) == DENSE_YAW_INTEGRAL_CUTS
	for name in ("yaw_err", "yaw_i_pa", "gyro_z", "roll_err"):
		assert (_dense_inner_for(name, spec, True, True, yaw_integral_dense=True)
		        == _dense_inner_for(name, spec, True, True)), name
	assert ThresholdCalibration().token() == ThresholdCalibration(yaw_integral_dense=False).token()
	assert ThresholdCalibration(yaw_integral_dense=True).token() == ThresholdCalibration().token() + ",yawi:dense"
	print("✓ the fit refuses bits_per_feature != 8 and a layout without yaw_err_i; only yaw_err_i takes the "
	      "fixed cuts; the token gains yawi:dense only when set")


def test_registry_and_adapter_refusal():
	from wnn.control.flow_adapter import _refuse_unbuilt_keys
	from wnn.ram.experiments.params import KNOWN_PARAMS
	key = "controller_yaw_integral_ladder"
	assert key in KNOWN_PARAMS, key
	_refuse_unbuilt_keys({key: None})
	assert _refuses(lambda: _refuse_unbuilt_keys({key: "dense"}), ValueError), "the adapter must refuse it"
	print("✓ registry knows controller_yaw_integral_ladder; the dashboard adapter refuses it (phased_ga-only)")


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--dump":
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2:])
		print(json.dumps(d))
		sys.exit(0)
	test_fit_refusals_and_calibration()
	test_cli_refusals()
	test_registry_and_adapter_refusal()
	test_flag_on_moves_yaw_err_i_only()
	test_composes_with_gyro_xy()
	test_flag_off_is_gt()
	print("ALL CTRL-79 yaw-integral-ladder tests passed")
