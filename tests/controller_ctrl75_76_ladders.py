"""CTRL-75 (L2, --gyro-xy-ladder dense) and CTRL-76 (L3, --yaw-err-outer-cut DEG), Python side.

  * flags OFF are today's G+T, measured across two source trees on the SAME wheel: for the
    flown G+T recipe (the CTRL-71/72 s1 argv below, save paths dropped) the fitted thresholds,
    the calibration token, the resolved ctbr dict and every parsed argument are EQUAL between
    $CTRL75_BASE_PYTHONPATH (the pre-CTRL-75 tree) and this tree;
  * --gyro-xy-ladder dense moves the gyro_x / gyro_y blocks ONLY, to exactly +-0.004, +-0.013,
    +-0.05, +-0.80 rad/s; provenance says gyroxy:dense;
  * --gyro-xy-ladder dense_hold (L2-b) likewise, to +-0.004, +-0.013, +-0.026, +-0.80 rad/s;
    provenance gyroxy:dense_hold;
  * --yaw-err-outer-cut 4.5 moves ONLY the outermost yaw_err pair, to +-4.5 deg (the inner
    +-0.10 / +-0.25 deg and the quantile pair, and yaw_err_i, unchanged); provenance yawout:4.5;
  * both flags together = the union of the two moves; the fixed-outer helper and the
    calibration validation; the flow-param registry knows both keys, the adapter refuses them.

Run: CTRL75_BASE_PYTHONPATH=<pre-CTRL-75 src> PYTHONPATH=src python tests/controller_ctrl75_76_ladders.py
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys

import numpy as np

# The flown G+T s1 argv (logs/controller/ctrl48/..._a62_..._L1_yld_n0g3gz_full_s1.out), save paths dropped.
GT_ARGV = "--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --test-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived --fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1 --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 --receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --drop-constant-features --test-seeds 99990101 99990102 99990103 99990104 99990105 99990106 99990107 99990108 99990109 99990110 99990111 99990112 99990113 99990114 99990115 99990116 99990117 99990118 99990119 99990120 99990121 99990122 99990123 99990124 99990125 99990126 99990127 99990128 99990129 99990130 99990131 99990132 99990133 99990134 99990135 --base-seed 31337001 --obs-att-bias-pred --write-rule vote --vote-share 0.1 --val-units 20 --select-tail p99 --yaw-centred-ladder dense --yaw-integral-leak 0.9995 --obs-tilt-ref-err --zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5 --ctbr-gamma 1 2 2 3 --gyro-z-ladder dense"
GT_TOKEN = "eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense,gyroz:dense"
XY_FLAG = ["--gyro-xy-ladder", "dense"]
YO_FLAG = ["--yaw-err-outer-cut", "4.5"]
XY_CUTS = (0.004, 0.013, 0.05, 0.80)
XYH_FLAG = ["--gyro-xy-ladder", "dense_hold"]
XYH_CUTS = (0.004, 0.013, 0.026, 0.80)
# delta_leak_yaw (CTRL-74, merged after this base) and yaw_integral_ladder (CTRL-79) are also absent
# from the pre-CTRL-75 tree; each is pinned by its own test.
NEW_KEYS = ("gyro_xy_ladder", "yaw_err_outer_cut", "delta_leak_yaw", "yaw_integral_ladder", "obs_yaw_lead", "obs_yaw_lead_tau")


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
	        "ctbr": repr(sorted(args._ctbr.items())), "args": public}


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


def test_flags_off_is_gt():
	base_pp = os.environ.get("CTRL75_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL75_BASE_PYTHONPATH to the pre-CTRL-75 src tree")
	old, new = _dump_in(base_pp, []), _dump_in(_here(), [])
	assert set(old) == set(new)
	for key in old:
		assert old[key] == new[key], f"flags OFF: {key} moved between the pre-CTRL-75 tree and this tree"
	assert new["token"] == GT_TOKEN, new["token"]
	print(f"✓ flags OFF == G+T: {len(new['thresholds'])} thresholds, calib token, ctbr dict and "
	      f"{len(new['args'])} parsed arguments identical across the two trees")


def _check_xy(off: dict, on: dict, cuts: tuple = XY_CUTS) -> None:
	want = [-c for c in reversed(cuts)] + list(cuts)
	b_on = _blocks(on)
	assert b_on["gyro_x"] == want and b_on["gyro_y"] == want, (b_on["gyro_x"], b_on["gyro_y"])


def _check_yaw_outer(off: dict, on: dict) -> None:
	b_off, b_on = _blocks(off), _blocks(on)
	c = math.radians(4.5)
	assert b_on["yaw_err"][1:-1] == b_off["yaw_err"][1:-1], "yaw_err inner/quantile cuts moved"
	assert b_on["yaw_err"][0] == -c and b_on["yaw_err"][-1] == c, b_on["yaw_err"]
	assert b_off["yaw_err"][-1] < c, "the replaced pair must lie inside the new outer cut"


def test_gyro_xy_flag_moves_gyro_xy_only():
	off, on = _dump_in(_here(), []), _dump_in(_here(), XY_FLAG)
	assert _moved(off, on) == ["gyro_x", "gyro_y"], _moved(off, on)
	_check_xy(off, on)
	assert on["token"] == off["token"] + ",gyroxy:dense", on["token"]
	assert on["args"] == off["args"] and on["ctbr"] == off["ctbr"], "flag ON changed another resolved setting"
	print(f"✓ --gyro-xy-ladder dense: gyro_x {_blocks(off)['gyro_x']} -> +-{XY_CUTS}; the other "
	      f"{len(_blocks(off)) - 2} ladders byte-identical; token {on['token']}")


def test_gyro_xy_dense_hold_moves_gyro_xy_only():
	off, on = _dump_in(_here(), []), _dump_in(_here(), XYH_FLAG)
	assert _moved(off, on) == ["gyro_x", "gyro_y"], _moved(off, on)
	_check_xy(off, on, XYH_CUTS)
	assert on["token"] == off["token"] + ",gyroxy:dense_hold", on["token"]
	assert on["args"] == off["args"] and on["ctbr"] == off["ctbr"], "flag ON changed another resolved setting"
	print(f"✓ --gyro-xy-ladder dense_hold: gyro_x/gyro_y -> +-{XYH_CUTS}; the other ladders byte-identical")


def test_yaw_outer_flag_moves_yaw_err_outer_only():
	off, on = _dump_in(_here(), []), _dump_in(_here(), YO_FLAG)
	assert _moved(off, on) == ["yaw_err"], _moved(off, on)
	_check_yaw_outer(off, on)
	assert on["token"] == off["token"] + ",yawout:4.5", on["token"]
	assert on["args"] == off["args"] and on["ctbr"] == off["ctbr"], "flag ON changed another resolved setting"
	deg = [round(math.degrees(t), 3) for t in _blocks(off)["yaw_err"]]
	print(f"✓ --yaw-err-outer-cut 4.5: yaw_err {deg} deg -> outer pair +-4.5 deg; yaw_err_i and the "
	      f"other ladders byte-identical; token {on['token']}")


def test_both_flags_union():
	off, on = _dump_in(_here(), []), _dump_in(_here(), XY_FLAG + YO_FLAG)
	assert _moved(off, on) == ["gyro_x", "gyro_y", "yaw_err"], _moved(off, on)
	_check_xy(off, on)
	_check_yaw_outer(off, on)
	assert on["token"] == off["token"] + ",gyroxy:dense,yawout:4.5", on["token"]
	print("✓ both flags: exactly the union of the two moves")


def test_fixed_outer_helper_and_validation():
	from wnn.control.evaluator import (DENSE_GYRO_XY_HOLD_INNER_RAD_S, DENSE_GYRO_XY_INNER_RAD_S,
	                                   _dense_thresholds, _with_fixed_outer)
	from wnn.control.threshold_calibration import ThresholdCalibration
	assert DENSE_GYRO_XY_INNER_RAD_S == XY_CUTS and DENSE_GYRO_XY_HOLD_INNER_RAD_S == XYH_CUTS
	ts = _dense_thresholds(np.random.default_rng(0).normal(0.0, 0.03, 20000), 8, (0.002, 0.004), None)
	out = _with_fixed_outer(ts, 0.08, 2)
	assert list(out[1:-1]) == list(ts[1:-1]) and out[0] == -0.08 and out[-1] == 0.08
	for bad in ((ts, float(ts[-2]) * 0.5, 2), (ts, 0.08, 4)):
		try:
			_with_fixed_outer(*bad)
		except ValueError:
			continue
		raise AssertionError(f"_with_fixed_outer must refuse {bad[1:]}")
	for kw in ({"yaw_err_outer_deg": 4.5}, {"yaw_centred": True, "yaw_dense": True, "yaw_err_outer_deg": 0.0},
	           {"yaw_centred": True, "yaw_dense": True, "yaw_trim_dense": True, "yaw_err_outer_deg": 4.5},
	           {"gyro_xy_ladder": "quantile"}):
		try:
			ThresholdCalibration(**kw)
		except ValueError:
			continue
		raise AssertionError(f"ThresholdCalibration must refuse {kw}")
	print("✓ fixed-outer helper keeps every inner cut, refuses a non-outer value / no quantile pair; "
	      "calibration refuses the yaw outer cut off the dense yaw ladder")


def test_registry_and_adapter_refusal():
	from wnn.control.flow_adapter import _refuse_unbuilt_keys
	from wnn.ram.experiments.params import KNOWN_PARAMS
	for key, val in (("controller_gyro_xy_ladder", "dense"), ("controller_yaw_err_outer_cut", 4.5)):
		assert key in KNOWN_PARAMS, key
		_refuse_unbuilt_keys({key: None})
		try:
			_refuse_unbuilt_keys({key: val})
		except ValueError:
			continue
		raise AssertionError(f"the dashboard adapter must refuse {key}")
	print("✓ registry knows both keys; the dashboard adapter refuses them (phased_ga-only)")


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--dump":
		import contextlib
		import io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2:])
		print(json.dumps(d))
		sys.exit(0)
	test_fixed_outer_helper_and_validation()
	test_registry_and_adapter_refusal()
	test_gyro_xy_flag_moves_gyro_xy_only()
	test_gyro_xy_dense_hold_moves_gyro_xy_only()
	test_yaw_outer_flag_moves_yaw_err_outer_only()
	test_both_flags_union()
	test_flags_off_is_gt()
	print("ALL CTRL-75/76 ladder tests passed")
