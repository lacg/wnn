"""CTRL-71 (--gyro-z-ladder dense): the fixed yaw-rate ladder, Python side.

  * flag OFF is today's N0, measured across two source trees on the SAME wheel: for the
    flown N0 recipe (the CTRL-67 s1 argv below, save paths dropped) the fitted thresholds,
    the calibration token, the resolved ctbr dict and every parsed argument are EQUAL
    between $CTRL71_BASE_PYTHONPATH (the pre-CTRL-71 tree) and this tree;
  * flag ON moves the gyro_z block ONLY, to exactly +-0.011, +-0.05, +-0.15, +-0.45 rad/s;
    every other feature's ladder is byte-identical to flag OFF; provenance says gyroz:dense;
  * the ladder itself (_dense_thresholds with the fixed cuts) at other widths;
  * the flow-param registry knows the key and the dashboard adapter refuses it.

Run: CTRL71_BASE_PYTHONPATH=/tmp/ctrl71_base/src PYTHONPATH=src python tests/controller_ctrl71_gyro_z_ladder.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import numpy as np

# The flown CTRL-67 N0 s1 argv (logs/controller/ctrl48/..._a61_..._L1_yld_n0_full_s1.out), save paths dropped.
N0_ARGV = "--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --test-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived --fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1 --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 --receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --drop-constant-features --test-seeds 99990101 99990102 99990103 99990104 99990105 99990106 99990107 99990108 99990109 99990110 99990111 99990112 99990113 99990114 99990115 99990116 99990117 99990118 99990119 99990120 99990121 99990122 99990123 99990124 99990125 99990126 99990127 99990128 99990129 99990130 99990131 99990132 99990133 99990134 99990135 --base-seed 31337001 --obs-att-bias-pred --write-rule vote --vote-share 0.1 --val-units 20 --select-tail p99 --yaw-centred-ladder dense --yaw-integral-leak 0.9995 --obs-tilt-ref-err --zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5"
FLAG = ["--gyro-z-ladder", "dense"]
CUTS = (0.011, 0.05, 0.15, 0.45)
# Flags added at or after CTRL-71 (off by default; their flag-off pins are their own tests,
# e.g. tests/controller_ctrl75_76_ladders.py): absent from the pre-CTRL-71 tree's namespace.
LATER_KEYS = ("gyro_z_ladder", "gyro_xy_ladder", "yaw_err_outer_cut")


def _probe(extra: list[str]):
	"""(args, ec, spec) resolved the way phased_ga resolves a run before its grid."""
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	args = pg.build_arg_parser().parse_args(shlex.split(N0_ARGV) + extra)
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
	public = {k: repr(v) for k, v in sorted(vars(args).items()) if not k.startswith("_") and k not in LATER_KEYS}
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


def test_flag_off_is_n0():
	base_pp = os.environ.get("CTRL71_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL71_BASE_PYTHONPATH to the pre-CTRL-71 src tree")
	old, new = _dump_in(base_pp, []), _dump_in(_here(), [])
	assert set(old) == set(new)
	for key in old:
		assert old[key] == new[key], f"flag OFF: {key} moved between the pre-CTRL-71 tree and this tree"
	assert new["token"] == "eps:200,regime:flown,ladder:quantile,yaw:dense,tilt:dense", new["token"]
	print(f"✓ flag OFF == N0: {len(new['thresholds'])} thresholds, calib token, ctbr dict and "
	      f"{len(new['args'])} parsed arguments identical across the two trees")


def _blocks(d: dict) -> dict:
	bpf = d["bpf"]
	return {n: d["thresholds"][i * bpf:(i + 1) * bpf] for i, n in enumerate(d["names"])}


def test_flag_on_moves_gyro_z_only():
	off, on = _dump_in(_here(), []), _dump_in(_here(), FLAG)
	b_off, b_on = _blocks(off), _blocks(on)
	moved = [n for n in b_off if b_off[n] != b_on[n]]
	assert moved == ["gyro_z"], f"flag ON moved {moved}"
	want = [-c for c in reversed(CUTS)] + list(CUTS)
	assert [float(t) for t in b_on["gyro_z"]] == want, b_on["gyro_z"]
	assert on["token"] == off["token"] + ",gyroz:dense", on["token"]
	assert on["args"] == off["args"] and on["ctbr"] == off["ctbr"], "flag ON changed another resolved setting"
	print(f"✓ flag ON: gyro_z {[float(t) for t in b_off['gyro_z']]} -> {want}; the other "
	      f"{len(b_off) - 1} ladders byte-identical; token {on['token']}")


def test_ladder_widths():
	from wnn.control.evaluator import DENSE_GYRO_Z_INNER_RAD_S, _dense_thresholds
	assert DENSE_GYRO_Z_INNER_RAD_S == CUTS
	arr = np.random.default_rng(0).normal(0.0, 0.3, 20000)
	t8 = _dense_thresholds(arr, 8, CUTS, None)
	assert list(t8) == [-c for c in reversed(CUTS)] + list(CUTS), "bpf 8: the four fixed pairs, sample-independent"
	t6 = _dense_thresholds(arr, 6, CUTS, None)
	assert list(t6) == [-0.15, -0.05, -0.011, 0.011, 0.05, 0.15], "bpf 6: the inner three pairs"
	t10 = _dense_thresholds(arr, 10, CUTS, None)
	assert list(t10[1:9]) == list(t8) and t10[9] > 0.45 and t10[0] == -t10[9], "bpf 10: one quantile pair beyond 0.45"
	t9 = _dense_thresholds(arr, 9, CUTS, None)
	assert t9[4] == 0.0 and list(t9[:4]) + list(t9[5:]) == list(t8), "odd bpf: one cut at 0"
	for t in (t6, t8, t9, t10):
		assert np.all(np.diff(t) > 0), "strictly increasing"
	print("✓ ladder: fixed pairs at bpf 6/8, a 0 cut at odd bpf, quantile pair(s) beyond the last cut at bpf 10")


def test_registry_and_adapter_refusal():
	from wnn.control.flow_adapter import _refuse_unbuilt_keys
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert "controller_gyro_z_ladder" in KNOWN_PARAMS
	_refuse_unbuilt_keys({"controller_gyro_z_ladder": None})
	try:
		_refuse_unbuilt_keys({"controller_gyro_z_ladder": "dense"})
	except ValueError:
		print("✓ registry knows controller_gyro_z_ladder; the dashboard adapter refuses it (phased_ga-only)")
		return
	raise AssertionError("the dashboard adapter must refuse controller_gyro_z_ladder")


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--dump":
		import contextlib
		import io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2:])
		print(json.dumps(d))
		sys.exit(0)
	test_ladder_widths()
	test_registry_and_adapter_refusal()
	test_flag_on_moves_gyro_z_only()
	test_flag_off_is_n0()
	print("ALL CTRL-71 gyro_z ladder tests passed")
