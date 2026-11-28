"""CTRL-86 S1+S2 (ABI 65): the cf21 FLAG-OFF pin vs ABI 64, measured across the two builds on
the flown G+T / l2b recipe (the CTRL-71/72 G+T s1 argv, save paths dropped, shrunk for a probe
by trailing overrides — the CTRL-80 harness):
  ABI 64: $CTRL86_BASE_PYTHONPATH (the ABI-64 src tree; the ABI-64 wheel = the venv's);
  ABI 65: this tree with the ABI-65 wheel first on PYTHONPATH.
Every dump is EQUAL: the feature names, the fitted thresholds, the calibration token, the
resolved ctbr dict, every parsed argument (minus the new keys, which must resolve to 0), the
provenance tokens, 40 steps of features, the Lamarckian write-back digest after one short
DAgger and the scorer rows. A second ABI-65 dump spells S1 out as --motor-lag-down-s 0.20
(down == up) and --pwm-min 0: every behaviour key equal again.

Run: CTRL86_BASE_PYTHONPATH=/tmp/ctrl86_base/src \\
     PYTHONPATH=/tmp/ctrl86_site:src python tests/controller_ctrl86_flag_off.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

GT_ARGV = "--levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --test-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived --fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1 --xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 --fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 --receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --drop-constant-features --test-seeds 99990101 99990102 99990103 99990104 99990105 99990106 99990107 99990108 99990109 99990110 99990111 99990112 99990113 99990114 99990115 99990116 99990117 99990118 99990119 99990120 99990121 99990122 99990123 99990124 99990125 99990126 99990127 99990128 99990129 99990130 99990131 99990132 99990133 99990134 99990135 --base-seed 31337001 --obs-att-bias-pred --write-rule vote --vote-share 0.1 --val-units 20 --select-tail p99 --yaw-centred-ladder dense --yaw-integral-leak 0.9995 --obs-tilt-ref-err --zrank-pos-floor auto --obs-yaw-from-att-est --delta-max-yaw 0.5 --ctbr-gamma 1 2 2 3 --gyro-z-ladder dense"
# Probe-sized overrides (argparse: the last occurrence wins) — the same on both builds.
SHRINK = ("--grid-output-neurons 64 --max-output-neurons 64 --steps 400 --eval-episodes 4 --num-eval-folds 1 "
          "--rg-rounds 1 --rg-episodes-per-round 2 --rg-eval-episodes 2 --pop 4 --train-workers 2 "
          "--calib-episodes 20")
VARIANTS = {"off": [], "down_eq_up": ["--motor-lag-down-s", "0.20", "--pwm-min", "0"]}
BEHAVIOUR = ("names", "thresholds", "token", "ctbr", "features", "train_digest", "scorer_rows", "prov")
NEW_KEYS = ("motor_lag_down_s", "pwm_min", "pwm_min_source")


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


def _prov(args) -> dict:
	from wnn.control import phased_ga as pg
	p = {k: v for k, v in pg.block10_tokens(args).items() if v != ""}
	act = getattr(pg, "actuator_token", None)
	if act is not None and act(args):
		p["actuator"] = act(args)
	return p


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
	new = {k: repr(getattr(args, k)) for k in NEW_KEYS if hasattr(args, k)}
	return {"names": list(feature_names(gs._probe_spec)), "bpf": int(gs._probe_spec.bits_per_feature),
	        "thresholds": [repr(float(t)) for t in gs.thresholds], "token": pg.calibration_from_args(args).token(),
	        "ctbr": repr(sorted(args._ctbr.items())), "args": public, "prov": _prov(args),
	        "features": feats, "train_digest": digest, "scorer_rows": rows,
	        "_new": new, "_ec": repr((getattr(ec, "motor_lag_down_s", None), getattr(ec, "pwm_min", None))),
	        "_abi": int(ra.EXPECTED_ABI), "_src": os.path.dirname(wnn.__file__)}


def _dump_in(pythonpath: str, variant: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def _here() -> str:
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_flag_off_is_abi64() -> dict:
	base_pp = os.environ.get("CTRL86_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL86_BASE_PYTHONPATH to the ABI-64 src tree (its wheel = the venv's)")
	new_pp = os.environ.get("PYTHONPATH", _here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	assert (old["_abi"], new["_abi"]) == (64, 65), (old["_abi"], new["_abi"])
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI 64 and ABI 65"
	assert old["_new"] == {} and new["_new"] == {"motor_lag_down_s": "0.0", "pwm_min": "0.0", "pwm_min_source": "'off'"}, new["_new"]
	assert new["_ec"] == "(0.0, 0.0)", new["_ec"]
	print(f"✓ flag off == ABI 64: names ({len(new['names'])}), thresholds ({len(new['thresholds'])}), token, ctbr, "
	      f"args ({len(new['args'])}), provenance, features ({len(new['features'])} steps), train digest "
	      f"{new['train_digest']}, scorer rows; the new args resolve to 0 / 'off'")
	return new


def test_down_equals_up(off: dict):
	spelled = _dump_in(os.environ.get("PYTHONPATH", _here()), "down_eq_up")
	for key in BEHAVIOUR:
		if key != "prov":
			assert spelled[key] == off[key], f"--motor-lag-down-s 0.20 --pwm-min 0 moved {key}"
	# Provenance records the spelling honestly — and nothing else.
	assert spelled["prov"] == {**off["prov"], "actuator": "down=0.2s"}, spelled["prov"]
	assert spelled["_ec"] == "(0.2, 0.0)", spelled["_ec"]
	print("✓ --motor-lag-down-s 0.20 (== up) --pwm-min 0: every behaviour key equal to flag off")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	off = test_flag_off_is_abi64()
	test_down_equals_up(off)
	print("\nCTRL-86 S1+S2: cf21 flag-off pin vs ABI 64 passed.")
