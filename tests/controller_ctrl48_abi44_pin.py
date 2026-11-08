"""ABI-45 flags OFF == ABI 44, measured across the two builds (CTRL-48).

Runs one recipe-shaped probe TWICE in subprocesses and compares the dumps exactly:
  * ABI 44: the committed tree (git HEAD of $CTRL48_BASE_REPO, default /Users/lacg/wnn)
    with the INSTALLED controller wheel;
  * ABI 45: this tree with whatever wheel is first on PYTHONPATH.
The probe (cascade, translation, posx2 obs flags, BINARY, mpcof teacher; once with
the CTRL-38 rate-I + d-hat blocks, once without) dumps:
  thresholds    the grid's ladder (the ControllerGridSearch fit, default calibration)
  features      40 steps of get_last_feature_vector on fixed sentinel inputs
  train_digest  the Lamarckian write-back cells' fingerprint after one short DAgger
  scorer_rows   score_genomes metrics of the trained genome

Run: PYTHONPATH=/tmp/ctrl48_site:src python tests/controller_ctrl48_abi44_pin.py
     (the --dump mode is the subprocess entry point)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

BASE_REPO = os.environ.get("CTRL48_BASE_REPO", "/Users/lacg/wnn")
PY = sys.executable
RECIPE = ["--airframe", "cf21_brushless", "--translation", "--output-mode", "ctbr", "--teacher", "mpcof",
          "--obs-peraxis-p", "--obs-peraxis-i", "--no-obs-peraxis-yaw", "--obs-yaw-err", "--obs-yaw-err-i",
          "--obs-collective-cmd", "--obs-alt-err", "--obs-vz", "--obs-pos-err-xy", "--obs-vel-xy",
          "--xy-offset", "0.5", "--att-err-ref", "tilt", "--memory-mode", "BINARY", "--disturbance", "L4C",
          "--motor-lag-s", "0.20", "--action-repeat", "10", "--ctbr-delta-law", "--delta-leak", "0.7",
          "--delta-max", "0.6", "--rg-gate", "off", "--tilt", "5.0", "--rotor-drag", "0.78",
          "--grid-state-neurons", "0", "--max-state-neurons", "0", "--grid-bits", "24",
          "--grid-output-neurons", "64", "--max-output-neurons", "64", "--steps", "400",
          "--eval-episodes", "4", "--num-eval-folds", "1", "--rg-rounds", "1",
          "--rg-episodes-per-round", "2", "--rg-eval-episodes", "2", "--pop", "4", "--train-workers", "2"]
VARIANTS = {"base": [], "cascade_blocks": ["--obs-rate-i", "--obs-cascade-dhat"]}


def _grid(argv):
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	args = pg.build_arg_parser().parse_args(argv)
	args.cli_argv = argv
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return ControllerGridSearch(args, ec, 31337002)


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
		out.append([float(v) for v in c.get_last_feature_vector()])
	return out


def _metrics(m) -> dict:
	keep = ("reward", "stable_rate", "mean_attitude_error_deg", "mean_steady_error_deg",
	        "mean_altitude_error_m", "mean_position_error_m", "fitness")
	return {k: (None if getattr(m, k, None) is None else float(getattr(m, k))) for k in keep}


def dump(variant: str) -> dict:
	gs = _grid(RECIPE + VARIANTS[variant])
	g = gs._random_genome(gs._enumerate_points()[0])
	feats = _features(g, gs)
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	digest = list(g.cells.fingerprint()) if g.cells is not None else None
	rows = [_metrics(m) for m in gs._evaluator.score_genomes([g])]
	return {"thresholds": [float(t) for t in gs.thresholds], "features": feats,
	        "train_digest": digest, "scorer_rows": rows}


def _run(variant: str, pythonpath: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([PY, __file__, "--dump", variant], env=env, capture_output=True, text=True,
	                   cwd=os.path.dirname(os.path.abspath(__file__)))
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flags_off_is_abi44():
	here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
	new_pp = os.environ.get("PYTHONPATH", here)
	for variant in VARIANTS:
		old = _run(variant, os.path.join(BASE_REPO, "src"))
		new = _run(variant, new_pp)
		for key in old:
			assert old[key] == new[key], f"{variant}: {key} moved between ABI 44 and ABI 45 (flags off)"
		print(f"✓ {variant}: thresholds ({len(new['thresholds'])}), features ({len(new['features'])} steps), "
		      f"train digest {new['train_digest']}, scorer rows identical to ABI 44")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
	else:
		test_flags_off_is_abi44()
		print("\nABI-45 flags-off pin vs ABI 44 passed.")
