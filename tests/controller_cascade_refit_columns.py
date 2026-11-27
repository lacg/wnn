"""The cascade refit replaces the cascade blocks' columns BY LAYOUT, not by tail arithmetic.

evaluator._cascade_columns_from_training_cascade used range(nf - n_cas, nf). The CTRL-46
bias predictor ([abp_r, abp_p]) sits AFTER the cascade blocks (controller.rs: append_cascade_
features, then append_att_bias_features), so with both on that range was the predictor's two
columns plus all but the first two cascade columns.

  * every cascade-block combination x predictor on/off: the refit replaces exactly the columns
    feature_layout names as the cascade block (FAILS on the pre-fix tree for every combination
    with the predictor on; the old range is checked here to be wrong exactly there);
  * a sampler that returns another column count is refused;
  * flown recipes are untouched, measured across two source trees on the SAME wheel: for the
    flown G+T argv, the CTRL-69 first arm (cascade inputs + predictor) and G+T + the cascade
    residual, the fitted thresholds and the calibration token are EQUAL between
    $CASFIX_BASE_PYTHONPATH (the pre-fix tree) and this tree, and the refit is never entered
    (--translation fits every column on the Rust training cascade before it).

Run: CASFIX_BASE_PYTHONPATH=<pre-fix src> PYTHONPATH=src python tests/controller_cascade_refit_columns.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import subprocess
import sys

import numpy as np

from controller_ctrl79_yaw_integral_ladder import GT_ARGV

CASCADE_NAMES = ("rate_i_", "rerr_", "cdhat_", "cresid_")
CTRL69 = ["--obs-rate-err", "--obs-rate-err-tau", "0.05", "--obs-rate-err-axes", "roll,pitch",
          "--obs-rate-i", "--obs-rate-i-axes", "yaw"]
COMBOS: dict[str, list[str]] = {
	"rate_i": ["--obs-rate-i"],
	"rate_i(yaw)": ["--obs-rate-i", "--obs-rate-i-axes", "yaw"],
	"rate_err": ["--obs-rate-err"],
	"ctrl69": CTRL69,
	"dhat": ["--obs-cascade-dhat"],
	"resid": ["--obs-cascade-dhat-residual"],
	"resid+rate_i": ["--obs-cascade-dhat-residual", "--obs-rate-i"],
	"dhat+resid": ["--obs-cascade-dhat", "--obs-cascade-dhat-residual"],
	"all four": ["--obs-rate-i", "--obs-rate-err", "--obs-cascade-dhat", "--obs-cascade-dhat-residual"],
}
# Flags that need the predictor: dropped with it for the predictor-off half.
NEEDS_ABP = ("--obs-att-bias-pred", "--obs-tilt-ref-err")
# Recipes of the two-tree pin: the base, and flown-style cascade + predictor recipes.
PINNED: dict[str, list[str]] = {"G+T": [], "G+T + CTRL-69 first arm": CTRL69,
                                "G+T + cascade residual": ["--obs-cascade-dhat-residual"]}


def _tokens(extra: list[str], abp: bool) -> list[str]:
	return [t for t in shlex.split(GT_ARGV) if abp or t not in NEEDS_ABP] + extra


def _probe(tokens: list[str]):
	"""(args, ec, spec) resolved the way phased_ga resolves a run before its grid."""
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	args = pg.build_arg_parser().parse_args(tokens)
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = "spread", 2
	pg.resolve_ctrl46_args(args)
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._rng_master = args, ec, np.random.default_rng(7)
	with contextlib.redirect_stdout(io.StringIO()):
		gs._valid_pairs, spec = gs._compute_valid_pairs()
	return args, ec, spec


def _refit_with(spec, ec, sampler) -> list[list[float]]:
	"""Run the refit on marker columns [f] with `sampler` standing in for the Rust one."""
	from wnn.control import evaluator as ev
	from wnn.control.threshold_calibration import ThresholdCalibration
	samples = [[float(f)] for f in range(spec.num_features())]
	real = ev._training_cascade_samples
	ev._training_cascade_samples = sampler
	try:
		ev._cascade_columns_from_training_cascade(samples, spec, ec, ThresholdCalibration(episodes=1), 0)
	finally:
		ev._training_cascade_samples = real
	return samples


def _replaced_names(spec, ec) -> list[str]:
	from wnn.control.feature_layout import feature_names
	nf = spec.num_features()
	out = _refit_with(spec, ec, lambda s, e, c, seed: [[1000.0 + f] for f in range(nf)])
	names = feature_names(spec)
	assert all(col in ([float(f)], [1000.0 + f]) for f, col in enumerate(out))
	return [names[f] for f, col in enumerate(out) if col == [1000.0 + f]]


def _old_tail_names(spec) -> list[str]:
	"""What range(nf - n_cas, nf) named (the pre-fix columns)."""
	from wnn.control.evaluator import ctbr_cascade_feature_count
	from wnn.control.feature_layout import feature_names
	names = feature_names(spec)
	return names[len(names) - ctbr_cascade_feature_count(spec.ctbr):]


def test_refit_replaces_the_cascade_block_by_name():
	from wnn.control.feature_layout import feature_names
	wrong_before = 0
	for label, extra in COMBOS.items():
		for abp in (True, False):
			_, ec, spec = _probe(_tokens(extra, abp))
			names = feature_names(spec)
			want = [n for n in names if n.startswith(CASCADE_NAMES)]
			assert want and (names[-2:] == ["abp_r", "abp_p"]) == abp, (label, abp, names)
			got = _replaced_names(spec, ec)
			assert got == want, f"{label} (predictor {'on' if abp else 'off'}): refit replaced {got}, cascade block is {want}"
			assert (_old_tail_names(spec) != want) == abp, (label, abp)
			wrong_before += abp
	print(f"✓ refit replaces exactly the cascade block in {2 * len(COMBOS)} layouts "
	      f"({len(COMBOS)} cascade combinations x predictor on/off); the old tail range was wrong in the {wrong_before} with the predictor on")


def test_no_cascade_inputs_is_a_no_op():
	_, ec, spec = _probe(_tokens([], True))

	def never(s, e, c, seed):
		raise AssertionError("the sampler must not run without cascade inputs")
	assert _refit_with(spec, ec, never) == [[float(f)] for f in range(spec.num_features())]
	print("✓ G+T (no cascade inputs): the refit touches nothing and never samples")


def test_wrong_column_count_is_refused():
	_, ec, spec = _probe(_tokens(CTRL69, True))
	nf = spec.num_features()
	try:
		_refit_with(spec, ec, lambda s, e, c, seed: [[0.0] for _ in range(nf - 2)])
	except ValueError as e:
		print(f"✓ a sampler with {nf - 2} columns for a {nf}-feature layout is refused ({e})")
		return
	raise AssertionError("a short sampler result must be refused")


def dump(extra: list[str]) -> dict:
	"""The address function a run gets (thresholds by repr, calibration token), and whether
	the calibration entered the cascade refit."""
	from wnn.control import evaluator as ev
	from wnn.control import phased_ga as pg
	from wnn.control.feature_layout import feature_names
	from wnn.seeds import resolve_seed_set
	args, ec, spec = _probe(_tokens(extra, True))
	entered = []
	real = ev._training_cascade_samples

	def spy(s, e, c, seed):
		import traceback
		entered.append(any(f.name == "_cascade_columns_from_training_cascade" for f in traceback.extract_stack()))
		return real(s, e, c, seed)
	ev._training_cascade_samples = spy
	try:
		s = resolve_seed_set(base=args.base_seed, run_index=0, train=args.train_seed, test=args.test_seed, val=args.val_seed)
		thr = pg._test_thresholds(args, ec, spec, 0, s.train, use_score=True)
	finally:
		ev._training_cascade_samples = real
	return {"thresholds": [repr(float(t)) for t in thr], "names": list(feature_names(spec)),
	        "token": pg.calibration_from_args(args).token(), "sampler_calls": len(entered),
	        "refit_entered": any(entered), "_src": os.path.dirname(pg.__file__)}


def _dump_in(pythonpath: str, extra: list[str]) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", *extra], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"dump @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def _here() -> str:
	return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_flown_recipes_identical_across_trees():
	import hashlib
	base_pp = os.environ.get("CASFIX_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CASFIX_BASE_PYTHONPATH to the pre-fix src tree")
	for label, extra in PINNED.items():
		old, new = _dump_in(base_pp, extra), _dump_in(_here(), extra)
		assert old["_src"] != new["_src"], "both dumps imported the same source tree"
		for key in ("thresholds", "names", "token"):
			assert old[key] == new[key], f"{label}: {key} moved between the pre-fix tree and this tree"
		for d in (old, new):
			assert d["sampler_calls"] >= 1 and not d["refit_entered"], f"{label}: the cascade refit was entered ({d['_src']})"
		sha = hashlib.sha256("\n".join(new["thresholds"]).encode()).hexdigest()[:16]
		print(f"✓ {label}: {len(new['thresholds'])} thresholds (sha256 {sha}) and token {new['token']} identical across "
		      f"the two trees; refit never entered (every column from the Rust training cascade)")


if __name__ == "__main__":
	if len(sys.argv) >= 2 and sys.argv[1] == "--dump":
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2:])
		print(json.dumps(d))
		sys.exit(0)
	test_no_cascade_inputs_is_a_no_op()
	test_wrong_column_count_is_refused()
	test_refit_replaces_the_cascade_block_by_name()
	test_flown_recipes_identical_across_trees()
	print("ALL cascade-refit column tests passed")
