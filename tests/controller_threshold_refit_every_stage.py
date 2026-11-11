"""--threshold-refit-from-student reaches EVERY fit site, on the real-life plant (ABI 49).

Two defects, both measured before the fix (docs/ctrl_dres_fix_offline_tests.md A.4):
  1. the student collector refused the ABI-32 plant (translation, specific-force IMU,
     rotor drag, Mahony, ctbr delta law, d-hat residual) and phased_ga printed
     "keeping the teacher fit" and flew on — a refit run that was not one;
  2. even when it ran, the refit ladder reached the GRID regrid only: NEURONS, MEMORY,
     every held-out and stage-select re-fit the TEACHER ladder (the loose
     `extra_samples` argument, the same 2-of-5 class as the outer quantile, 8542f4c1);
and a third found while fixing them:
  3. the grid winner carries NO cells, so the "student" rolled out was an EMPTY memory.

Sections (recipe = the rf_c200_dres s1 [argv], budgets scaled down):
  A  refit OFF == the pre-change tree (ctrl-dres-ideas 6dcf8f35 + the ABI-48 wheel):
     grid ladder, NEURONS/MEMORY fit, held-out ladder, features, training fingerprint,
     scorer rows — exact, across two subprocesses.
  B  refit ON: the collector runs on this plant, flies a TRAINED student, the refit
     ladder differs from the teacher ladder, and the SAME refit ladder is handed to the
     GRID regrid, NEURONS, MEMORY, the held-out scorer AND its train-seed trainer
     (captured at the real ControllerEvaluator constructors, not recomputed).
  C  every failure stops the run: collector exception, empty pool, no grid winner,
     resume / --recalc-headline with the flag, overactuated runs.

Run (ABI-49 wheel first on the path):
  PYTHONPATH=/tmp/ctrl_refit_site:src/wnn:src \\
  REFIT_BASE_SRC=/tmp/ctrl_refit_base/src REFIT_BASE_SITE=/tmp/ctrl_dres_site \\
    python tests/controller_threshold_refit_every_stage.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BASE_SRC = os.environ.get("REFIT_BASE_SRC", "/tmp/ctrl_refit_base/src")
BASE_SITE = os.environ.get("REFIT_BASE_SITE", "/tmp/ctrl_dres_site")
SEED = 31337001

# logs/controller/ctrl48/C-S2-d_wposx2_a45_rf_c200_dres_full_s1.out [argv], budgets scaled
# (pop/episodes/steps/output neurons/test seeds); every plant, feature and law flag kept.
RECIPE = shlex.split(
	"--levels 16 --lamarckian --max-cells-strict --neurons-gens 1 --neurons-patience 1 "
	"--memory-gens 1 --memory-patience 1 --pop 4 --num-eval-folds 5 --check-interval 2 "
	"--magnitude-aware-patience --eval-episodes 4 --memory-eval-episodes 4 --steps 400 --tilt 5.0 "
	"--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 "
	"--grid-bits 24 --grid-output-neurons 64 --test-episodes 4 --holdout-pop-sample 2 --runs 1 "
	"--memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof "
	"--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i "
	"--obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 "
	"--grid-state-neurons 0 --max-state-neurons 0 --skip-stages connections,bits "
	"--max-output-neurons 64 --max-cells 1000000000 --teacher-hover derived "
	"--xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt "
	"--rotor-drag 0.78 --motor-lag-s 0.20 --output-mode ctbr --action-repeat 10 --ctbr-delta-law "
	"--delta-leak 0.7 --delta-max 0.6 --rg-gate off --fit-weight-err-sq 0.2557 "
	"--fit-weight-stable 0.2045 --fit-weight-steady 0.3580 --fit-weight-pos 0.1818 "
	"--receptive-fields axis --rf-cross 0.2 --calib-episodes 200 --obs-cascade-dhat-residual "
	"--test-seeds 99990101 99990102 --base-seed 31337001 "
	"--rg-rounds 1 --rg-episodes-per-round 2 --rg-eval-episodes 2 --train-workers 2")
REFIT = ["--threshold-refit-from-student", "--threshold-refit-episodes", "4"]


def _prepared(argv):
	"""(args, ec) set up the way phased_ga.main does for this recipe."""
	from wnn.control import phased_ga as pg
	from wnn.control.recurrent_genome import conn_policy_from_name
	args = pg.build_arg_parser().parse_args(argv)
	args.cli_argv = argv
	ec = pg.episode_config_from_args(args)
	args._conn_policy, args._conn_policy_min = conn_policy_from_name(args.conn_policy)
	pg._resolve_ctbr(args, ec)
	args._dhat_b = None
	return args, ec


def _seeds(args):
	from wnn.seeds import resolve_seed_set
	return resolve_seed_set(base=args.base_seed, run_index=0, train=args.train_seed,
	                        test=args.test_seed, val=args.val_seed)


def _floats(xs) -> list:
	return [float(x) for x in xs]


# ---------------------------------------------------------------- A: refit OFF pin
def dump_off() -> dict:
	"""Everything the refit could touch, with the flag OFF (runs in either tree)."""
	from wnn.control import phased_ga as pg
	from wnn.control.controller_grid_search import ControllerGridSearch
	from wnn.control.evaluator import fit_thresholds_from_pid_rollouts
	from wnn.control.threshold_calibration import calibration_from_args
	from controller_ctrl48_abi44_pin import _features, _metrics
	args, ec = _prepared(RECIPE)
	train = _seeds(args).train
	gs = ControllerGridSearch(args, ec, train)
	g = gs._random_genome(gs._enumerate_points()[0])
	phase = fit_thresholds_from_pid_rollouts(gs._probe_spec, calibration=calibration_from_args(args),
		seed=train, geometry=None, alloc=None, episode_config=pg._calib_ec(args, ec))
	report = pg._test_thresholds(args, ec, gs._probe_spec, 99990101, train, True)
	feats = _features(g, gs)
	gs._evaluator._evaluate_core([g], write_back=True, return_stats=True)
	return {"grid": _floats(gs.thresholds), "phase": _floats(phase), "report": _floats(report),
	        "features": feats, "train_digest": list(g.cells.fingerprint()),
	        "scorer_rows": [_metrics(m) for m in gs._evaluator.score_genomes([g])]}


def _run_dump(pythonpath: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([PY, __file__, "--dump-off"], env=env, capture_output=True, text=True, cwd=HERE)
	if r.returncode != 0:
		raise RuntimeError(f"dump @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_a_refit_off_is_unchanged():
	# Absolute: the subprocess runs in tests/, where a relative src/wnn would miss this tree.
	new_pp = ":".join(os.path.abspath(p) for p in os.environ["PYTHONPATH"].split(":") if p)
	old = _run_dump(f"{BASE_SITE}:{BASE_SRC}/wnn:{BASE_SRC}")
	new = _run_dump(new_pp)
	for key in old:
		assert old[key] == new[key], f"refit OFF: {key} moved vs the pre-change tree"
	assert old["grid"] == old["phase"] == old["report"], "teacher ladder differs between fit sites"
	print(f"✓ A refit OFF: grid/phase/report ladders ({len(new['grid'])}), features, train digest "
	      f"{new['train_digest']}, scorer rows identical to ctrl-dres-ideas + ABI 48")


# ---------------------------------------------------------------- B: refit ON
class _Captured(Exception):
	"""Raised by the evaluator stub once the fit site's thresholds are recorded."""


def _capture_evaluators(pg, seen: list, stop: bool):
	"""Replace phased_ga.ControllerEvaluator with a recorder of its `thresholds`."""
	real = pg.ControllerEvaluator

	def stub(spec, **kw):
		seen.append(list(kw["thresholds"]))
		if stop:
			raise _Captured()
		return real(spec, **kw)
	pg.ControllerEvaluator = stub
	return real


def _site(pg, fn, seen: list) -> list:
	"""Thresholds the first ControllerEvaluator built by `fn` received."""
	n = len(seen)
	real = _capture_evaluators(pg, seen, stop=True)
	try:
		fn()
	except _Captured:
		pass
	finally:
		pg.ControllerEvaluator = real
	assert len(seen) > n, "the fit site built no evaluator"
	return seen[n]


def _refit_run():
	"""Stage 0 with the refit ON; returns (pg, args, ec, seeds, grid 5-tuple, collected cg)."""
	from wnn.control import phased_ga as pg
	args, ec = _prepared(RECIPE + REFIT)
	seeds = _seeds(args)
	got = {}
	real = pg.collect_student_feature_samples

	def spy(cg, ec_, eps, seed):
		got["cg"] = cg
		return real(cg, ec_, eps, seed)
	pg.collect_student_feature_samples = spy
	try:
		grid = pg._stage0_with_refit(args, ec, seeds, seeds.train)
	finally:
		pg.collect_student_feature_samples = real
	return pg, args, ec, seeds, grid, got["cg"]


def _teacher_ladder(pg, args, ec, spec, seed) -> list:
	from wnn.control.evaluator import fit_thresholds_from_pid_rollouts
	from wnn.control.threshold_calibration import calibration_from_args
	import copy
	plain = copy.copy(args)
	plain._threshold_refit_samples = None
	return fit_thresholds_from_pid_rollouts(spec, calibration=calibration_from_args(plain), seed=seed,
		geometry=None, alloc=None, episode_config=pg._calib_ec(plain, ec))


def test_b_refit_reaches_every_stage():
	from wnn.ram.strategies.optimization_dimension import OptimizationDimension
	pg, args, ec, seeds, grid, cg = _refit_run()
	winner_spec, pop, _m, _dt, grid_thr = grid
	pool = args._threshold_refit_samples
	assert pool is not None and len(pool) == winner_spec.num_features() == 25, "no 25-feature student pool"
	assert cg.cells_handle is not None, "the refit flew an UNTRAINED grid winner (no cells)"
	teacher = _teacher_ladder(pg, args, ec, winner_spec, seeds.train)
	assert list(grid_thr) != list(teacher), "the refit ladder equals the teacher ladder"
	seen: list = []
	neurons = _site(pg, lambda: pg._run_arch_phase(args, ec, winner_spec, OptimizationDimension.NEURONS,
		1, 1, seeds.train, initial_population=pop), seen)
	memory = _site(pg, lambda: pg._run_memory_phase(args, ec, winner_spec, 1, 1, seeds.train,
		initial_population=pop), seen)
	assert neurons == memory == list(grid_thr), "NEURONS / MEMORY do not fly the GRID's refit ladder"
	_check_holdout(pg, args, ec, seeds, winner_spec, pop, list(grid_thr))
	print(f"✓ B refit ON: {len(pool[0])} trained-student samples/feature on the dres plant; refit ladder "
	      f"!= teacher ({sum(a != b for a, b in zip(grid_thr, teacher))}/{len(teacher)} cuts moved); "
	      f"GRID == NEURONS == MEMORY == held-out train == held-out score == stage-select")


def _check_holdout(pg, args, ec, seeds, spec, pop, ladder: list):
	"""The held-out (and stage-select, the same call on the val seeds) builds a train-seed
	trainer for an arch-only winner and a scorer: both must get the refit ladder."""
	seen: list = []
	real = _capture_evaluators(pg, seen, stop=False)
	try:
		g = pop[0].clone()
		g.cells = None
		pg._holdout_report(args, ec, spec, g, [g], 99990101, seeds.train, "GRID")
	finally:
		pg.ControllerEvaluator = real
	assert len(seen) == 2, f"expected scorer + train-seed trainer, got {len(seen)}"
	assert seen[0] == seen[1] == ladder, "held-out train/score ladder != the refit ladder"


# ---------------------------------------------------------------- C: failures stop the run
def _expect(exc, fn, needle: str):
	try:
		fn()
	except exc as e:
		assert needle in str(e), f"{exc.__name__} without {needle!r}: {e}"
		return
	raise AssertionError(f"expected {exc.__name__} ({needle})")


def test_c_failures_stop_the_run():
	from wnn.control import phased_ga as pg
	from wnn.control.evaluator import collect_student_feature_samples
	args, ec = _prepared(RECIPE + REFIT)
	seeds = _seeds(args)
	_expect(SystemExit, lambda: pg._refit_thresholds_from_student(args, ec, seeds, None, []), "no winner")
	real = pg.collect_student_feature_samples
	for fake, exc, needle in ((lambda *a: (_ for _ in ()).throw(ValueError("boom")), RuntimeError, "boom"),
	                          (lambda *a: [[] for _ in range(25)], SystemExit, "no samples")):
		pg.collect_student_feature_samples = fake
		try:
			_expect(exc, lambda: pg._stage0_with_refit(args, ec, seeds, seeds.train), needle)
		finally:
			pg.collect_student_feature_samples = real
	_expect(SystemExit, lambda: pg._refuse_refit_without_grid(args, "--resume"), "cannot run with --resume")
	import dataclasses
	_expect(NotImplementedError, lambda: collect_student_feature_samples(
		None, dataclasses.replace(ec, geometry=object()), 1, 0), "overactuated")
	print("✓ C no winner / collector error / empty pool / resume / overactuated: each STOPS the run")


if __name__ == "__main__":
	if len(sys.argv) == 2 and sys.argv[1] == "--dump-off":
		import contextlib, io
		sys.path.insert(0, HERE)
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump_off()
		print(json.dumps(d))
	else:
		sys.path.insert(0, HERE)
		test_c_failures_stop_the_run()
		test_b_refit_reaches_every_stage()
		test_a_refit_off_is_unchanged()
		print("\n--threshold-refit-from-student: every-stage + OFF-pin tests passed.")
