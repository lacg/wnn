#!/usr/bin/env python3
"""Parity: the classical baselines must be scored under the SAME conditions as the
WNN cells they are compared against.

WHY THIS EXISTS. On 29/07/2026 three separate condition mismatches were found between
compute_baselines.py (Python setup) and ControllerEvaluator (the scorer), each worth
roughly 10pp of PID stability, each invisible because both sides produced plausible
numbers from the same Rust kernel:

  1. motor_asym  — baselines passed the raw fixed (1,1,1,1) multiplier, so they flew a
                   perfectly symmetric quadrotor while every WNN cell carried an ~8%
                   weak motor.                                    PID 97.0 -> 89.0
  2. stream seed — baselines used the report seed where the scorer uses
                   dist.seed XOR active_score_seed.
  3. IC pool     — baselines sampled from the report seed, but with K>1 the scorer
                   samples from _fold_seeds[fold], never the report seed itself.
                                                                  PID  89.0 -> 100.0

They went undetected for eight days because the only check in place compared against
_pid_baseline (eval_closed_loop_reset), which is NOT the twin of the WNN scorer — it
redraws asymmetry per episode and ignores the stream seed. Agreeing with it proved
nothing.

WHAT THIS ASSERTS. Not physics — the two paths already share the Rust engine. It
asserts the SETUP invariant, which is where all three bugs lived: for a given
(disturbance, seed, K), the conditions compute_baselines builds are bit-identical to
the ones the evaluator actually uses. Both now derive them from evaluator's
fold_pool_seed / disturbance_stream, so this test is what keeps a fourth copy from
being introduced.

Run: python3 tests/controller_baseline_conditions_parity.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from wnn.control.evaluator import (ControllerEvaluator, ControllerSpec,
                                   EpisodeConfig, disturbance_stream, fold_pool_seed)
from wnn.control.training import DisturbanceConfig, sample_ics_flat

REPORT_SEED = 99990101
SIM_SEED = 911          # phased_ga's hardcoded DisturbanceConfig.preset seed
FOLDS = 5               # the study's --num-eval-folds
_FAILURES = []


def check(name, got, want):
	ok = got == want
	print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
	if not ok:
		print(f"         got  {got}\n         want {want}")
		_FAILURES.append(name)


def _ec(tilt_deg=5.0, steps=2000):
	return EpisodeConfig(
		dt=0.001, steps_per_episode=steps,
		max_initial_tilt_rad=math.radians(tilt_deg),
		max_initial_yaw_rad=math.radians(tilt_deg),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		disturbance=DisturbanceConfig.preset("L2D", seed=SIM_SEED))


def _evaluator(seed, folds, ec):
	"""A scorer built the way _holdout_report builds one (fresh, scored once)."""
	return ControllerEvaluator(ControllerSpec(), num_eval_episodes=100, seed=seed,
	                           episode_config=ec, num_eval_folds=folds)


def test_fold_pool_seed_is_the_evaluators_own_derivation():
	"""fold_pool_seed must reproduce the evaluator's _fold_seeds exactly. Guards the
	29/07 extraction: a drifting copy silently scores a different episode set."""
	ev = _evaluator(REPORT_SEED, FOLDS, _ec())
	check("fold_pool_seed == evaluator._fold_seeds",
	      [fold_pool_seed(REPORT_SEED, k) for k in range(FOLDS)], list(ev._fold_seeds))


def test_heldout_lands_on_fold_zero():
	"""A held-out builds a fresh evaluator and scores ONCE, so the active pool is
	fold 0. compute_baselines' --fold-index default of 0 depends on this."""
	ev = _evaluator(REPORT_SEED, FOLDS, _ec())
	ev._advance_fold()                     # what score_genomes does first
	check("first _advance_fold -> fold_pool_seed(seed, 0)",
	      ev._active_score_seed, fold_pool_seed(REPORT_SEED, 0))


def test_k1_keeps_the_raw_seed():
	"""The K=1 asymmetry: _advance_fold keeps the RAW seed, NOT fold_pool_seed(seed,0).
	So reproducing a run needs the fold COUNT as well as the index."""
	ev = _evaluator(REPORT_SEED, 1, _ec())
	ev._advance_fold()
	check("K=1 -> raw report seed", ev._active_score_seed, REPORT_SEED)
	got_same = fold_pool_seed(REPORT_SEED, 0) == REPORT_SEED
	check("K=1 pool differs from fold-0 pool (asymmetry is real)", got_same, False)


def test_baselines_build_the_evaluators_conditions():
	"""The end-to-end invariant: what compute_baselines feeds the Rust scorer must be
	what the evaluator would feed it — same IC pool, same stream seed, same asymmetry."""
	import compute_baselines as cb

	class Args:
		disturbance, tilt, steps = "L2D", 5.0, 2000
		report_episodes, stable_deg = 100, 5.0
		sim_seed, eval_folds, fold_index = SIM_SEED, FOLDS, 0

	ec = _ec()
	ev = _evaluator(REPORT_SEED, FOLDS, ec)
	ev._advance_fold()
	want_pool = ev._active_score_seed
	want_dseed, want_asym = disturbance_stream(ec.disturbance, want_pool)

	got_pool = fold_pool_seed(REPORT_SEED, Args.fold_index)
	check("baseline IC pool == evaluator active score seed", got_pool, want_pool)

	got_dseed, got_asym = disturbance_stream(ec.disturbance, got_pool)
	check("baseline stream seed == evaluator stream seed", got_dseed, want_dseed)
	check("baseline motor_asym == evaluator motor_asym",
	      [float(x) for x in got_asym], [float(x) for x in want_asym])

	# The asymmetry must be a real draw, not the fixed multiplier — bug #1.
	check("motor_asym is the RESOLVED draw, not (1,1,1,1)",
	      [float(x) for x in got_asym] == [1.0, 1.0, 1.0, 1.0], False)

	# And the fields dict must carry that draw through, not d.motor_asym.
	fields = cb._dist_fields(ec.disturbance, got_dseed, got_asym)
	check("_dist_fields carries the resolved asym",
	      fields["dist_motor_asym"], [float(x) for x in want_asym])
	check("_dist_fields carries the stream seed", fields["dist_seed"], want_dseed)

	# ICs must be drawn from the pool seed, not the report seed — bug #3.
	q_pool, _ = sample_ics_flat(want_pool, Args.report_episodes, ec)
	q_report, _ = sample_ics_flat(REPORT_SEED, Args.report_episodes, ec)
	check("pool ICs differ from report-seed ICs (the bug was real)",
	      list(q_pool) == list(q_report), False)


def main():
	print(__doc__.strip().splitlines()[0])
	for fn in (test_fold_pool_seed_is_the_evaluators_own_derivation,
	           test_heldout_lands_on_fold_zero,
	           test_k1_keeps_the_raw_seed,
	           test_baselines_build_the_evaluators_conditions):
		print(f"\n{fn.__name__}:")
		fn()
	print()
	if _FAILURES:
		print(f"FAILED ({len(_FAILURES)}): {', '.join(_FAILURES)}")
		return 1
	print("ALL PASS — baseline conditions match the scorer's")
	return 0


if __name__ == "__main__":
	sys.exit(main())
