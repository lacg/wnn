#!/usr/bin/env python3
"""The held-out threshold gate: score a trained genome through the address function
it was TRAINED under. UNCONDITIONAL since 03/08/2026 — there is no flag any more.

WHY: thresholds are the per-feature thermometer cut-points for the INPUT sensors,
and connections + thresholds together decide WHICH ADDRESS each neuron reads
(evaluator.py:10-16). A genome's cells are written at addresses computed under the
TRAIN-seed thresholds; refitting on the report seed re-quantizes the inputs, so the
same physical state maps elsewhere and the trained memory is read where nothing was
written. Measured cost on frozen winners over 5 report seeds:
    1layer_9feat_BINARY_s31337003  48.0+-13.8  ->  86.8+-1.7
    dfa_9feat_BINARY_s31337003     67.0+- 6.2  ->  87.6+-1.9

WHY THE FLAG IS GONE (03/08/2026, Luiz): it shipped default-OFF on 01/08, then
default-ON earlier the same day. Luiz asked the right question — why is a bug fix
optional at all, and is there ever a run that wants the bug? There is not. A flag
implies a choice; this is whether a genome is read at addresses it was written to.
The old default cost 26 dfa1l cells reporting 0-27% stable for architectures that
measure 98-100% once aligned.

There is deliberately NO escape hatch. Reproducing a pre-fix published number is a
git operation — check out the commit that produced it — not a flag that lets today's
code emit yesterday's bug. The both-ways COMPARISON that documents the gap lives in
scripts/rescore_winners.py, which fits both threshold sets itself and never used
this flag.

The gate is exercised through the real _report_thresholds() with the real
fit_thresholds_from_pid_rollouts patched out, so the test asserts WHICH SEED the fit
is asked for — the only thing the gate decides — without paying for PID rollouts.
"""
import sys, pathlib, types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from wnn.control import phased_ga

TRAIN_SEED, REPORT_SEED = 3072558954, 99990101
FAILS = []


class _Args:
	def __init__(self, fixed):
		self.holdout_fixed_thresholds = fixed


class _Ec:
	geometry = None
	alloc_residual = None


def _capture_seed(args, use_score):
	"""Run the real gate; return the seed it asked the fitter for."""
	seen = {}

	def _fake_fit(spec, num_episodes, seed, geometry=None, alloc=None):
		seen["seed"] = seed
		return [0.0]

	orig = phased_ga.fit_thresholds_from_pid_rollouts
	phased_ga.fit_thresholds_from_pid_rollouts = _fake_fit
	try:
		phased_ga._report_thresholds(args, _Ec(), object(), REPORT_SEED, TRAIN_SEED, use_score)
	finally:
		phased_ga.fit_thresholds_from_pid_rollouts = orig
	return seen.get("seed")


def check(label, got, expect):
	ok = got == expect
	name = {TRAIN_SEED: "TRAIN", REPORT_SEED: "REPORT"}
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<52} -> {name.get(got, got)}")
	if not ok:
		FAILS.append(label)


print("=== the score-only path ALWAYS uses the TRAIN seed ===")
check("score-only  -> reuse TRAIN seed",
      _capture_seed(_Args(True), True), TRAIN_SEED)
check("train-fresh -> still REPORT seed (trains fresh under what it is handed)",
      _capture_seed(_Args(True), False), REPORT_SEED)

print("\n=== the old opt-out no longer has any effect ===")
# _Args(False) sets holdout_fixed_thresholds=False, i.e. exactly what the removed
# --no-holdout-fixed-thresholds used to produce. The gate must ignore it now: a
# stale attribute lying around must not be able to resurrect the bug.
check("score-only  + stale False attr -> STILL the TRAIN seed",
      _capture_seed(_Args(False), True), TRAIN_SEED)
check("train-fresh + stale False attr -> REPORT seed",
      _capture_seed(_Args(False), False), REPORT_SEED)

print("\n=== a missing attribute is fine — nothing is consulted ===")
check("args without the attribute -> TRAIN seed",
      _capture_seed(types.SimpleNamespace(), True), TRAIN_SEED)

print("\n=== the CLI flag is GONE — asking for the bug is not expressible ===")
ap = phased_ga.build_arg_parser()


def _rejects(argv):
    import contextlib, io
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ap.parse_args(argv)
        return False
    except SystemExit:
        return True


check("--holdout-fixed-thresholds is rejected", _rejects(["--holdout-fixed-thresholds"]), True)
check("--no-holdout-fixed-thresholds is rejected", _rejects(["--no-holdout-fixed-thresholds"]), True)
check("parser still builds and parses an empty argv",
      phased_ga.build_arg_parser().parse_args([]) is not None, True)

print()
if FAILS:
	print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
	sys.exit(1)
print("ALL PASS — the fix is unconditional and the bug is no longer expressible")
