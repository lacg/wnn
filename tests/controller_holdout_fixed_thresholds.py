#!/usr/bin/env python3
"""--holdout-fixed-thresholds: score a trained genome through the address function
it was TRAINED under. DEFAULT ON since 03/08/2026 — see the note at the end.

WHY: thresholds are the per-feature thermometer cut-points for the INPUT sensors,
and connections + thresholds together decide WHICH ADDRESS each neuron reads
(evaluator.py:10-16). A genome's cells are written at addresses computed under the
TRAIN-seed thresholds; refitting on the report seed re-quantizes the inputs, so the
same physical state maps elsewhere and the trained memory is read where nothing was
written. Measured cost on frozen winners over 5 report seeds:
    1layer_9feat_BINARY_s31337003  48.0+-13.8  ->  86.8+-1.7
    dfa_9feat_BINARY_s31337003     67.0+- 6.2  ->  87.6+-1.9

WHY THE DEFAULT MOVED (03/08/2026, Luiz): this shipped default-OFF on 01/08 to keep
an in-flight campaign internally consistent. That was wrong. Thresholds are part of
the ADDRESS function, so refitting them is not a calibration preference a run may
hold either way — it reads a trained memory where nothing was written. Leaving the
bug as the default meant new cohorts kept producing wrong numbers by omission: the
dfa1l sweep ran 26 cells reporting 0-27% stable for architectures that measure
98-100% once aligned. The opt-out survives for reproducing pre-fix published numbers,
so choosing the bug is now explicit rather than automatic.

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


print("=== FIXED (the default since 03/08/2026): only the score-only path changes ===")
check("score-only  + fixed -> reuse TRAIN seed",
      _capture_seed(_Args(True), True), TRAIN_SEED)
check("train-fresh + fixed -> still REPORT seed (trains fresh)",
      _capture_seed(_Args(True), False), REPORT_SEED)

print("\n=== OPT-OUT (--no-holdout-fixed-thresholds): the legacy refit, on request ===")
check("score-only  + opt-out -> refit on REPORT seed",
      _capture_seed(_Args(False), True), REPORT_SEED)
check("train-fresh + opt-out -> refit on REPORT seed",
      _capture_seed(_Args(False), False), REPORT_SEED)

print("\n=== a missing attribute must not crash, and must FAIL SAFE to fixed ===")
# getattr's fallback is the safety net for any caller that predates the flag. It
# defaults to the CORRECT axis now: an old caller should not silently get the bug.
check("args without the attribute -> TRAIN seed (fail safe)",
      _capture_seed(types.SimpleNamespace(), True), TRAIN_SEED)

print("\n=== the CLI defaults ON and the bug must be asked for explicitly ===")
ap = phased_ga.build_arg_parser()
REQUIRED = ["--winner", "logs/x_winner.yaml.gz"]   # satisfy any required args


def _parse(extra):
	try:
		return ap.parse_args(extra)
	except SystemExit:
		return None


d = _parse([])
check("default (no flag) -> True", getattr(d, "holdout_fixed_thresholds", "MISSING") if d else "PARSE-FAIL", True)
o = _parse(["--holdout-fixed-thresholds"])
check("--holdout-fixed-thresholds -> True", getattr(o, "holdout_fixed_thresholds", "MISSING") if o else "PARSE-FAIL", True)
n = _parse(["--no-holdout-fixed-thresholds"])
check("--no-holdout-fixed-thresholds -> False", getattr(n, "holdout_fixed_thresholds", "MISSING") if n else "PARSE-FAIL", False)

print()
if FAILS:
	print(f"FAILED ({len(FAILS)}): " + "; ".join(FAILS))
	sys.exit(1)
print("ALL PASS — flag off is byte-identical legacy; flag on fixes only the score-only path")
