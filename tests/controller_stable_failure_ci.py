"""R11 failure-count export + exact CI reader.

Run: PYTHONPATH=src python tests/controller_stable_failure_ci.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_spec = importlib.util.spec_from_file_location("sfc", os.path.join(ROOT, "scripts", "stable_failure_ci.py"))
sfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfc)

FAILS = 0


def check(label, got, want):
	global FAILS
	ok = got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<64} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


RESULT = "  RESULT — during-search winner (held-out):  stable={:.1f}%  err=1.00°  steady=0.50°"
MULTI = "  [report-seeds] {} MULTI-SEED held-out (5 seeds [1, 2, 3, 4, 5]): stable=99.0±0.5%  err=1.00±0.10°{}"


def synthetic_out(headline_rescore: bool, crowned: str) -> str:
	lines = []
	for pct in (99.0, 100.0, 98.0, 99.0, 99.0):        # GRID: 5 failures
		lines.append(RESULT.format(pct))
	lines.append(MULTI.format("GRID", ""))
	for pct in (100.0, 100.0, 99.0, 100.0, 100.0):     # MEMORY: 1 failure
		lines.append(RESULT.format(pct))
	lines.append(MULTI.format("MEMORY", ""))
	for pct in (90.0,) * 45:                            # stage-select val scoring noise
		lines.append(RESULT.format(pct))
	lines.append(f"  [stage-select] HEADLINE stage={crowned.split('#')[0]} genome={crowned} (union rank)")
	if headline_rescore:
		for pct in (97.0, 98.0, 99.0, 100.0, 96.0):     # HEADLINE: 10 failures
			lines.append(RESULT.format(pct))
		lines.append(MULTI.format(f"HEADLINE-{crowned}", ""))
	return "\n".join(lines) + "\n"


def with_tree(out_text: str, marker: dict):
	td = tempfile.mkdtemp()
	os.makedirs(os.path.join(td, "experiments", "x_markers"))
	os.makedirs(os.path.join(td, "logs", "controller", "x"))
	json.dump(marker, open(os.path.join(td, "experiments", "x_markers", "T.json"), "w"))
	open(os.path.join(td, "logs", "controller", "x", "T.out"), "w").write(out_text)
	return td


def test_blocks_take_exactly_n_seeds_and_alias_headline():
	td = with_tree(synthetic_out(headline_rescore=True, crowned="MEMORY#1"), {"tag": "T"})
	fc = sfc.failure_count(td, "T", "GRID", 100)
	check("GRID failures from 5 trailing RESULT lines", (fc.failures, fc.episodes, fc.per_seed), (5, 500, (1, 0, 2, 1, 1)))
	fc = sfc.failure_count(td, "T", "MEMORY", 100)
	check("MEMORY failures", (fc.failures, fc.per_seed), (1, (0, 0, 1, 0, 0)))
	fc = sfc.failure_count(td, "T", "HEADLINE", 100)
	check("HEADLINE re-score ignores the 45 val-scoring lines before it", (fc.failures, fc.per_seed), (10, (3, 2, 1, 0, 4)))
	check("source is the .out for a pre-R11 marker", fc.source, ".out")
	td = with_tree(synthetic_out(headline_rescore=False, crowned="MEMORY#0"), {"tag": "T"})
	fc = sfc.failure_count(td, "T", "HEADLINE", 100)
	check("crowned pop[0] is not re-scored -> HEADLINE aliases MEMORY", (fc.failures, fc.per_seed), (1, (0, 0, 1, 0, 0)))


def test_marker_field_takes_precedence():
	td = with_tree(synthetic_out(True, "MEMORY#1"),
	               {"tag": "T", "held_memory_multiseed": "stable=99.4±0.5% err=1.6° stable_fail=7/500"})
	fc = sfc.failure_count(td, "T", "MEMORY", 100)
	check("stable_fail=k/n from the marker wins over the .out", (fc.failures, fc.episodes, fc.source), (7, 500, "marker"))


def test_wrong_report_episodes_is_refused():
	td = with_tree(synthetic_out(True, "MEMORY#1"), {"tag": "T"})
	try:
		sfc.failure_count(td, "T", "GRID", 37)
		check("N=37 puts 99.0% off the grid -> refuses", "returned", "SystemExit")
	except SystemExit as e:
		check("N=37 puts 99.0% off the grid -> refuses", "not on the 100/37 grid" in str(e), True)


def test_clopper_pearson_edges():
	fc = sfc.FailureCount("t", "MEMORY", 0, 500, (), "marker")
	lo, hi = fc.ci95()
	check("k=0: lower bound exactly 0", lo, 0.0)
	check("k=0 of 500: upper bound ~0.74% (rule of three-ish)", round(hi * 100, 2), 0.74)
	fc = sfc.FailureCount("t", "MEMORY", 3, 500, (), "marker")
	lo, hi = fc.ci95()
	check("k=3 of 500: CI ~[0.12, 1.74]%", (round(lo * 100, 2), round(hi * 100, 2)), (0.12, 1.74))


def test_phased_ga_export_helper():
	import wnn.control.phased_ga as p
	rs = [SimpleNamespace(acc=0.99), SimpleNamespace(acc=1.0), SimpleNamespace(acc=0.93)]
	check("counts failures over report_episodes", p._stable_failure_count(rs, SimpleNamespace(report_episodes=100, eval_episodes=10)), (8, 300))
	check("falls back to eval_episodes", p._stable_failure_count(rs, SimpleNamespace(report_episodes=None, eval_episodes=200)), (16, 600))


if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_") and callable(fn):
			print(f"=== {name}")
			fn()
	print()
	if FAILS:
		print(f"FAILED ({FAILS})"); sys.exit(1)
	print("ALL PASS — stable is a count with an exact interval, for every marker ever banked")
