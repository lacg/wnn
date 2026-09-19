"""paired_power.py: --welch (UNPAIRED, anchor n may exceed condition n) + a byte-level
regression of the PAIRED path on the banked _pipeN vs _hd29 markers, seeds 31337002-05.

Run: PYTHONPATH=src/wnn python tests/paired_power_welch.py
"""

import importlib.util
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "paired_power.py")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "paired_power_pipeN_vs_hd29_s02_05.txt")

_spec = importlib.util.spec_from_file_location("pp", SCRIPT)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

from scipy import stats  # noqa: E402  (the venv has it; the tool's exact path uses it)

FAILS = 0


def check(label, got, want):
	global FAILS
	ok = got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<64} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


def close(label, got, want, tol=1e-9):
	check(label, abs(got - want) <= tol, True) if not (math.isnan(got) and math.isnan(want)) else check(label, True, True)


BASE = "SL_C_b24n256_cf21_brushless_L4C_g10_s{seed}"
SEEDS = ["31337002", "31337003", "31337004", "31337005"]


def _cli(extra):
	argv = [sys.executable, SCRIPT, "--arm", "_pipeN", "--base", BASE, "--control-suffix", "_hd29"]
	for s in SEEDS:
		argv += ["--seed", s]
	env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src", "wnn"))
	return subprocess.run(argv + extra, capture_output=True, text=True, env=env, cwd=ROOT)


def _strip_marker_count(text):
	# The first line counts EVERY banked marker in experiments/*_markers; that number
	# grows as arms bank. Everything else is fixed by the four named seed pairs.
	return re.sub(r"^markers loaded: \d+$", "markers loaded: N", text, count=1, flags=re.M)


def test_paired_cli_is_byte_identical_to_the_banked_fixture():
	r = _cli([])
	check("paired CLI exits 0", r.returncode, 0)
	want = open(FIXTURE).read()
	check("paired output == fixture (modulo the marker count line)",
	      _strip_marker_count(r.stdout) == _strip_marker_count(want), True)
	check("paired output prints the sign-test gate", "the win/loss gate" in r.stdout, True)
	check("paired output has NO Welch block", "Welch" in r.stdout, False)


def test_welch_cli_runs_on_the_banked_markers():
	r = _cli(["--welch", "--primary", "err", "--anchor-seed", "31337002", "--anchor-seed", "31337003",
	          "--anchor-seed", "31337004", "--anchor-seed", "31337005", "--anchor-seed", "31337006"])
	check("welch CLI exits 0", r.returncode, 0)
	check("names the test", "UNPAIRED Welch two-sample t" in r.stdout, True)
	check("an unbanked anchor seed is SKIPPED, not silently dropped",
	      "anchor seed 31337006: SKIPPED" in r.stdout, True)
	check("reports both group sizes", "condition seeds: 4   anchor seeds: 4" in r.stdout, True)
	check("PRIMARY tag lands on err", bool(re.search(r"^  err .*\[PRIMARY\]", r.stdout, re.M)), True)
	check("no sign-test gate in unpaired mode", "the win/loss gate" in r.stdout, False)
	check("seeds-needed block is the CONDITION-seeds one", "CONDITION SEEDS NEEDED" in r.stdout, True)


def _marker(tag, stable, err, steady, alt):
	row = (f" [report-seeds] MEMORY MULTI-SEED held-out (5 seeds [1, 2, 3, 4, 5]): stable={stable}±1.0% "
	       f"err={err}±0.10° steady={steady}±0.10° jerk=0.0100±0.0001 mono_viol=6±0 alt={alt}±0.010m")
	return {"tag": tag, "held_memory_multiseed": row}


ARM = [(98.0, 1.50, 1.00, 0.30), (99.0, 1.40, 0.90, 0.28), (97.0, 1.70, 1.10, 0.26), (99.0, 1.60, 1.00, 0.29)]
ANCHOR = [(97.0, 1.80, 1.20, 0.33), (98.0, 1.70, 1.30, 0.35), (96.0, 2.10, 1.10, 0.30), (99.0, 1.60, 1.25, 0.32),
          (97.0, 1.90, 1.15, 0.34), (98.0, 1.75, 1.35, 0.31)]


def _synthetic():
	markers = {}
	for i, v in enumerate(ARM):
		markers[f"B_s{i}_arm"] = _marker(f"B_s{i}_arm", *v)
	for i, v in enumerate(ANCHOR):
		markers[f"B_s{i}_ctl"] = _marker(f"B_s{i}_ctl", *v)
	return markers


def test_welch_row_matches_scipy_unequal_n():
	markers = _synthetic()
	arm_rows = pp.group_values(markers, pp.arm_tags("B_s{seed}", "_arm", [str(i) for i in range(4)]), pp.METRICS)
	ctl_rows = pp.group_values(markers, pp.control_tags("B_s{seed}", [str(i) for i in range(6)], {}, "_ctl"), pp.METRICS)
	check("4 condition rows extracted", len([r for r in arm_rows if "vals" in r]), 4)
	check("6 anchor rows extracted", len([r for r in ctl_rows if "vals" in r]), 6)
	for m, idx in (("stable", 0), ("err", 1), ("steady", 2), ("alt", 3)):
		xs, ys = [v[idx] for v in ARM], [v[idx] for v in ANCHOR]
		row = pp.welch_row(m, arm_rows, ctl_rows, 4, 0.8)
		ref = stats.ttest_ind(xs, ys, equal_var=False)
		sign = 1.0 if pp.LOWER_BETTER[m] else -1.0
		close(f"{m}: signed mean difference", row["d"], sign * (sum(xs) / 4 - sum(ys) / 6))
		close(f"{m}: Welch-Satterthwaite df == scipy", row["df"], float(ref.df), 1e-9)
		close(f"{m}: t statistic (d/SE) == scipy (up to sign)", abs(row["d"] / row["se"]), abs(float(ref.statistic)), 1e-9)
		lo, hi = ref.confidence_interval(0.95)
		close(f"{m}: 95% CI width == scipy", row["hi"] - row["lo"], float(hi - lo), 1e-9)
	row = pp.welch_row("stable", arm_rows, ctl_rows, 4, 0.8)
	check("stable is sign-flipped (higher stable, arm better -> NEGATIVE d)", row["d"] < 0, True)


def test_missing_seed_is_reported_not_dropped():
	markers = _synthetic()
	rows = pp.group_values(markers, pp.arm_tags("B_s{seed}", "_arm", ["0", "1", "9"]), pp.METRICS)
	check("missing tag reported", rows[2].get("missing"), "B_s9_arm")
	markers["B_s7_arm"] = {"tag": "B_s7_arm", "held_memory_multiseed": ""}
	rows = pp.group_values(markers, pp.arm_tags("B_s{seed}", "_arm", ["7"]), pp.METRICS)
	check("marker without a MEMORY row reported", rows[0].get("missing"), "MEMORY row")


def test_welch_power_sizing():
	# Equal SDs and equal n: Welch df = 2n-2 and power matches the pooled two-sample t.
	sd, n, effect = 0.3, 6, 0.5
	se, df = pp.welch_se_df(n, sd, n, sd)
	close("equal-variance df collapses to 2n-2", df, 2 * n - 2)
	close("SE = sd*sqrt(2/n)", se, sd * math.sqrt(2.0 / n))
	p6 = pp.welch_power(n, sd, n, sd, effect)
	p12 = pp.welch_power(2 * n, sd, n, sd, effect)
	check("more condition seeds -> more power", p12 > p6, True)
	check("power in (0,1)", 0.0 < p6 < 1.0, True)
	# Anchor-capped: with the anchor at n=4, SD 0.35, no condition n resolves d=0.2.
	check("anchor-capped target -> None (extend the ANCHOR)", pp.welch_seeds_needed(0.2, 4, 0.35, 0.2, 0.8), None)
	need = pp.welch_seeds_needed(0.2, 8, 0.35, 0.5, 0.8)
	check("reachable target -> a finite condition n", isinstance(need, int) and need >= 2, True)
	check("power at the returned n reaches the target", pp.welch_power(need, 0.2, 8, 0.35, 0.5) >= 0.8, True)
	check("power one seed below does NOT", need == 2 or pp.welch_power(need - 1, 0.2, 8, 0.35, 0.5) < 0.8, True)
	mde = pp.welch_mde(4, 0.2, 8, 0.35, 0.8)
	close("MDE has exactly the target power", pp.welch_power(4, 0.2, 8, 0.35, mde), 0.8, 1e-6)
	check("paired power still routes through the shared nct core",
	      abs(pp.paired_t_power(5, 0.3, 0.5) - pp.nct_power(4, 0.5 / (0.3 / math.sqrt(5)))) < 1e-12, True)
	check("seeds_cell renders None as anchor-capped", pp.seeds_cell(None), "anchor-capped")


if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_") and callable(fn):
			print(f"=== {name}")
			fn()
	print()
	if FAILS:
		print(f"FAILED ({FAILS})"); sys.exit(1)
	print("ALL PASS — paired path byte-identical, Welch matches scipy at unequal n, anchor cap named")
