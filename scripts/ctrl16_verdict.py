"""CTRL-16 verdict: does the FULL five-stage pipeline replace the anchor?

Pre-registered 25/09/2026 in experiments/ctrl16_rule.json, BEFORE any _full30
marker existed. Applies the promotion gate's bar unchanged (same HEADLINE rows,
same four columns, same n=4 paired mean) plus one parsimony clause:

  REPLACE iff >=3/4 columns favour _full30 (stable up, err down, steady down,
              alt down) on the mean paired delta vs _pipeN30
          AND err delta <= +0.10 deg
          AND mean TRUE keys of the _full30 winners <= that of the _pipeN30 winners
  otherwise the simpler pipeline stands (a tie goes to _pipeN30).

The TRUE-key count is the deployable footprint (experiments/h743_keys.json,
via scripts/count_true_keys.py). Usage: python scripts/ctrl16_verdict.py
Writes experiments/ctrl16_verdict.json; exits 1 if any marker is missing.
"""
import json
import os
import re
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = os.path.join(ROOT, "experiments", "sweepladder_markers")
LOGS = os.path.join(ROOT, "logs", "controller", "sweep_ladder")
KEYS = os.path.join(ROOT, "experiments", "h743_keys.json")
OUT = os.path.join(ROOT, "experiments", "ctrl16_verdict.json")
BASE = "SL_C_b24n256_cf21_brushless_L4C_g10"
SEEDS = ["31337002", "31337003", "31337004", "31337005"]
ARM, ANCHOR = "_full30", "_pipeN30"
ERR_CEILING_DEG = 0.10


def tag(seed: str, suffix: str) -> str:
	return f"{BASE}_s{seed}{suffix}"


def headline(t: str) -> list[float]:
	"""stable %, err deg, steady deg, alt m off the marker's HEADLINE held-out row."""
	d = json.load(open(os.path.join(MARK, f"{t}.json")))
	m = re.search(r"stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?alt=([\d.]+)m",
	              d["headline_holdout"])
	return [float(x) for x in m.groups()]


def true_keys(t: str) -> int:
	"""Winner TRUE keys from the cache; counts (and caches) it on a miss."""
	cache = json.load(open(KEYS)) if os.path.exists(KEYS) else {}
	if t not in cache:
		winner = os.path.join(LOGS, f"{t}_winner.yaml.gz")
		subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "count_true_keys.py"),
		                "--winner", winner], check=True, stdout=subprocess.DEVNULL)
		cache = json.load(open(KEYS))
	return int(cache[t]["true_keys"])


def missing_markers() -> list[str]:
	return [tag(s, x) for s in SEEDS for x in (ARM, ANCHOR)
	        if not os.path.exists(os.path.join(MARK, f"{tag(s, x)}.json"))]


def paired_deltas() -> list[list[float]]:
	return [[a - c for a, c in zip(headline(tag(s, ARM)), headline(tag(s, ANCHOR)))]
	        for s in SEEDS]


def footprint() -> dict:
	arm = {s: true_keys(tag(s, ARM)) for s in SEEDS}
	anchor = {s: true_keys(tag(s, ANCHOR)) for s in SEEDS}
	return dict(arm=arm, anchor=anchor,
	            arm_mean=st.mean(arm.values()), anchor_mean=st.mean(anchor.values()))


def verdict() -> dict:
	d = paired_deltas()
	mean = [st.mean(col) for col in zip(*d)]
	fav = [mean[0] > 0, mean[1] < 0, mean[2] < 0, mean[3] < 0]
	gate = sum(fav) >= 3 and mean[1] <= ERR_CEILING_DEG
	fp = footprint()
	smaller = fp["arm_mean"] <= fp["anchor_mean"]
	return dict(
		rule=json.load(open(os.path.join(ROOT, "experiments", "ctrl16_rule.json")))["rule"],
		mean_delta=dict(stable_pp=round(mean[0], 3), err_deg=round(mean[1], 3),
		                steady_deg=round(mean[2], 3), alt_m=round(mean[3], 4)),
		per_seed=dict(zip(SEEDS, [[round(x, 3) for x in row] for row in d])),
		favourable=fav, gate_bar_passed=gate,
		true_keys=fp, footprint_not_larger=smaller,
		replace_anchor=gate and smaller)


def main() -> int:
	gone = missing_markers()
	if gone:
		print("NOT READY — missing markers:\n  " + "\n  ".join(gone), file=sys.stderr)
		return 1
	res = verdict()
	json.dump(res, open(OUT, "w"), indent=1)
	print(json.dumps(res, indent=1))
	print("REPLACE — _full30 becomes the recipe" if res["replace_anchor"]
	      else "KEEP — _pipeN30 stands", file=sys.stderr)
	return 0


if __name__ == "__main__":
	sys.exit(main())
