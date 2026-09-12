#!/usr/bin/env python3
"""Exact binomial CI on the STABLE column — multi-axis spec R11.

stable% is a bounded per-episode pass rate; a t-CI on its per-seed mean is
approximate and its paired SD swings 0.4-3.4 pp between arms. R11 asks for the
FAILURE COUNT with an exact (Clopper-Pearson) interval instead.

Where the count comes from, in order of preference:
  1. the marker's MULTI-SEED line carries `stable_fail=k/n` (runs after 11/09/2026);
  2. otherwise the run's .out: the report-seed RESULT lines immediately before each
     `MULTI-SEED held-out` line are one per seed at N=--report-episodes each, and
     `stable=99.0%` at N=100 is exactly 99/100. Every value must sit on the 100/N
     grid or the script REFUSES — a value off the grid means N is wrong, and
     rounding it would print a count that never happened.
So every marker ever banked — the four multi-axis anchors included — gets an exact
interval without a re-fly.

The statistical unit is a DESIGN CHOICE and is printed, not hidden: the interval
pools all n episodes as Bernoulli trials, but episodes within one report seed share
that seed's disturbance stream, so the honest replication unit is the seed. The
per-seed counts are printed beside the pooled interval so clustering is visible.

Usage:
  PYTHONPATH=src python scripts/stable_failure_ci.py --tag <tag> [--tag ...] \\
      [--stage MEMORY|CONNECTIONS|GRID|HEADLINE] [--report-episodes 100] [--vs A B] [--root DIR]
  --vs A B  adds Fisher's exact test on the two runs' failure counts (same stage).
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

from scipy.stats import beta, fisher_exact

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_FIELD = {"GRID": "held_grid_multiseed", "CONNECTIONS": "held_neurons_multiseed",
               "NEURONS": "held_neurons_multiseed", "MEMORY": "held_memory_multiseed"}
_RESULT = re.compile(r"RESULT — during-search winner \(held-out\):\s+stable=([\d.]+)%")
_MULTI = re.compile(r"\[report-seeds\] (\S+) MULTI-SEED held-out \((\d+) seeds")
_FAIL = re.compile(r"stable_fail=(\d+)/(\d+)")
_CROWNED = re.compile(r"\[stage-select\] HEADLINE stage=(\w+) genome=\w+#(\d+)")


@dataclass(frozen=True)
class FailureCount:
	tag: str
	stage: str
	failures: int
	episodes: int
	per_seed: tuple      # failures per report seed, in .out order
	source: str          # "marker" | ".out"

	def rate(self) -> float:
		return self.failures / self.episodes

	def ci95(self) -> tuple[float, float]:
		"""Clopper-Pearson 95% interval on the failure rate."""
		k, n = self.failures, self.episodes
		lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
		hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
		return float(lo), float(hi)


def parse_args():
	ap = argparse.ArgumentParser(description="Exact binomial CI on stable (R11)")
	ap.add_argument("--tag", action="append", required=True, help="marker tag; repeatable")
	ap.add_argument("--stage", default="MEMORY", choices=sorted(set(STAGE_FIELD) | {"HEADLINE"}),
	                help="which held-out row (default MEMORY — the comparison surface, R7)")
	ap.add_argument("--report-episodes", type=int, default=100,
	                help="episodes per report seed when reconstructing from the .out")
	ap.add_argument("--vs", nargs=2, metavar=("A", "B"), default=None,
	                help="two tags to compare with Fisher's exact test")
	ap.add_argument("--root", default=DEFAULT_ROOT,
	                help="tree holding experiments/*_markers and logs/controller (default: this checkout)")
	return ap.parse_args()


def load_marker(root: str, tag: str) -> dict:
	for path in glob.glob(os.path.join(root, "experiments", "*_markers", "*.json")):
		if os.path.basename(path) == tag + ".json":
			return json.load(open(path))
	raise SystemExit(f"no marker for {tag}")


def find_out(root: str, tag: str) -> str:
	for path in glob.glob(os.path.join(root, "logs", "controller", "*", "*.out")):
		if os.path.basename(path) == tag + ".out":
			return path
	raise SystemExit(f"no .out for {tag}")


def count_from_marker(tag: str, marker: dict, stage: str):
	field = STAGE_FIELD.get(stage)
	if field is None:
		return None
	m = _FAIL.search(marker.get(field, "") or "")
	if not m:
		return None
	return FailureCount(tag, stage, int(m.group(1)), int(m.group(2)), (), "marker")


def stage_blocks(out_path: str) -> dict:
	"""{stage label: [per-seed stable%]} — the RESULT lines that precede each
	MULTI-SEED line belong to that line's stage — exactly as many as the line's own
	seed count, because the stage-select val scoring prints RESULT lines too (45 of
	them sit between the MEMORY row and the HEADLINE re-score). HEADLINE-<stage>#k
	labels collapse to HEADLINE. When stage-select crowns a stage's pop[0] it is NOT
	re-scored (the headline IS that stage's row), so HEADLINE then aliases it."""
	blocks, pending, crowned = {}, [], None
	for line in open(out_path, errors="replace"):
		r = _RESULT.search(line)
		if r:
			pending.append(float(r.group(1)))
			continue
		m = _MULTI.search(line)
		if m:
			label = "HEADLINE" if m.group(1).startswith("HEADLINE") else m.group(1)
			n_seeds = int(m.group(2))
			if len(pending) < n_seeds:
				raise SystemExit(f"{out_path}: {label} MULTI-SEED claims {n_seeds} seeds but only "
				                 f"{len(pending)} RESULT lines precede it")
			blocks[label] = pending[-n_seeds:]
			pending = []
			continue
		c = _CROWNED.search(line)
		if c:
			crowned = (c.group(1), int(c.group(2)))
	if "HEADLINE" not in blocks and crowned and crowned[1] == 0 and crowned[0] in blocks:
		blocks["HEADLINE"] = blocks[crowned[0]]
	return blocks


def count_from_out(root: str, tag: str, stage: str, n_per_seed: int) -> FailureCount:
	blocks = stage_blocks(find_out(root, tag))
	if stage not in blocks:
		raise SystemExit(f"{tag}: no {stage} MULTI-SEED block in the .out (have {sorted(blocks)})")
	per_seed = []
	for pct in blocks[stage]:
		passed = pct * n_per_seed / 100.0
		if abs(passed - round(passed)) > 1e-6:
			raise SystemExit(f"{tag}/{stage}: stable={pct}% is not on the 100/{n_per_seed} grid — "
			                 f"wrong --report-episodes; refusing to round a count")
		per_seed.append(n_per_seed - int(round(passed)))
	return FailureCount(tag, stage, sum(per_seed), n_per_seed * len(per_seed), tuple(per_seed), ".out")


def failure_count(root: str, tag: str, stage: str, n_per_seed: int) -> FailureCount:
	fc = count_from_marker(tag, load_marker(root, tag), stage)
	return fc if fc is not None else count_from_out(root, tag, stage, n_per_seed)


def print_row(fc: FailureCount) -> None:
	lo, hi = fc.ci95()
	seeds = "/".join(str(k) for k in fc.per_seed) if fc.per_seed else "—"
	print(f"  {fc.tag:<58} {fc.stage:<11} fail {fc.failures:>3}/{fc.episodes:<4} "
	      f"stable {100 * (1 - fc.rate()):5.1f}%  fail-rate 95% CI [{100 * lo:4.1f}, {100 * hi:4.1f}] %  "
	      f"per-seed {seeds:<12} ({fc.source})")


def main():
	args = parse_args()
	print(f"exact (Clopper-Pearson) 95% CI on the FAILURE rate; unit = pooled episodes "
	      f"(seed clustering visible in per-seed) — stage {args.stage}")
	rows = {t: failure_count(args.root, t, args.stage, args.report_episodes) for t in args.tag}
	for fc in rows.values():
		print_row(fc)
	if args.vs:
		a, b = (rows.get(t) or failure_count(args.root, t, args.stage, args.report_episodes) for t in args.vs)
		table = [[a.failures, a.episodes - a.failures], [b.failures, b.episodes - b.failures]]
		_, p = fisher_exact(table)
		print(f"\n  Fisher exact, {a.tag} vs {b.tag}: failures {a.failures} vs {b.failures} "
		      f"of {a.episodes}/{b.episodes}  p = {p:.3f}"
		      f"  ({'indistinguishable' if p >= 0.05 else 'differs'} at 0.05; pooled-episode unit)")


if __name__ == "__main__":
	main()
