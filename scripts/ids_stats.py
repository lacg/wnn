#!/usr/bin/env python3
"""Statistical analysis for IDS cohorts: bootstrap BCa CIs + rank tests.

Addresses RAID'26 Review C major issue 2 (statistical protocol):
- (C2.3) parametric normal-approximation CIs misbehave for bounded, skewed
  metrics (F1 near 1, FPR near 0) -> bias-corrected accelerated (BCa)
  bootstrap CIs on the cohort mean.
- (C2.2) no significance tests -> Mann-Whitney U for INDEPENDENT cohort
  comparisons (different runs), Wilcoxon signed-rank for PAIRED comparisons
  (same flows under two threshold modes), Holm correction across families.

Reads held-out metrics from validation_summaries.threshold_metadata (the
full 7-mode table; best_genomes is incomplete — see memory
reference_validation_summaries_table).

Usage:
  ids_stats.py ci      --like 'XDS-unsw-temporal-16b-Wb%' [--genome-type best_f1] [--mode val_cal]
  ids_stats.py compare --like-a 'UNSW-fitfix-t16b-temporal%' --like-b 'UNSW-fitfix-t8b-temporal%' \
                       --genome-type best_f1 --mode val_cal
  ids_stats.py modes   --like 'XDS-unsw-temporal-16b-Wb%' --genome-type best_f1 --metric f1
"""

import argparse
import json
import sqlite3
from collections import defaultdict

import numpy as np
from scipy import stats

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
METRICS = ("f1", "fpr", "acc")
MODES = ("train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal")
N_BOOT = 10_000
SEED = 20260711


def fetch(db_path: str, like: str, phase_prefix: str) -> dict:
	"""Per-flow held-out metrics: {(genome_type, mode): {flow: {metric: value}}}."""
	con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
	con.row_factory = sqlite3.Row
	rows = con.execute(
		"""SELECT f.name AS flow, vs.genome_type, e.phase_type, vs.threshold_metadata
		FROM validation_summaries vs
		JOIN experiments e ON vs.experiment_id = e.id
		JOIN flows f ON e.flow_id = f.id
		WHERE f.name LIKE ? AND f.status = 'completed' AND e.phase_type LIKE ?
		AND f.name NOT LIKE '%PREEMP%'""",
		(like, phase_prefix + "%"),
	).fetchall()
	con.close()

	out: dict = defaultdict(dict)
	for r in rows:
		try:
			tm = json.loads(r["threshold_metadata"])
		except (TypeError, json.JSONDecodeError):
			continue
		for mode, vals in tm.items():
			if not isinstance(vals, dict):
				continue
			point = {m: vals[m] for m in METRICS if m in vals and vals[m] is not None}
			if point:
				out[(r["genome_type"], mode)][r["flow"]] = point
	return out


def bca_ci(values: np.ndarray, alpha: float = 0.05) -> tuple:
	"""BCa bootstrap CI for the mean. Returns (lo, hi)."""
	if len(values) < 3 or np.ptp(values) == 0.0:
		return (float(values.mean()), float(values.mean()))
	res = stats.bootstrap(
		(values,), np.mean, n_resamples=N_BOOT, confidence_level=1 - alpha,
		method="BCa", random_state=np.random.default_rng(SEED),
	)
	return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def holm(pvals: list) -> list:
	"""Holm step-down adjusted p-values (same order as input)."""
	m = len(pvals)
	order = sorted(range(m), key=lambda i: pvals[i])
	adjusted = [0.0] * m
	running = 0.0
	for rank, i in enumerate(order):
		running = max(running, (m - rank) * pvals[i])
		adjusted[i] = min(1.0, running)
	return adjusted


def cmd_ci(args) -> None:
	data = fetch(args.db, args.like, args.phase)
	print(f"# BCa 95% CIs — cohort LIKE '{args.like}' (phase {args.phase}*, {N_BOOT} resamples)")
	print(f"{'genome_type':<14} {'mode':<21} {'metric':<7} {'n':>4} {'mean%':>8} {'std%':>7} {'BCa 95% CI %':>20}")
	for (gt, mode), flows in sorted(data.items()):
		if args.genome_type and gt != args.genome_type:
			continue
		if args.mode and mode != args.mode:
			continue
		for metric in METRICS:
			vals = np.array([p[metric] for p in flows.values() if metric in p]) * 100.0
			if len(vals) == 0:
				continue
			lo, hi = bca_ci(vals)
			print(f"{gt:<14} {mode:<21} {metric:<7} {len(vals):>4} {vals.mean():>8.2f} {vals.std(ddof=1):>7.2f} [{lo:>8.2f}, {hi:>8.2f}]")


def cmd_compare(args) -> None:
	a = fetch(args.db, args.like_a, args.phase).get((args.genome_type, args.mode), {})
	b = fetch(args.db, args.like_b, args.phase).get((args.genome_type, args.mode), {})
	print(f"# Mann-Whitney U (independent, two-sided) — {args.genome_type}/{args.mode}")
	print(f"# A: '{args.like_a}' (n={len(a)})   B: '{args.like_b}' (n={len(b)})")
	pvals, lines = [], []
	for metric in METRICS:
		va = np.array([p[metric] for p in a.values() if metric in p]) * 100.0
		vb = np.array([p[metric] for p in b.values() if metric in p]) * 100.0
		if len(va) < 3 or len(vb) < 3:
			continue
		u, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
		# rank-biserial effect size: r = 1 - 2U/(n1*n2)
		r_rb = 1.0 - 2.0 * u / (len(va) * len(vb))
		pvals.append(p)
		lines.append((metric, va, vb, u, p, r_rb))
	adj = holm(pvals)
	print(f"{'metric':<7} {'mean_A%':>8} {'mean_B%':>8} {'U':>10} {'p':>12} {'p_holm':>12} {'r_rb':>7}")
	for (metric, va, vb, u, p, r_rb), p_adj in zip(lines, adj):
		print(f"{metric:<7} {va.mean():>8.2f} {vb.mean():>8.2f} {u:>10.1f} {p:>12.3e} {p_adj:>12.3e} {r_rb:>7.3f}")


def cmd_modes(args) -> None:
	data = fetch(args.db, args.like, args.phase)
	series = {}
	for mode in MODES:
		flows = data.get((args.genome_type, mode), {})
		series[mode] = {f: p[args.metric] for f, p in flows.items() if args.metric in p}
	print(f"# Wilcoxon signed-rank (paired by flow, two-sided) — {args.genome_type}, metric={args.metric}")
	print(f"# cohort LIKE '{args.like}'")
	pairs, pvals, lines = [], [], []
	for i, m1 in enumerate(MODES):
		for m2 in MODES[i + 1:]:
			shared = sorted(set(series[m1]) & set(series[m2]))
			if len(shared) < 6:
				continue
			v1 = np.array([series[m1][f] for f in shared]) * 100.0
			v2 = np.array([series[m2][f] for f in shared]) * 100.0
			if np.ptp(v1 - v2) == 0.0:
				continue
			w, p = stats.wilcoxon(v1, v2, alternative="two-sided")
			pairs.append((m1, m2))
			pvals.append(p)
			lines.append((m1, m2, len(shared), v1.mean(), v2.mean(), w, p))
	adj = holm(pvals)
	print(f"{'mode_1':<21} {'mode_2':<21} {'n':>4} {'mean_1%':>8} {'mean_2%':>8} {'W':>9} {'p':>12} {'p_holm':>12}")
	for (m1, m2, n, mu1, mu2, w, p), p_adj in zip(lines, adj):
		print(f"{m1:<21} {m2:<21} {n:>4} {mu1:>8.2f} {mu2:>8.2f} {w:>9.1f} {p:>12.3e} {p_adj:>12.3e}")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--db", default=DB_PATH)
	sub = ap.add_subparsers(dest="cmd", required=True)

	p_ci = sub.add_parser("ci", help="BCa bootstrap CIs per genome_type x mode x metric")
	p_ci.add_argument("--like", required=True)
	p_ci.add_argument("--phase", default="ga")
	p_ci.add_argument("--genome-type", default=None)
	p_ci.add_argument("--mode", default=None)
	p_ci.set_defaults(fn=cmd_ci)

	p_cmp = sub.add_parser("compare", help="Mann-Whitney U between two independent cohorts")
	p_cmp.add_argument("--like-a", required=True)
	p_cmp.add_argument("--like-b", required=True)
	p_cmp.add_argument("--phase", default="ga")
	p_cmp.add_argument("--genome-type", default="best_f1")
	p_cmp.add_argument("--mode", default="val_cal")
	p_cmp.set_defaults(fn=cmd_compare)

	p_modes = sub.add_parser("modes", help="Wilcoxon signed-rank across threshold modes (paired)")
	p_modes.add_argument("--like", required=True)
	p_modes.add_argument("--phase", default="ga")
	p_modes.add_argument("--genome-type", default="best_f1")
	p_modes.add_argument("--metric", default="f1", choices=METRICS)
	p_modes.set_defaults(fn=cmd_modes)

	args = ap.parse_args()
	args.fn(args)


if __name__ == "__main__":
	main()
