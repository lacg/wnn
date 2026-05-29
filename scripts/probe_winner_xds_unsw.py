"""Pick the winning (thermo-width, weight-set) for the UNSW-temp XDS probe.

The probe runs `widths × {Wa, Wb} × 3 shared seeds = 30 flows` whose only job
is to choose the (width, weight) that Cohort 2 (97 fresh seeds) will use.
This script aggregates the validation_summaries from completed XDS-unsw-temporal
flows and ranks the (width, weight) groups under a user-defined criterion.

The aggregation is mechanical. The ranking *rule* is a research call — that
lives in `select_winner()` near the bottom of this file.

Usage:
  python3 scripts/probe_winner_xds_unsw.py            # default: ranking + table
  python3 scripts/probe_winner_xds_unsw.py --strict   # error if any (w, wt) cell has <3 seeds
  python3 scripts/probe_winner_xds_unsw.py --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
PREFIX = "XDS-unsw-temporal-"
# Restrict to the locked architecture for UNSW-temp (500n × 34b). The earlier
# 250n × 100b probe runs were superseded after the architecture-derivation
# size mismatch was identified — see [[project_cross_dataset_oi_cohorts]].
LOCKED_ARCH = "500n34b"
# 29/05/2026: extended to 3 weight sets {a, b, c}. Wa = CIC-IoT (CE-leaning,
# ce=0.35); Wb = paper "balanced" (f1+fpr heavy, ce=0.1); Wc = CE-heavy NEW
# (ce=0.7) — tests the pre-fix-bug-recreation hypothesis under fixed code.
NAME_RE = re.compile(rf"^XDS-unsw-temporal-(\d+)b-W([abc])-C35-{LOCKED_ARCH}-OI-r(\d+)$")
WEIGHT_LABELS = {"a": "CIC-IoT", "b": "paper", "c": "CE-heavy"}

GENOMES = ["best_fitness", "best_f1", "best_fpr", "best_acc", "best_ce"]
MODES = ["train_cal", "fixed_05", "platt", "beta", "empirical", "empirical_cumulative", "val_cal"]


def load_probe_rows():
	"""Return list of dicts: one per (flow, genome_type, mode) validation cell."""
	con = sqlite3.connect(DB)
	con.row_factory = sqlite3.Row
	cur = con.cursor()
	cur.execute(
		"""SELECT f.id AS flow_id, f.name, e.phase_type,
		          vs.genome_type, vs.threshold_metadata
		   FROM flows f
		   JOIN experiments e ON e.flow_id = f.id
		   JOIN validation_summaries vs ON vs.experiment_id = e.id
		   WHERE f.name LIKE ?
		     AND f.status = 'completed'
		     AND vs.validation_point = 'final'
		     AND e.phase_type = 'ga_neurons'""",
		(PREFIX + "%",),
	)
	rows = []
	for r in cur:
		m = NAME_RE.match(r["name"])
		if not m:
			continue
		width = int(m.group(1))
		weight = m.group(2)            # 'a' or 'b'
		seed = int(m.group(3))
		try:
			tm = json.loads(r["threshold_metadata"] or "{}")
		except Exception:
			continue
		if r["genome_type"] not in GENOMES:
			continue
		for mode in MODES:
			entry = tm.get(mode)
			if not isinstance(entry, dict) or entry.get("f1") is None:
				continue
			rows.append({
				"flow_id": r["flow_id"],
				"width": width,
				"weight": weight,
				"seed": seed,
				"genome_type": r["genome_type"],
				"mode": mode,
				"f1":  entry["f1"]  * 100.0,
				"fpr": entry["fpr"] * 100.0,
				"acc": entry["acc"] * 100.0,
			})
	con.close()
	return rows


def group_by_cell(rows):
	"""Return {(width, weight): {'rows': [...], 'seeds': set(), 'flows': set()}}."""
	groups = defaultdict(lambda: {"rows": [], "seeds": set(), "flows": set()})
	for r in rows:
		key = (r["width"], r["weight"])
		groups[key]["rows"].append(r)
		groups[key]["seeds"].add(r["seed"])
		groups[key]["flows"].add(r["flow_id"])
	return groups


def fmt_pct(xs):
	if not xs:
		return "   —   "
	mean = statistics.fmean(xs)
	sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
	return f"{mean:5.2f}±{sd:4.2f}"


def print_coverage(groups):
	print("\nCoverage by (width, weight):")
	print(f"  {'width':>5} {'wt':>3} | {'flows':>5} {'seeds':>5} {'rows':>5}")
	for (w, wt) in sorted(groups):
		g = groups[(w, wt)]
		print(f"  {w:>5} {wt:>3} | {len(g['flows']):>5} {len(g['seeds']):>5} {len(g['rows']):>5}")


def print_metric_table(groups, mode="train_cal", genome_type="best_fitness"):
	"""Slice the rows down to a single (mode, genome_type) and print mean±std per (w, wt)."""
	print(f"\nMean±std at mode={mode!r}, genome_type={genome_type!r}:")
	print(f"  {'width':>5} {'wt':>3} | {'F1':>10} {'FPR':>10} {'Acc':>10}   n")
	for (w, wt) in sorted(groups):
		rs = [r for r in groups[(w, wt)]["rows"]
		      if r["mode"] == mode and r["genome_type"] == genome_type]
		f1s  = [r["f1"]  for r in rs]
		fprs = [r["fpr"] for r in rs]
		accs = [r["acc"] for r in rs]
		print(f"  {w:>5} {wt:>3} | {fmt_pct(f1s):>10} {fmt_pct(fprs):>10} {fmt_pct(accs):>10}   {len(rs)}")


# ─────────────────────────────────────────────────────────────────────────────
# YOUR CALL — what defines "the winner" of this probe?
# ─────────────────────────────────────────────────────────────────────────────
#
# `groups` is {(width:int, weight:'a'|'b'): {"rows": [...], "seeds": set, "flows": set}}.
# Each row in `rows` is one (genome_type × threshold_mode) cell for one seed:
#   {"width", "weight", "seed", "genome_type", "mode", "f1", "fpr", "acc"}
#
# Return a list of (key, score, reason_str) sorted best→worst.
#   key:    (width, weight)  e.g. (16, 'a')
#   score:  float — higher is better in your ordering
#   reason: short str — printed as the explanation (≤60 chars)
#
# Things to weigh:
#   • Filter to ONE genome_type (best_fitness? best_f1? best_fpr?) or aggregate?
#   • Pick ONE threshold mode (train_cal is the GA's objective; empirical_cumulative
#     is the conservative-FPR alternative; val_cal peeks at the held-out)?
#   • Score = mean F1, or F1 − k·FPR, or rank-product over (F1, -FPR)?
#   • Require all 3 seeds present for a width to be eligible?
#
# Memory says: weight-set Wa is "locked" for Cohort 2, so the meaningful
# remaining decision is *width* — but be explicit about why and don't ignore
# what the Wb numbers tell you.
# ─────────────────────────────────────────────────────────────────────────────
FPR_CEILING_PCT = 10.0     # IDS deployment ceiling (literature-standard)
FPR_TIEBREAK_PCT = 0.10    # if F1 within 0.10, break ties by lower mean FPR


def select_winner(groups):
	"""Rule: per-seed best-F1 within the deployable region (FPR < FPR_CEILING_PCT),
	then mean over seeds. Require all 3 seeds reach feasibility.

	29/05/2026: now ranks ALL three weight sets (a, b, c). The earlier Wa-lock
	is dropped because (i) the empirical-brittleness fix changed the score
	distributions enough that prior probes aren't directly comparable, and
	(ii) the new Wc weight set explicitly tests whether CE-pressure selection
	hits paper-era operating points under post-fix code.

	Ties (F1 within FPR_TIEBREAK_PCT) → lower mean FPR wins.
	"""
	rankings = []
	for key, g in groups.items():
		width, weight = key
		feas = [r for r in g["rows"] if r["fpr"] < FPR_CEILING_PCT]
		by_seed = {}
		for r in feas:
			cur = by_seed.get(r["seed"])
			if cur is None or r["f1"] > cur["f1"]:
				by_seed[r["seed"]] = r
		if len(by_seed) < 3:                     # all 3 seeds must reach feasibility
			continue
		mean_f1  = statistics.fmean(v["f1"]  for v in by_seed.values())
		mean_fpr = statistics.fmean(v["fpr"] for v in by_seed.values())
		# Score = F1 primary, −0.01·FPR as a tiebreaker hint (keeps ordering stable).
		score = mean_f1 - 0.01 * mean_fpr
		w_label = WEIGHT_LABELS.get(weight, weight)
		reason = (f"mean best-F1@FPR<{FPR_CEILING_PCT:g}% = {mean_f1:.2f}, "
		          f"mean FPR = {mean_fpr:.2f}  [W{weight}={w_label}]")
		rankings.append((key, score, reason))
	rankings.sort(key=lambda x: -x[1])
	return rankings


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--strict", action="store_true",
	                help="Error if any (width, weight) cell has fewer than 3 seeds.")
	ap.add_argument("--show-mode", default="train_cal",
	                help="Threshold mode for the diagnostic table (default: train_cal).")
	ap.add_argument("--show-genome", default="best_fitness",
	                help="Genome type for the diagnostic table (default: best_fitness).")
	args = ap.parse_args()

	rows = load_probe_rows()
	if not rows:
		print(f"No completed XDS-unsw-temporal probe rows found in {DB}.", file=sys.stderr)
		sys.exit(2)
	groups = group_by_cell(rows)

	print_coverage(groups)
	if args.strict:
		missing = [k for k, g in groups.items() if len(g["seeds"]) < 3]
		if missing:
			print(f"\nERROR (--strict): {len(missing)} cells with <3 seeds: {missing}", file=sys.stderr)
			sys.exit(3)

	print_metric_table(groups, mode=args.show_mode, genome_type=args.show_genome)

	print(f"\nRanking (via select_winner):")
	ranking = select_winner(groups)
	if not ranking:
		print("  (empty — select_winner returned no candidates)")
		return
	for rank, (key, score, reason) in enumerate(ranking, 1):
		marker = "  ★" if rank == 1 else "   "
		print(f"{marker} {rank:>2}. {key[0]:>3}b W{key[1]}  score={score:7.3f}   {reason}")
	w, wt = ranking[0][0]
	print(f"\nWINNER: width={w}b, weight=W{wt}")
	print(f"  → fire Cohort 2 with:")
	print(f"    python scripts/queue_cross_dataset.py --dataset unsw-temporal "
	      f"--phase cohort --width {w} --weight {wt} --execute")


if __name__ == "__main__":
	main()
