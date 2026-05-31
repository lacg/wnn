"""Final XDS cohort analysis — ranks (width × weight) configs across 4 user-defined criteria.

Use after the XDS-temporal queue drains. Produces a single-screen ranked summary +
the per-config breakdown the 5-table report doesn't surface (deployability,
genome-size penalty, duration tie-break).

Criteria (user-specified 30/05/2026):
  1. Best F1:       highest mean F1 (val_cal threshold, GA Neurons phase, best-CE genome)
  2. Best deployability: F1 - λ·FPR - μ·log(neurons) — favors low-FPR + small-genome
  3. Mean ± std:    F1/FPR/Acc reproducibility across seeds (tightest std = winner)
  4. Duration:      tiebreaker — fastest wall-time wins on ties

Output:
  - Per-config (width × weight) table with all 4 dimensions
  - Top-3 ranking by each criterion
  - Overall "deployable champion" recommendation

Usage:
  python3 scripts/analyze_xds_final.py                    # XDS-unsw-temporal (default)
  python3 scripts/analyze_xds_final.py --cohort unsw-random
  python3 scripts/analyze_xds_final.py --lambda 2.0       # FPR penalty
  python3 scripts/analyze_xds_final.py --mu 0.5           # genome-size penalty
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")

WEIGHT_LABELS = {
	"a": "Wa (CIC-IoT, ce=0.35)",
	"b": "Wb (paper, ce=0.10)",
	"c": "Wc (CE-heavy, ce=0.70)",
}


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--cohort", default="unsw-temporal",
	                choices=["unsw-temporal", "unsw-random", "cicids"])
	ap.add_argument("--lambda", dest="lam", type=float, default=2.0,
	                help="FPR penalty in deployability score (default 2.0 = 1pp FPR = 2pp F1)")
	ap.add_argument("--mu", type=float, default=0.5,
	                help="Genome-size (log neurons) penalty in deployability score (default 0.5)")
	ap.add_argument("--threshold", default="val_cal",
	                help="Threshold mode to extract metrics at (default val_cal)")
	ap.add_argument("--phase", default="ga_neurons",
	                choices=["ga_neurons", "grid_search"],
	                help="Which phase's best-genome to report (default ga_neurons)")
	ap.add_argument("--metric-selector", default="ce",
	                choices=["f1_macro", "fpr", "accuracy", "ce", "fitness"],
	                help="Which best-X genome to read from each flow (default ce)")
	ap.add_argument("--db", default=str(DB))
	args = ap.parse_args()

	prefix = f"XDS-{args.cohort}-"
	# NOTE 31/05/2026: switched literal `500n34b` to `\d+n\d+b` so cohorts can be
	# resized without breaking this matcher (UNSW-random was 50n34b on 31/05).
	name_re = re.compile(rf"^{re.escape(prefix)}(\d+)b-W([abc])-C35-\d+n\d+b-OI-r(\d+)$")

	con = sqlite3.connect(args.db)
	con.row_factory = sqlite3.Row
	cur = con.cursor()

	# Pull validation_summaries (full 7-threshold JSON) joined with flows for
	# wall time. Neuron counts come from a separate best_genomes → genomes join
	# because validation_summaries only has genome_hash, not genome_id.
	cur.execute("""
		SELECT f.name AS fname, f.id AS flow_id,
		       (julianday(f.completed_at) - julianday(f.started_at)) * 1440 AS wall_min,
		       e.phase_type AS phase,
		       vs.genome_type AS gt,
		       vs.threshold_metadata AS tm,
		       vs.genome_hash AS gh
		FROM validation_summaries vs
		JOIN experiments e ON e.id = vs.experiment_id
		JOIN flows f ON f.id = vs.flow_id
		WHERE f.name LIKE ? AND f.name NOT LIKE '%PREEMP-OLD%' AND f.status = 'completed'
		  AND vs.validation_point = 'final'
		""", (f"{prefix}%",))
	rows = cur.fetchall()

	# Build genome_hash → total_neurons lookup via best_genomes for this prefix.
	cur.execute("""
		SELECT bg.genome_hash, g.total_neurons
		FROM best_genomes bg
		JOIN genomes g ON g.id = bg.genome_id
		JOIN flows f ON f.id = bg.flow_id
		WHERE f.name LIKE ? AND f.name NOT LIKE '%PREEMP-OLD%' AND f.status = 'completed'
		""", (f"{prefix}%",))
	hash_to_neurons = {r["genome_hash"]: r["total_neurons"] for r in cur.fetchall() if r["genome_hash"]}

	# Nest: data[(w, wt)] = list of per-seed dicts {f1, fpr, acc, neurons, wall_min}
	# We pick the genome_type == f'best_{args.metric_selector}' for the chosen phase.
	target_gt = f"best_{args.metric_selector}"
	per_config = defaultdict(list)
	for r in rows:
		m = name_re.match(r["fname"])
		if not m:
			continue
		w, wt, seed = int(m.group(1)), m.group(2), int(m.group(3))
		if r["phase"] != args.phase or r["gt"] != target_gt:
			continue
		tm = json.loads(r["tm"])
		md = tm.get(args.threshold)
		if not isinstance(md, dict) or md.get("f1") is None:
			continue
		per_config[(w, wt)].append({
			"seed": seed,
			"f1": md["f1"] * 100,
			"fpr": md["fpr"] * 100,
			"acc": md["acc"] * 100,
			"neurons": hash_to_neurons.get(r["gh"], 0),
			"wall_min": r["wall_min"] or 0,
		})

	if not per_config:
		print(f"No completed flows matching {prefix}*  (phase={args.phase}, genome={target_gt})")
		return

	# Aggregate per config
	agg = []
	for (w, wt), seeds in sorted(per_config.items(), key=lambda c: (c[0][0], c[0][1])):
		f1s = [s["f1"] for s in seeds]
		fprs = [s["fpr"] for s in seeds]
		accs = [s["acc"] for s in seeds]
		neurons = [s["neurons"] for s in seeds if s["neurons"]]
		walls = [s["wall_min"] for s in seeds if s["wall_min"] > 0]
		n = len(seeds)
		mean_f1 = statistics.mean(f1s)
		mean_fpr = statistics.mean(fprs)
		mean_acc = statistics.mean(accs)
		std_f1 = statistics.stdev(f1s) if n > 1 else 0.0
		std_fpr = statistics.stdev(fprs) if n > 1 else 0.0
		std_acc = statistics.stdev(accs) if n > 1 else 0.0
		mean_neurons = statistics.mean(neurons) if neurons else 0
		median_wall = statistics.median(walls) if walls else 0
		# Sum-of-stds compactness score (tighter = better)
		repro_score = std_f1 + std_fpr + std_acc
		# Deployability: F1 - λ·FPR - μ·log(neurons+1).
		# μ·log(neurons+1) is in F1-percentage-points. log(100)=4.6, log(500)=6.2 — small but tilts toward smaller arch on ties.
		deploy_score = mean_f1 - args.lam * mean_fpr - args.mu * math.log(max(1, mean_neurons))
		agg.append({
			"w": w, "wt": wt, "n": n,
			"f1": mean_f1, "f1_std": std_f1,
			"fpr": mean_fpr, "fpr_std": std_fpr,
			"acc": mean_acc, "acc_std": std_acc,
			"neurons": mean_neurons,
			"wall": median_wall,
			"repro": repro_score,
			"deploy": deploy_score,
		})

	# Print per-config table
	print(f"\n{'='*100}")
	print(f"  XDS-{args.cohort} — per-(width, weight) summary")
	print(f"  Threshold: {args.threshold}  |  Phase: {args.phase}  |  Genome: {target_gt}")
	print(f"  Deployability: F1 - {args.lam:.1f}·FPR - {args.mu:.2f}·log(neurons)")
	print(f"{'='*100}")
	print(f"  {'config':<10} {'n':<3} {'F1':<14} {'FPR':<14} {'Acc':<14} {'neur':<6} {'wall':<7} {'repro':<7} {'deploy':<7}")
	print("  " + "-" * 98)
	for r in agg:
		f1s = f"{r['f1']:5.2f}±{r['f1_std']:4.2f}"
		fprs = f"{r['fpr']:5.2f}±{r['fpr_std']:4.2f}"
		accs = f"{r['acc']:5.2f}±{r['acc_std']:4.2f}"
		print(f"  {r['w']:>3}b-W{r['wt']}     "
		      f"{r['n']:<3} {f1s:<14} {fprs:<14} {accs:<14} "
		      f"{r['neurons']:>5.0f} {r['wall']:>6.1f}m {r['repro']:>5.2f}  {r['deploy']:>6.2f}")
	print()

	def top3(key, label, reverse=True):
		s = sorted(agg, key=lambda r: r[key], reverse=reverse)
		print(f"  [{label}]")
		for i, r in enumerate(s[:3], 1):
			marker = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
			print(f"    {marker} {r['w']:>3}b-W{r['wt']}  "
			      f"({key}={r[key]:.2f}, F1={r['f1']:.2f}, FPR={r['fpr']:.2f}, neur={r['neurons']:.0f})")

	print(f"{'='*100}\n  RANKINGS\n{'='*100}\n")
	top3("f1", "1. Best F1 (highest mean F1)", reverse=True)
	print()
	top3("deploy", "2. Best deployability (F1 - λ·FPR - μ·log·neurons)", reverse=True)
	print()
	top3("repro", "3. Tightest std (most reproducible across seeds)", reverse=False)
	print()
	top3("wall", "4. Fastest training (lowest median wall time)", reverse=False)

	# Final recommendation: deployability winner with duration as tie-breaker
	# Among the top 3 by deployability, pick the one with lowest wall time
	top_deploy = sorted(agg, key=lambda r: r["deploy"], reverse=True)[:3]
	winner = sorted(top_deploy, key=lambda r: r["wall"])[0]
	print()
	print(f"{'='*100}")
	print(f"  🏆 OVERALL RECOMMENDATION (best deployability with duration tie-break)")
	print(f"{'='*100}")
	print(f"  Config:        {winner['w']}b-W{winner['wt']} ({WEIGHT_LABELS[winner['wt']]})")
	print(f"  F1 mean±std:   {winner['f1']:.2f}±{winner['f1_std']:.2f}%")
	print(f"  FPR mean±std:  {winner['fpr']:.2f}±{winner['fpr_std']:.2f}%")
	print(f"  Acc mean±std:  {winner['acc']:.2f}±{winner['acc_std']:.2f}%")
	print(f"  Avg neurons:   {winner['neurons']:.0f}")
	print(f"  Median wall:   {winner['wall']:.1f} min")
	print(f"  Deployability: {winner['deploy']:.2f}")
	print(f"  Seeds:         {len(per_config[(winner['w'], winner['wt'])])}")


if __name__ == "__main__":
	main()
