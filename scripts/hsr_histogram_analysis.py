#!/usr/bin/env python3
"""Per-genome-shape side-by-side HSR comparison + winner analysis.

Reads genome_evaluations directly from the dashboard SQLite DB.
For each (neurons, bits) genome configuration observed across multiple HSR
values, computes timing statistics (mean, std, median, min/max) per HSR and
identifies the fastest ratio.

Usage:
  python3 scripts/hsr_histogram_analysis.py
  python3 scripts/hsr_histogram_analysis.py --min-samples 5 --bin-mode exact
  python3 scripts/hsr_histogram_analysis.py --csv hsr_results.csv
"""
import argparse
import csv
import json
import sqlite3
import statistics
import sys
from collections import defaultdict

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
HSR_VALUES = [1, 2, 3, 5, 7, 8, 10]


def fetch_rows(db_path: str) -> list[dict]:
	"""Pull every genome_evaluation with eval_time_ms set across all HSR cohort flows."""
	conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
	conn.row_factory = sqlite3.Row
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT
		    f.id                                      AS flow_id,
		    f.name                                    AS flow_name,
		    json_extract(f.config_json, '$.params.wnn_hybrid_speed_ratio') AS hsr,
		    json_extract(f.config_json, '$.params.seed')                   AS seed,
		    e.id                                      AS experiment_id,
		    e.phase_type                              AS phase,
		    ge.eval_time_ms                           AS time_ms,
		    g.total_neurons                           AS neurons,
		    g.tiers_json                              AS tiers_json,
		    ge.role                                   AS role,
		    ge.f1_macro                               AS f1_macro,
		    ge.fpr                                    AS fpr,
		    ge.ce                                     AS ce,
		    ge.accuracy                               AS accuracy
		FROM flows f
		JOIN experiments e            ON e.flow_id      = f.id
		JOIN iterations i             ON i.experiment_id = e.id
		JOIN genome_evaluations ge    ON ge.iteration_id = i.id
		JOIN genomes g                ON g.id           = ge.genome_id
		WHERE f.name LIKE 'WSWEEP-T20-96b-C35-250n100b-OI-HSR%'
		  AND ge.eval_time_ms IS NOT NULL
		"""
	)
	rows = []
	for r in cur.fetchall():
		try:
			tiers = json.loads(r["tiers_json"])
			bits = tiers[0]["bits"] if tiers else None
		except Exception:
			bits = None
		rows.append({
			"flow_id": r["flow_id"],
			"hsr": int(r["hsr"]) if r["hsr"] is not None else None,
			"seed": r["seed"],
			"phase": r["phase"],
			"time_ms": r["time_ms"],
			"neurons": r["neurons"],
			"bits": bits,
			"role": r["role"],
			"f1_macro": r["f1_macro"],
			"fpr": r["fpr"],
			"ce": r["ce"],
			"accuracy": r["accuracy"],
		})
	conn.close()
	return rows


def bin_shape(neurons: int, bits: int, mode: str) -> tuple[int, int]:
	"""Quantize the (neurons, bits) shape so similar genomes group together.

	- 'exact': return (neurons, bits) unchanged.
	- 'coarse': round neurons to nearest 25, bits to nearest 4. Groups small-variance shapes.
	"""
	if mode == "exact":
		return (neurons, bits)
	# coarse: 25-neuron buckets, 4-bit buckets
	n_bucket = int(round(neurons / 25.0) * 25) or 5
	b_bucket = int(round(bits / 4.0) * 4) or 4
	return (n_bucket, b_bucket)


def text_histogram(values: list[int], width: int = 30) -> str:
	"""Single-line ASCII histogram of percentile positions."""
	if not values:
		return ""
	mn, mx = min(values), max(values)
	if mx == mn:
		return f"|{'*':>{width}}|"
	bins = [0] * width
	for v in values:
		idx = min(width - 1, int((v - mn) / (mx - mn) * width))
		bins[idx] += 1
	peak = max(bins)
	if peak == 0:
		return "|" + " " * width + "|"
	out = []
	for c in bins:
		if c == 0:
			out.append(" ")
		else:
			frac = c / peak
			out.append("▁▂▃▄▅▆▇█"[min(7, int(frac * 7))])
	return "|" + "".join(out) + "|"


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--min-samples", type=int, default=3,
	                help="Minimum samples per (shape, HSR) cell to include in the table.")
	ap.add_argument("--bin-mode", choices=["exact", "coarse"], default="coarse",
	                help="'exact' = (neurons, bits) unchanged; 'coarse' = bucket by 25 neurons / 4 bits.")
	ap.add_argument("--csv", type=str, default=None,
	                help="If set, also write per-(shape, HSR) summary rows here.")
	ap.add_argument("--db", type=str, default=DB_PATH)
	args = ap.parse_args()

	rows = fetch_rows(args.db)
	if not rows:
		print("No HSR cohort data with eval_time_ms yet.")
		print("Once the first HSR flow has some evaluated genomes, this will populate.")
		sys.exit(0)

	# Cohort summary
	total_rows = len(rows)
	per_hsr = defaultdict(int)
	per_hsr_flows = defaultdict(set)
	for r in rows:
		per_hsr[r["hsr"]] += 1
		per_hsr_flows[r["hsr"]].add(r["flow_id"])

	print(f"=== HSR cohort timing data ===")
	print(f"Total rows with eval_time_ms: {total_rows}")
	print()
	print(f"  {'HSR':>4} | {'rows':>6} | flows seen / 8 expected")
	print(f"  {'-'*4}-+-{'-'*6}-+-------------------------")
	for h in HSR_VALUES:
		print(f"  {h:>4} | {per_hsr[h]:>6} | {len(per_hsr_flows[h])}/8")
	print()

	# Bucket by genome shape
	buckets: dict[tuple[int, int], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
	for r in rows:
		if r["neurons"] is None or r["bits"] is None or r["hsr"] is None:
			continue
		shape = bin_shape(r["neurons"], r["bits"], args.bin_mode)
		buckets[shape][r["hsr"]].append(r["time_ms"])

	# Filter to shapes that appear across ≥2 HSR values with ≥min_samples each
	eligible_shapes = []
	for shape, hsr_map in buckets.items():
		qualifying_hsr = [h for h, vs in hsr_map.items() if len(vs) >= args.min_samples]
		if len(qualifying_hsr) >= 2:
			eligible_shapes.append((shape, qualifying_hsr))

	if not eligible_shapes:
		print(f"No genome shape yet has ≥{args.min_samples} samples across ≥2 HSR values.")
		print("Data is still sparse — re-run later.")
		sys.exit(0)

	# Side-by-side per shape
	print(f"=== Per-shape side-by-side ({args.bin_mode} binning, min {args.min_samples} samples) ===")
	print()
	winners = []
	csv_rows = []
	for shape, qualifying in sorted(eligible_shapes, key=lambda x: (x[0][0], x[0][1])):
		n, b = shape
		hsr_map = buckets[shape]
		print(f"[shape ≈ {n} neurons × {b} bits]")
		print(f"  {'HSR':>4} | {'n':>4} | {'mean ms':>9} | {'median':>7} | {'std':>7} | {'min':>6} | {'max':>6} | Δ vs winner | distribution (min={'min':>4}…max={'max':>4})")
		print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+--{'-'*10}-+-------------------------")

		shape_stats = {}
		for h in HSR_VALUES:
			vs = hsr_map.get(h, [])
			if len(vs) < args.min_samples:
				continue
			shape_stats[h] = {
				"n": len(vs),
				"mean": statistics.mean(vs),
				"median": statistics.median(vs),
				"std": statistics.stdev(vs) if len(vs) > 1 else 0.0,
				"min": min(vs),
				"max": max(vs),
				"values": vs,
			}

		if not shape_stats:
			print("  (no qualifying HSR groups)\n")
			continue

		winner_hsr = min(shape_stats.keys(), key=lambda h: shape_stats[h]["mean"])
		winner_mean = shape_stats[winner_hsr]["mean"]
		all_min = min(s["min"] for s in shape_stats.values())
		all_max = max(s["max"] for s in shape_stats.values())

		for h in HSR_VALUES:
			s = shape_stats.get(h)
			if s is None:
				continue
			delta = s["mean"] - winner_mean
			delta_pct = (delta / winner_mean * 100) if winner_mean > 0 else 0
			win_flag = " ★" if h == winner_hsr else "  "
			delta_str = f"{delta:+7.1f} ({delta_pct:+5.1f}%)" if h != winner_hsr else "       —     "
			# Stretched histogram across the full shape's range so bars align across HSR rows
			# Bin values into a 25-cell range from all_min..all_max
			width = 25
			if all_max > all_min:
				bins = [0] * width
				for v in s["values"]:
					idx = min(width - 1, int((v - all_min) / (all_max - all_min) * width))
					bins[idx] += 1
				peak = max(bins) or 1
				hist = "".join(("▁▂▃▄▅▆▇█"[min(7, int(c / peak * 7))] if c else " ") for c in bins)
			else:
				hist = " " * width
			print(f"  {h:>4} | {s['n']:>4} | {s['mean']:>9.1f} | {s['median']:>7.1f} | {s['std']:>7.1f} | {s['min']:>6.0f} | {s['max']:>6.0f} | {delta_str}{win_flag}|{hist}|")

			csv_rows.append({
				"shape_neurons": n, "shape_bits": b, "hsr": h,
				"n_samples": s["n"], "mean_ms": s["mean"], "median_ms": s["median"],
				"std_ms": s["std"], "min_ms": s["min"], "max_ms": s["max"],
				"is_winner": h == winner_hsr,
				"delta_vs_winner_ms": delta,
				"delta_vs_winner_pct": delta_pct,
			})
		print(f"  Range: {all_min:.0f}–{all_max:.0f} ms.  Winner: HSR={winner_hsr} ({winner_mean:.1f} ms mean).")
		print()
		winners.append((shape, winner_hsr, winner_mean, shape_stats))

	# Summary winners table
	print("=== Winner per genome shape ===")
	print()
	print(f"  {'shape':>14} | winner | {'mean ms':>9} | {'2nd':>8} | {'gap':>6} | regime hint")
	print(f"  {'-'*14}-+--------+-{'-'*9}-+-{'-'*8}-+-{'-'*6}-+----------------")
	for shape, winner_hsr, winner_mean, stats_map in winners:
		n, b = shape
		sorted_hsrs = sorted(stats_map.items(), key=lambda kv: kv[1]["mean"])
		runner_up_h, runner_up = sorted_hsrs[1] if len(sorted_hsrs) > 1 else (None, None)
		runner_up_mean = runner_up["mean"] if runner_up else None
		gap = (runner_up_mean - winner_mean) if runner_up_mean else None
		# Regime hint: lower HSR favored = GPU-friendly; higher HSR favored = CPU-friendly
		if winner_hsr <= 2:
			hint = "GPU-leaning (small HSR = aggressive GPU dispatch)"
		elif winner_hsr >= 8:
			hint = "CPU-leaning (large HSR = stay on CPU)"
		else:
			hint = "mid-ratio"
		shape_str = f"{n}n × {b}b"
		gap_str = f"{gap:>+6.1f}" if gap is not None else "    --"
		runner_str = f"HSR={runner_up_h}" if runner_up_h is not None else "  --"
		print(f"  {shape_str:>14} | HSR={winner_hsr:<2}  | {winner_mean:>9.1f} | {runner_str:>8} | {gap_str} | {hint}")

	print()

	# Aggregate verdict
	winner_counts = defaultdict(int)
	for _, w, _, _ in winners:
		winner_counts[w] += 1
	print("=== Aggregate verdict ===")
	print(f"  Shapes analyzed: {len(winners)}")
	print(f"  Winner-count distribution:")
	for h in HSR_VALUES:
		c = winner_counts.get(h, 0)
		bar = "█" * c
		print(f"    HSR={h:>2}: {c} shape{'s' if c != 1 else ''} {bar}")

	# Optional CSV
	if args.csv and csv_rows:
		with open(args.csv, "w", newline="") as fh:
			w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
			w.writeheader()
			w.writerows(csv_rows)
		print(f"\nCSV written: {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
	main()
