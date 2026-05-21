#!/usr/bin/env python3
"""Cross-seed heterogeneous-bits feature-usage analysis.

Finds all GA-phase best-genome picks in a cohort, separates them into
homogeneous (every neuron at the same bit-width) and heterogeneous,
then aggregates per-feature connectivity slot counts across all
heterogeneous genomes, broken down by bit-width. Compares each
bit-width's feature distribution to a homo-64 baseline (sampled
control).

Used to test cross-seed reproducibility of the "Std-overrepresentation
in non-64-bit neurons" pattern initially observed on r58879. See
memory `project_cascade_pipeline.md` (follow-up paper plan) for
context.

Usage:
  python3 cohort_hetero_bits.py
  python3 cohort_hetero_bits.py --cohort-prefix "WSWEEP-T20-96b-C35-250n100b-OI"
  python3 cohort_hetero_bits.py --homo-baseline-n 20
"""
import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
FEATURES_FILE = Path("/Users/lacg/wnn/data/top20_canonical_neto_full.json")
BITS_PER_FEATURE = 96
NUM_FEATURES = 20


def load_features():
	with FEATURES_FILE.open() as f:
		top20 = json.load(f)
	features = top20["top20"]
	importances = dict(top20["all_ranked"])
	return features, importances


def collect_genomes(con, cohort_prefix, phase):
	"""Yield (genome_id, flow_id, flow_name, tiers_dict) for each GA-phase genome.

	Skips genomes whose tiers_json is the legacy list shape (older runs)
	rather than the per-neuron dict shape used by this cohort.
	"""
	rows = con.execute(
		"""
		SELECT g.id, e.flow_id, f.name, g.tiers_json
		FROM genomes g
		JOIN experiments e ON e.id = g.experiment_id
		JOIN flows f ON f.id = e.flow_id
		WHERE e.phase_type = ?
		  AND f.name LIKE ?
		""",
		(phase, cohort_prefix + "%"),
	).fetchall()
	for row in rows:
		try:
			tiers = json.loads(row["tiers_json"])
		except (TypeError, json.JSONDecodeError):
			continue
		if not isinstance(tiers, dict) or "bits_per_neuron" not in tiers:
			continue
		yield row["id"], row["flow_id"], row["name"], tiers


def main():
	p = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--cohort-prefix", default="WSWEEP-T20-96b-C35-250n100b-OI",
		help="Match flows whose name starts with this prefix")
	p.add_argument("--phase", default="ga_neurons",
		help="Experiment phase to inspect (default: ga_neurons)")
	p.add_argument("--homo-baseline-n", type=int, default=10,
		help="How many homo-64 genomes to sample for the baseline (default: 10)")
	args = p.parse_args()

	features, importances = load_features()
	con = sqlite3.connect(DB)
	con.row_factory = sqlite3.Row

	all_genomes = list(collect_genomes(con, args.cohort_prefix, args.phase))
	if not all_genomes:
		sys.exit(f"No {args.phase} genomes found for cohort {args.cohort_prefix!r}")

	print(f"Cohort prefix : {args.cohort_prefix}")
	print(f"Phase         : {args.phase}")
	print(f"Total genomes : {len(all_genomes)}")
	print(f"Unique flows  : {len({g[1] for g in all_genomes})}")

	hetero = []
	homo_widths = Counter()
	for gid, flow_id, flow_name, tiers in all_genomes:
		uniq = set(tiers["bits_per_neuron"])
		if len(uniq) == 1:
			homo_widths[next(iter(uniq))] += 1
		else:
			hetero.append((gid, flow_id, flow_name, tiers))

	print(f"\nHomogeneous bit-widths:")
	for width, count in sorted(homo_widths.items()):
		print(f"  b={width}: {count}")
	print(f"Heterogeneous bits   : {len(hetero)}")

	if not hetero:
		print("\nNo heterogeneous-bits genomes — pattern doesn't exist in this cohort.")
		return

	# Per-flow breakdown
	hetero_by_flow = defaultdict(list)
	for gid, flow_id, flow_name, tiers in hetero:
		hetero_by_flow[(flow_id, flow_name)].append((gid, tiers))

	print(f"\nFlows with at least one heterogeneous GA genome: {len(hetero_by_flow)}")
	for (flow_id, flow_name), entries in sorted(hetero_by_flow.items()):
		widths = Counter()
		for _, t in entries:
			widths.update(t["bits_per_neuron"])
		width_str = ", ".join(f"b{w}:{c}" for w, c in sorted(widths.items()))
		print(f"  flow {flow_id} ({flow_name}): {len(entries)} hetero genomes, slots {width_str}")

	# Build homo-64 baseline (sample N from the homo-64 pool)
	homo64_genomes = [g for g in all_genomes if set(g[3]["bits_per_neuron"]) == {64}]
	homo_sample = homo64_genomes[:args.homo_baseline_n]
	print(f"\nHomo-64 baseline: {len(homo_sample)} of {len(homo64_genomes)} homo-64 genomes sampled")

	# Aggregate slot counts per bit-width across all hetero genomes
	agg = defaultdict(Counter)        # bits -> Counter(feat_rank -> slots)
	agg_neurons = Counter()           # bits -> neurons-aggregated
	per_genome_widths = Counter()     # bits -> # genomes containing that width

	for gid, flow_id, flow_name, tiers in hetero:
		row = con.execute("SELECT connections_json FROM genomes WHERE id=?", (gid,)).fetchone()
		if not row or not row["connections_json"]:
			continue
		flat = [int(x) for x in row["connections_json"].split(",")]
		bpn = tiers["bits_per_neuron"]
		if sum(bpn) != len(flat):
			print(f"  SKIP genome {gid}: bpn sum={sum(bpn)} != conn len={len(flat)}")
			continue
		per_genome_widths.update(set(bpn))
		idx = 0
		for nb in bpn:
			for bit in flat[idx:idx + nb]:
				agg[nb][bit // BITS_PER_FEATURE] += 1
			agg_neurons[nb] += 1
			idx += nb

	homo_agg = Counter()
	homo_neurons = 0
	for gid, _, _, tiers in homo_sample:
		row = con.execute("SELECT connections_json FROM genomes WHERE id=?", (gid,)).fetchone()
		if not row or not row["connections_json"]:
			continue
		flat = [int(x) for x in row["connections_json"].split(",")]
		bpn = tiers["bits_per_neuron"]
		if sum(bpn) != len(flat):
			continue
		for bit in flat:
			homo_agg[bit // BITS_PER_FEATURE] += 1
		homo_neurons += len(bpn)

	homo_total = sum(homo_agg.values())
	print(f"\nWidth coverage across hetero genomes: {dict(agg_neurons)}")
	print(f"Genomes containing each width       : {dict(per_genome_widths)}")
	print(f"Homo-64 control                     : {homo_neurons} neurons, {homo_total} slots")

	for width in sorted(agg.keys()):
		total = sum(agg[width].values())
		if total == 0:
			continue
		print(f"\n-- Aggregate b={width}b neurons (n_neurons={agg_neurons[width]}, slots={total})")
		print(f"   Rank Feature              RF-imp    b{width}%   homo64%   Dpp     Drel")
		for rank, feat in enumerate(features):
			w_pct = 100 * agg[width].get(rank, 0) / total
			h_pct = 100 * homo_agg.get(rank, 0) / homo_total if homo_total else 0
			delta_pp = w_pct - h_pct
			delta_rel = (delta_pp / h_pct * 100) if h_pct > 0 else float("nan")
			marker = " *" if abs(delta_pp) > 1.0 else "  "
			print(f"   {rank + 1:>3}  {feat:<20} {importances[feat]:.4f}  {w_pct:>5.1f}    {h_pct:>5.1f}  {delta_pp:+6.2f}  {delta_rel:+6.1f}%{marker}")


if __name__ == "__main__":
	main()
