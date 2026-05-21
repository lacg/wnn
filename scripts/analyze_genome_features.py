#!/usr/bin/env python3
"""Per-feature breakdown of a GA-phase genome's connectivity.

For each neuron-bit-width within a genome, computes how the neuron
connections distribute across the 20 thermometer-encoded features.
Useful for inspecting whether high/low-bit neurons concentrate on
particular features (e.g., the heterogeneous-bits investigation that
checks whether 80-bit neurons specialize on high-RF-importance features).

The thermometer layout is assumed to be `BITS_PER_FEATURE` consecutive
bits per feature, so `feature_id = bit_idx // BITS_PER_FEATURE`.
For the 96b TOP20 CIC-IoT-2023 cohort: 20 features x 96 bits = 1920
input bits, feature_id in [0, 20).

Usage:
  python3 analyze_genome_features.py --genome-id 770352 --label "r58879"
  python3 analyze_genome_features.py --genome-id 770352 --genome-id 772296
  python3 analyze_genome_features.py --flow-seed 58879 \\
      --metric ce --threshold empirical_cumulative
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
	assert len(features) == NUM_FEATURES, f"expected 20 features, got {len(features)}"
	return features, importances


def resolve_flow_seed(con, seed, metric, threshold):
	"""Find the GA-phase best-genome ID matching a flow seed + metric + threshold."""
	row = con.execute(
		"""
		SELECT bg.genome_id, bg.f1_macro, bg.fpr, bg.accuracy, f.id, f.name
		FROM best_genomes bg
		JOIN flows f ON f.id = bg.flow_id
		JOIN genomes g ON g.id = bg.genome_id
		JOIN experiments e ON e.id = g.experiment_id
		WHERE f.name LIKE ?
		  AND bg.metric = ?
		  AND bg.threshold_mode = ?
		  AND e.phase_type = 'ga_neurons'
		ORDER BY bg.f1_macro DESC
		LIMIT 1
		""",
		(f"%r{seed}", metric, threshold),
	).fetchone()
	if not row:
		sys.exit(f"No GA-phase genome found for seed r{seed} / metric={metric} / threshold={threshold}")
	return row["genome_id"], f"r{seed} ({metric}/{threshold}, F1={row['f1_macro']:.4f})"


def fetch_genome(con, gid):
	row = con.execute(
		"SELECT tiers_json, connections_json FROM genomes WHERE id=?",
		(gid,),
	).fetchone()
	if not row or not row["tiers_json"] or not row["connections_json"]:
		sys.exit(f"Genome {gid} missing tiers_json or connections_json")
	tiers = json.loads(row["tiers_json"])
	if not isinstance(tiers, dict) or "bits_per_neuron" not in tiers:
		sys.exit(f"Genome {gid} has unexpected tiers_json shape: {type(tiers).__name__}")
	conns = [int(x) for x in row["connections_json"].split(",")]
	return tiers, conns


def analyze(gid, label, features, importances, tiers, flat_conns):
	bpn = tiers["bits_per_neuron"]
	expected = sum(bpn)
	print(f"\n{'=' * 78}\n{label} (genome id {gid})")
	print(f"  neurons_per_cluster: {tiers['neurons_per_cluster'][0]}")
	print(f"  bits_per_neuron len: {len(bpn)}  (sum={expected}, conn_list_len={len(flat_conns)})")
	if expected != len(flat_conns):
		print(f"  WARNING: sum(bits) != len(connections) — analysis may be corrupted")
		return
	print(f"  bit-width composition: {dict(Counter(bpn))}")

	per_neuron_conns = []
	idx = 0
	for nb in bpn:
		per_neuron_conns.append(flat_conns[idx:idx + nb])
		idx += nb

	width_groups = defaultdict(list)
	for n_idx, (nb, cs) in enumerate(zip(bpn, per_neuron_conns)):
		width_groups[nb].append((n_idx, cs))

	# Per-width feature-usage histogram
	for width in sorted(width_groups.keys()):
		neurons = width_groups[width]
		feat_count = Counter()
		for _, cs in neurons:
			for bit in cs:
				feat_count[bit // BITS_PER_FEATURE] += 1
		total_bits = sum(feat_count.values())
		print(f"\n  -- neurons with bits={width} (count={len(neurons)}, total slots={total_bits})")
		print(f"      Rank Feature              RF-imp     Slots    Slot%   Per-neuron")
		for rank, feat in enumerate(features):
			cnt = feat_count.get(rank, 0)
			pct = 100 * cnt / total_bits if total_bits else 0
			per_neuron = cnt / len(neurons) if neurons else 0
			print(f"      {rank + 1:>3}  {feat:<20} {importances[feat]:.4f}    {cnt:>5}    {pct:>5.1f}%   {per_neuron:>5.2f}")

	# Cross-width comparison if both 64-bit and another width exist
	if 64 in width_groups and len(width_groups) > 1:
		for other in sorted(w for w in width_groups if w != 64):
			feat_hi = Counter()
			feat_lo = Counter()
			for _, cs in width_groups[other]:
				for bit in cs:
					feat_hi[bit // BITS_PER_FEATURE] += 1
			for _, cs in width_groups[64]:
				for bit in cs:
					feat_lo[bit // BITS_PER_FEATURE] += 1
			hi_total = sum(feat_hi.values())
			lo_total = sum(feat_lo.values())
			print(f"\n  -- {other}-bit vs 64-bit feature concentration (per-slot %)")
			print(f"      Rank Feature              RF-imp    {other}b%   64b%   Dpp    Drel")
			for rank, feat in enumerate(features):
				hi_pct = 100 * feat_hi.get(rank, 0) / hi_total if hi_total else 0
				lo_pct = 100 * feat_lo.get(rank, 0) / lo_total if lo_total else 0
				delta_pp = hi_pct - lo_pct
				delta_rel = (delta_pp / lo_pct * 100) if lo_pct > 0 else float("nan")
				marker = " *" if abs(delta_pp) > 0.5 else "  "
				print(f"      {rank + 1:>3}  {feat:<20} {importances[feat]:.4f}  {hi_pct:>5.1f} {lo_pct:>5.1f}  {delta_pp:+5.2f}  {delta_rel:+6.1f}%{marker}")

		# Neuron positions for non-64 widths
		for other in sorted(w for w in width_groups if w != 64):
			idxs = [i for i, nb in enumerate(bpn) if nb == other]
			print(f"\n  -- {other}-bit neuron positions ({len(idxs)} neurons): {idxs}")


def main():
	p = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	p.add_argument("--genome-id", type=int, action="append", default=[],
		help="Genome ID to analyze (repeatable)")
	p.add_argument("--label", action="append", default=[],
		help="Optional label per --genome-id (zip-paired with --genome-id)")
	p.add_argument("--flow-seed", type=int, default=None,
		help="Alternative: resolve a GA-phase genome by flow seed (e.g. 58879)")
	p.add_argument("--metric", default="ce",
		help="With --flow-seed: which metric (default: ce)")
	p.add_argument("--threshold", default="empirical_cumulative",
		help="With --flow-seed: which threshold mode (default: empirical_cumulative)")
	args = p.parse_args()

	if not args.genome_id and args.flow_seed is None:
		p.error("Pass --genome-id (repeatable) or --flow-seed")

	features, importances = load_features()
	con = sqlite3.connect(DB)
	con.row_factory = sqlite3.Row

	targets = []
	for i, gid in enumerate(args.genome_id):
		label = args.label[i] if i < len(args.label) else f"genome {gid}"
		targets.append((gid, label))
	if args.flow_seed is not None:
		gid, label = resolve_flow_seed(con, args.flow_seed, args.metric, args.threshold)
		targets.append((gid, label))

	for gid, label in targets:
		tiers, conns = fetch_genome(con, gid)
		analyze(gid, label, features, importances, tiers, conns)


if __name__ == "__main__":
	main()
