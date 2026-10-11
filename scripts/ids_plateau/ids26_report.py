"""IDS-26: render the per-dataset diversity tables from ids26_population_diversity.py output.

Usage: python scripts/ids_plateau/ids26_report.py results.json
"""
import sys, json, collections
import numpy as np

ORDER = [("SP100", "unswr", "quad"), ("SWEEP", "unswr", "quad"), ("SWEEP-QSR", "unswr", "qsr"),
	("SP100", "unswt", "quad"), ("SWEEP", "unswt", "quad"),
	("SP100", "cicids", "quad"), ("SWEEP", "cicids", "quad"),
	("SP100", "ciciot", "quad"), ("SWEEP", "ciciot", "quad")]
ROWS = [
	("pop size", "size", 0), ("distinct genomes", "distinct", 1),
	("neurons mean", "n_mean", 0), ("neurons SD (within pop)", "n_sd", 1), ("neurons range (max-min)", "n_rng", 0),
	("bits mean", "b_mean", 2), ("bits SD (within pop)", "b_sd", 2), ("bits range (max-min)", "b_rng", 2),
	("J_neu within-pop", "j_neu", 3), ("J_in within-pop", "j_in", 3), ("J_use within-pop", "j_use", 3),
	("gene pool (genome-equiv.)", "pool_ratio", 1), ("basins (J_neu>=0.5)", "basins", 1), ("basins (J_neu>=0.2)", "basins_loose", 1), ("arch cells", "arch_cells", 1),
	("k-fold F1 range in pop (pp)", "f1_kfold_range", 3), ("best genome neurons", "best_n", 0), ("best genome bits", "best_b", 1),
]


def ms(xs, dp):
	x = np.asarray(xs, float)
	return f"{x.mean():.{dp}f}±{x.std(ddof=1):.{dp}f}" if len(x) > 1 else f"{x.mean():.{dp}f}"


def augment(s):
	s["n_rng"] = s["n_max"] - s["n_min"]
	s["b_rng"] = s["b_max"] - s["b_min"]
	return s


def per_dataset(runs):
	g = collections.defaultdict(list)
	for r in runs:
		augment(r["GS"])
		augment(r["GA"])
		g[(r["fam"], r["ds"], r["mem"])].append(r)
	for key in ORDER:
		rs = g.get(key)
		if not rs:
			continue
		print(f"\n{key[1]} / {key[0]} ({key[2]})  runs={len(rs)}  arms={len({r['arm'] for r in rs})}")
		print(f"  {'metric':30s} | {'Grid (seed pop)':>16s} | {'GA final pop':>16s}")
		print("  " + "-" * 68)
		for label, k, dp in ROWS:
			print(f"  {label:30s} | {ms([r['GS'][k] for r in rs], dp):>16s} | {ms([r['GA'][k] for r in rs], dp):>16s}")
		anc = [r["anc"] for r in rs]
		print(f"  GA best: neurons verbatim from grid {ms([a['inherited'] for a in anc], 3)}, "
			f"grid ancestors {ms([a['n_ancestors'] for a in anc], 1)}, top-ancestor share {ms([a['top_ancestor'] for a in anc], 3)}, "
			f"grid genomes still alive in GA pop {ms([a['pop_alive_ancestors'] for a in anc], 1)}/50, "
			f"top ancestor = grid pop[0..4] in {sum(0 <= a['top_ancestor_rank'] < 5 for a in anc)}/{len(anc)}")
		print(f"  GA gens run {ms([r['GA']['iterations'] for r in rs], 0)}; "
			f"runs with GA basins==1: {sum(r['GA']['basins'] == 1 for r in rs)}/{len(rs)}")


def cross(xr):
	print("\nCross-run best-genome similarity (mean pairwise; n pairs)")
	for key in ORDER:
		k = "|".join(key)
		if k not in xr:
			continue
		print(f"\n{key[1]} / {key[0]} ({key[2]})")
		print(f"  {'pair relation':22s} | {'phase':5s} | {'J_neu':>7s} | {'J_in':>7s} | {'J_use':>7s} | n")
		for rel in ("same-arm/diff-seed", "same-seed/diff-arm", "diff-both"):
			for ph in ("GS", "GA"):
				v = xr[k].get(f"{rel}|{ph}")
				if v:
					print(f"  {rel:22s} | {ph:5s} | {v['j_neu'][0]:7.3f} | {v['j_in'][0]:7.3f} | {v['j_use'][0]:7.3f} | {v['j_neu'][1]}")


def main():
	d = json.load(open(sys.argv[1]))
	per_dataset(d["runs"])
	cross(d["cross"])


if __name__ == "__main__":
	main()
