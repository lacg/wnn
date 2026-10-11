"""IDS-24: Grid-phase best vs GA-phase best on held-out TEST (Protocol v2 _3way), paired per run.

View A: per run, each phase's best-everywhere cell over 35 cells (5 genome types x 7 modes):
        F1 max, FPR min among cells with F1 >= dataset floor, Acc max -> paired GA-GS deltas.
View B: best-of-best single cell per dataset per phase (n=1, located).
Conventional: val_cal of best_f1 and best_fitness, paired.
Read-only. Usage: python scripts/ids_plateau/ids24_grid_vs_ga.py
"""
import sys, os
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import connect, select_flows, load_cells, genome_size, MODES, GT, DSETS

RNG = np.random.default_rng(20261010)


def ms(x):
	x = np.asarray(x, float)
	if len(x) == 0: return "n/a"
	return f"{x.mean():6.2f}±{x.std(ddof=1):.2f}" if len(x) > 1 else f"{x[0]:6.2f}"


def boot_ci(d, B=10000):
	d = np.asarray(d, float)
	if len(d) < 2: return (np.nan, np.nan)
	m = RNG.choice(d, (B, len(d))).mean(1)
	return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(d, lower_better=False):
	d = np.asarray(d, float)
	better = (d < -1e-9).sum() if lower_better else (d > 1e-9).sum()
	worse = (d > 1e-9).sum() if lower_better else (d < -1e-9).sum()
	ties = len(d) - better - worse
	try:
		p = stats.wilcoxon(d, zero_method="wilcox").pvalue if (np.abs(d) > 1e-9).sum() >= 5 else np.nan
	except ValueError:
		p = np.nan
	lo, hi = boot_ci(d)
	return f"{d.mean():+6.3f}±{d.std(ddof=1):.3f}  CI[{lo:+.3f},{hi:+.3f}]  GA-better {better}/{len(d)} worse {worse} tie {ties}  Wilcoxon p={p:.2g}"


def all_cells(D, fid, ph):
	out = []
	for gt in GT:
		for md, v in D.get((fid, ph, gt), {}).items():
			out.append((v, gt, md))
	return out


def complete(D, fid):
	return all(len(D.get((fid, ph, gt), {})) == 7 for ph in ("GS", "GA") for gt in GT)


def best_view(cells, floor):
	f1 = max(cells, key=lambda c: (c[0][0], -c[0][1]))
	ok = [c for c in cells if c[0][0] >= floor]
	fpr = min(ok, key=lambda c: (c[0][1], -c[0][0])) if ok else None
	acc = max(cells, key=lambda c: (c[0][2], c[0][0]))
	return f1, fpr, acc


def floors(D, runs_by_ds):
	"""(median GA best_f1 val_cal F1, floor = median-0.5, grid-reachable floor = min(floor, median GS best-F1 - 0.5))."""
	fl = {}
	for ds, fids in runs_by_ds.items():
		v = float(np.median([D[(f, "GA", "best_f1")]["val_cal"][0] for f in fids]))
		g = float(np.median([max(c[0][0] for c in all_cells(D, f, "GS")) for f in fids]))
		fl[ds] = (v, round(v - 0.5, 1), min(round(v - 0.5, 1), round(g - 0.5, 1)))
	return fl


def main():
	con = connect()
	flows = select_flows(con)
	D, GH, nin = load_cells(con, flows)
	runs_by_ds = {ds: [] for ds in DSETS}
	dropped = {ds: [] for ds in DSETS}
	for fid, f in sorted(flows.items()):
		(runs_by_ds if complete(D, fid) else dropped)[f["ds"]].append(fid)
	print(f"inherited rows excluded: {nin}")
	for ds in DSETS:
		fam = {}
		for f in runs_by_ds[ds]: fam[flows[f]["fam"]] = fam.get(flows[f]["fam"], 0) + 1
		print(f"{ds}: runs complete (both phases x 5 gt x 7 modes own) = {len(runs_by_ds[ds])} {fam}; dropped (missing own rows) = {len(dropped[ds])}")
	fl = floors(D, runs_by_ds)
	print("\nFPR floors (median GA best_f1 val_cal F1 - 0.5pp; second = grid-reachable floor):")
	for ds in DSETS: print(f"  {ds}: median {fl[ds][0]:.2f} -> floor {fl[ds][1]:.1f} | grid-reachable floor {fl[ds][2]:.1f}")
	for scope in ["ALL", "SP100", "SWEEP"]:
		print(f"\n================ VIEW A — paired per run, scope={scope} ================")
		for ds in DSETS:
			fids = [f for f in runs_by_ds[ds] if scope == "ALL" or flows[f]["fam"] == scope]
			if len(fids) < 2: continue
			floor = fl[ds][1]; floor2 = fl[ds][2]
			rec = {k: [] for k in ["F1gs", "F1ga", "FPgs", "FPga", "ACgs", "ACga", "dF1", "dFP", "dAC", "f1src", "fpsrc", "FP2gs", "FP2ga", "dFP2", "fp2src", "nogs"]}
			for f in fids:
				g = best_view(all_cells(D, f, "GS"), floor); a = best_view(all_cells(D, f, "GA"), floor)
				rec["F1gs"].append(g[0][0][0]); rec["F1ga"].append(a[0][0][0]); rec["dF1"].append(a[0][0][0] - g[0][0][0])
				rec["ACgs"].append(g[2][0][2]); rec["ACga"].append(a[2][0][2]); rec["dAC"].append(a[2][0][2] - g[2][0][2])
				rec["f1src"].append("GS" if g[0][0][0] > a[0][0][0] + 1e-9 else ("GA" if a[0][0][0] > g[0][0][0] + 1e-9 else "tie"))
				g2 = best_view(all_cells(D, f, "GS"), floor2)[1]; a2 = best_view(all_cells(D, f, "GA"), floor2)[1]
				if g2 and a2:
					rec["FP2gs"].append(g2[0][1]); rec["FP2ga"].append(a2[0][1]); rec["dFP2"].append(a2[0][1] - g2[0][1])
					rec["fp2src"].append("GS" if g2[0][1] < a2[0][1] - 1e-9 else ("GA" if a2[0][1] < g2[0][1] - 1e-9 else "tie"))
				if a[1] and not g[1]: rec["nogs"].append(f)
				if g[1] and a[1]:
					rec["FPgs"].append(g[1][0][1]); rec["FPga"].append(a[1][0][1]); rec["dFP"].append(a[1][0][1] - g[1][0][1])
					rec["fpsrc"].append("GS" if g[1][0][1] < a[1][0][1] - 1e-9 else ("GA" if a[1][0][1] < g[1][0][1] - 1e-9 else "tie"))
			print(f"### {ds}  (N={len(fids)} runs; FPR floor F1>={floor}, N paired={len(rec['dFP'])})")
			print(f"    Metric            | Grid best      | GA best        | delta GA-Grid (pp), sign, test")
			print(f"    F1 max            | {ms(rec['F1gs'])}   | {ms(rec['F1ga'])}   | {paired(rec['dF1'])}")
			print(f"    FPR min @F1>={floor:<5} | {ms(rec['FPgs'])}   | {ms(rec['FPga'])}   | {paired(rec['dFP'], True) if len(rec['dFP']) > 1 else 'n/a'}  (N paired={len(rec['dFP'])})")
			print(f"    FPR min @F1>={floor2:<5} | {ms(rec['FP2gs'])}   | {ms(rec['FP2ga'])}   | {paired(rec['dFP2'], True) if len(rec['dFP2']) > 1 else 'n/a'}  (N paired={len(rec['dFP2'])})")
			print(f"    Acc max           | {ms(rec['ACgs'])}   | {ms(rec['ACga'])}   | {paired(rec['dAC'])}")
			cnt = lambda L: {k: L.count(k) for k in ("GS", "GA", "tie")}
			print(f"    holder of run's best F1 cell: {cnt(rec['f1src'])}  | best FPR@floor cell: {cnt(rec['fpsrc'])} | @grid-reachable floor: {cnt(rec['fp2src'])}")
			print(f"    runs where GA reaches F1>={floor} but Grid never does (any of 35 cells): {len(rec['nogs'])}/{len(fids)}")
		print(f"\n---- Conventional cell (val_cal), paired, scope={scope} ----")
		for ds in DSETS:
			fids = [f for f in runs_by_ds[ds] if scope == "ALL" or flows[f]["fam"] == scope]
			if len(fids) < 2: continue
			for gt in ["best_f1", "best_fitness"]:
				g = np.array([D[(f, "GS", gt)]["val_cal"] for f in fids]); a = np.array([D[(f, "GA", gt)]["val_cal"] for f in fids])
				print(f"  {ds:6s} {gt:12s} N={len(fids):3d} | F1 GS {ms(g[:,0])} GA {ms(a[:,0])} d {paired(a[:,0]-g[:,0])}")
				print(f"  {'':6s} {'':12s}        | FPR GS {ms(g[:,1])} GA {ms(a[:,1])} d {paired(a[:,1]-g[:,1], True)}")
				print(f"  {'':6s} {'':12s}        | Acc GS {ms(g[:,2])} GA {ms(a[:,2])} d {paired(a[:,2]-g[:,2])}")
	print("\n================ VIEW B — best-of-best single cell per dataset per phase (n=1, ALL scope) ================")
	for ds in DSETS:
		floor = fl[ds][1]
		print(f"### {ds}  (pool: {len(runs_by_ds[ds])} runs x 35 cells per phase; FPR floor F1>={floor})")
		print("    Phase | Metric              |    F1   |   FPR   |   Acc   | Source (run / genome_type / mode / size)")
		for ph in ("GS", "GA"):
			cells = [(v, gt, md, f) for f in runs_by_ds[ds] for (v, gt, md) in all_cells(D, f, ph)]
			b1 = max(cells, key=lambda c: (c[0][0], -c[0][1]))
			bac = max(cells, key=lambda c: (c[0][2], c[0][0]))
			L = [("Best F1 (any FPR)", b1)]
			for fl_ in sorted({floor, fl[ds][2]}, reverse=True):
				ok = [c for c in cells if c[0][0] >= fl_]
				L.append((f"Best FPR (F1>={fl_})", min(ok, key=lambda c: (c[0][1], -c[0][0])) if ok else None))
			L.append(("Best Acc", bac))
			for lab, c in L:
				if c is None:
					print(f"    {ph:5s} | {lab:19s} |   —     |   —     |   —     | no cell reaches the floor"); continue
				sz = genome_size(con, *GH[(c[3], ph, c[1])])
				szs = f"{sz[0]}n x {sz[1]:.1f}b" if sz else "size n/a"
				print(f"    {ph:5s} | {lab:19s} | {c[0][0]:7.3f} | {c[0][1]:7.3f} | {c[0][2]:7.3f} | {flows[c[3]]['name']} / {c[1]} / {c[2]} / {szs}")
	print("\n================ Phase best_f1 genome size (mean±SD over runs) ================")
	for ds in DSETS:
		for ph in ("GS", "GA"):
			s = [genome_size(con, *GH[(f, ph, "best_f1")]) for f in runs_by_ds[ds]]
			s = [x for x in s if x]
			print(f"  {ds:6s} {ph}: {ms([x[0] for x in s])} neurons | {ms([x[1] for x in s])} bits  (N={len(s)})")


if __name__ == "__main__":
	main()
