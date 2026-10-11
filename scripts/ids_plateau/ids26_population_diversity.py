"""IDS-26: final GA population diversity vs the grid population (search-collapse test, spike IDS-23).

Read-only: DB opened mode=ro, checkpoints streamed one run at a time.

Per run (Grid Search final_population = the GA's seed population; GA Neurons final_population):
  - pop size, total neurons, mean bits/neuron, distinct fingerprints
  - within-population mean pairwise Jaccard, three granularities:
      J_neu  : neuron identity  — Jaccard of the SETS of neurons, a neuron = (bits, sorted input-bit multiset).
               Crossover/mutation in the NEURONS phase copy neurons verbatim, so J_neu measures shared lineage.
      J_in   : input-bit set    — Jaccard of the union of input bits observed by the genome (saturates for big genomes).
      J_use  : input-bit usage  — weighted (Ruzicka) Jaccard of the normalized per-input-bit usage histograms
               (feature-selection similarity; the grid's random draws give the chance baseline).
  - basins: single-linkage components over the edge rule
      J_neu >= 0.5  AND  |dN|/max(N) <= 0.10  AND  |d mean bits| <= 1
    (same lineage AND same architecture). Also reported at J_neu >= 0.2 (sensitivity) and as architecture
    cells (modal bits, neurons in 50-neuron buckets).
  - pool_ratio: |union of distinct neurons in the population| / mean neurons per genome — the gene-pool size in
    genome-equivalents (1.0 = every genome is the same neuron set; ~pop size = no sharing).
  - ancestry (GA): fraction of the GA best genome's neurons found verbatim in the grid population, number of grid
    genomes contributing, share of the single largest contributor.
Cross-run: best genome (phase_result.best_genome) of each phase compared pairwise within a dataset, split into
  same-arm/different-seed, same-seed/different-arm, different-both.

Usage: python scripts/ids_plateau/ids26_population_diversity.py [--out results.json]
"""
import sys, os, re, gzip, json, argparse, itertools, collections
import sqlite3
import numpy as np

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"

ROOT = "/Users/lacg/wnn"
FAMILIES = [
	("SP100", re.compile(r"^SP100-(cicids|ciciot|unswr|unswt)-(quad)-\d+b(W[abc])-r(\d+)$")),
	("SWEEP", re.compile(r"^IDSXD-(unswr|unswt)-(quad)-\d+b-(?!B34-CTRL-r\d+$)(.+?)-r(\d+)(?:-immfix)?$")),
	("SWEEP-QSR", re.compile(r"^IDSXD-(unswr)-(qsr)-\d+b-(.+?)-r(\d+)-abi13$")),
	("SWEEP", re.compile(r"^IDSAGGX?-(cicids|ciciot)-(quad)-96b-(.+?)-r(\d+)$")),
]
J_LINEAGE = 0.5
J_LOOSE = 0.2


def connect():
	return sqlite3.connect(DB, uri=True)


def select_runs(con):
	"""Completed flows of the four cohorts -> list of run dicts with grid/GA checkpoint paths."""
	runs = []
	for fid, name in con.execute("SELECT id,name FROM flows WHERE status='completed' ORDER BY id"):
		for fam, rx in FAMILIES:
			m = rx.match(name)
			if m and _keep(name):
				run = dict(fid=fid, name=name, fam=fam, ds=m.group(1), mem=m.group(2), arm=m.group(3), seed=int(m.group(4)))
				run.update(_ckpt_paths(con, fid))
				if run.get("GS") and run.get("GA"):
					runs.append(run)
				break
	return runs


def _keep(name):
	# QUAD B34-CTRL on unswr: only the -immfix reruns (pre-fix r20401-05 are 34/8 hybrids, IDS-17)
	if name.startswith("IDSXD-unswr-quad-64b-B34-CTRL") and not name.endswith("-immfix"):
		return False
	return not name.startswith("zINVALID")


def _ckpt_paths(con, fid):
	q = """SELECT e.phase_type, c.file_path FROM checkpoints c JOIN experiments e ON e.id=c.experiment_id
		WHERE e.flow_id=? AND c.checkpoint_type='experiment_end' ORDER BY c.id"""
	out = {}
	for ph, fp in con.execute(q, (fid,)):
		key = {"grid_search": "GS", "ga_neurons": "GA"}.get(ph)
		if key and os.path.exists(os.path.join(ROOT, fp)):
			out[key] = os.path.join(ROOT, fp)
	return out


class GenomeView:
	"""Hashable neuron multiset + input-bit usage of one genome."""

	def __init__(self, g):
		bits = g["bits_per_neuron"]
		conns = g["connections"]
		self.n = len(bits)
		self.mean_bits = sum(bits) / max(1, self.n)
		self.modal_bits = collections.Counter(bits).most_common(1)[0][0] if bits else 0
		self.neurons = self._neuron_keys(bits, conns)
		self.inputs = set(conns)
		self.usage = collections.Counter(conns)
		self.fp = hash((tuple(bits), tuple(conns)))
		self.metrics = g.get("cached_metrics") or {}

	@staticmethod
	def _neuron_keys(bits, conns):
		keys, off = set(), 0
		for b in bits:
			keys.add((b, tuple(sorted(conns[off:off + b]))))
			off += b
		return keys


def jac(a, b):
	u = len(a | b)
	return len(a & b) / u if u else 1.0


def wjac(ca, cb):
	ta, tb = sum(ca.values()), sum(cb.values())
	keys = set(ca) | set(cb)
	mn = sum(min(ca.get(k, 0) / ta, cb.get(k, 0) / tb) for k in keys)
	mx = sum(max(ca.get(k, 0) / ta, cb.get(k, 0) / tb) for k in keys)
	return mn / mx if mx else 1.0


def same_basin(a, b, jthr):
	if abs(a.n - b.n) / max(a.n, b.n, 1) > 0.10 or abs(a.mean_bits - b.mean_bits) > 1.0:
		return False
	return jac(a.neurons, b.neurons) >= jthr


def components(views, jthr):
	parent = list(range(len(views)))

	def find(i):
		while parent[i] != i:
			parent[i] = parent[parent[i]]
			i = parent[i]
		return i
	for i, j in itertools.combinations(range(len(views)), 2):
		if find(i) != find(j) and same_basin(views[i], views[j], jthr):
			parent[find(i)] = find(j)
	return len({find(i) for i in range(len(views))})


def pop_stats(views):
	pairs = list(itertools.combinations(views, 2))
	ns = np.array([v.n for v in views], float)
	bs = np.array([v.mean_bits for v in views], float)
	f1 = np.array([v.metrics.get("f1", np.nan) for v in views], float) * 100
	return dict(
		size=len(views), distinct=len({v.fp for v in views}),
		n_mean=ns.mean(), n_sd=ns.std(ddof=1), n_min=ns.min(), n_max=ns.max(),
		b_mean=bs.mean(), b_sd=bs.std(ddof=1), b_min=bs.min(), b_max=bs.max(),
		j_neu=float(np.mean([jac(a.neurons, b.neurons) for a, b in pairs])),
		j_in=float(np.mean([jac(a.inputs, b.inputs) for a, b in pairs])),
		j_use=float(np.mean([wjac(a.usage, b.usage) for a, b in pairs])),
		basins=components(views, J_LINEAGE), basins_loose=components(views, J_LOOSE),
		arch_cells=len({(v.modal_bits, int(v.n // 50)) for v in views}),
		f1_kfold_range=float(np.nanmax(f1) - np.nanmin(f1)),
		pool_ratio=len(set().union(*(v.neurons for v in views))) / ns.mean(),
	)


def ancestry(best, grid_views):
	"""How much of the GA best genome is verbatim grid material, and from how many grid genomes."""
	owner = collections.defaultdict(set)
	for gi, gv in enumerate(grid_views):
		for k in gv.neurons:
			owner[k].add(gi)
	contrib = collections.Counter()
	inherited = 0
	for k in best.neurons:
		if k in owner:
			inherited += 1
			for gi in owner[k]:
				contrib[gi] += 1
	top_idx, top_cnt = contrib.most_common(1)[0] if contrib else (-1, 0)
	return dict(inherited=inherited / max(1, len(best.neurons)), n_ancestors=len(contrib),
		top_ancestor=top_cnt / max(1, len(best.neurons)), top_ancestor_rank=top_idx)


def pop_ancestry(ga_views, grid_views):
	"""Across the WHOLE final GA population: how many grid genomes still contribute any neuron."""
	owner = {}
	for gi, gv in enumerate(grid_views):
		for k in gv.neurons:
			owner.setdefault(k, set()).add(gi)
	alive = set()
	for v in ga_views:
		for k in v.neurons:
			alive |= owner.get(k, set())
	return len(alive)


def analyze_run(run):
	out = dict({k: run[k] for k in ("fid", "name", "fam", "ds", "mem", "arm", "seed")})
	best = {}
	grid_views = None
	for ph in ("GS", "GA"):
		with gzip.open(run[ph], "rt") as fh:
			d = json.load(fh)
		pr = d["phase_result"]
		views = [GenomeView(g) for g in pr["final_population"]]
		out[ph] = pop_stats(views)
		out[ph]["iterations"] = pr.get("iterations_run")
		best[ph] = GenomeView(pr["best_genome"])
		out[ph]["best_n"], out[ph]["best_b"] = best[ph].n, best[ph].mean_bits
		if ph == "GS":
			grid_views = views
		else:
			out["anc"] = ancestry(best["GA"], grid_views)
			out["anc"]["pop_alive_ancestors"] = pop_ancestry(views, grid_views)
		del d, pr, views
	return out, best


def cross_run(results, bests):
	"""Pairwise best-genome similarity within each (family, dataset, mem), by pair relation."""
	groups = collections.defaultdict(list)
	for i, r in enumerate(results):
		groups[(r["fam"], r["ds"], r["mem"])].append(i)
	out = {}
	for key, idx in groups.items():
		rel = collections.defaultdict(lambda: collections.defaultdict(list))
		for i, j in itertools.combinations(idx, 2):
			ri, rj = results[i], results[j]
			r = "same-arm/diff-seed" if ri["arm"] == rj["arm"] else ("same-seed/diff-arm" if ri["seed"] == rj["seed"] else "diff-both")
			for ph in ("GS", "GA"):
				a, b = bests[i][ph], bests[j][ph]
				rel[(r, ph)]["j_neu"].append(jac(a.neurons, b.neurons))
				rel[(r, ph)]["j_in"].append(jac(a.inputs, b.inputs))
				rel[(r, ph)]["j_use"].append(wjac(a.usage, b.usage))
		out["|".join(key)] = {f"{r}|{ph}": {m: (float(np.mean(v)), len(v)) for m, v in ms.items()} for (r, ph), ms in rel.items()}
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", default="ids26_results.json")
	args = ap.parse_args()
	con = connect()
	runs = select_runs(con)
	print(f"{len(runs)} runs", flush=True)
	results, bests = [], []
	for k, run in enumerate(runs):
		res, best = analyze_run(run)
		results.append(res)
		bests.append(best)
		print(f"[{k + 1}/{len(runs)}] {run['name']}  GA J_neu={res['GA']['j_neu']:.3f} basins={res['GA']['basins']}", flush=True)
	xr = cross_run(results, bests)
	with open(args.out, "w") as fh:
		json.dump(dict(runs=results, cross=xr), fh, indent=1)
	print("wrote", args.out)


if __name__ == "__main__":
	main()
