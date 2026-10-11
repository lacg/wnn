"""IDS-26 follow-up: verbatim-duplicate neurons inside one genome.

A neuron = (bits, sorted input-bit multiset). Two identical neurons in one single-cluster discriminator train
identically and answer identically, so a duplicate adds no feature — it only double-weights one vote.
Reports, per cohort, distinct/total neurons for the phase best genome and the mean over the final population.

Usage: python scripts/ids_plateau/ids26_duplicate_neurons.py
"""
import sys, os, gzip, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ids26_population_diversity import connect, select_runs


def distinct_frac(g):
	keys, off = [], 0
	for b in g["bits_per_neuron"]:
		keys.append((b, tuple(sorted(g["connections"][off:off + b]))))
		off += b
	return len(set(keys)) / max(1, len(keys)), len(keys)


def run_fracs(path):
	with gzip.open(path, "rt") as fh:
		pr = json.load(fh)["phase_result"]
	best, n = distinct_frac(pr["best_genome"])
	pop = float(np.mean([distinct_frac(g)[0] for g in pr["final_population"]]))
	return best, n, pop


def main():
	runs = select_runs(connect())
	agg = collections.defaultdict(lambda: collections.defaultdict(list))
	for r in runs:
		key = (r["fam"], r["ds"], r["mem"])
		for ph in ("GS", "GA"):
			best, n, pop = run_fracs(r[ph])
			agg[key][ph + "_best"].append(best)
			agg[key][ph + "_n"].append(n)
			agg[key][ph + "_pop"].append(pop)
	print(f"{'cohort':28s} | {'runs':>4s} | {'GS best distinct':>16s} | {'GA best distinct':>16s} | {'GA best n (distinct n)':>22s} | {'GA pop distinct':>15s}")
	for key, d in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
		f = lambda k: f"{np.mean(d[k]):.3f}±{np.std(d[k], ddof=1):.3f}"
		nd = np.array(d["GA_n"]) * np.array(d["GA_best"])
		print(f"{'/'.join(key):28s} | {len(d['GA_best']):4d} | {f('GS_best'):>16s} | {f('GA_best'):>16s} | "
			f"{np.mean(d['GA_n']):9.0f} ({np.mean(nd):5.0f})       | {f('GA_pop'):>15s}")


if __name__ == "__main__":
	main()
