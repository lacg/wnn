"""Box table of the IDS-80 TEST read (gate_hybrid_test.jsonl): one row per random genome, E | H | B."""
import json, numpy as np
R = [json.loads(l) for l in open("/Users/lacg/wnn/logs/ids/ids80/gate_hybrid_test.jsonl")]
A = {a: {r["seed"]: r for r in R if r["arm"] == a} for a in "EHB"}
HEADER = ("genome (draw)", "F1 % (E | H | B)", "FPR % (E | H | B)", "acc % (E | H | B)", "cells/neuron (E | H | B)", "bits (E | H | B)")
def row(label, get):
	return (label, " | ".join("%.2f" % get(a, "f1") for a in "EHB"), " | ".join("%.3f" % get(a, "fpr") for a in "EHB"),
		" | ".join("%.2f" % get(a, "acc") for a in "EHB"), " | ".join("%.0f" % get(a, "cpn") for a in "EHB"),
		" | ".join("%d" % get(a, "bits") for a in "EHB"))
def val(r, k):
	return r["logical_cells"] / r["logical_neurons"] if k == "cpn" else r[k]
rows = [row("s%d" % s, lambda a, k, s=s: val(A[a][s], k)) for s in sorted(A["E"])]
mean = row("mean (10)", lambda a, k: np.mean([val(r, k) for r in A[a].values()]))
allr = rows + [mean, HEADER]
w = [max(len(r[i]) for r in allr) + 2 for i in range(len(HEADER))]
rule = lambda l, m, r_: l + m.join("─" * x for x in w) + r_
line = lambda r: "│" + "│".join((" " + r[i]).ljust(w[i]) if i == 0 else r[i].center(w[i]) for i in range(len(HEADER))) + "│"
print("\n".join([rule("┌", "┬", "┐"), line(HEADER), rule("├", "┼", "┤")] + [line(r) for r in rows] + [rule("├", "┼", "┤"), line(mean), rule("└", "┴", "┘")]))
