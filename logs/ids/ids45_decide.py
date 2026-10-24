"""IDS-45 budget readout + PRE-REGISTERED cap decision for the IDS-47 BINARY arms.

Luiz 24/10/2026: "choose for me and in the morning I will validate". Rule written 24/10 ~16:00 EDT,
BEFORE any IDS-45 flow ran. Arms (unswt, BINARY lattice 2m, blend_neuron, rate 0.25, same 10 seeds):
  max250 = max_neurons 250 per cluster (sum up to 500)  — IDS-44/46, the config IDS-47 was cloned with
  tot250 = max_neurons_total 250 (free split, sum <= 250) — IDS-45 7333-7342
  tot500 = max_neurons_total 500 (free split, sum <= 500) — IDS-45 7208-7217
Metric: best_f1 genome, GA Neurons final, margin_val_cal (held-out), paired by seed vs max250.
RULE (first that holds wins):
  1. tot250 if mean dF1 >= -0.10 pp AND mean dFPR <= +0.20 pp   (same total budget as QUAD 1m = fair A/B, cheaper)
  2. tot500 if dF1 95% CI lower bound > 0 AND mean dFPR <= +0.20 pp
  3. else max250 (no change)
FAIL-SAFE: fewer than 10 completed pairs per arm -> max250 (the worker keeps running; Luiz decides).
Prints the readout; last line 'DECISION <cap> <reason>'. Writes logs/ids/ids45_decision.json.
"""
import json, re, sqlite3, statistics as st, sys

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
SEEDS = ["98954", "77715", "25052", "20419", "20420", "20421", "20422", "20423", "20424", "20425"]
ARMS = {
	"max250": "IDS4_-unswt-BINARY-lattice-2m-max250-xo_blend_neuron-16bWb-r%-rate0.25",
	"tot250": "IDS45-unswt-BINARY-lattice-2m-tot250-xo_blend_neuron-16bWb-r%-rate0.25",
	"tot500": "IDS45-unswt-BINARY-lattice-2m-tot500-xo_blend_neuron-16bWb-r%-rate0.25",
}
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
OUT = "/Users/lacg/wnn/logs/ids/ids45_decision.json"


def flow_metrics(con, fid, hours):
	r = con.execute("""SELECT v.physical_neurons, v.physical_cells, v.threshold_metadata FROM validation_summaries v
		JOIN experiments e ON e.id=v.experiment_id WHERE v.flow_id=? AND e.name='GA Neurons'
		AND v.validation_point='final' AND v.genome_type='best_f1'""", (fid,)).fetchone()
	if not r:
		return None
	mv = json.loads(r[2])["margin_val_cal"]
	(tn, fp), (fn, tp) = mv["confusion"]
	return dict(fid=fid, f1=mv["f1"] * 100, fpr=mv["fpr"] * 100, acc=(tn + tp) / (tn + fp + fn + tp) * 100,
	            n=r[0], cells=r[1], h=hours)


def load(con):
	D = {}
	for arm, pat in ARMS.items():
		for fid, name, h in con.execute("""SELECT id, name, (julianday(completed_at)-julianday(started_at))*24 FROM flows
				WHERE name LIKE ? AND status='completed'""", (pat,)):
			seed = re.search(r"-r(\d+)-rate", name).group(1)
			m = flow_metrics(con, fid, h)
			if seed in SEEDS and m:
				D[(arm, seed)] = m
	return D


def ci(x):
	return st.stdev(x) / len(x) ** .5 * T95.get(len(x), 2.2) if len(x) > 1 else float("inf")


def paired(D, arm):
	ks = [s for s in SEEDS if (arm, s) in D and ("max250", s) in D]
	d = {k: [D[(arm, s)][k] - D[("max250", s)][k] for s in ks] for k in ("f1", "fpr", "acc", "h", "n")}
	return ks, d


def summary(D):
	print("arm      n   F1            FPR           Acc           neurons  cells(M)  hours")
	for arm in ARMS:
		v = [D[(arm, s)] for s in SEEDS if (arm, s) in D]
		if not v:
			print(f"{arm}   0"); continue
		f = lambda k: f"{st.mean([x[k] for x in v]):6.2f}±{(st.stdev([x[k] for x in v]) if len(v) > 1 else 0):.2f}"
		print(f"{arm}  {len(v):2d}  {f('f1')}  {f('fpr')}  {f('acc')}  {st.mean([x['n'] for x in v]):7.0f}  "
		      f"{st.mean([x['cells'] for x in v]) / 1e6:8.2f}  {st.mean([x['h'] for x in v]):5.2f}")


def decide(D):
	out = {}
	for arm in ("tot250", "tot500"):
		ks, d = paired(D, arm)
		out[arm] = dict(n=len(ks), dF1=st.mean(d["f1"]) if ks else None, dF1_ci=ci(d["f1"]) if ks else None,
		                dFPR=st.mean(d["fpr"]) if ks else None, dFPR_ci=ci(d["fpr"]) if ks else None,
		                dAcc=st.mean(d["acc"]) if ks else None, dHours=st.mean(d["h"]) if ks else None,
		                dNeurons=st.mean(d["n"]) if ks else None)
		if ks:
			o = out[arm]
			print(f"{arm} - max250  n={len(ks):2d}  dF1 {o['dF1']:+.2f}±{o['dF1_ci']:.2f}  dFPR {o['dFPR']:+.2f}±{o['dFPR_ci']:.2f}"
			      f"  dAcc {o['dAcc']:+.2f}  dNeurons {o['dNeurons']:+.0f}  dHours {o['dHours']:+.2f}")
	if any(out[a]["n"] < len(SEEDS) for a in out):
		return "max250", f"FAIL-SAFE incomplete pairs {out['tot250']['n']}/{out['tot500']['n']} of {len(SEEDS)}", out
	t2, t5 = out["tot250"], out["tot500"]
	if t2["dF1"] >= -0.10 and t2["dFPR"] <= 0.20:
		return "tot250", f"rule 1: tot250 dF1 {t2['dF1']:+.2f} >= -0.10 and dFPR {t2['dFPR']:+.2f} <= +0.20", out
	if t5["dF1"] - t5["dF1_ci"] > 0 and t5["dFPR"] <= 0.20:
		return "tot500", f"rule 2: tot500 dF1 {t5['dF1']:+.2f}±{t5['dF1_ci']:.2f} CI excludes 0, dFPR {t5['dFPR']:+.2f}", out
	return "max250", "rule 3: neither budget arm qualified — keep max250", out


def main():
	con = sqlite3.connect(DB, uri=True)
	D = load(con)
	con.close()
	summary(D)
	print()
	cap, why, out = decide(D)
	json.dump(dict(cap=cap, reason=why, paired=out), open(OUT, "w"), indent=1)
	print(f"DECISION {cap} {why}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
