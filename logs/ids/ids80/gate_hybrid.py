"""IDS-80 offline VAL gate (prereg docs/ids80_hybrid_cut_gate.md). Arms E / H (hybrid) / B(ref).
Random 250x250x80 genomes, seeds 1000-1009, paired by uniform stream. TRAIN fit, VAL scoring only."""
import sys, json, gc, sqlite3, time
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ids57_abi_equivalence as H
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.representations import supervised_cut_fitter as SCF
from wnn.representations.cut_method import CutMethod
from wnn.representations.xgb_split_cuts import xgb_split_cuts

OUT = "/Users/lacg/wnn/logs/ids/ids80"
TIMING = ("dur", "Sintpkt", "Sjit", "Dintpkt")
SEEDS = range(1000, 1010)
HYBRID = {"on": False}
_orig_fit = SCF.SupervisedCutFitter.fit


def _merge(e, b, x):
	"""Union of E and B cuts; a B cut is dropped if it splits TRAIN the same as a kept cut."""
	vals = np.unique(x[np.isfinite(x)])
	kept = list(np.sort(e)); added = 0
	for c in np.sort(b):
		dup = False
		for k in kept:
			lo, hi = min(k, c), max(k, c)
			if not np.any((vals >= lo) & (vals < hi)):
				dup = True; break
		if not dup:
			kept.append(c); added += 1
	return np.sort(np.array(kept, dtype=np.float64)), added


def _fit(self, df, feature_types, quantile_bits, y_binary, y_multi):
	cuts = _orig_fit(self, df, feature_types, quantile_bits, y_binary, y_multi)
	if HYBRID["on"] and self.method == CutMethod.INFO_GAIN_MC:
		numeric = [f for f, t in feature_types.items() if t == "numeric"]
		bc = dict(zip(numeric, xgb_split_cuts(df[numeric].values.astype(np.float64), y_binary, self.seed)))
		info = {}
		for f in TIMING:
			x = df[f].values.astype(np.float64)
			cuts.thresholds[f], added = _merge(cuts.thresholds[f], bc[f], x)
			info[f] = dict(E=None, B=len(bc[f]), added=added, total=len(cuts.thresholds[f]))
		print("  [HYBRID] timing merge:", json.dumps(info), flush=True)
	return cuts


SCF.SupervisedCutFitter.fit = _fit
db = sqlite3.connect(H.DB_URI, uri=True)
factory = IdsEvaluatorFactory(log=lambda m: None)


def draw(seed, T, npc=(250, 250), bits=80):
	u = np.random.default_rng(seed).random(sum(npc) * bits)
	c = np.minimum((u * T).astype(np.int64), T - 1)
	return ClusterGenome(bits_per_neuron=[bits] * sum(npc), neurons_per_cluster=list(npc), connections=[int(x) for x in c])


def readout(ev, g, yb, yv, fuz):
	P = int((yb == 1).sum()); N = int((yb == 0).sum())
	ex = ev.export_neuron_outputs(g, partition="val", packed=True)
	assert np.array_equal(np.asarray(ex.labels), yb)
	sc = np.asarray(ex.scores, dtype=np.float64); m = sc[:, 1] - sc[:, 0]
	o = np.argsort(-m, kind="stable"); ms = m[o]; ys = yb[o]
	tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0); last = np.r_[ms[1:] != ms[:-1], True]
	th, tp, fp = ms[last], tp[last], fp[last]; fn = P - tp; tn = N - fp
	f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
	i = int(np.argmax(f1)); j = int(np.where(fp / N <= 0.0025)[0][-1])
	return dict(f1=f1[i] * 100, fpr=fp[i] / N * 100, rec=tp[i] / P * 100,
		rec25=tp[j] / P * 100, fpr25=fp[j] / N * 100, f1_25=f1[j] * 100, fuz25=(m[yv == fuz] >= th[j]).mean() * 100)


out = open(f"{OUT}/gate_hybrid.jsonl", "a")
for arm, flow, hyb in (("E", 7591, False), ("H", 7591, True), ("B", 7594, False)):
	HYBRID["on"] = hyb
	params = H.flow_params(db, flow)
	dataset = factory.load(H.settings_for(params))
	val = H.val_as_report(dataset)
	dataset = None; gc.collect()
	T = int(val.X_train.total_bits)
	yb = np.asarray(val.y_val_binary); yv = np.asarray(val.y_val_multi); fuz = list(val.category_names).index("Fuzzers")
	print(f"[ARM] {arm} total_bits={T}", flush=True)
	H.apply_production_env(params); s = H.settings_for(params)
	opt, ev = factory.flat(val, s)
	for seed in SEEDS:
		t0 = time.time(); g = draw(seed, T)
		r = readout(ev, g, yb, yv, fuz)
		if seed == 1000:
			fp_ = ev.evaluate_multiclass_at_thresholds(g)["footprint"]
			r["footprint"] = str(fp_)
		rec = dict(arm=arm, bits=T, seed=seed, secs=round(time.time() - t0, 1), **r)
		print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
	del opt, ev, val; gc.collect()
