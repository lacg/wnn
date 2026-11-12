"""IDS-57 probe 4: per-category recall on VAL (tau fit on VAL), random [250,250] genomes at 34b vs 64b, lattice. VAL only."""
import os, sys, json, sqlite3, dataclasses
import numpy as np
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, "/Users/lacg/wnn/src")
db = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True)
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
params = json.loads(db.execute("select config_json from flows where id=7501").fetchone()[0])["params"]
IdsEvaluatorSettings.apply_oi_env(params)
s = IdsEvaluatorSettings.from_params(params)
ds = IdsEvaluatorFactory(log=lambda m: None).load(s)
T = int(ds.X_train.total_bits); yv = np.asarray(ds.y_val_binary); yvm = np.asarray(ds.y_val_multi); cats = list(ds.category_names)
pv = np.load("/tmp/ids57/xgb_pv.npy"); pred = (pv >= 0.43).astype(int)
print(json.dumps({"part": "xgb_val_percat", **{cats[c]: float((pred[yvm == c] == (0 if cats[c] == "Normal" else 1)).mean()) for c in np.unique(yvm)}}), flush=True)
subs = {}
for c in np.unique(yvm):
	idx = np.where(yvm == c)[0]
	d = dataclasses.replace(ds, X_test=ds.X_val.row_subset(idx), y_test_binary=yv[idx], y_test_multi=yvm[idx])
	e = IdsEvaluatorFactory._evaluator(d, s, 1, 1, 1, None); e._cache.set_write_rule("lattice"); subs[cats[c]] = (e, len(idx))
for bits in (34, 64):
	for seed in (1000, 1001):
		rng = np.random.default_rng(seed)
		g = ClusterGenome(bits_per_neuron=[bits] * 500, neurons_per_cluster=[250, 250], connections=[int(x) for x in rng.integers(0, T, 500 * bits)])
		out = {}
		for name, (e, n) in subs.items():
			cf = e.evaluate_multiclass_at_thresholds(g)["modes"]["margin_val_cal"]["confusion"]
			out[name] = (cf[0][0] if name == "Normal" else cf[1][1]) / n
		print(json.dumps({"part": "wnn_val_percat", "bits": bits, "seed": seed, **out}), flush=True)
