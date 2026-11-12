"""IDS-57 pre-check probe 3 (read-only on DB; no flow touched). VAL-ONLY reads, as probe 2.
usage: probe3.py unswr|unswt"""
import os, sys, json, sqlite3, dataclasses
import numpy as np
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, "/Users/lacg/wnn/src")
db = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True)
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
def emit(**k): print(json.dumps(k), flush=True)
which = sys.argv[1]; fid = {"unswr": 7501, "unswt": 7484}[which]
params = json.loads(db.execute("select config_json from flows where id=?", (fid,)).fetchone()[0])["params"]
IdsEvaluatorSettings.apply_oi_env(params)
s = IdsEvaluatorSettings.from_params(params)
ds = IdsEvaluatorFactory(log=lambda m: None).load(s)
T = int(ds.X_train.total_bits)
ds_val = dataclasses.replace(ds, X_test=ds.X_val, y_test_binary=ds.y_val_binary, y_test_multi=ds.y_val_multi)
mk = lambda st: IdsEvaluatorFactory._evaluator(ds_val, st, 1, 1, 1, None)
EV = {}
if which == "unswr":
	from sklearn.metrics import f1_score
	yv = np.asarray(ds.y_val_binary); pv = np.load("/tmp/ids57/xgb_pv.npy")
	emit(part="xgb_val", f1=float(f1_score(yv, (pv >= 0.43).astype(int), average="macro")), fpr=float(((pv >= 0.43) & (yv == 0)).sum() / (yv == 0).sum()))
	e = mk(s); e._cache.set_write_rule("lattice"); EV["lattice"] = e
	e = mk(dataclasses.replace(s, balance_classes=False)); e._cache.set_write_rule("vote", vote_share=0.02); EV["u002"] = e
	pass
	CASES = [(250, b) for b in (64, 80, 96, 128)] + [(125, 64), (500, 64)]
else:
	e = mk(s); e._cache.set_write_rule("lattice"); EV["lattice"] = e
	e = mk(s); e._cache.set_write_rule("vote", vote_share=0.1); EV["ms01"] = e
	CASES = [(250, b) for b in (24, 34, 48, 64, 96)] + [(500, 34), (500, 48)]
emit(part="data", which=which, total_bits=T, rate=s.neuron_sample_rate, balance=s.balance_classes, dedupe=s.encoding_dedupe)
def genome(npc, bits, seed):
	rng = np.random.default_rng(seed); nn = 2 * npc
	return ClusterGenome(bits_per_neuron=[bits] * nn, neurons_per_cluster=[npc, npc], connections=[int(x) for x in rng.integers(0, T, nn * bits)])
def read(e, g):
	r = e.evaluate_multiclass_at_thresholds(g); m = r["modes"]["margin_val_cal"]; c = m["confusion"]
	return [m["f1"], m["fpr"], c[1][1] / sum(c[1]), int(r["footprint"].logical_cells)]
for npc, bits in CASES:
	for seed in range(5):
		g = genome(npc, bits, 1000 + seed); row = dict(part="val", which=which, npc=npc, bits=bits, seed=seed)
		for name, e in EV.items():
			try: row[name] = read(e, g)
			except Exception as ex: row[name + "_error"] = str(ex)[:200]
		emit(**row)
