"""IDS-57 diagnosis probe (read-only on DB; no flow touched). Worker construction path
(IdsEvaluatorFactory -> IDSCacheWrapper), Rust training/scoring only (evaluate_multiclass_at_thresholds).
XGBoost = scripts/verify_ids_baselines.py recipe on the SAME encoded matrix. WNN per-category recall and the
WNN-vs-XGB error overlap come from TEST row-subset caches (val unchanged => tau identical), as in /tmp/ids56/probe2.py."""
import os, sys, json, sqlite3, dataclasses, time
import numpy as np
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, "/Users/lacg/wnn/src")
db = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True)
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ids.dataset import TOP20_RF_FEATURES
OUT = "/tmp/ids57"
def emit(**k): print(json.dumps(k), flush=True)
FLOWS = [7501, 7505, 7509, 7513, 7517]
params = json.loads(db.execute("select config_json from flows where id=7501").fetchone()[0])["params"]
IdsEvaluatorSettings.apply_oi_env(params)
s = IdsEvaluatorSettings.from_params(params)
ds = IdsEvaluatorFactory(log=lambda m: None).load(s)
enc = ds.encoder
rng = enc.feature_bit_ranges()
emit(part="features", wnn_used=list(ds.feature_names), legacy_list=list(TOP20_RF_FEATURES),
	same_set=sorted(ds.feature_names) == sorted(TOP20_RF_FEATURES), same_order=list(ds.feature_names) == list(TOP20_RF_FEATURES),
	bit_ranges={k: list(v) for k, v in rng.items()}, total_bits=int(ds.X_train.total_bits),
	types={k: enc.feature_types_.get(k) for k in ds.feature_names})
yt = np.asarray(ds.y_test_binary); ytm = np.asarray(ds.y_test_multi); cats = list(ds.category_names)
yv = np.asarray(ds.y_val_binary); ytr = np.asarray(ds.y_train_binary)
emit(part="support", n_train=int(len(ytr)), n_val=int(len(yv)), n_test=int(len(yt)),
	test={cats[c]: int((ytm == c).sum()) for c in np.unique(ytm)},
	train={cats[c]: int((np.asarray(ds.y_train_multi) == c).sum()) for c in np.unique(ds.y_train_multi)})

# ---------- XGBoost on the same encoded bits ----------
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
Xtr = ds.X_train.to_numpy_bool(); Xte = ds.X_test.to_numpy_bool(); Xva = ds.X_val.to_numpy_bool()
def score(y, pred):
	tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
	fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
	return {"f1": float(f1_score(y, pred, average="macro")), "fpr": fp / (fp + tn), "tpr": tp / (tp + fn), "conf": [tn, fp, fn, tp]}
def val_thr(p):
	grid = np.linspace(0.01, 0.99, 99)
	return float(max(grid, key=lambda t: f1_score(yv, (p >= t).astype(int), average="macro")))
def percat(pred):
	return {cats[c]: float((pred[ytm == c] == (0 if cats[c] == "Normal" else 1)).mean()) for c in np.unique(ytm)}
def tpr_at_fpr(p_te, targets):
	neg = np.sort(p_te[yt == 0])[::-1]; out = {}
	for f in targets:
		k = int(round(f * len(neg))); thr = neg[k] if k < len(neg) else 0.0
		out[str(f)] = float((p_te[yt == 1] > thr).mean())
	return out
xgb_pred = None
for depth in (6, 1, 2, 3):
	t0 = time.time()
	m = XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=0.1, n_jobs=4, random_state=42, eval_metric="logloss", verbosity=0)
	m.fit(Xtr, ytr)
	pv = m.predict_proba(Xva)[:, 1]; pt = m.predict_proba(Xte)[:, 1]
	thr = val_thr(pv); pred = (pt >= thr).astype(int)
	row = dict(part="xgb", depth=depth, thr=thr, secs=round(time.time() - t0, 1), **score(yt, pred), percat=percat(pred),
		tpr_at_fpr=tpr_at_fpr(pt, [0.001, 0.0019, 0.0035, 0.0042, 0.006, 0.01]))
	if depth == 6:
		xgb_pred = pred; np.save(f"{OUT}/xgb_pt.npy", pt); np.save(f"{OUT}/xgb_pv.npy", pv)
		gain = m.get_booster().get_score(importance_type="total_gain")
		tot = sum(gain.values()); per = {}
		for f, (a, b) in rng.items():
			per[f] = sum(v for k, v in gain.items() if a <= int(k[1:]) < b) / tot
		row["gain_share_by_feature"] = per
		row["n_bits_used"] = len(gain)
	emit(**row)
del Xtr, Xva

# ---------- WNN ----------
mk = lambda d: IdsEvaluatorFactory._evaluator(d, s, 1, 1, 1, None)
def genome_for(fid, phase="ga_neurons", gtype="best_f1"):
	h, = db.execute("""select t.genome_hash from test_summaries t join experiments e on e.id=t.experiment_id
		where t.flow_id=? and e.phase_type=? and t.genome_type=? and t.test_point='final'""", (fid, phase, gtype)).fetchone()
	tj, cj = db.execute("select tiers_json, connections_json from genomes where genome_hash=? and connections_json is not null limit 1", (h,)).fetchone()
	t = json.loads(tj)
	return ClusterGenome(bits_per_neuron=t["bits_per_neuron"], neurons_per_cluster=t["neurons_per_cluster"], connections=[int(x) for x in cj.split(",")])
def sub_ds(idx):
	return dataclasses.replace(ds, X_test=ds.X_test.row_subset(idx), y_test_binary=yt[idx], y_test_multi=ytm[idx])
ev_full = mk(ds)
subs = {}
for c in np.unique(ytm):
	idx = np.where(ytm == c)[0]; subs["cat:" + cats[c]] = (mk(sub_ds(idx)), len(idx))
xgb_ok = xgb_pred == yt
for nm, mask in (("ben_xgbOK", (yt == 0) & xgb_ok), ("ben_xgbFP", (yt == 0) & ~xgb_ok), ("att_xgbOK", (yt == 1) & xgb_ok), ("att_xgbFN", (yt == 1) & ~xgb_ok)):
	idx = np.where(mask)[0]; subs["ov:" + nm] = (mk(sub_ds(idx)), len(idx))
bit_feat = np.empty(ds.X_train.total_bits, dtype=object)
for f, (a, b) in rng.items(): bit_feat[a:b] = f
for fid in FLOWS:
	g = genome_for(fid)
	t0 = time.time()
	full = ev_full.evaluate_multiclass_at_thresholds(g)["modes"]
	m = full["margin_val_cal"]
	row = dict(part="wnn", flow=fid, neurons=list(g.neurons_per_cluster), mean_bits=float(np.mean(g.bits_per_neuron)),
		bits_hist={str(k): int(v) for k, v in zip(*np.unique(g.bits_per_neuron, return_counts=True))},
		f1=m["f1"], fpr=m["fpr"], tau=m["tau"], conf=m["confusion"], full_secs=round(time.time() - t0, 1),
		modes={k: [v["f1"], v["fpr"], v["confusion"][1][1] / sum(v["confusion"][1])] for k, v in full.items()})
	conn = np.asarray(g.connections); off = np.cumsum([0] + list(g.bits_per_neuron)); n0 = g.neurons_per_cluster[0]
	share = {}
	for ci, (a, b) in enumerate(((0, n0), (n0, len(g.bits_per_neuron)))):
		cc = conn[off[a]:off[b]]; fs, cn = np.unique(bit_feat[cc], return_counts=True)
		share[f"cluster{ci}"] = {str(f): float(c / len(cc)) for f, c in zip(fs, cn)}
		share[f"cluster{ci}_distinct_bits"] = int(len(np.unique(cc)))
	row["conn_share"] = share
	out = {}
	for nm, (e, n) in subs.items():
		mm = e.evaluate_multiclass_at_thresholds(g)["modes"]["margin_val_cal"]; cf = mm["confusion"]
		attack_sub = not (nm == "cat:Normal" or nm.startswith("ov:ben"))
		out[nm] = {"n": n, "correct": (cf[1][1] if attack_sub else cf[0][0]) / max(1, n), "tau": mm["tau"]}
	row["subsets"] = out; row["secs"] = round(time.time() - t0, 1)
	emit(**row)
