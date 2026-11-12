"""IDS-57 pre-check probe 2 (read-only on DB; no flow touched). VAL-ONLY reads (tau fit on VAL, scored on VAL): the
choosing read. No TEST number is computed for any exploratory genome. Genomes are built like grid genomes
(random connections); training/scoring = Rust via the worker construction path."""
import os, sys, json, sqlite3, dataclasses, hashlib
import numpy as np
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, "/Users/lacg/wnn/src")
db = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True)
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from sklearn.metrics import f1_score
def emit(**k): print(json.dumps(k), flush=True)
params = json.loads(db.execute("select config_json from flows where id=7501").fetchone()[0])["params"]
IdsEvaluatorSettings.apply_oi_env(params)
s = IdsEvaluatorSettings.from_params(params); s_u = dataclasses.replace(s, balance_classes=False)
ds = IdsEvaluatorFactory(log=lambda m: None).load(s)
rngs = ds.encoder.feature_bit_ranges(); feats = list(ds.feature_names); T = int(ds.X_train.total_bits)
yt = np.asarray(ds.y_test_binary)
# (a) XGB (probe1, thermo64 depth 6) macro-F1 at the WNN's FPR levels — TEST, the comparator's own curve
pt = np.load("/tmp/ids57/xgb_pt.npy"); neg = np.sort(pt[yt == 0])[::-1]; out = {}
for fp in (295, 306, 343, 309, 635):
	thr = neg[fp]; pred = (pt > thr).astype(int)
	out[str(fp)] = {"fpr": float(((pred == 1) & (yt == 0)).sum() / (yt == 0).sum()), "tpr": float(pred[yt == 1].mean()), "f1": float(f1_score(yt, pred, average="macro"))}
emit(part="xgb_at_wnn_fpr", rows=out)
# (b) distinct informative bits per feature on TRAIN
Xtr = ds.X_train.to_numpy_bool(); seen = {}; distinct = []; per = {}
for f in feats:
	a, b = rngs[f]; k = 0
	for j in range(a, b):
		col = Xtr[:, j]
		if col.all() or not col.any(): continue
		h = hashlib.md5(np.packbits(col).tobytes()).hexdigest()
		if h in seen: continue
		seen[h] = j; distinct.append(j); k += 1
	per[f] = k
del Xtr
distinct = np.array(distinct)
emit(part="distinct_bits", total=int(len(distinct)), per_feature=per)
# (c) connectivity / capacity pre-check on VAL
ds_val = dataclasses.replace(ds, X_test=ds.X_val, y_test_binary=ds.y_val_binary, y_test_multi=ds.y_val_multi)
mk = lambda st: IdsEvaluatorFactory._evaluator(ds_val, st, 1, 1, 1, None)
ev_l = mk(s); ev_l._cache.set_write_rule("lattice")
ev_u = mk(s_u); ev_u._cache.set_write_rule("vote", vote_share=0.02)
gain = [json.loads(l) for l in open("/tmp/ids57/probe1.jsonl") if l.startswith('{"part": "xgb", "depth": 6')][0]["gain_share_by_feature"]
gw = np.array([np.sqrt(gain[f]) for f in feats]); gw /= gw.sum()
def draw(kind, n, rng):
	if kind == "U": return rng.integers(0, T, n)
	if kind == "D": return distinct[rng.integers(0, len(distinct), n)]
	if kind in ("FU", "G"):
		fi = rng.integers(0, len(feats), n) if kind == "FU" else rng.choice(len(feats), n, p=gw)
		return np.array([rng.integers(*rngs[feats[i]]) for i in fi])
	if kind == "GH":  # half sqrt-gain-weighted features, half uniform bits
		h = n // 2; return np.concatenate([draw("G", h, rng), draw("U", n - h, rng)])
	if kind == "FD":  # feature-uniform over DISTINCT bits
		byf = {f: distinct[(distinct >= rngs[f][0]) & (distinct < rngs[f][1])] for f in feats}
		ok = [f for f in feats if len(byf[f])]
		return np.array([rng.choice(byf[ok[i]]) for i in rng.integers(0, len(ok), n)])
def genome(kind, npc, bits, seed):
	rng = np.random.default_rng(seed); nn = 2 * npc
	conns = np.concatenate([draw(kind, bits, rng) for _ in range(nn)])
	return ClusterGenome(bits_per_neuron=[bits] * nn, neurons_per_cluster=[npc, npc], connections=[int(x) for x in conns])
def read(e, g):
	m = e.evaluate_multiclass_at_thresholds(g)["modes"]["margin_val_cal"]; c = m["confusion"]
	return [m["f1"], m["fpr"], c[1][1] / sum(c[1])]
CASES = [("U", 250, b) for b in (24, 34, 48, 64)] + [("U", 500, 34), ("U", 500, 48)] + \
	[(k, 250, b) for k in ("D", "FU", "FD", "G", "GH") for b in (34, 48)]
for kind, npc, bits in CASES:
	for seed in range(5):
		g = genome(kind, npc, bits, 1000 + seed)
		try:
			emit(part="val", kind=kind, npc=npc, bits=bits, seed=seed, lattice=read(ev_l, g), u002=read(ev_u, g))
		except Exception as ex:
			emit(part="val", kind=kind, npc=npc, bits=bits, seed=seed, error=str(ex)[:200])
