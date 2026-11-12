"""IDS-57 arm F gate: shared helpers (read-only on DB; no flow touched; VAL-only reads).

Training and scoring are Rust only, through the worker's construction path:
IdsEvaluatorFactory -> IDSEvaluator -> (a) IDSCacheWrapper.evaluate_multiclass_at_thresholds (K x K confusion at
the cache's own tau) and (b) IDSEvaluator._build_streamer -> IDSGenomeStreamerWrapper (train_chunk / score_chunk /
take_scores), the one path in the ABI 28 wheel that hands back the per-row K-vector of cluster scores.
The tau sweep and the metrics are Rust too (ram_accelerator.multiclass_modes_from_scores on a 2-column
[benign, attack-aggregate] score matrix). Python only reduces the attack columns (max / sum / z-normalised max):
that reduction IS the decision rule under test, not existing accelerator logic."""
import os, sys, json, sqlite3, dataclasses
import numpy as np
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, "/Users/lacg/wnn/src")
import ram_accelerator
from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.architecture.ids_evaluator import _compute_class_weights
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
CHUNK = 250_000

def emit(**k): print(json.dumps(k), flush=True)

def load(flow_id=7501):
	"""The control flow's dataset + settings; 'test' slot rebound to VAL so nothing here can read TEST."""
	db = sqlite3.connect(DB, uri=True)
	params = json.loads(db.execute("select config_json from flows where id=?", (flow_id,)).fetchone()[0])["params"]
	IdsEvaluatorSettings.apply_oi_env(params)
	s = IdsEvaluatorSettings.from_params(params)
	ds = IdsEvaluatorFactory(log=lambda m: None).load(s)
	ds_val = dataclasses.replace(ds, X_test=ds.X_val, y_test_binary=ds.y_val_binary, y_test_multi=ds.y_val_multi)
	return s, ds_val

def relabel(ds_val, groups):
	"""groups: list of lists of category names; groups[0] must be ['Normal']. Returns a dataset whose multi labels
	are the group index (cluster-label mapping), plus the group names."""
	cats = list(ds_val.category_names)
	lut = np.full(len(cats), -1, dtype=np.int64)
	for gi, g in enumerate(groups):
		for name in g: lut[cats.index(name)] = gi
	assert (lut >= 0).all(), "every category must be mapped"
	names = ["+".join(g) if len(g) <= 2 else f"other{len(g)}" for g in groups]
	m = lambda y: lut[np.asarray(y)]
	return dataclasses.replace(ds_val, y_train_multi=m(ds_val.y_train_multi), y_val_multi=m(ds_val.y_val_multi),
		y_test_multi=m(ds_val.y_test_multi), category_names=names), names

def evaluator(ds_k, s, k):
	"""K-cluster evaluator with the control flow's training settings (lattice, BINARY, rate, OI, balance)."""
	if k == 2:
		st = s
	else:
		st = dataclasses.replace(s, classification="multi")
	ev = IdsEvaluatorFactory._evaluator(ds_k, st, 1, 1, 1, None)
	ev._cache.set_write_rule("lattice")
	ev._streaming_class_weights = (_compute_class_weights(np.asarray(ev._y_train, dtype=np.int64), ev._num_classes,
		st.class_weight_multiplier) if st.balance_classes else None)
	return ev

def streamer_scores(ev, ds_k, genome):
	"""Train ONCE on the full train partition through the streamer, score VAL; return (n_val x K scores, footprint)."""
	st = ev._build_streamer(genome)
	tf = ev._total_features
	packed = ds_k.X_train.as_packed_uint8(); y = ev._y_train
	for a in range(0, packed.shape[0], CHUNK):
		b = min(a + CHUNK, packed.shape[0])
		st.train_chunk(np.ascontiguousarray(packed[a:b].ravel()), y[a:b], tf)
	st.seal_for_scoring()
	fp = st.footprint()
	pv = ds_k.X_val.as_packed_uint8(); yv = ev._y_val
	for a in range(0, pv.shape[0], CHUNK):
		b = min(a + CHUNK, pv.shape[0])
		st.score_chunk(np.ascontiguousarray(pv[a:b].ravel()), yv[a:b], tf)
	flat = np.asarray(st.take_scores(), dtype=np.float64)
	return flat.reshape(len(yv), ev._num_classes), fp

def rust_modes(scores, labels, k, normal=0):
	"""Rust decode modes; VAL passed as eval, train and val (tuning read: fit on VAL, scored on VAL)."""
	fl = [float(x) for x in scores.ravel()]; lb = [int(x) for x in labels]
	_, modes = ram_accelerator.multiclass_modes_from_scores(fl, lb, fl, lb, k, normal, val_scores=fl, val_labels=lb)
	return {m[0]: m for m in modes}

def binary_read(benign, attack_agg, y_bin):
	"""2-class macro-F1 / FPR / tau at the VAL-optimal margin threshold (Rust sweep + Rust metrics)."""
	two = np.stack([benign, attack_agg], axis=1)
	m = rust_modes(two, y_bin, 2)["margin_val_cal"]
	conf = np.asarray(m[7], dtype=np.int64).reshape(2, 2)
	return {"f1": m[4], "fpr": m[6], "tau": m[1], "tpr": float(conf[1, 1] / conf[1].sum()), "conf": conf.tolist()}

def draw_genome(spec, total_bits, seed):
	"""spec: [(neurons, bits)] per cluster, cluster 0 = benign. One sequential stream per seed, neuron by neuron
	(the A1 gate's draw: probe2.genome('U', ...)), so a 250 x 34 benign cluster is the SAME neurons in every variant."""
	rng = np.random.default_rng(seed); bits = []; conns = []
	for n, b in spec:
		for _ in range(n):
			bits.append(b); conns.append(rng.integers(0, total_bits, b))
	return ClusterGenome(bits_per_neuron=bits, neurons_per_cluster=[n for n, _ in spec],
		connections=[int(x) for x in np.concatenate(conns)])
