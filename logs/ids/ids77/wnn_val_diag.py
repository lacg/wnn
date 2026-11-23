"""IDS-77: VAL-only diagnosis of the unswr cut-E winners (GA + grid best_f1 final genomes).
TEST is never scored: the report slot is replaced by VAL (val_as_report). Rust train+score only.
Saves per-row VAL margins, per-neuron cell codes summary and genome stats to /Users/lacg/wnn/logs/ids/ids77/.
Usage: wnn_val_diag.py 7591,7592,7593"""
import sys, json, gc, sqlite3
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ids57_abi_equivalence as H
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory

OUT = "/Users/lacg/wnn/logs/ids/ids77"
flows = [int(f) for f in sys.argv[1].split(",")]
db = sqlite3.connect(H.DB_URI, uri=True)
H.check_shared_dataset(db, flows)
factory = IdsEvaluatorFactory(log=lambda m: None)
s0 = H.settings_for(H.flow_params(db, flows[0]))
dataset = factory.load(s0)
names = list(dataset.category_names)
enc = dataset.encoder
feat_of_bit = []
for f in dataset.feature_names:
	feat_of_bit += [f] * int(enc.per_feature_bits_[f])
print("total bits", len(feat_of_bit), "features", len(dataset.feature_names), flush=True)
val = H.val_as_report(dataset)
yv = np.asarray(dataset.y_val_multi); ybv = np.asarray(dataset.y_val_binary)
np.savez(f"{OUT}/val_labels.npz", y_multi=yv, y_bin=ybv, y_train_multi=np.asarray(dataset.y_train_multi),
	y_train_bin=np.asarray(dataset.y_train_binary))
json.dump(dict(names=names, feat_of_bit=feat_of_bit, features=list(dataset.feature_names),
	bits_per_feature={f: int(enc.per_feature_bits_[f]) for f in dataset.feature_names}), open(f"{OUT}/meta.json", "w"))


def genome_of(flow, phase):
	row = db.execute("select t.genome_hash from test_summaries t join experiments e on e.id=t.experiment_id "
		"where t.flow_id=? and e.phase_type=? and t.test_point='final' and t.genome_type='best_f1'", (flow, phase)).fetchone()
	return row[0], H.stored_genome(db, row[0])


out = open(f"{OUT}/wnn_val_diag.jsonl", "a")
for f in flows:
	params = H.flow_params(db, f)
	H.apply_production_env(params)
	s = H.settings_for(params)
	H.check_retrain_settings(f, params, s)
	opt, ev = factory.flat(val, s)
	for phase in ("ga_neurons", "grid_search"):
		gh, g = genome_of(f, phase)
		res = ev.evaluate_multiclass_at_thresholds(g)
		modes = res["modes"]; vc = modes["margin_val_cal"]; tau = vc["tau"]
		ex = ev.export_neuron_outputs(g, partition="val", packed=False)
		sc = np.asarray(ex.scores, dtype=np.float64); m = sc[:, 1] - sc[:, 0]
		assert np.array_equal(np.asarray(ex.labels), ybv), "label order"
		cells = ex.unpacked_cells()
		npc = ex.neurons_per_cluster
		codes = {int(c): int((cells == c).sum()) for c in np.unique(cells)}
		# per-row fraction of neurons answering each code, per cluster
		sl = ex.cluster_slices()
		frac = {}
		for ci, ss in enumerate(sl):
			for c in np.unique(cells):
				frac[f"c{ci}_code{int(c)}"] = (cells[:, ss] == c).mean(axis=1).astype(np.float32)
		np.savez_compressed(f"{OUT}/val_{f}_{phase}.npz", margin=m, scores=sc, **frac)
		pred = m >= tau
		cm = [[int(((ybv == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]
		cat = {names[c]: [int(pred[yv == c].sum()), int((yv == c).sum())] for c in np.unique(yv)}
		bpn = list(g.bits_per_neuron); conns = np.asarray(g.connections)
		fu = {}
		for b in conns:
			fu[feat_of_bit[b]] = fu.get(feat_of_bit[b], 0) + 1
		rec = dict(flow=f, seed=params["seed"], phase=phase, genome_hash=gh, tau=tau,
			val_f1=vc["macro_f1"] * 100, val_fpr=vc["fpr"] * 100, val_acc=vc["acc"] * 100, rust_cm=vc["confusion"], py_cm=cm,
			cat=cat, cell_weights=ex.cell_weights, code_counts=codes, neurons_per_cluster=list(g.neurons_per_cluster),
			bits_min=int(min(bpn)), bits_max=int(max(bpn)), n_neurons=len(bpn), distinct_bits_used=int(len(np.unique(conns))),
			feature_conn_counts=fu, footprint=str(res["footprint"]),
			val_modes={k: [v["macro_f1"] * 100, v["fpr"] * 100] for k, v in modes.items()},
			val_partition_valcal=(res["val_partition"].get("margin_val_cal", {}) or {}).get("macro_f1"))
		print(json.dumps({k: rec[k] for k in ("flow", "phase", "tau", "val_f1", "val_fpr", "rust_cm", "py_cm", "neurons_per_cluster", "code_counts")}), flush=True)
		out.write(json.dumps(rec) + "\n"); out.flush()
		del ex, cells, frac; gc.collect()
	del opt, ev; gc.collect()
