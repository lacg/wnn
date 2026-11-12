"""IDS-57 arm F gate, step 1: can today's wheel (ABI 28) train a K-cluster memory with per-cluster (neurons, bits)
on per-category labels and hand back per-row cluster scores? Checks, on VAL only:
  F1  2-cluster 250x34: streamer scores -> Rust modes == IDSCacheWrapper.evaluate_multiclass_at_thresholds
  F2  the same read reproduces the A1 gate baseline (probe2, kind U, 34 bits, seed 0)
  F3  10-cluster (9 attack categories), mixed widths (Fuzzers 64 bits): streamer == cache, mode by mode"""
import time
from armf_common import *
s, ds = load()
T = int(ds.X_train.total_bits); cats = list(ds.category_names)
yv_bin = np.asarray(ds.y_val_binary); yv_cat = np.asarray(ds.y_val_multi)
emit(part="data", abi=ram_accelerator.ABI_VERSION, total_bits=T, cats=cats, n_train=int(len(ds.y_train_binary)), n_val=int(len(yv_bin)),
	val_support={c: int((yv_cat == i).sum()) for i, c in enumerate(cats)},
	train_support={c: int((np.asarray(ds.y_train_multi) == i).sum()) for i, c in enumerate(cats)})
def same(a, b): return all(abs(x - y) < 1e-12 for x, y in zip(a, b))
def cmp(ev, ds_k, g, k, tag):
	t0 = time.time(); ref = ev.evaluate_multiclass_at_thresholds(g); t1 = time.time()
	sc, fp = streamer_scores(ev, ds_k, g); t2 = time.time()
	got = rust_modes(sc, ev._y_val, k)
	rows = {}
	for name, r in ref["modes"].items():
		m = got[name]; conf = [list(map(int, m[7][i * k:(i + 1) * k])) for i in range(k)]
		rows[name] = {"cache": [r["f1"], r["fpr"]], "streamer": [m[4], m[6]], "confusion_equal": conf == r["confusion"],
			"tau_equal": ("tau" not in r) or abs(r["tau"] - m[1]) < 1e-12}
	emit(part="fidelity", tag=tag, k=k, cache_secs=round(t1 - t0, 1), streamer_secs=round(t2 - t1, 1),
		cells_cache=int(ref["footprint"].logical_cells), cells_streamer=int(fp[2]), modes=rows,
		all_equal=all(v["confusion_equal"] and v["tau_equal"] for k_, v in rows.items() if k_ not in ("margin_train_cal", "margin_train_f1")))
	return sc
# F1 / F2
ev2 = evaluator(ds, s, 2)
g = draw_genome([(250, 34), (250, 34)], T, 1000)
sc = cmp(ev2, ds, g, 2, "2cluster_250x34_seed0")
emit(part="baseline_read", **binary_read(sc[:, 0], sc[:, 1], yv_bin))
# F3
groups = [["Normal"]] + [[c] for c in cats if c != "Normal"]
ds10, names = relabel(ds, groups)
ev10 = evaluator(ds10, s, len(groups))
spec = [(250, 34)] + [(28, 64 if n == "Fuzzers" else 34) for n in names[1:]]
g10 = draw_genome(spec, T, 1000)
cmp(ev10, ds10, g10, len(groups), "10cluster_mixed_widths_seed0")
