"""IDS-77 offline VAL gate (random unsearched genomes, cut-E encoding, flow 7591 params; TEST never scored).
Variants differ only in how connections are drawn / theta / neurons. Paired by draw seed.
Readout per genome: F1-optimal threshold swept on VAL margins (= margin_val_cal rule), FPR, recall, Fuzzers recall."""
import sys, json, gc, sqlite3, time
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import ids57_abi_equivalence as H
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

OUT = "/Users/lacg/wnn/logs/ids/ids77"
which = sys.argv[1] if len(sys.argv) > 1 else "conn"
db = sqlite3.connect(H.DB_URI, uri=True)
base = H.flow_params(db, 7591)
factory = IdsEvaluatorFactory(log=lambda m: None)
dataset = factory.load(H.settings_for(base))
val = H.val_as_report(dataset)
meta = json.load(open(f"{OUT}/meta.json")); feats = meta["features"]; bpf = meta["bits_per_feature"]
yv = np.asarray(dataset.y_val_multi); yb = np.asarray(dataset.y_val_binary); names = list(dataset.category_names)
FUZ = names.index("Fuzzers"); P = int((yb == 1).sum()); N = int((yb == 0).sum())
starts = np.cumsum([0] + [bpf[f] for f in feats])[:-1]; T = int(sum(bpf.values()))
rfimp = np.mean([list(json.loads(l)["importances"].values()) for l in open(f"{OUT}/rf_val_cat.jsonl") if '"rf_raw"' in l], axis=0)
rf_names = list(json.loads(open(f"{OUT}/rf_val_cat.jsonl").readline())["importances"].keys())
imp = np.array([rfimp[rf_names.index(f)] for f in feats])  # TRAIN-fit RF importances


def draw(seed, scheme, npc=(250, 250), bits=80):
	rng = np.random.default_rng(seed)
	n = sum(npc) * bits
	if scheme == "bit":
		c = rng.integers(0, T, n)
	else:
		w = {"feat": np.ones(len(feats)), "sqrt": np.sqrt([bpf[f] for f in feats]),
			"imp50": 0.5 / len(feats) + 0.5 * imp / imp.sum(), "imp": imp}[scheme]
		w = w / w.sum()
		fi = rng.choice(len(feats), size=n, p=w)
		sizes = np.array([bpf[f] for f in feats])
		c = starts[fi] + (rng.random(n) * sizes[fi]).astype(np.int64)
	return ClusterGenome(bits_per_neuron=[bits] * sum(npc), neurons_per_cluster=list(npc), connections=[int(x) for x in c])


def readout(ev, g):
	ex = ev.export_neuron_outputs(g, partition="val", packed=True)
	assert np.array_equal(np.asarray(ex.labels), yb)
	sc = np.asarray(ex.scores, dtype=np.float64); m = sc[:, 1] - sc[:, 0]
	o = np.argsort(-m, kind="stable"); ms = m[o]; ys = yb[o]
	tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0); last = np.r_[ms[1:] != ms[:-1], True]
	tp, fp = tp[last], fp[last]; fn = P - tp; tn = N - fp
	f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
	i = int(np.argmax(f1)); tau = ms[last][i]
	pred = m >= tau
	return dict(f1=f1[i] * 100, fpr=fp[i] / N * 100, rec=tp[i] / P * 100, fuz=pred[yv == FUZ].mean() * 100)


variants = {
	"conn": [("bit", {}, {}), ("feat", {}, {}), ("sqrt", {}, {}), ("imp50", {}, {})],
	"theta": [("bit", {"ids_vote_share": 0.02}, {}), ("bit", {"ids_vote_share": 0.05}, {}), ("bit", {"ids_vote_share": 0.1}, {}),
		("bit", {"ids_vote_share": 0.01}, {})],
	"size": [("bit", {}, {"npc": (500, 500)}), ("feat", {}, {"npc": (500, 500)}), ("bit", {}, {"bits": 64}), ("feat", {}, {"bits": 64})],
	"size2": [("sqrt", {}, {"npc": (500, 500)}), ("bit", {}, {"npc": (1000, 1000)}), ("bit", {}, {"npc": (500, 500), "bits": 96}),
		("bit", {}, {"npc": (750, 750)})],
}[which]
out = open(f"{OUT}/gate_{which}.jsonl", "a")
for scheme, over, shape in variants:
	params = dict(base, **over); H.apply_production_env(params); s = H.settings_for(params)
	opt, ev = factory.flat(val, s)
	for seed in (1000, 1001, 1002, 1003, 1004):
		t0 = time.time(); r = readout(ev, draw(seed, scheme, **shape))
		rec = dict(which=which, scheme=scheme, over=over, shape={k: list(v) if isinstance(v, tuple) else v for k, v in shape.items()},
			seed=seed, secs=round(time.time() - t0, 1), **r)
		print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
	del opt, ev; gc.collect()
