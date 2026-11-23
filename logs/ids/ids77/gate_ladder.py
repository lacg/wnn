"""IDS-77 offline VAL gate: a theta LADDER across neurons (graded per-address share readout at EQUAL cells).
A 2x250x80b random genome (production per-bit draw) is cut into 5 sub-genomes of 2x50 neurons; each sub-genome is
trained by Rust under its own vote share theta; cluster score = mean of the 5 sub-scores (= the fraction-of-neurons
score of the whole genome). Control = the same 5 sub-genomes all at theta 0.02 (production). TEST never scored."""
import sys, json, gc, sqlite3
sys.path.insert(0, "/Users/lacg/wnn/scripts"); sys.path.insert(0, "/Users/lacg/wnn/logs/ids/ids77")
import numpy as np
import ids57_abi_equivalence as H
from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

OUT = "/Users/lacg/wnn/logs/ids/ids77"
db = sqlite3.connect(H.DB_URI, uri=True)
base = H.flow_params(db, 7591)
factory = IdsEvaluatorFactory(log=lambda m: None)
dataset = factory.load(H.settings_for(base))
val = H.val_as_report(dataset)
yv = np.asarray(dataset.y_val_multi); yb = np.asarray(dataset.y_val_binary); names = list(dataset.category_names)
FUZ = names.index("Fuzzers"); P = int((yb == 1).sum()); N = int((yb == 0).sum()); T = 1104
SEEDS = (1000, 1001, 1002, 1003, 1004); K = 5; B = 80; NPC = 250


def sub(seed, k):
	rng = np.random.default_rng(seed)
	c = rng.integers(0, T, 2 * NPC * B).reshape(2, NPC, B)
	s = c[:, k * (NPC // K):(k + 1) * (NPC // K), :]
	return ClusterGenome(bits_per_neuron=[B] * (2 * NPC // K), neurons_per_cluster=[NPC // K] * 2,
		connections=[int(x) for x in s.reshape(-1)])


def readout(sc):
	m = sc[:, 1] - sc[:, 0]
	o = np.argsort(-m, kind="stable"); ms = m[o]; ys = yb[o]
	tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0); last = np.r_[ms[1:] != ms[:-1], True]
	tp, fp = tp[last], fp[last]; fn = P - tp; tn = N - fp
	f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
	i = int(np.argmax(f1)); pred = m >= ms[last][i]
	return dict(f1=f1[i] * 100, fpr=fp[i] / N * 100, rec=tp[i] / P * 100, fuz=pred[yv == FUZ].mean() * 100, distinct=int(last.sum()))


THETAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.35, 0.5)
scores = {}
for th in THETAS:
	params = dict(base, ids_vote_share=th); H.apply_production_env(params); s = H.settings_for(params)
	opt, ev = factory.flat(val, s)
	for seed in SEEDS:
		for k in range(K):
			ex = ev.export_neuron_outputs(sub(seed, k), partition="val", packed=True)
			scores[(th, seed, k)] = np.asarray(ex.scores, dtype=np.float32)
	print("theta done", th, flush=True)
	del opt, ev; gc.collect()
LADDERS = {"ctrl_all_0.02": (0.02,) * 5, "all_0.05": (0.05,) * 5, "all_0.1": (0.1,) * 5, "all_0.5": (0.5,) * 5,
	"ladder_A_.02-.5": (0.02, 0.1, 0.2, 0.35, 0.5), "ladder_B_.02-.3": (0.02, 0.05, 0.1, 0.2, 0.3),
	"ladder_C_.02x3+.2+.5": (0.02, 0.02, 0.02, 0.2, 0.5)}
out = open(f"{OUT}/gate_ladder.jsonl", "w")
for name, ths in LADDERS.items():
	for seed in SEEDS:
		sc = np.mean([scores[(th, seed, k)] for k, th in enumerate(ths)], axis=0).astype(np.float64)
		rec = dict(variant=name, thetas=ths, seed=seed, **readout(sc))
		print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n")
