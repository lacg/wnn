"""IDS-57 arm F gate, step 2 (VAL only; tau fitted on VAL and scored on VAL = a tuning read, differences only).
Random unsearched genomes, uniform random connections, the A1 gate's draws (seeds 1000..1004): the benign cluster
(250 x 34) is the same neurons in every variant that keeps it. One process, 4 threads.
usage: armf_gate.py [variant ...]   (default: all)"""
import time
from armf_common import *
from wnn.ram.experiments.tier_sizing import allocate_neurons
s, ds = load()
T = int(ds.X_train.total_bits); cats = list(ds.category_names)
yv_bin = np.asarray(ds.y_val_binary); yv_cat = np.asarray(ds.y_val_multi)
train_n = {c: int((np.asarray(ds.y_train_multi) == i).sum()) for i, c in enumerate(cats)}
MAJOR = ["Fuzzers", "Exploits", "Reconnaissance", "DoS", "Generic"]; RARE = ["Analysis", "Backdoor", "Shellcode", "Worms"]
ATT = [c for c in cats if c != "Normal"]
MAPS = {
	"two": None,
	"fz": [["Normal"], ["Fuzzers"], [c for c in ATT if c != "Fuzzers"]],
	"top3": [["Normal"], ["Fuzzers"], ["Exploits"], ["Reconnaissance"], [c for c in ATT if c not in ("Fuzzers", "Exploits", "Reconnaissance")]],
	"maj5": [["Normal"]] + [[c] for c in MAJOR] + [RARE],
	"all9": [["Normal"]] + [[c] for c in MAJOR] + [[c] for c in RARE],
}
def by_support(groups, cap=250, floor=10):
	return allocate_neurons([sum(train_n[c] for c in g) for g in groups[1:]], cap, floor)
def spec_for(groups, neurons, bits_of):
	return [(250, 34)] + [(n, bits_of(g)) for n, g in zip(neurons, groups[1:])]
fz64 = lambda g: 64 if g == ["Fuzzers"] else 34
def tier_bits(g):  # Luiz's "easy categories 16-34 bits": Fuzzers 64, hard majors 34, Generic 24, rare 20
	if g == ["Fuzzers"]: return 64
	if g == ["Generic"]: return 24
	if all(c in RARE for c in g): return 20
	return 34
V = {  # name -> (map, spec)
	"a_b34": ("two", [(250, 34), (250, 34)]),
	"b_A1_b64": ("two", [(250, 64), (250, 64)]),
	"m_benign64": ("two", [(250, 64), (250, 34)]),
	"m_attack64": ("two", [(250, 34), (250, 64)]),
	"fz_250_64": ("fz", [(250, 34), (250, 64), (250, 34)]),
	"fz_125_64": ("fz", [(250, 34), (125, 64), (125, 34)]),
	"fz_250_34": ("fz", [(250, 34), (250, 34), (250, 34)]),
	"fz_250_64_ben64": ("fz", [(250, 64), (250, 64), (250, 34)]),
	"top3_sup": ("top3", spec_for(MAPS["top3"], by_support(MAPS["top3"]), fz64)),
	"top3_125": ("top3", spec_for(MAPS["top3"], [125] * 4, fz64)),
	"maj5_sup": ("maj5", spec_for(MAPS["maj5"], by_support(MAPS["maj5"]), fz64)),
	"all9_sup": ("all9", spec_for(MAPS["all9"], by_support(MAPS["all9"]), fz64)),
	"maj5_60": ("maj5", spec_for(MAPS["maj5"], [60] * 6, fz64)),
	"all9_60": ("all9", spec_for(MAPS["all9"], [60] * 9, fz64)),
	"all9_sup_tierbits": ("all9", spec_for(MAPS["all9"], by_support(MAPS["all9"]), tier_bits)),
	# follow-ups after the first read (bits pay on the BENIGN side): category structure at A1's widths,
	# and cheaper benign clusters. Not arm F as proposed; run to size the alternative.
	"fz_all64": ("fz", [(250, 64), (250, 64), (250, 64)]),
	"fz_125_all64": ("fz", [(250, 64), (125, 64), (125, 64)]),
	"all9_sup_all64": ("all9", [(250, 64)] + [(n, 64) for n in by_support(MAPS["all9"])]),
	"s_b125x64_a250x64": ("two", [(125, 64), (250, 64)]),
	"s_b250x64_a125x64": ("two", [(250, 64), (125, 64)]),
	"s_b250x48_a250x64": ("two", [(250, 48), (250, 64)]),
	"s_b250x48_a250x48": ("two", [(250, 48), (250, 48)]),
}
want = sys.argv[1:] or list(V)
EVS = {}
def ev_for(mapname):
	if mapname not in EVS:
		if MAPS[mapname] is None: EVS[mapname] = (evaluator(ds, s, 2), ds, ["Normal", "attack"])
		else:
			dk, names = relabel(ds, MAPS[mapname]); EVS[mapname] = (evaluator(dk, s, len(names)), dk, names)
	return EVS[mapname]
def zcols(sc):
	mu = sc.mean(axis=0); sd = sc.std(axis=0); return (sc - mu) / np.maximum(sd, 1e-6), sd
for name in want:
	mapname, spec = V[name]; ev, dk, names = ev_for(mapname); k = len(names)
	emit(part="variant", name=name, map=mapname, clusters=names, spec=spec, class_weights=ev._streaming_class_weights)
	for seed in range(5):
		t0 = time.time(); g = draw_genome(spec, T, 1000 + seed); sc, fp = streamer_scores(ev, dk, g)
		ben = sc[:, 0]; att = sc[:, 1:]; nn = np.array([n for n, _ in spec[1:]], dtype=np.float64)
		rules = {"max": att.max(axis=1), "sum": att.sum(axis=1), "pooled": (att * nn).sum(axis=1) / nn.sum()}
		z, sd = zcols(sc); out = {r: binary_read(ben, a, yv_bin) for r, a in rules.items()}
		out["zmax"] = binary_read(z[:, 0], z[:, 1:].max(axis=1), yv_bin)
		# at the 'max' rule's tau: who is caught, and which attack cluster takes the benign rows
		tau = out["max"]["tau"]; flag = (rules["max"] - ben) >= tau; win = att.argmax(axis=1)
		rec = {c: float(flag[yv_cat == i].mean()) for i, c in enumerate(cats) if c != "Normal"}
		fp_by = {names[j + 1]: int((flag & (yv_bin == 0) & (win == j)).sum()) for j in range(k - 1)}
		tp_by = {names[j + 1]: int((flag & (yv_bin == 1) & (win == j)).sum()) for j in range(k - 1)}
		# ROC of the max-rule margin at exact false-alarm budgets. Margins are discrete (votes / neurons), so the
		# curve is interpolated between adjacent thresholds (the unsearched 34-bit baseline calibrates to the
		# high-recall point; recall at each variant's own tau is not comparable).
		marg = rules["max"] - ben; thr = np.unique(marg)[::-1]; isb = yv_bin == 0; isf = yv_cat == cats.index("Fuzzers")
		srt = lambda m: np.sort(marg[m]); cnt = lambda sv: (len(sv) - np.searchsorted(sv, thr, side="left")) / len(sv)
		rf, rt, rz = cnt(srt(isb)), cnt(srt(~isb)), cnt(srt(isf))
		at = {str(f): {"tpr": float(np.interp(f, rf, rt)), "fuzzers": float(np.interp(f, rf, rz))} for f in (0.002, 0.003, 0.005, 0.009)}
		emit(part="val", name=name, seed=seed, rules=out, recall=rec, benign_rows_won=fp_by, attack_rows_won=tp_by, at_fpr=at,
			min_sigma=float(sd.min()), sigma={names[j]: float(sd[j]) for j in range(k)},
			neurons=int(sum(n for n, _ in spec)), cells_logical=int(fp[2]), cells_physical=int(fp[3]), secs=round(time.time() - t0, 1))
