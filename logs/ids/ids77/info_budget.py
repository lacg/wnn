"""IDS-77 Q3: TRAIN-only information each numeric feature keeps under the cut-E cuts vs a 2x / 4x E budget vs raw.
I(bin; attack_cat) and I(bin; binary label), bits. Uses the pre-check's cut code (logs/ids/encoding_B/cuts.py)."""
import sys, json
sys.path.insert(0, "/Users/lacg/wnn/scripts"); sys.path.insert(0, "/Users/lacg/wnn/logs/ids/encoding_B")
import numpy as np
import cuts as K
import verify_ids_baselines as V
OUT = "/Users/lacg/wnn/logs/ids/ids77"
lab = np.load(f"{OUT}/val_labels.npz"); ym = lab["y_train_multi"]; yb = lab["y_train_bin"]
Xtr, ytr, Xte, yte, Xva, yva = V._load_raw("unsw", "random_3way"); del Xte, Xva
assert np.array_equal(np.asarray(ytr), yb)
feats = list(V.TOP20["unsw"]); E = json.load(open("/Users/lacg/wnn/logs/ids/encoding_B/cuts_E.json"))
cat = {"proto", "service", "state"}
num = [f for f in feats if f not in cat]
def mi(u, counts, cuts):
	b = np.searchsorted(np.asarray(cuts), u, side="right") if len(cuts) else np.zeros(len(u), int)
	tab = np.zeros((b.max() + 1, counts.shape[1])); np.add.at(tab, b, counts)
	n = tab.sum(); py = tab.sum(0) / n; px = tab.sum(1) / n
	hy = -(py[py > 0] * np.log2(py[py > 0])).sum()
	hyx = (px * K.entropy_rows(tab)).sum()
	return hy - hyx
tabs = {f: K.class_table(Xtr[:, feats.index(f)].astype(np.float64), ym, int(ym.max()) + 1) for f in num}
tabsb = {f: K.class_table(Xtr[:, feats.index(f)].astype(np.float64), yb, 2) for f in num}
budget_num = sum(len(E[f]) for f in num)
big = {m: K.budget_cuts(tabs, m * budget_num) for m in (2, 4)}
print(f"numeric bits in E: {budget_num}; 2x/4x budgets reallocated over the same 17 numeric features (TRAIN only)")
print(f"{'feature':18s} uniq   E-bits  I_E/I_raw(cat)  2x-bits I_2x/raw  4x-bits I_4x/raw | I_E/I_raw(bin)  I_raw(cat) bits")
rows = []
for f in num:
	u, c = tabs[f]; ub, cb = tabsb[f]
	raw = mi(u, c, 0.5 * (u[1:] + u[:-1])); rawb = mi(ub, cb, 0.5 * (ub[1:] + ub[:-1]))
	e = mi(u, c, E[f]); eb = mi(ub, cb, E[f]); b2 = mi(u, c, big[2][f]); b4 = mi(u, c, big[4][f])
	rows.append((f, raw))
	print(f"{f:18s} {len(u):6d} {len(E[f]):6d}  {e/raw*100:12.2f}%  {len(big[2][f]):6d} {b2/raw*100:7.2f}%  {len(big[4][f]):6d} {b4/raw*100:7.2f}% | {eb/rawb*100:12.2f}%  {raw:.4f}")
