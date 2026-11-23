"""IDS-77: RF/XGB raw top-20 (and RF on cut-E bits) per-attack-category recall on VAL, same train partition.
Protocol = logs/ids/unsw_headroom_1611/seed_baselines.py (RF 100 trees depth None; XGB 100/6/0.1; VAL F1-optimal
threshold on linspace(0.01,0.99,99)). TEST is not touched. Also saves VAL probabilities + RF feature importances."""
import sys, json, time
sys.path.insert(0, "/Users/lacg/wnn/scripts")
import numpy as np
import verify_ids_baselines as V
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

OUT = "/Users/lacg/wnn/logs/ids/ids77"
lab = np.load(f"{OUT}/val_labels.npz"); meta = json.load(open(f"{OUT}/meta.json"))
names = meta["names"]; yv = lab["y_multi"]; ybv = lab["y_bin"]
Xtr, ytr, Xte, yte, Xva, yva = V._load_raw("unsw", "random_3way")
Xtr, Xte, Xva = V._impute(Xtr, Xte, Xva)
assert np.array_equal(np.asarray(yva), ybv), "VAL row order differs from the worker dataset"
assert np.array_equal(np.asarray(ytr), lab["y_train_bin"]), "TRAIN row order differs"
del Xte, yte
feats = [f for f in V.TOP20["unsw"]]
out = open(f"{OUT}/rf_val_cat.jsonl", "a")


def report(tag, seed, pv, extra=None):
	thr = V._val_threshold(ybv, pv)
	f1, fpr, acc, cm = V._score(ybv, pv, thr)
	pred = pv >= thr
	cat = {names[c]: [int(pred[yv == c].sum()), int((yv == c).sum())] for c in np.unique(yv)}
	rec = dict(model=tag, seed=seed, thr=thr, val_f1=f1 * 100, val_fpr=fpr * 100, val_acc=acc * 100,
		val_cm=[int(c) for c in cm], cat=cat, **(extra or {}))
	print(json.dumps({k: rec[k] for k in ("model", "seed", "thr", "val_f1", "val_fpr", "val_cm")}), flush=True)
	out.write(json.dumps(rec) + "\n"); out.flush()
	np.save(f"{OUT}/pv_{tag}_{seed}.npy", pv.astype(np.float32))


for seed in (20421, 20422, 20423):
	t0 = time.time(); m = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=3, random_state=seed).fit(Xtr, ytr)
	report("rf_raw", seed, m.predict_proba(Xva)[:, 1], dict(importances=dict(zip(feats, map(float, m.feature_importances_))),
		fit_s=round(time.time() - t0, 1)))
	del m
m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=3, random_state=20421, eval_metric="logloss",
	verbosity=0).fit(Xtr, ytr)
report("xgb_raw", 20421, m.predict_proba(Xva)[:, 1], dict(gain=m.get_booster().get_score(importance_type="total_gain")))
del m, Xtr, Xva
z = np.load("/tmp/encB/E.npz"); total = json.load(open("/tmp/encB/E.json"))["total_bits"]
Etr = np.unpackbits(z["train"], axis=1, count=total, bitorder="little").astype(bool)
Eva = np.unpackbits(z["val"], axis=1, count=total, bitorder="little").astype(bool)
m = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=3, random_state=20421).fit(Etr, ytr)
report("rf_E", 20421, m.predict_proba(Eva)[:, 1])
