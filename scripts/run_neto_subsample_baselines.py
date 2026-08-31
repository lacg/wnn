"""All baselines on lacg030175/CIC-IoT-2023-neto-subsample (1.43M, 46 feat).

Runs RF + XGBoost + AdaBoost (top-20) + AdaBoost (all 46) + Perceptron.
Top-20 features for RF/XGBoost/Perceptron (matches WNN's input subset).
AdaBoost both ways for the paper's transparency comparison.
random_3way (test+val merged). CPU-only.
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

TOP20_CICIOT = [
	"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
	"Rate", "IAT", "ack_count", "Header_Length", "Min",
	"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
	"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
]


def metrics(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return {
		"f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
		"acc": accuracy_score(y_true, y_pred),
		"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
	}


def per_class(y_pred, attack_classes):
	ac = np.asarray(attack_classes)
	out = {}
	for cls in sorted(set(ac.tolist())):
		mask = (ac == cls)
		n = int(mask.sum())
		if n == 0: continue
		n_pred = int(((y_pred == 1) & mask).sum())
		out[cls] = {"count": n, "predicted_attack": n_pred, "rate": n_pred / n}
	return out


def run_clf(name, clf, X_tr, y_tr, X_ev, y_ev, attack_classes):
	print(f"\n  Training {name}...", flush=True)
	t0 = time.time()
	clf.fit(X_tr, y_tr)
	train_s = time.time() - t0
	t0 = time.time()
	y_pred = clf.predict(X_ev)
	infer_s = time.time() - t0
	m = metrics(y_ev, y_pred)
	m["train_s"] = train_s; m["infer_s"] = infer_s
	m["per_class"] = per_class(y_pred, attack_classes)
	print(f"    F1: {m['f1']*100:.2f}%   FPR: {m['fpr']*100:.2f}%   Acc: {m['acc']*100:.2f}%   "
		  f"(train {train_s:.1f}s, infer {infer_s:.2f}s)")
	return m


def main():
	from datasets import load_dataset
	repo = "lacg030175/CIC-IoT-2023-neto-subsample"
	print(f"Loading {repo}...", flush=True)
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	# WHICH PARTITION IS THE EVAL SET (added 31/08/2026).
	# The historical default MERGES test+validation into one 20% eval set, which is
	# a 2-way evaluation carried out on a 3-way dataset. Protocol v2 (worker ABI 3,
	# 11/07/2026) does NOT merge them: on a _3way dataset the WNN calibrates its
	# threshold modes on VAL and reports on the 10% TEST partition alone. So a
	# baseline meant to sit beside a Protocol-v2 WNN number has to be scored on
	# TEST ONLY, or the two are measured on different sets.
	# Default stays "merged" so every previously banked baseline reproduces
	# bit-for-bit; pass --eval-split test for the Protocol-v2-comparable number.
	eval_split = "merged"
	if "--eval-split" in sys.argv:
		eval_split = sys.argv[sys.argv.index("--eval-split") + 1]
	if eval_split == "merged":
		df_eval = pd.concat([ds["test"].to_pandas(), ds["validation"].to_pandas()], ignore_index=True)
	elif eval_split in ("test", "validation"):
		df_eval = ds[eval_split].to_pandas()
	else:
		raise SystemExit(f"--eval-split must be test|validation|merged, got {eval_split!r}")
	print(f"  Train: {len(df_train):,}  Eval: {len(df_eval):,}  (eval-split={eval_split})")

	non_features = {"Label", "Label_orig", "label", "attack_class"}
	all_features = [c for c in df_train.columns if c not in non_features]
	top20 = [f for f in TOP20_CICIOT if f in df_train.columns]
	print(f"  All features: {len(all_features)}, Top-20: {len(top20)}")

	# Top-20 feature matrices (for RF/XGB/Perceptron + AdaBoost variant 1)
	X_tr_t20 = df_train[top20].values.astype(np.float32)
	X_ev_t20 = df_eval[top20].values.astype(np.float32)
	# All-46 feature matrices (for AdaBoost variant 2)
	X_tr_all = df_train[all_features].values.astype(np.float32)
	X_ev_all = df_eval[all_features].values.astype(np.float32)
	y_tr = df_train["label"].values.astype(np.int32)
	y_ev = df_eval["label"].values.astype(np.int32)
	attack_classes = df_eval["attack_class"].values

	# Replace inf with NaN
	for X in (X_tr_t20, X_ev_t20, X_tr_all, X_ev_all):
		X[np.isinf(X)] = np.nan

	# Median impute (RF, AdaBoost, Perceptron need this; XGBoost handles NaN natively)
	imp_t20 = SimpleImputer(strategy="median")
	X_tr_t20_imp = imp_t20.fit_transform(X_tr_t20)
	X_ev_t20_imp = imp_t20.transform(X_ev_t20)
	imp_all = SimpleImputer(strategy="median")
	X_tr_all_imp = imp_all.fit_transform(X_tr_all)
	X_ev_all_imp = imp_all.transform(X_ev_all)

	# Perceptron also benefits from scaling
	scaler = StandardScaler()
	X_tr_t20_scaled = scaler.fit_transform(X_tr_t20_imp)
	X_ev_t20_scaled = scaler.transform(X_ev_t20_imp)

	results = {}

	# 1. RF (top-20)
	results["rf"] = run_clf(
		"RF (top-20, NaN-imputed)",
		RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
		X_tr_t20_imp, y_tr, X_ev_t20_imp, y_ev, attack_classes)

	# 2. XGBoost (top-20, native NaN)
	results["xgb"] = run_clf(
		"XGBoost (top-20, native NaN)",
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
					  n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0),
		X_tr_t20, y_tr, X_ev_t20, y_ev, attack_classes)

	# 3. AdaBoost (top-20)
	results["ada_t20"] = run_clf(
		"AdaBoost (top-20, NaN-imputed)",
		AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
		X_tr_t20_imp, y_tr, X_ev_t20_imp, y_ev, attack_classes)

	# 4. AdaBoost (all 46)
	results["ada_all"] = run_clf(
		"AdaBoost (all 46, NaN-imputed)",
		AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
		X_tr_all_imp, y_tr, X_ev_all_imp, y_ev, attack_classes)

	# 5. Perceptron (top-20, scaled)
	results["perceptron"] = run_clf(
		"Perceptron (top-20, scaled, NaN-imputed)",
		Perceptron(random_state=42, n_jobs=-1),
		X_tr_t20_scaled, y_tr, X_ev_t20_scaled, y_ev, attack_classes)

	# Summary
	print(f"\n{'='*78}")
	print(f"  CIC-IoT-2023 NETO-SUBSAMPLE (1.43M, 46 feat) — Summary")
	print(f"{'='*78}")
	print(f"  {'Model':<32s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s}")
	print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
	for k, lbl in [("rf", "Random Forest (top-20)"),
				   ("xgb", "XGBoost (top-20)"),
				   ("ada_t20", "AdaBoost (top-20)"),
				   ("ada_all", "AdaBoost (all 46)"),
				   ("perceptron", "Perceptron (top-20)")]:
		r = results[k]
		print(f"  {lbl:<32s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}%")

	print(f"\n{'='*78}")
	print(f"  PER-CLASS BREAKDOWN (Detection rate; Benign row is FPR)")
	print(f"{'='*78}")
	classes = sorted(set().union(*(r["per_class"].keys() for r in results.values())))
	if "Benign" in classes:
		classes = ["Benign"] + [c for c in classes if c != "Benign"]
	hdr = f"  {'Class':<15s} {'Count':>10s}"
	for k in ("rf", "xgb", "ada_t20", "ada_all", "perceptron"):
		hdr += f"  {k:>10s}"
	print(hdr)
	for cls in classes:
		row = f"  {cls:<15s}"
		any_r = next(iter(results.values()))
		count = any_r["per_class"].get(cls, {}).get("count", 0)
		row += f" {count:>10,}"
		for k in ("rf", "xgb", "ada_t20", "ada_all", "perceptron"):
			rate = results[k]["per_class"].get(cls, {}).get("rate", 0)
			row += f"  {rate*100:>9.2f}%"
		print(row)


if __name__ == "__main__":
	main()
