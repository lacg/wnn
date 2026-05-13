"""RF + XGBoost baselines on lacg030175/CIC-IoT-2023-neto-full (46.7M, 46 features).

Same protocol as run_ciciot_canonical_baselines.py: top-20 features (apples-to-apples
with r98), random_3way (test+val merged), per-class breakdown.

CPU-only — safe to run alongside the worker.
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

# Canonical TOP20 RF importance on neto-full (re-derived 13/05/2026 after the
# bencorn-fabricated Time_To_Live was identified as a missing-feature bug).
from wnn.ids.ciciot2023 import TOP20_RF_FEATURES as TOP20_CICIOT


def compute_metrics(y_true, y_pred):
	f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	return {"f1": f1, "fpr": fpr, "acc": acc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_classifier(name, clf, X_train, y_train, X_eval, y_eval, attack_classes):
	print(f"\n  Training {name}...", flush=True)
	t0 = time.time()
	clf.fit(X_train, y_train)
	train_s = time.time() - t0
	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0
	m = compute_metrics(y_eval, y_pred)
	m["train_s"] = train_s; m["infer_s"] = infer_s
	print(f"    F1: {m['f1']*100:.2f}%  FPR: {m['fpr']*100:.2f}%  Acc: {m['acc']*100:.2f}%")
	print(f"    Train: {train_s:.1f}s  Infer: {infer_s:.2f}s")

	# Per-class breakdown
	print(f"    Per-class:")
	ac = np.asarray(attack_classes)
	per_class = {}
	for cls in sorted(set(ac.tolist())):
		mask = (ac == cls)
		n = int(mask.sum())
		if n == 0: continue
		n_pred_attack = int(((y_pred == 1) & mask).sum())
		rate = n_pred_attack / n
		per_class[cls] = {"count": n, "predicted_attack": n_pred_attack, "rate": rate}
		typ = "FPR" if cls == "Benign" else "recall"
		print(f"      {cls:<15s}: {typ} = {rate*100:>6.2f}%  ({n_pred_attack:,}/{n:,})")
	m["per_class"] = per_class
	return m


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("CIC-IoT-2023 NETO-FULL (46.7M, 46 features) — RF + XGBoost baselines")
	print("Top-20 features (same as r98), random_3way (test+val merged)")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"\nLoading {repo} (random_3way)...", flush=True)
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas()
	df_eval = pd.concat([df_test, df_val], ignore_index=True)
	print(f"  Train: {len(df_train):,}  Eval (test+val): {len(df_eval):,}")

	features = [f for f in TOP20_CICIOT if f in df_train.columns]
	missing = [f for f in TOP20_CICIOT if f not in df_train.columns]
	print(f"  Features: {len(features)}/20 found{f' (missing: {missing})' if missing else ''}")

	X_train_raw = df_train[features].values.astype(np.float32)
	y_train = df_train["label"].values.astype(np.int32)
	X_eval_raw = df_eval[features].values.astype(np.float32)
	y_eval = df_eval["label"].values.astype(np.int32)
	eval_attack_classes = df_eval["attack_class"].values

	X_train_raw = np.where(np.isinf(X_train_raw), np.nan, X_train_raw)
	X_eval_raw = np.where(np.isinf(X_eval_raw), np.nan, X_eval_raw)
	n_nan_tr = int(np.isnan(X_train_raw).any(axis=1).sum())
	n_nan_ev = int(np.isnan(X_eval_raw).any(axis=1).sum())
	print(f"  NaN/Inf: train {n_nan_tr:,} ({n_nan_tr/len(X_train_raw)*100:.4f}%), "
		  f"eval {n_nan_ev:,} ({n_nan_ev/len(X_eval_raw)*100:.4f}%)")

	results = {}

	print("\n  Median-imputing NaN/Inf for RF (XGBoost handles natively)...")
	imputer = SimpleImputer(strategy="median")
	X_train_imp = imputer.fit_transform(X_train_raw)
	X_eval_imp = imputer.transform(X_eval_raw)
	results["RF"] = run_classifier(
		"RF (raw, median-imputed NaN)",
		RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
		X_train_imp, y_train, X_eval_imp, y_eval, eval_attack_classes)
	results["XGB"] = run_classifier(
		"XGBoost (raw, native NaN)",
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
					  n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0),
		X_train_raw, y_train, X_eval_raw, y_eval, eval_attack_classes)

	# Summary
	print(f"\n{'='*78}")
	print(f"  CIC-IoT-2023 NETO-FULL (46.7M, 46 features) — Summary")
	print(f"{'='*78}")
	print(f"  {'Model':<25s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s} {'Train':>7s}")
	print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
	for k, lbl in [("RF", "RF"), ("XGB", "XGBoost")]:
		r = results[k]
		print(f"  {lbl:<25s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}% {r['train_s']:>6.1f}s")

	print(f"\n{'='*78}")
	print(f"  PER-CLASS BREAKDOWN — RF vs XGBoost (NETO-FULL, 46.7M)")
	print(f"{'='*78}")
	classes = sorted(set(results["RF"]["per_class"].keys()) | set(results["XGB"]["per_class"].keys()))
	if "Benign" in classes:
		classes = ["Benign"] + [c for c in classes if c != "Benign"]
	print(f"  {'Class':<15s} {'Count':>10s}   {'RF rate':>8s}   {'XGB rate':>8s}   {'Type':>8s}")
	print(f"  {'-'*15} {'-'*10}   {'-'*8}   {'-'*8}   {'-'*8}")
	for cls in classes:
		rf_r = results["RF"]["per_class"].get(cls, {}).get("rate", 0)
		xgb_r = results["XGB"]["per_class"].get(cls, {}).get("rate", 0)
		count = results["RF"]["per_class"].get(cls, {}).get("count", 0)
		typ = "FPR" if cls == "Benign" else "recall"
		print(f"  {cls:<15s} {count:>10,}   {rf_r*100:>7.2f}%   {xgb_r*100:>7.2f}%   {typ:>8s}")


if __name__ == "__main__":
	main()
