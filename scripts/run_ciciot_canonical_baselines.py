"""Run RF + XGBoost baselines on lacg030175/CIC-IoT-2023-canonical-neto (45M).

Mirrors run_all_baselines.py for ciciot, but on the new canonical-neto data
that preserves NaN/inf. Sklearn's RF doesn't handle NaN natively, so we
median-impute before fitting (XGBoost handles NaN natively → no impute).

Uses the SAME top-20 feature list as run_all_baselines.py so the comparison
to the WNN paper baseline (r98) is apples-to-apples on features.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

# Match the top-20 in run_all_baselines.py exactly (apples-to-apples vs r98).
TOP20_CICIOT = [
	"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
	"Rate", "IAT", "ack_count", "Header_Length", "Min",
	"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
	"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
]


def compute_metrics(y_true, y_pred):
	f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	cm = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = cm.ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	return {"f1": f1, "fpr": fpr, "acc": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_classifier(name, clf, X_train, y_train, X_eval, y_eval, attack_classes=None):
	print(f"\n  Training {name}...", flush=True)
	t0 = time.time()
	clf.fit(X_train, y_train)
	train_s = time.time() - t0
	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0
	m = compute_metrics(y_eval, y_pred)
	m["train_s"] = train_s
	m["infer_s"] = infer_s
	print(f"    F1:  {m['f1']*100:.2f}%  |  FPR: {m['fpr']*100:.2f}%  |  Acc: {m['acc']*100:.2f}%")
	print(f"    Confusion: TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,}")
	print(f"    Train: {train_s:.1f}s  |  Infer: {infer_s:.2f}s ({1e6*infer_s/len(X_eval):.1f} us/sample)")

	# Per-attack-class recall (detection rate for attack classes; false-positive rate for benign)
	if attack_classes is not None:
		print(f"    Per-class breakdown:")
		ac = np.asarray(attack_classes)
		per_class = {}
		for cls in sorted(set(ac.tolist())):
			mask = (ac == cls)
			n_total = int(mask.sum())
			if n_total == 0: continue
			n_predicted_attack = int(((y_pred == 1) & mask).sum())
			rate = n_predicted_attack / n_total
			per_class[cls] = {"count": n_total, "predicted_attack": n_predicted_attack, "rate": rate}
			label = "FPR" if cls == "Benign" else "recall"
			print(f"      {cls:<15s}: {label:6s} = {rate*100:>6.2f}%  ({n_predicted_attack:,}/{n_total:,})")
		m["per_class"] = per_class
	return m


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("CIC-IoT-2023 CANONICAL (Neto, 45M) — RF + XGBoost baselines")
	print("Top-20 features (same as r98), random_3way (test+val merged)")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-canonical-neto"
	print(f"\nLoading {repo} (random_3way)...")
	ds = load_dataset(repo, "random_3way")

	# Load splits
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas()
	df_eval = pd.concat([df_test, df_val], ignore_index=True)
	print(f"  Train: {len(df_train):,}  |  Eval (test+val): {len(df_eval):,}")

	# Top-20 feature filter
	features = [f for f in TOP20_CICIOT if f in df_train.columns]
	missing = [f for f in TOP20_CICIOT if f not in df_train.columns]
	print(f"  Features: {len(features)}/20 found{f' (missing: {missing})' if missing else ''}")

	X_train_raw = df_train[features].values.astype(np.float32)
	y_train = df_train["label"].values.astype(np.int32)
	X_eval_raw = df_eval[features].values.astype(np.float32)
	y_eval = df_eval["label"].values.astype(np.int32)
	# attack_class strings for per-class breakdown (Benign + 7 attack classes)
	eval_attack_classes = df_eval["attack_class"].values

	# Replace ±Inf with NaN so imputer can see them uniformly
	X_train_raw = np.where(np.isinf(X_train_raw), np.nan, X_train_raw)
	X_eval_raw = np.where(np.isinf(X_eval_raw), np.nan, X_eval_raw)

	n_nan_tr = np.isnan(X_train_raw).any(axis=1).sum()
	n_nan_ev = np.isnan(X_eval_raw).any(axis=1).sum()
	print(f"  NaN/Inf rows: train {n_nan_tr:,} ({n_nan_tr/len(X_train_raw)*100:.4f}%), "
		  f"eval {n_nan_ev:,} ({n_nan_ev/len(X_eval_raw)*100:.4f}%)")
	print(f"  Train: {X_train_raw.shape} ({y_train.sum():,} attack / {(y_train==0).sum():,} normal)")
	print(f"  Eval:  {X_eval_raw.shape} ({y_eval.sum():,} attack / {(y_eval==0).sum():,} normal)")

	results = {}

	# ── RF: requires median imputation (sklearn doesn't handle NaN natively) ──
	print("\n  Median-imputing NaN/Inf for RF (XGBoost handles NaN natively)...")
	imputer = SimpleImputer(strategy="median")
	X_train_imp = imputer.fit_transform(X_train_raw)
	X_eval_imp = imputer.transform(X_eval_raw)
	results["rf_raw"] = run_classifier(
		"RF (raw, median-imputed NaN)",
		RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
		X_train_imp, y_train, X_eval_imp, y_eval,
		attack_classes=eval_attack_classes)

	# ── XGBoost: native NaN handling, no impute ──
	results["xgb_raw"] = run_classifier(
		"XGBoost (raw, native NaN)",
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
					  n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0),
		X_train_raw, y_train, X_eval_raw, y_eval,
		attack_classes=eval_attack_classes)

	# Final summary
	print(f"\n{'='*78}")
	print(f"  CIC-IoT-2023 CANONICAL (Neto, 45M) — Summary")
	print(f"{'='*78}")
	print(f"  {'Model':<35s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s} {'Train':>7s}")
	print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
	for key, label in [("rf_raw", "RF (median-imputed NaN)"),
					   ("xgb_raw", "XGBoost (native NaN)")]:
		r = results[key]
		print(f"  {label:<35s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}% {r['train_s']:>6.1f}s")
	print()
	print("Compare: r98 ran on lacg030175/CIC-IoT-2023-full (38.5M, NaN dropped).")
	print("  Old baselines (38.5M): see scripts/run_all_baselines.py output.")
	print("  Old WNN r98 result:    F1 87.77% / FPR 7.52% / Acc 98.37%")

	# Per-class comparison table (for paper/analysis use)
	print(f"\n{'='*78}")
	print(f"  PER-CLASS BREAKDOWN — RF vs XGBoost")
	print(f"{'='*78}")
	classes_union = sorted(set(results["rf_raw"]["per_class"].keys()) |
						   set(results["xgb_raw"]["per_class"].keys()))
	# Benign first (FPR), then attack classes (recall)
	if "Benign" in classes_union:
		classes_union = ["Benign"] + [c for c in classes_union if c != "Benign"]
	print(f"  {'Class':<15s} {'Count':>10s}   {'RF rate':>8s}   {'XGB rate':>8s}   {'Type':>8s}")
	print(f"  {'-'*15} {'-'*10}   {'-'*8}   {'-'*8}   {'-'*8}")
	for cls in classes_union:
		rf_r = results["rf_raw"]["per_class"].get(cls, {}).get("rate", 0)
		xgb_r = results["xgb_raw"]["per_class"].get(cls, {}).get("rate", 0)
		count = results["rf_raw"]["per_class"].get(cls, {}).get("count", 0)
		typ = "FPR" if cls == "Benign" else "recall"
		print(f"  {cls:<15s} {count:>10,}   {rf_r*100:>7.2f}%   {xgb_r*100:>7.2f}%   {typ:>8s}")


if __name__ == "__main__":
	main()
