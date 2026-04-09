"""
Classical ML baselines for CIC-IoT-2023 (RF + XGBoost).

Runs on the same random 80/20 split and top-20 features used by
the WNN PUB50 batches:
  - RF and XGBoost on RAW numeric features (their natural format)
  - RF and XGBoost on 8-bit thermometer-encoded features (WNN's format)

Both use the merged test+val set (20%) for evaluation, matching the
WNN protocol exactly. Comparison baseline for RAID 2026 paper.

Usage:
    source wnn/bin/activate
    python scripts/run_ciciot_baselines.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))


def compute_metrics(y_true, y_pred):
	"""Compute F1-macro, FPR, and accuracy for binary classification."""
	f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	cm = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = cm.ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	return {"f1": f1, "fpr": fpr, "acc": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def load_raw_ciciot(split="random_3way", feature_selection="top20"):
	"""Load CIC-IoT-2023 raw numeric features (no thermometer encoding)."""
	from datasets import load_dataset

	print(f"Loading CIC-IoT-2023 ({split}) from HuggingFace...")
	ds = load_dataset("lacg030175/CIC-IoT-2023", split)

	df_train = ds["train"].to_pandas()

	# Top-20 features (same as WNN)
	TOP20 = [
		"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
		"Rate", "IAT", "ack_count", "Header_Length", "Min",
		"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
		"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
	]

	if feature_selection == "top20":
		features = [f for f in TOP20 if f in df_train.columns]
	else:
		exclude = {"Label", "label", "attack_class"}
		features = sorted(set(df_train.columns) - exclude)

	print(f"  Features: {len(features)} ({feature_selection})")

	X_train = df_train[features].values.astype(np.float32)
	y_train = df_train["label"].values.astype(np.int32)

	# Merge test + val into single 20% set (matching WNN protocol)
	if "validation" in ds:
		df_test = ds["test"].to_pandas()
		df_val = ds["validation"].to_pandas()
		df_eval = __import__("pandas").concat([df_test, df_val], ignore_index=True)
		print(f"  Merged test ({len(df_test):,}) + val ({len(df_val):,}) = {len(df_eval):,} eval samples")
	else:
		df_eval = ds["test"].to_pandas()
		print(f"  Test: {len(df_eval):,} samples")

	X_eval = df_eval[features].values.astype(np.float32)
	y_eval = df_eval["label"].values.astype(np.int32)

	print(f"  Train: {X_train.shape} ({y_train.sum():,} attack / {(y_train==0).sum():,} normal)")
	print(f"  Eval:  {X_eval.shape} ({y_eval.sum():,} attack / {(y_eval==0).sum():,} normal)")

	return X_train, y_train, X_eval, y_eval, features


def load_thermo_ciciot(n_bits=8):
	"""Load CIC-IoT-2023 thermometer-encoded (same as WNN)."""
	from wnn.ids.ciciot2023 import load_ciciot2023

	print(f"\nLoading CIC-IoT-2023 (thermometer {n_bits}-bit, top20)...")
	ds = load_ciciot2023(n_bits=n_bits, split="random_3way", feature_selection="top20")

	# Merge test+val
	if ds.X_val is not None:
		X_eval = np.concatenate([ds.X_test, ds.X_val])
		y_eval = np.concatenate([ds.y_test_binary, ds.y_val_binary])
	else:
		X_eval = ds.X_test
		y_eval = ds.y_test_binary

	print(f"  Train: {ds.X_train.shape} (thermo-encoded)")
	print(f"  Eval:  {X_eval.shape}")

	return ds.X_train, ds.y_train_binary, X_eval, y_eval


def run_classifier(name, clf, X_train, y_train, X_eval, y_eval):
	"""Train and evaluate a classifier, return metrics dict."""
	print(f"\n  Training {name}...")
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

	return m


def main():
	print("=" * 70)
	print("CIC-IoT-2023 Classical ML Baselines for RAID 2026")
	print("=" * 70)

	results = {}

	# === RAW FEATURES (RF + XGBoost natural format) ===
	print("\n" + "=" * 70)
	print("PART 1: Raw numeric features (top-20)")
	print("=" * 70)
	X_train_raw, y_train, X_eval_raw, y_eval, features = load_raw_ciciot()

	results["rf_raw"] = run_classifier(
		"Random Forest (raw)",
		RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
		X_train_raw, y_train, X_eval_raw, y_eval,
	)
	results["xgb_raw"] = run_classifier(
		"XGBoost (raw)",
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
			random_state=42, eval_metric="logloss", verbosity=0),
		X_train_raw, y_train, X_eval_raw, y_eval,
	)

	# === THERMOMETER-ENCODED FEATURES (same as WNN) ===
	print("\n" + "=" * 70)
	print("PART 2: Thermometer-encoded features (8-bit, top-20)")
	print("=" * 70)
	X_train_thermo, y_train_t, X_eval_thermo, y_eval_t = load_thermo_ciciot(n_bits=8)

	results["rf_thermo"] = run_classifier(
		"Random Forest (8b thermo)",
		RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
		X_train_thermo, y_train_t, X_eval_thermo, y_eval_t,
	)
	results["xgb_thermo"] = run_classifier(
		"XGBoost (8b thermo)",
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
			random_state=42, eval_metric="logloss", verbosity=0),
		X_train_thermo, y_train_t, X_eval_thermo, y_eval_t,
	)

	# === SUMMARY TABLE ===
	print("\n" + "=" * 70)
	print("SUMMARY — CIC-IoT-2023 Random 80/20 (Top-20 Features)")
	print("=" * 70)
	print(f"{'Model':<30s} {'F1':>8s} {'FPR':>8s} {'Acc':>8s} {'Train':>8s} {'Infer':>8s}")
	print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

	for key, label in [
		("rf_raw",     "RF (raw features)"),
		("xgb_raw",    "XGBoost (raw features)"),
		("rf_thermo",  "RF (8b thermometer)"),
		("xgb_thermo", "XGBoost (8b thermometer)"),
	]:
		r = results[key]
		print(f"{label:<30s} {r['f1']*100:>7.2f}% {r['fpr']*100:>7.2f}% {r['acc']*100:>7.2f}% {r['train_s']:>7.1f}s {r['infer_s']:>7.2f}s")

	# WNN comparison line (from PUB50 n=54 mean, best_fitness GA Neurons train_cal)
	print(f"{'WNN (PUB50 8b, n=54 mean)':<30s} {'80.07':>7s}% {'4.32':>7s}% {'86.85':>7s}%     {'—':>4s}     {'—':>4s}")
	print(f"{'WNN fixed_05 (n=54 mean)':<30s} {'81.61':>7s}% {'9.12':>7s}% {'88.56':>7s}%     {'—':>4s}     {'—':>4s}")

	print("\nNote: WNN uses train_cal (fitness-aligned) threshold by default.")
	print("      RF/XGBoost use predict() which is equivalent to fixed_05 (0.5 threshold).")
	print("      For fair comparison: XGBoost raw vs WNN fixed_05.")


if __name__ == "__main__":
	main()
