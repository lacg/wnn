"""Verify RF + XGBoost baselines on UNSW-NB15 temporal binary classification.

Current paper-plan numbers (RF/XGB at ~87% F1 / ~12% FPR on temporal) are
estimates labeled "Zoghi 2024" but Zoghi & Serpen 2024 reports a DNN baseline,
not RF/XGB. We need measured numbers to make a defensible "WNN beats RF/XGB"
claim in the paper.

This script:
  1. Loads UNSW-NB15 temporal binary with the SAME top-20 features our WNN uses
     (via load_unsw_nb15(split="temporal", feature_selection="top20", raw=True)).
  2. Trains RF (100 estimators, max_depth=None) and XGBoost (100 estimators,
     max_depth=6) with random_state=42 (deterministic).
  3. Reports F1-macro and FPR (binary "Attack" class FPR) on the temporal
     held-out test set — directly comparable to our XDS-unsw-temporal cohort
     results (which also score on the temporal held-out set via val_cal).
  4. Also reports model size (pickled bytes) for the "WNN < 1 KB vs RF 50 MB"
     framing.

Usage:
  python3 scripts/verify_unsw_temporal_baselines.py
  # ~3-5 min total (1-2 min RF train, 1-2 min XGBoost train, fast inference)
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from xgboost import XGBClassifier

from wnn.ids.dataset import load_unsw_nb15


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n-bits", type=int, default=8,
	                help="Thermometer bits per feature (default 8; pass 16 to match our WNN 16b-Wb cohort).")
	args = ap.parse_args()

	print("=" * 78)
	print(f"  UNSW-NB15 temporal binary — RF + XGBoost verification ({args.n_bits}-bit thermo)")
	print("=" * 78)

	# The HF UNSW-NB15 is pre-encoded; only thermometer variants are available
	# (no raw numerical). Use --n-bits 16 to match our WNN 16b-Wb cohort.
	N_BITS = args.n_bits
	print(f"\nLoading UNSW-NB15 temporal split (top-20 features, {N_BITS}-bit thermo)...")
	t0 = time.time()
	ds = load_unsw_nb15(split="temporal", feature_selection="top20", n_bits=N_BITS)
	print(f"  Loaded in {time.time() - t0:.1f}s")
	X_train = ds.X_train.to_numpy_bool()
	X_test = ds.X_test.to_numpy_bool()
	print(f"  Train shape: {X_train.shape}")
	print(f"  Test shape:  {X_test.shape}")
	print(f"  Train class balance: "
	      f"Normal={np.mean(ds.y_train_binary == 0):.1%}, "
	      f"Attack={np.mean(ds.y_train_binary == 1):.1%}")
	print(f"  Test  class balance: "
	      f"Normal={np.mean(ds.y_test_binary == 0):.1%}, "
	      f"Attack={np.mean(ds.y_test_binary == 1):.1%}")
	y_train = ds.y_train_binary
	y_test  = ds.y_test_binary

	results = []

	# ── Random Forest ──────────────────────────────────────────────────
	print("\n" + "-" * 78)
	print("  Random Forest (100 estimators, max_depth=None)")
	print("-" * 78)
	rf = RandomForestClassifier(
		n_estimators=100, max_depth=None, n_jobs=-1, random_state=42,
	)
	t0 = time.time()
	rf.fit(X_train, y_train)
	rf_train_s = time.time() - t0

	t0 = time.time()
	y_pred = rf.predict(X_test)
	rf_infer_s = time.time() - t0

	cm = confusion_matrix(y_test, y_pred)
	tn, fp, fn, tp = cm.ravel()
	rf_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	rf_acc = accuracy_score(y_test, y_pred)
	rf_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
	rf_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
	rf_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
	rf_size = len(pickle.dumps(rf))

	print(f"  Train time:  {rf_train_s:.1f}s")
	print(f"  Infer time:  {rf_infer_s:.3f}s  ({1e6 * rf_infer_s / len(X_test):.1f} µs/sample)")
	print(f"  Model size:  {rf_size:,} bytes ({rf_size / 1024 / 1024:.2f} MB)")
	print(f"  Confusion:   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
	print(f"  F1-macro:    {rf_f1:.4f}  ({100 * rf_f1:.2f}%)")
	print(f"  FPR:         {rf_fpr:.4f}  ({100 * rf_fpr:.2f}%)")
	print(f"  Accuracy:    {rf_acc:.4f}  ({100 * rf_acc:.2f}%)")
	print(f"  Precision:   {rf_prec:.4f}")
	print(f"  Recall:      {rf_rec:.4f}")
	results.append(("RF", rf_f1, rf_fpr, rf_acc, rf_size))

	# ── XGBoost ────────────────────────────────────────────────────────
	print("\n" + "-" * 78)
	print("  XGBoost (100 estimators, max_depth=6, lr=0.1)")
	print("-" * 78)
	xgb = XGBClassifier(
		n_estimators=100, max_depth=6, learning_rate=0.1,
		n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0,
	)
	t0 = time.time()
	xgb.fit(X_train, y_train)
	xgb_train_s = time.time() - t0

	t0 = time.time()
	y_pred = xgb.predict(X_test)
	xgb_infer_s = time.time() - t0

	cm = confusion_matrix(y_test, y_pred)
	tn, fp, fn, tp = cm.ravel()
	xgb_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
	xgb_acc = accuracy_score(y_test, y_pred)
	xgb_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
	xgb_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
	xgb_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
	xgb_size = len(pickle.dumps(xgb))

	print(f"  Train time:  {xgb_train_s:.1f}s")
	print(f"  Infer time:  {xgb_infer_s:.3f}s  ({1e6 * xgb_infer_s / len(X_test):.1f} µs/sample)")
	print(f"  Model size:  {xgb_size:,} bytes ({xgb_size / 1024 / 1024:.2f} MB)")
	print(f"  Confusion:   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
	print(f"  F1-macro:    {xgb_f1:.4f}  ({100 * xgb_f1:.2f}%)")
	print(f"  FPR:         {xgb_fpr:.4f}  ({100 * xgb_fpr:.2f}%)")
	print(f"  Accuracy:    {xgb_acc:.4f}  ({100 * xgb_acc:.2f}%)")
	print(f"  Precision:   {xgb_prec:.4f}")
	print(f"  Recall:      {xgb_rec:.4f}")
	results.append(("XGBoost", xgb_f1, xgb_fpr, xgb_acc, xgb_size))

	# ── Comparison to our WNN ──────────────────────────────────────────
	print("\n" + "=" * 78)
	print("  Final comparison vs XDS-unsw-temporal-16b-Wb WNN cohort")
	print("=" * 78)
	print(f"  {'Method':<14}{'F1':>10}{'FPR':>10}{'Acc':>10}{'Size':>16}")
	print(f"  {'-' * 60}")
	for name, f1, fpr, acc, size in results:
		size_mb = size / 1024 / 1024
		print(f"  {name:<14}{f1 * 100:>9.2f}%{fpr * 100:>9.2f}%{acc * 100:>9.2f}%   {size_mb:>9.2f} MB")
	print(f"  {'WNN (16b-Wb)':<14}{'88.94%':>10}{'7.86%':>10}{'88.98%':>10}{'~0.001 MB':>16}")
	print(f"\n  WNN deltas vs RF:    F1 {(88.94 - results[0][1]*100):+.2f}pp  "
	      f"FPR {(7.86 - results[0][2]*100):+.2f}pp")
	if len(results) > 1:
		print(f"  WNN deltas vs XGB:   F1 {(88.94 - results[1][1]*100):+.2f}pp  "
		      f"FPR {(7.86 - results[1][2]*100):+.2f}pp")


if __name__ == "__main__":
	main()
