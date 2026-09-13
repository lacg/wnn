"""Verify RF + XGBoost baselines on UNSW-NB15 binary classification.

Measured numbers for the "WNN vs RF/XGB" claim. The 31/05/2026 run on the
legacy 2-way `temporal` split gave RF F1 85.83 / FPR 25.99 and XGB 84.62 / 29.18
(paper-plan estimates of ~87/12 were wrong on FPR by 2-3x).

13/09/2026: `--split temporal_3way` (Protocol v2, 80/10/10). On a `_3way` split
the 10% VAL partition calibrates a second threshold (F1-optimal on val) so both
the WNN `fixed_05` and `val_cal` rows have a like-for-like comparator; scoring is
always on the TEST partition. Same top-20 features and thermometer encoding the
WNN sees.

Usage:
  python3 scripts/verify_unsw_temporal_baselines.py                       # legacy 2-way temporal
  python3 scripts/verify_unsw_temporal_baselines.py --split temporal_3way  # Protocol v2
  # ~3-5 min total (1-2 min RF train, 1-2 min XGBoost train, fast inference)
"""
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

from wnn.ids.dataset import load_unsw_nb15


@dataclass
class Row:
	method: str
	mode: str          # fixed_05 | val_cal
	threshold: float
	f1: float
	fpr: float
	acc: float
	size_bytes: int


def _score(y_true: np.ndarray, p_attack: np.ndarray, threshold: float) -> tuple[float, float, float, tuple]:
	y_pred = (p_attack >= threshold).astype(int)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
	return (f1_score(y_true, y_pred, average="macro", zero_division=0),
	        fpr, accuracy_score(y_true, y_pred), (tn, fp, fn, tp))


def _val_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
	"""F1-optimal threshold on the VAL partition — the WNN's val_cal rule."""
	grid = np.linspace(0.01, 0.99, 99)
	return float(max(grid, key=lambda t: _score(y_val, p_val, t)[0]))


def _evaluate(name: str, model, X_train, y_train, X_test, y_test, X_val, y_val) -> list[Row]:
	print("\n" + "-" * 78)
	print(f"  {name}")
	print("-" * 78)
	t0 = time.time()
	model.fit(X_train, y_train)
	print(f"  Train time:  {time.time() - t0:.1f}s")
	t0 = time.time()
	p_test = model.predict_proba(X_test)[:, 1]
	infer_s = time.time() - t0
	print(f"  Infer time:  {infer_s:.3f}s  ({1e6 * infer_s / len(X_test):.1f} µs/sample)")
	size = len(pickle.dumps(model))
	print(f"  Model size:  {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

	thresholds = [("fixed_05", 0.5)]
	if X_val is not None:
		thresholds.append(("val_cal", _val_threshold(y_val, model.predict_proba(X_val)[:, 1])))
	rows = []
	for mode, thr in thresholds:
		f1, fpr, acc, (tn, fp, fn, tp) = _score(y_test, p_test, thr)
		print(f"  [{mode:<8} thr={thr:.2f}]  TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}  "
		      f"F1 {100 * f1:.2f}%  FPR {100 * fpr:.2f}%  Acc {100 * acc:.2f}%")
		rows.append(Row(name.split(" ")[0], mode, thr, f1, fpr, acc, size))
	return rows


def _load(split: str, n_bits: int):
	print(f"\nLoading UNSW-NB15 {split} split (top-20 features, {n_bits}-bit thermo)...")
	t0 = time.time()
	ds = load_unsw_nb15(split=split, feature_selection="top20", n_bits=n_bits)
	print(f"  Loaded in {time.time() - t0:.1f}s")
	X_train, X_test = ds.X_train.to_numpy_bool(), ds.X_test.to_numpy_bool()
	X_val = ds.X_val.to_numpy_bool() if ds.X_val is not None else None
	for label, X, y in (("Train", X_train, ds.y_train_binary), ("Test", X_test, ds.y_test_binary),
	                    ("Val", X_val, ds.y_val_binary)):
		if X is not None:
			print(f"  {label:<5} {X.shape}  Normal={np.mean(y == 0):.1%} Attack={np.mean(y == 1):.1%}")
	return X_train, ds.y_train_binary, X_test, ds.y_test_binary, X_val, ds.y_val_binary


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n-bits", type=int, default=8,
	                help="Thermometer bits per feature (default 8; 16 matches the WNN 16b-Wb cohort).")
	ap.add_argument("--split", default="temporal", help="temporal (legacy 2-way) | temporal_3way | random | random_3way")
	args = ap.parse_args()

	print("=" * 78)
	print(f"  UNSW-NB15 {args.split} binary — RF + XGBoost ({args.n_bits}-bit thermo)")
	print("=" * 78)
	X_train, y_train, X_test, y_test, X_val, y_val = _load(args.split, args.n_bits)

	rows = []
	rows += _evaluate("RF (100 estimators, max_depth=None)",
	                  RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
	                  X_train, y_train, X_test, y_test, X_val, y_val)
	rows += _evaluate("XGBoost (100 estimators, max_depth=6, lr=0.1)",
	                  XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
	                                random_state=42, eval_metric="logloss", verbosity=0),
	                  X_train, y_train, X_test, y_test, X_val, y_val)

	print("\n" + "=" * 78)
	print(f"  UNSW-NB15 {args.split} — scored on the TEST partition")
	print("=" * 78)
	print(f"  {'Method':<9}{'mode':<10}{'thr':>5}{'F1':>10}{'FPR':>10}{'Acc':>10}{'Size':>12}")
	print(f"  {'-' * 66}")
	for r in rows:
		print(f"  {r.method:<9}{r.mode:<10}{r.threshold:>5.2f}{r.f1 * 100:>9.2f}%{r.fpr * 100:>9.2f}%"
		      f"{r.acc * 100:>9.2f}%{r.size_bytes / 1024 / 1024:>9.2f} MB")


if __name__ == "__main__":
	main()
