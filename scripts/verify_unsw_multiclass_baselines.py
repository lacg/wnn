"""RF + XGBoost MULTICLASS baselines on UNSW-NB15 temporal_3way (K=10).

Counterpart of verify_unsw_temporal_baselines.py for the multiclass IDS
protocol (ids_classification='multi', Protocol v2): same top-20 features and
thermometer encoding the WNN sees, same temporal_3way 80/10/10 partitions.
RF/XGB decode by argmax (no threshold calibration), so VAL is unused for
fitting; metrics are reported on BOTH the report-only TEST partition and VAL
so the numbers are comparable whichever partition a WNN table cites.

Metrics per model x partition: macro-F1, weighted-F1, accuracy, benign-FPR
(fraction of true-benign rows predicted as any attack), per-class
precision/recall/F1/support, and the KxK confusion matrix — the same fields
the worker writes into validation_summaries.threshold_metadata modes.

Usage:
  python3 scripts/verify_unsw_multiclass_baselines.py [--n-bits 16]
  # RF ~2-4 min, XGB ~2-6 min (10-class softprob) on M4 Max
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from wnn.ids.dataset import load_unsw_nb15


def benign_fpr(y_true: np.ndarray, y_pred: np.ndarray, benign_idx: int = 0) -> float:
	"""Fraction of true-benign rows predicted as any attack class."""
	benign_mask = y_true == benign_idx
	if benign_mask.sum() == 0:
		return 0.0
	return float((y_pred[benign_mask] != benign_idx).mean())


def report_partition(name: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: list[str]) -> dict:
	macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
	weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
	acc = accuracy_score(y_true, y_pred)
	bfpr = benign_fpr(y_true, y_pred)
	prec, rec, f1c, sup = precision_recall_fscore_support(
		y_true, y_pred, labels=range(len(class_names)), zero_division=0)
	cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

	print(f"\n  [{model_name} @ {name}]")
	print(f"  macro-F1:    {macro:.4f}  ({100 * macro:.2f}%)")
	print(f"  weighted-F1: {weighted:.4f}  ({100 * weighted:.2f}%)")
	print(f"  accuracy:    {acc:.4f}  ({100 * acc:.2f}%)")
	print(f"  benign-FPR:  {bfpr:.4f}  ({100 * bfpr:.2f}%)")
	print(f"  {'class':<16} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>9}")
	for i, cls in enumerate(class_names):
		print(f"  {cls:<16} {prec[i]:>8.4f} {rec[i]:>8.4f} {f1c[i]:>8.4f} {sup[i]:>9,}")
	print(f"  confusion (rows=true, label-index order {class_names}):")
	for i, row in enumerate(cm):
		print(f"    {class_names[i]:<16} {' '.join(f'{v:>7,}' for v in row)}")

	return {
		"partition": name,
		"macro_f1": float(macro),
		"weighted_f1": float(weighted),
		"acc": float(acc),
		"benign_fpr": float(bfpr),
		"per_class": {
			cls: {"precision": float(prec[i]), "recall": float(rec[i]),
			      "f1": float(f1c[i]), "support": int(sup[i])}
			for i, cls in enumerate(class_names)
		},
		"confusion": cm.tolist(),
	}


def run_model(model, model_name: str, X_train, y_train, partitions: dict,
              class_names: list[str]) -> dict:
	print("\n" + "-" * 78)
	print(f"  {model_name}")
	print("-" * 78)
	t0 = time.time()
	model.fit(X_train, y_train)
	train_s = time.time() - t0
	size = len(pickle.dumps(model))
	print(f"  Train time:  {train_s:.1f}s")
	print(f"  Model size:  {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

	out = {"model": model_name, "train_s": train_s, "size_bytes": size, "partitions": []}
	for pname, (X, y) in partitions.items():
		t0 = time.time()
		y_pred = model.predict(X)
		infer_s = time.time() - t0
		print(f"\n  Infer {pname}: {infer_s:.3f}s ({1e6 * infer_s / len(X):.1f} µs/sample)")
		out["partitions"].append(report_partition(pname, model_name, y, y_pred, class_names))
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n-bits", type=int, default=16,
	                help="Thermometer bits per feature (default 16, matching the multiclass WNN runs).")
	ap.add_argument("--json-out", default=None,
	                help="Optional path to dump results as JSON (for tables).")
	args = ap.parse_args()

	print("=" * 78)
	print(f"  UNSW-NB15 temporal_3way MULTICLASS — RF + XGBoost baselines ({args.n_bits}-bit thermo)")
	print("=" * 78)

	t0 = time.time()
	ds = load_unsw_nb15(split="temporal_3way", feature_selection="top20", n_bits=args.n_bits)
	print(f"  Loaded in {time.time() - t0:.1f}s")
	X_train = ds.X_train.to_numpy_bool()
	X_test = ds.X_test.to_numpy_bool()
	X_val = ds.X_val.to_numpy_bool()
	class_names = list(ds.category_names)
	print(f"  Train {X_train.shape}  Test {X_test.shape}  Val {X_val.shape}")
	print(f"  Classes ({len(class_names)}): {class_names}")
	for pname, y in (("train", ds.y_train_multi), ("test", ds.y_test_multi), ("val", ds.y_val_multi)):
		counts = np.bincount(y, minlength=len(class_names))
		print(f"  {pname} distribution: " + ", ".join(f"{c}={n:,}" for c, n in zip(class_names, counts)))

	partitions = {
		"test (report-only)": (X_test, ds.y_test_multi),
		"val": (X_val, ds.y_val_multi),
	}

	results = []
	results.append(run_model(
		RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
		"Random Forest (100 est, depth=None)", X_train, ds.y_train_multi, partitions, class_names))
	results.append(run_model(
		XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
		              n_jobs=-1, random_state=42, eval_metric="mlogloss", verbosity=0),
		"XGBoost (100 est, depth=6, lr=0.1)", X_train, ds.y_train_multi, partitions, class_names))

	print("\n" + "=" * 78)
	print("  Summary (test = Protocol-v2 report partition)")
	print("=" * 78)
	print(f"  {'Model':<16} {'part':<20} {'mF1':>8} {'wF1':>8} {'Acc':>8} {'bFPR':>8}")
	for r in results:
		short = "RF" if r["model"].startswith("Random") else "XGB"
		for p in r["partitions"]:
			print(f"  {short:<16} {p['partition']:<20} {100 * p['macro_f1']:>7.2f}% "
			      f"{100 * p['weighted_f1']:>7.2f}% {100 * p['acc']:>7.2f}% {100 * p['benign_fpr']:>7.2f}%")

	if args.json_out:
		from pathlib import Path
		Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
		with open(args.json_out, "w") as fh:
			json.dump({"n_bits": args.n_bits, "class_names": class_names, "results": results}, fh, indent=1)
		print(f"\n  JSON written to {args.json_out}")


if __name__ == "__main__":
	main()
