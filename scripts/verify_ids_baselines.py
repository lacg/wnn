"""Verify RF + XGBoost + AdaBoost baselines on the IDS binary datasets (UNSW / CICIDS / CIC-IoT).

Measured comparators for the "WNN vs RF/XGB" claim, on the SAME top-20 features and
thermometer encoding the WNN sees. Was scripts/verify_unsw_temporal_baselines.py
(31/05/2026: legacy 2-way UNSW temporal, RF F1 85.83 / FPR 25.99, XGB 84.62 / 29.18).

13/09/2026: `--dataset` + `--split`. On a `_3way` split (Protocol v2, 80/10/10) the 10%
VAL partition calibrates a second threshold (F1-optimal on val) so both the WNN
`fixed_05` and `val_cal` rows have a like-for-like comparator; scoring is always on
the TEST partition. This is what showed the banked "WNN -18pp FPR" read had paired
the WNN's val_cal row against trees at a fixed 0.5 (docs/ids_results.md section 0A).

Usage:
  python3 scripts/verify_ids_baselines.py --dataset unsw   --split temporal_3way [--n-bits 16]
  python3 scripts/verify_ids_baselines.py --dataset cicids --split random_3way
  python3 scripts/verify_ids_baselines.py --dataset ciciot --split random_3way   # neto_subsample
  python3 scripts/verify_ids_baselines.py --dataset ciciot --split random_3way --raw   # paper comparator
  python3 scripts/verify_ids_baselines.py --dataset ciciot46m --split random_3way --raw   # 46.7M, raw only, ~1 h
  # a few minutes per dataset; CIC-IoT is the slow one
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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer

from wnn.ids import cicids2017, ciciot2023, dataset as unsw_dataset
from wnn.ids.cicids2017 import load_cicids2017
from wnn.ids.ciciot2023 import load_ciciot2023
from wnn.ids.dataset import load_unsw_nb15

# RAW top-20 = the SAME lists the WNN loader resolves for feature_selection="top20"
# (run_all_baselines.py carries a stale bencorn-era ciciot list with Time_To_Live — not this).
TOP20 = {
	"unsw": unsw_dataset.TOP20_RF_FEATURES,
	"cicids": cicids2017.TOP20_RF_FEATURES,
	"ciciot": ciciot2023.TOP20_RF_FEATURES,
	"ciciot46m": ciciot2023.TOP20_RF_FEATURES,
}
# the same repos the loaders read — ciciot is the NETO SUBSAMPLE (lacg030175/CIC-IoT-2023 is the
# bencorn-era mirror: 13/20 canonical features, different rows — not what the WNN sweeps on)
HF_REPOS = {
	"unsw": "lacg030175/UNSW-NB15",
	"cicids": "lacg030175/CICIDS2017",
	"ciciot": ciciot2023.HF_DATASET_NETO_SUBSAMPLE_ID,
	"ciciot46m": ciciot2023.HF_DATASET_NETO_FULL_ID,   # 46.7M rows — RAW ONLY (never swept, never thermo'd here)
}

LOADERS = {
	"unsw": lambda split, n_bits: load_unsw_nb15(split=split, feature_selection="top20", n_bits=n_bits),
	"cicids": lambda split, n_bits: load_cicids2017(split=split, feature_selection="top20", n_bits=n_bits),
	# the SUBSAMPLE is the swept/reported CIC-IoT set (46M neto_full is never swept)
	"ciciot": lambda split, n_bits: load_ciciot2023(split=split, feature_selection="top20", n_bits=n_bits,
	                                                dataset_size="neto_subsample"),
}


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


def _describe(parts: list[tuple[str, np.ndarray | None, np.ndarray | None]]):
	for label, X, y in parts:
		if X is not None:
			print(f"  {label:<5} {X.shape}  Normal={np.mean(y == 0):.1%} Attack={np.mean(y == 1):.1%}")


def _load_thermo(dataset: str, split: str, n_bits: int):
	print(f"\nLoading {dataset} {split} split (top-20 features, {n_bits}-bit thermo)...")
	t0 = time.time()
	ds = LOADERS[dataset](split, n_bits)
	print(f"  Loaded in {time.time() - t0:.1f}s")
	X_train, X_test = ds.X_train.to_numpy_bool(), ds.X_test.to_numpy_bool()
	X_val = ds.X_val.to_numpy_bool() if ds.X_val is not None else None
	_describe([("Train", X_train, ds.y_train_binary), ("Test", X_test, ds.y_test_binary),
	           ("Val", X_val, ds.y_val_binary)])
	return X_train, ds.y_train_binary, X_test, ds.y_test_binary, X_val, ds.y_val_binary


def _impute(X_train, X_test, X_val):
	"""Inf -> NaN -> train-median, fit on train only (the 46M set carries a few Inf rows)."""
	clean = lambda X: None if X is None else np.where(np.isinf(X), np.nan, X)
	X_train, X_test, X_val = clean(X_train), clean(X_test), clean(X_val)
	n_bad = int(np.isnan(X_train).any(axis=1).sum())
	if n_bad == 0 and not np.isnan(X_test).any() and (X_val is None or not np.isnan(X_val).any()):
		return X_train, X_test, X_val
	print(f"  NaN/Inf rows in train: {n_bad:,} — median-imputing for the sklearn models")
	imp = SimpleImputer(strategy="median").fit(X_train)
	return imp.transform(X_train), imp.transform(X_test), None if X_val is None else imp.transform(X_val)


def _load_raw(dataset: str, split: str):
	"""RAW numeric top-20 straight from HuggingFace — the comparator the paper's rule 5 uses."""
	from datasets import load_dataset
	print(f"\nLoading {dataset} {split} split (top-20 features, RAW numeric) from {HF_REPOS[dataset]}...")
	t0 = time.time()
	ds = load_dataset(HF_REPOS[dataset], split)
	frames = {k: ds[k].to_pandas() for k in ("train", "test", "validation") if k in ds}
	feats = [f for f in TOP20[dataset] if f in frames["train"].columns]
	label = "label" if "label" in frames["train"].columns else "Label"
	print(f"  Loaded in {time.time() - t0:.1f}s; {len(feats)}/{len(TOP20[dataset])} top-20 features present")
	# categorical columns (UNSW proto/service/state) -> integer codes with TRAIN-fitted
	# categories; unseen levels code to -1. Trees are fine with ordinal codes.
	import pandas as pd
	cats = {f: pd.Categorical(frames["train"][f]).categories
	        for f in feats if not pd.api.types.is_numeric_dtype(frames["train"][f])}
	if cats:
		print(f"  categorical -> codes: {sorted(cats)}")
	def xy(k):
		if k not in frames:
			return None, None
		df = frames[k][feats].copy()
		for f, levels in cats.items():
			df[f] = pd.Categorical(df[f], categories=levels).codes
		return df.values.astype(np.float32), frames[k][label].values.astype(np.int32)
	(X_train, y_train), (X_test, y_test), (X_val, y_val) = xy("train"), xy("test"), xy("validation")
	_describe([("Train", X_train, y_train), ("Test", X_test, y_test), ("Val", X_val, y_val)])
	return X_train, y_train, X_test, y_test, X_val, y_val


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n-bits", type=int, default=8,
	                help="Thermometer bits per feature (default 8; 16 matches the WNN 16b-Wb cohort).")
	ap.add_argument("--dataset", default="unsw", choices=sorted(HF_REPOS))
	ap.add_argument("--split", default="temporal", help="temporal (legacy 2-way) | temporal_3way | random | random_3way")
	ap.add_argument("--raw", action="store_true", help="RAW numeric top-20 instead of the thermometer encoding")
	args = ap.parse_args()
	encoding = "raw numeric" if args.raw else f"{args.n_bits}-bit thermo"

	print("=" * 78)
	print(f"  {args.dataset} {args.split} binary — RF + XGBoost + AdaBoost ({encoding})")
	print("=" * 78)
	if args.dataset not in LOADERS and not args.raw:
		raise SystemExit(f"{args.dataset} is raw-only (pass --raw)")
	X_train, y_train, X_test, y_test, X_val, y_val = (_load_raw(args.dataset, args.split) if args.raw
	                                                  else _load_thermo(args.dataset, args.split, args.n_bits))
	# sklearn trees refuse NaN/Inf (46M has a few); XGBoost handles them natively
	X_train_sk, X_test_sk, X_val_sk = _impute(X_train, X_test, X_val) if args.raw else (X_train, X_test, X_val)

	rows = []
	rows += _evaluate("RF (100 estimators, max_depth=None)",
	                  RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
	                  X_train_sk, y_train, X_test_sk, y_test, X_val_sk, y_val)
	rows += _evaluate("XGBoost (100 estimators, max_depth=6, lr=0.1)",
	                  XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
	                                random_state=42, eval_metric="logloss", verbosity=0),
	                  X_train, y_train, X_test, y_test, X_val, y_val)
	rows += _evaluate("AdaBoost (100 depth-1 stumps)",
	                  AdaBoostClassifier(n_estimators=100, random_state=42),
	                  X_train_sk, y_train, X_test_sk, y_test, X_val_sk, y_val)

	print("\n" + "=" * 78)
	print(f"  {args.dataset} {args.split} ({encoding}) — scored on the TEST partition")
	print("=" * 78)
	print(f"  {'Method':<9}{'mode':<10}{'thr':>5}{'F1':>10}{'FPR':>10}{'Acc':>10}{'Size':>12}")
	print(f"  {'-' * 66}")
	for r in rows:
		print(f"  {r.method:<9}{r.mode:<10}{r.threshold:>5.2f}{r.f1 * 100:>9.2f}%{r.fpr * 100:>9.2f}%"
		      f"{r.acc * 100:>9.2f}%{r.size_bytes / 1024 / 1024:>9.2f} MB")


if __name__ == "__main__":
	main()
