"""RF/XGB multiclass baselines on the SCREENING protocol (MULTICLASS_DESIGN.md §5).

Matches the MCS screening arms exactly: same loader, same split (temporal_3way
80/10/10), same feature selection (top20), same class mapping (loader's own
attack_categories via map_multiclass_labels — benign index 0 by construction).
RAW feature values (classical models don't take thermometer bits — matches the
reference_measured_raw_baselines convention). Train on the 80%; TEST (10%) is
the report partition; VAL (10%) is untouched (no calibration for argmax models).

Usage:
  python scripts/run_multiclass_baselines.py --dataset unsw-nb15 --split temporal_3way
Outputs JSON to docs/multiclass_baselines/<dataset>_<split>.json + stdout report.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", default="unsw-nb15")
	ap.add_argument("--split", default="temporal_3way")
	ap.add_argument("--feature-selection", default="top20")
	ap.add_argument("--n-jobs", type=int, default=4, help="polite default: ladder + worker own the box")
	ap.add_argument("--out-dir", default="docs/multiclass_baselines")
	return ap.parse_args()


def load_raw_frames(dataset: str, split: str, feature_selection: str):
	"""Same loader the flows use, stopped BEFORE thermometer encoding."""
	from wnn.ids.loaders import get_loader, LoadSpec
	from wnn.ids.dataset import _select_top_features

	loader = get_loader(dataset)
	spec = LoadSpec(split=split, feature_selection=feature_selection)
	df_train, df_test, common_features, df_val = loader._fetch(spec)
	# The exact selection rule the encode path applies (top20 = first 20 of the
	# loader's ranked list intersected with the frame's columns).
	feats, _ = _select_top_features(
		feature_selection, list(common_features), loader._feature_list(spec) or [], None)
	return loader, df_train, df_test, df_val, feats


def fit_categorical_codes(df_train, feats):
	"""Ordinal codes for non-numeric feature columns, learned on TRAIN only.
	Unseen categories at test/val map to -1 (same spirit as the encoder's
	unseen-category handling; RF/XGB split on the code as an opaque ordinal)."""
	import pandas as pd

	codes = {}
	for f in feats:
		if not pd.api.types.is_numeric_dtype(df_train[f]):
			cats = df_train[f].astype(str).unique().tolist()
			codes[f] = {c: i for i, c in enumerate(cats)}
	return codes


def build_xy(loader, df, feats, class_to_idx, split_name, cat_codes):
	from wnn.ids.dataset import map_multiclass_labels

	cols = []
	for f in feats:
		if f in cat_codes:
			cols.append(df[f].astype(str).map(cat_codes[f]).fillna(-1).to_numpy(dtype=np.float32))
		else:
			cols.append(df[f].to_numpy(dtype=np.float32, copy=True))
	x = np.column_stack(cols)
	np.nan_to_num(x, copy=False)  # raw variants may carry NaN/inf; classical models will not
	y_bin = df[loader.binary_col].to_numpy(dtype=np.int64)
	y_multi = map_multiclass_labels(df[loader.multiclass_col], class_to_idx, split_name)
	return x, y_bin, y_multi


def metrics_multiclass(y_true, y_pred, names, benign_idx=0):
	from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

	cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
	benign_mask = y_true == benign_idx
	benign_fpr = float((y_pred[benign_mask] != benign_idx).mean()) if benign_mask.any() else None
	return {
		"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		"weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"benign_fpr": benign_fpr,
		"per_class_recall": {
			n: float(r) for n, r in zip(names, recall_score(
				y_true, y_pred, average=None, labels=range(len(names)), zero_division=0))
		},
		"support": {n: int(s) for n, s in zip(names, cm.sum(axis=1))},
		"confusion": cm.tolist(),
	}


def metrics_binary(y_true, y_pred):
	from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

	tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
	return {
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"fpr": float(fp / (fp + tn)) if (fp + tn) else None,
		"accuracy": float(accuracy_score(y_true, y_pred)),
	}


def _compact_labels(y_train, num_classes):
	"""XGBoost demands labels 0..K-1 CONTIGUOUS in y_train. A split whose train
	partition is missing classes (CICIDS temporal omits PortScan/DDoS/Bot entirely)
	hands it gaps and it raises instead of fitting. Map present labels down to a
	dense range and return the inverse so predictions come back in dataset space."""
	present = sorted(set(int(v) for v in y_train))
	if present == list(range(len(present))) and len(present) == num_classes:
		return y_train, None, num_classes
	forward = {lab: i for i, lab in enumerate(present)}
	inverse = np.array(present, dtype=np.int64)
	return np.array([forward[int(v)] for v in y_train], dtype=np.int64), inverse, len(present)


def fit_predict(model_name, task, x_train, y_train, x_test, n_jobs, num_classes):
	inverse = None
	if model_name == "rf":
		from sklearn.ensemble import RandomForestClassifier
		m = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42)
	else:
		from xgboost import XGBClassifier
		if task == "multi":
			y_train, inverse, num_classes = _compact_labels(y_train, num_classes)
		m = XGBClassifier(
			n_estimators=100, tree_method="hist", n_jobs=n_jobs, random_state=42,
			objective="multi:softmax" if task == "multi" else "binary:logistic",
			num_class=num_classes if task == "multi" else None,
			verbosity=0)
	t0 = time.time()
	m.fit(x_train, y_train)
	fit_s = time.time() - t0
	pred = m.predict(x_test)
	return (pred if inverse is None else inverse[pred]), fit_s


def fit_predict_cascade(model_name, x_train, yb_train, ym_train, x_test, n_jobs, num_classes):
	"""S0 binary gate -> S1 attack-type classifier trained on ATTACKS ONLY, with
	S1's predictions remapped into the full K-class label space.

	This mirrors the WNN `hierarchical` arm exactly (flow.py
	_compute_ids_hierarchical_combined: S0 binary, S1 9-class, remap to 1-9), and
	it is the ONLY honest comparator for that arm. Scoring a WNN cascade against
	a FLAT RF would credit the WNN architecture with a gain that comes from the
	cascade STRUCTURE — the classical models get the same structure or the
	comparison means nothing.

	Routing is by S0's PREDICTION, not by the true label: benign-predicted rows
	are emitted as class 0 and never reach S1, so a benign row S0 lets through
	still lands on some attack class. That is what makes the cascade's
	benign-FPR inherit S0's FPR, and it is the property under test.

	S1 TRAINS on true attacks (standard cascade practice, and what the WNN arm
	does) while it PREDICTS on routed rows — so S1 never sees benign rows in
	training and cannot learn to emit class 0.
	"""
	t0 = time.time()
	s0_pred, _ = fit_predict(model_name, "binary", x_train, yb_train, x_test, n_jobs, 2)
	pred = np.zeros(len(x_test), dtype=np.int64)
	routed = s0_pred == 1
	attack_train = yb_train == 1
	if routed.any() and attack_train.any():
		s1_pred, _ = fit_predict(model_name, "multi", x_train[attack_train],
		                         ym_train[attack_train], x_test[routed],
		                         n_jobs, num_classes)
		pred[routed] = s1_pred
	return pred, time.time() - t0, int(routed.sum()), int(attack_train.sum())


def main():
	args = parse_args()
	loader, df_train, df_test, df_val, feats = load_raw_frames(
		args.dataset, args.split, args.feature_selection)
	names = list(loader.attack_categories)
	class_to_idx = {n: i for i, n in enumerate(names)}
	print(f"dataset={args.dataset} split={args.split} feats={len(feats)} classes={len(names)}: {names}")
	print(f"rows: train={len(df_train)} test={len(df_test)} val={len(df_val) if df_val is not None else 0} (val untouched)")

	cat_codes = fit_categorical_codes(df_train, feats)
	if cat_codes:
		print(f"categorical features (train-fitted ordinal codes): {list(cat_codes)}")
	x_train, yb_train, ym_train = build_xy(loader, df_train, feats, class_to_idx, "train", cat_codes)
	x_test, yb_test, ym_test = build_xy(loader, df_test, feats, class_to_idx, "test", cat_codes)
	# The task loop below rebinds `y_test`; the cascade is scored in 10-class
	# space, so hold an unambiguous reference rather than trusting loop leakage.
	y_test_multi_ref = ym_test

	results = {"dataset": args.dataset, "split": args.split,
			   "feature_selection": args.feature_selection, "features": feats,
			   "classes": names, "rows": {"train": len(x_train), "test": len(x_test)},
			   "models": {}}
	for model_name in ("rf", "xgb"):
		for task, y_train, y_test in (("binary", yb_train, yb_test), ("multi", ym_train, ym_test)):
			y_pred, fit_s = fit_predict(model_name, task, x_train, y_train, x_test,
										args.n_jobs, len(names))
			m = metrics_binary(y_test, y_pred) if task == "binary" \
				else metrics_multiclass(y_test, y_pred, names)
			m["fit_seconds"] = round(fit_s, 1)
			results["models"][f"{model_name}_{task}"] = m
			head = {k: v for k, v in m.items() if k not in ("per_class_recall", "support", "confusion")}
			print(f"[{model_name} {task}] {head}")
			if task == "multi":
				print("  per-class recall: " + "  ".join(
					f"{n}={r:.3f}" for n, r in m["per_class_recall"].items()))

	# CASCADE arm — the comparator for the WNN `hierarchical` screening arm.
	for model_name in ("rf", "xgb"):
		y_pred, fit_s, n_routed, n_attack_train = fit_predict_cascade(
			model_name, x_train, yb_train, ym_train, x_test, args.n_jobs, len(names))
		m = metrics_multiclass(y_test_multi_ref, y_pred, names)
		m["fit_seconds"] = round(fit_s, 1)
		m["routed_to_s1"] = n_routed
		m["s1_train_rows"] = n_attack_train
		results["models"][f"{model_name}_cascade"] = m
		head = {k: v for k, v in m.items() if k not in ("per_class_recall", "support", "confusion")}
		print(f"[{model_name} cascade] {head}")
		print("  per-class recall: " + "  ".join(
			f"{n}={r:.3f}" for n, r in m["per_class_recall"].items()))

	out = Path(args.out_dir) / f"{args.dataset}_{args.split}.json"
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(results, indent=1))
	print(f"wrote {out}")


if __name__ == "__main__":
	main()
