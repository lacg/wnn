"""
Classical ML baselines for UNSW-NB15.

Runs Random Forest and XGBoost on both binary and multi-class tasks,
using the same thermometer-encoded features as the WNN.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	classification_report,
)

from .dataset import load_unsw_nb15, ATTACK_CATEGORIES


def run_baselines(n_bits: int = 8):
	"""Run RF and XGBoost baselines on UNSW-NB15."""

	ds = load_unsw_nb15(n_bits=n_bits)

	print("\n" + "=" * 70)
	print("CLASSICAL BASELINES ON UNSW-NB15")
	print("=" * 70)

	results = {}

	# ── Random Forest ──────────────────────────────────────────────────
	for task, y_train, y_test, names in [
		("binary", ds.y_train_binary, ds.y_test_binary, ["Normal", "Attack"]),
		("multi", ds.y_train_multi, ds.y_test_multi, ATTACK_CATEGORIES),
	]:
		print(f"\n--- Random Forest ({task}) ---")
		rf = RandomForestClassifier(
			n_estimators=100,
			max_depth=None,
			n_jobs=-1,
			random_state=42,
		)
		t0 = time.time()
		rf.fit(ds.X_train, y_train)
		train_time = time.time() - t0

		t0 = time.time()
		y_pred = rf.predict(ds.X_test)
		infer_time = time.time() - t0

		acc = accuracy_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
		rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
		f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

		# False positive rate (for binary)
		if task == "binary":
			cm = confusion_matrix(y_test, y_pred)
			tn, fp, fn, tp = cm.ravel()
			fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
			print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
			print(f"  FPR: {fpr:.4f}")
			results[f"rf_{task}_fpr"] = fpr

		print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
		print(f"  Precision: {prec:.4f}")
		print(f"  Recall:    {rec:.4f}")
		print(f"  F1 (macro):{f1:.4f}")
		print(f"  Train time: {train_time:.1f}s, Infer time: {infer_time:.3f}s")
		print(f"  Infer/sample: {1e6*infer_time/len(ds.X_test):.1f} µs")

		results[f"rf_{task}_acc"] = acc
		results[f"rf_{task}_f1"] = f1
		results[f"rf_{task}_train_s"] = train_time

		if task == "multi":
			print(f"\n  Per-class report:")
			print(classification_report(y_test, y_pred, target_names=names, zero_division=0))

	# ── XGBoost ────────────────────────────────────────────────────────
	try:
		from xgboost import XGBClassifier
		has_xgb = True
	except ImportError:
		print("\n⚠ XGBoost not installed — skipping. Install with: pip install xgboost")
		has_xgb = False

	if has_xgb:
		for task, y_train, y_test, names in [
			("binary", ds.y_train_binary, ds.y_test_binary, ["Normal", "Attack"]),
			("multi", ds.y_train_multi, ds.y_test_multi, ATTACK_CATEGORIES),
		]:
			print(f"\n--- XGBoost ({task}) ---")
			xgb = XGBClassifier(
				n_estimators=100,
				max_depth=6,
				learning_rate=0.1,
				n_jobs=-1,
				random_state=42,
				eval_metric="logloss" if task == "binary" else "mlogloss",
				verbosity=0,
			)
			t0 = time.time()
			xgb.fit(ds.X_train, y_train)
			train_time = time.time() - t0

			t0 = time.time()
			y_pred = xgb.predict(ds.X_test)
			infer_time = time.time() - t0

			acc = accuracy_score(y_test, y_pred)
			prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
			rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
			f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

			if task == "binary":
				cm = confusion_matrix(y_test, y_pred)
				tn, fp, fn, tp = cm.ravel()
				fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
				print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
				print(f"  FPR: {fpr:.4f}")
				results[f"xgb_{task}_fpr"] = fpr

			print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
			print(f"  Precision: {prec:.4f}")
			print(f"  Recall:    {rec:.4f}")
			print(f"  F1 (macro):{f1:.4f}")
			print(f"  Train time: {train_time:.1f}s, Infer time: {infer_time:.3f}s")
			print(f"  Infer/sample: {1e6*infer_time/len(ds.X_test):.1f} µs")

			results[f"xgb_{task}_acc"] = acc
			results[f"xgb_{task}_f1"] = f1
			results[f"xgb_{task}_train_s"] = train_time

			if task == "multi":
				print(f"\n  Per-class report:")
				print(classification_report(y_test, y_pred, target_names=names, zero_division=0))

	# ── Summary ────────────────────────────────────────────────────────
	print("\n" + "=" * 70)
	print("SUMMARY")
	print("=" * 70)
	print(f"{'Model':20s} {'Task':8s} {'Acc':>8s} {'F1':>8s} {'FPR':>8s} {'Train(s)':>8s}")
	print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
	for prefix, name in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
		for task in ["binary", "multi"]:
			key = f"{prefix}_{task}_acc"
			if key not in results:
				continue
			acc = results[key]
			f1 = results[f"{prefix}_{task}_f1"]
			fpr = results.get(f"{prefix}_{task}_fpr", None)
			train_s = results[f"{prefix}_{task}_train_s"]
			fpr_str = f"{fpr:.4f}" if fpr is not None else "--"
			print(f"{name:20s} {task:8s} {acc:>7.4f} {f1:>7.4f} {fpr_str:>8s} {train_s:>7.1f}")

	return results


def run_raw_baselines():
	"""Run RF and XGBoost on raw (non-encoded) features for fair comparison."""
	import pandas as pd
	from pathlib import Path
	from sklearn.preprocessing import LabelEncoder

	# Auto-detect data directory
	candidates = [
		Path(__file__).parents[4] / "data" / "unsw-nb15",
		Path.cwd() / "data" / "unsw-nb15",
	]
	data_dir = None
	for c in candidates:
		if c.exists():
			data_dir = c
			break
	if data_dir is None:
		raise FileNotFoundError("UNSW-NB15 data not found")
	df_train = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv")
	df_test = pd.read_csv(data_dir / "UNSW_NB15_testing-set.csv", encoding="utf-8-sig")

	exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
	train_features = set(df_train.columns) - exclude
	test_features = set(df_test.columns) - exclude
	common = sorted(train_features & test_features)

	# Encode categoricals as integers (RF/XGB handle this fine)
	for col in common:
		if df_train[col].dtype == object or hasattr(df_train[col], "cat"):
			# Convert to string first (handles Categorical dtype from parquet)
			train_vals = df_train[col].astype(str).fillna("?")
			test_vals = df_test[col].astype(str).fillna("?")
			le = LabelEncoder()
			le.fit(pd.concat([train_vals, test_vals]))
			df_train[col] = le.transform(train_vals)
			df_test[col] = le.transform(test_vals)
		elif hasattr(df_test[col], "cat"):
			# Parquet test column is Categorical but train isn't
			test_vals = df_test[col].astype(str).fillna("?")
			train_vals = df_train[col].astype(str).fillna("?")
			le = LabelEncoder()
			le.fit(pd.concat([train_vals, test_vals]))
			df_train[col] = le.transform(train_vals)
			df_test[col] = le.transform(test_vals)

	X_train = df_train[common].fillna(0).values.astype(np.float32)
	X_test = df_test[common].fillna(0).values.astype(np.float32)
	y_train = df_train["label"].values
	y_test = df_test["label"].values

	print(f"\n{'=' * 70}")
	print("RAW FEATURE BASELINES (no thermometer encoding)")
	print(f"{'=' * 70}")
	print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

	results = {}

	for Model, name in [
		(RandomForestClassifier, "Random Forest"),
	]:
		print(f"\n--- {name} (binary, raw features) ---")
		clf = Model(n_estimators=100, n_jobs=-1, random_state=42)
		t0 = time.time()
		clf.fit(X_train, y_train)
		train_time = time.time() - t0

		y_pred = clf.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		cm = confusion_matrix(y_test, y_pred)
		tn, fp, fn, tp = cm.ravel()
		fpr = fp / (fp + tn)
		f1 = f1_score(y_test, y_pred, average="macro")

		print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
		print(f"  F1 (macro):{f1:.4f}")
		print(f"  FPR:       {fpr:.4f}")
		print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
		print(f"  Train time: {train_time:.1f}s")
		results[f"rf_raw_binary_acc"] = acc

	try:
		from xgboost import XGBClassifier
		print(f"\n--- XGBoost (binary, raw features) ---")
		xgb = XGBClassifier(
			n_estimators=100, max_depth=6, learning_rate=0.1,
			n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0,
		)
		t0 = time.time()
		xgb.fit(X_train, y_train)
		train_time = time.time() - t0

		y_pred = xgb.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		cm = confusion_matrix(y_test, y_pred)
		tn, fp, fn, tp = cm.ravel()
		fpr = fp / (fp + tn)
		f1 = f1_score(y_test, y_pred, average="macro")

		print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
		print(f"  F1 (macro):{f1:.4f}")
		print(f"  FPR:       {fpr:.4f}")
		print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
		print(f"  Train time: {train_time:.1f}s")
		results[f"xgb_raw_binary_acc"] = acc
	except ImportError:
		pass

	return results


def run_random_split_baselines():
	"""Run RF and XGBoost with FWIW's random-split methodology.

	Replicates FWIW's evaluation protocol:
	1. Load full 2.28M records (HuggingFace Mouwiya/UNSW-NB15)
	2. Drop duplicates (~30% are dupes)
	3. Random 90/10 split (random_state=0)
	4. Optionally oversample minority class
	5. Drop nominal features (srcip, dstip, sport, dsport)
	"""
	import pandas as pd
	from pathlib import Path
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder
	from sklearn.utils import resample

	# Load full dataset (saved as parquet from HuggingFace)
	full_parquet = Path(__file__).parents[4] / "data" / "unsw-nb15" / "full_dataset.parquet"
	if not full_parquet.exists():
		full_parquet = Path.cwd() / "data" / "unsw-nb15" / "full_dataset.parquet"
	if not full_parquet.exists():
		raise FileNotFoundError(
			"Full UNSW-NB15 parquet not found. Run:\n"
			"  from datasets import load_dataset\n"
			"  ds = load_dataset('Mouwiya/UNSW-NB15', split='train')\n"
			"  ds.to_pandas().to_parquet('data/unsw-nb15/full_dataset.parquet')"
		)

	print(f"\n{'=' * 70}")
	print("RANDOM SPLIT BASELINES (FWIW methodology)")
	print(f"{'=' * 70}")

	df = pd.read_parquet(full_parquet)
	print(f"  Total: {len(df):,} records")

	# Drop duplicates
	n_before = len(df)
	df = df.drop_duplicates()
	print(f"  After dedup: {len(df):,} (dropped {n_before - len(df):,})")

	# Features (drop nominal IPs + ports + labels)
	exclude = {'srcip', 'dstip', 'sport', 'dsport', 'label', 'attack_cat'}
	feature_cols = [c for c in df.columns if c not in exclude]

	for col in feature_cols:
		if df[col].dtype == object:
			le = LabelEncoder()
			df[col] = le.fit_transform(df[col].astype(str).fillna("?"))

	X = df[feature_cols].fillna(0).values.astype(np.float32)
	y = df['label'].values.astype(np.int32)

	# Random 90/10 split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.1, random_state=0, stratify=y
	)
	print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

	results = {}

	for oversample in [True, False]:
		suffix = "os" if oversample else "noos"

		if oversample:
			mask_normal = y_train == 0
			mask_attack = y_train == 1
			n_normal = mask_normal.sum()
			X_attack_up = resample(X_train[mask_attack], n_samples=n_normal, random_state=0)
			X_tr = np.vstack([X_train[mask_normal], X_attack_up])
			y_tr = np.concatenate([y_train[mask_normal], np.ones(n_normal, dtype=np.int32)])
		else:
			X_tr, y_tr = X_train, y_train

		# Random Forest
		label = f"RF random split {'+ oversample' if oversample else '(natural)'}"
		print(f"\n--- {label} ---")
		rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
		t0 = time.time()
		rf.fit(X_tr, y_tr)
		train_time = time.time() - t0

		y_pred = rf.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
		cm = confusion_matrix(y_test, y_pred)
		tn, fp, fn, tp = cm.ravel()
		fpr = fp / (fp + tn)

		print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
		print(f"  F1 (macro):{f1:.4f}")
		print(f"  FPR:       {fpr:.4f}")
		print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
		print(f"  Train time: {train_time:.1f}s")

		results[f"rf_random_{suffix}_acc"] = acc
		results[f"rf_random_{suffix}_f1"] = f1
		results[f"rf_random_{suffix}_fpr"] = fpr

		# XGBoost
		try:
			from xgboost import XGBClassifier
			label = f"XGB random split {'+ oversample' if oversample else '(natural)'}"
			print(f"\n--- {label} ---")
			xgb = XGBClassifier(
				n_estimators=100, max_depth=6, learning_rate=0.1,
				n_jobs=-1, random_state=42, eval_metric="logloss", verbosity=0,
			)
			t0 = time.time()
			xgb.fit(X_tr, y_tr)
			train_time = time.time() - t0

			y_pred = xgb.predict(X_test)
			acc = accuracy_score(y_test, y_pred)
			f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
			cm = confusion_matrix(y_test, y_pred)
			tn, fp, fn, tp = cm.ravel()
			fpr = fp / (fp + tn)

			print(f"  Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
			print(f"  F1 (macro):{f1:.4f}")
			print(f"  FPR:       {fpr:.4f}")
			print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
			print(f"  Train time: {train_time:.1f}s")

			results[f"xgb_random_{suffix}_acc"] = acc
			results[f"xgb_random_{suffix}_f1"] = f1
			results[f"xgb_random_{suffix}_fpr"] = fpr
		except ImportError:
			pass

	return results


if __name__ == "__main__":
	import sys

	if "--random-split" in sys.argv:
		run_random_split_baselines()
	elif "--all" in sys.argv:
		run_baselines()
		run_raw_baselines()
		run_random_split_baselines()
	else:
		# Default: standard split baselines
		run_baselines()
		run_raw_baselines()
