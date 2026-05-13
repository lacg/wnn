"""Train + measure Table 5 baselines on canonical neto-full with TOP20.

End-to-end one-shot run: trains each model once with random_state=42, then
reports model size (joblib bytes) + F1/FPR/Acc + per-sample inference
latency on the 9.34M held-out set. Replaces the prior split between
run_neto_full_baselines.py (F1/FPR/Acc) and the size-only path — one
training pass covers both columns of Table 5.

Models measured:
- Random Forest (n_estimators=100, top-20)
- XGBoost (n_estimators=100, max_depth=6, lr=0.1, top-20)
- AdaBoost (top-20, n_estimators=50)
- AdaBoost (all 46 features, n_estimators=50)
- Perceptron (sklearn defaults + StandardScaler + median imputer, top-20)

CPU-only — safe to run alongside the worker.
"""

import io
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

# Canonical TOP20 RF importance on neto-full (re-derived 13/05/2026).
from wnn.ids.ciciot2023 import TOP20_RF_FEATURES as TOP20_CICIOT


def fmt_bytes(n: int) -> str:
	"""Human-friendly size string."""
	if n < 10_000:
		return f"{n:,} B"
	if n < 10_000_000:
		return f"{n / 1024:.1f} KB"
	if n < 10_000_000_000:
		return f"{n / (1024 * 1024):.1f} MB"
	return f"{n / (1024 * 1024 * 1024):.2f} GB"


def measure(name: str, clf, X_eval, y_eval) -> dict:
	# Size: joblib serialization (sklearn reference)
	jl_buf = io.BytesIO()
	joblib.dump(clf, jl_buf)
	jl_size = len(jl_buf.getvalue())

	# Inference: predict on full held-out set, time it for per-sample latency
	t0 = time.time()
	y_pred = clf.predict(X_eval)
	infer_s = time.time() - t0
	per_sample_ns = (infer_s / len(X_eval)) * 1e9

	# Metrics
	f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
	acc = accuracy_score(y_eval, y_pred)
	tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
	fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

	return {
		"name": name, "joblib_bytes": jl_size,
		"f1": f1, "fpr": fpr, "acc": acc,
		"infer_s": infer_s, "per_sample_ns": per_sample_ns,
	}


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("Table 5 baselines on canonical neto-full + canonical TOP20")
	print("  random_state=42, n_estimators=100 (RF/XGB) / 50 (AdaBoost)")
	print("  Dataset: lacg030175/CIC-IoT-2023-neto-full (46.7M canonical Neto)")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"\nLoading {repo} (random_3way)...", flush=True)
	t0 = time.time()
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	df_eval = pd.concat([ds["test"].to_pandas(), ds["validation"].to_pandas()], ignore_index=True)
	print(f"  Loaded {len(df_train):,} train + {len(df_eval):,} eval (test+val) rows in {time.time() - t0:.1f}s")

	# Binary label — already int (0=Benign, 1=Attack) in lacg030175 HF dataset
	y_train = df_train["label"].astype(np.int32).values
	y_eval = df_eval["label"].astype(np.int32).values

	# Top-20 features for everything except all-46 AdaBoost
	features_20 = [f for f in TOP20_CICIOT if f in df_train.columns]
	missing = [f for f in TOP20_CICIOT if f not in df_train.columns]
	print(f"  TOP20 features present: {len(features_20)} / 20")
	if missing:
		print(f"  ⚠ MISSING from dataset: {missing} — investigate before continuing")
		sys.exit(1)
	X_train_20 = df_train[features_20].values.astype(np.float32)
	X_eval_20 = df_eval[features_20].values.astype(np.float32)

	# All-46 (numeric-only columns; HF dataset includes string label/attack cols)
	all_features = [
		c for c in df_train.columns
		if c not in ("label", "attack_class", "is_benign") and pd.api.types.is_numeric_dtype(df_train[c])
	]
	print(f"  all-46 feature count: {len(all_features)}")
	X_train_46 = df_train[all_features].values.astype(np.float32)
	X_eval_46 = df_eval[all_features].values.astype(np.float32)

	# Impute NaN/Inf for RF / Perceptron / AdaBoost; XGBoost handles natively
	imp20 = SimpleImputer(strategy="median").fit(X_train_20)
	imp46 = SimpleImputer(strategy="median").fit(X_train_46)
	print("  Median-imputing NaN/Inf for RF/AdaBoost/Perceptron...")
	X_train_20_imp = imp20.transform(X_train_20)
	X_eval_20_imp = imp20.transform(X_eval_20)
	X_train_46_imp = imp46.transform(X_train_46)
	X_eval_46_imp = imp46.transform(X_eval_46)

	results = []

	# --- Random Forest (top-20) ---
	print("\n[1/5] Training Random Forest (n_estimators=100, top-20)...", flush=True)
	t0 = time.time()
	rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
	rf.fit(X_train_20_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("Random Forest", rf, X_eval_20_imp, y_eval))
	print(f"      F1={results[-1]['f1']*100:.2f}%  FPR={results[-1]['fpr']*100:.2f}%  Acc={results[-1]['acc']*100:.2f}%  "
	      f"per-sample={results[-1]['per_sample_ns']:.1f} ns  size={fmt_bytes(results[-1]['joblib_bytes'])}")
	del rf

	# --- XGBoost (top-20) ---
	print("\n[2/5] Training XGBoost (n_estimators=100, top-20)...", flush=True)
	t0 = time.time()
	xgb = XGBClassifier(
		n_estimators=100, max_depth=6, learning_rate=0.1,
		objective="binary:logistic", n_jobs=-1, random_state=42, verbosity=0,
	)
	xgb.fit(X_train_20, y_train)  # XGBoost handles NaN natively
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("XGBoost", xgb, X_eval_20, y_eval))
	print(f"      F1={results[-1]['f1']*100:.2f}%  FPR={results[-1]['fpr']*100:.2f}%  Acc={results[-1]['acc']*100:.2f}%  "
	      f"per-sample={results[-1]['per_sample_ns']:.1f} ns  size={fmt_bytes(results[-1]['joblib_bytes'])}")
	del xgb

	# --- AdaBoost (top-20) ---
	print("\n[3/5] Training AdaBoost (n_estimators=50, top-20)...", flush=True)
	t0 = time.time()
	ab20 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
	ab20.fit(X_train_20_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("AdaBoost (top-20)", ab20, X_eval_20_imp, y_eval))
	print(f"      F1={results[-1]['f1']*100:.2f}%  FPR={results[-1]['fpr']*100:.2f}%  Acc={results[-1]['acc']*100:.2f}%  "
	      f"per-sample={results[-1]['per_sample_ns']:.1f} ns  size={fmt_bytes(results[-1]['joblib_bytes'])}")
	del ab20

	# --- AdaBoost (all 46 features) ---
	print("\n[4/5] Training AdaBoost (n_estimators=50, all 46 feat)...", flush=True)
	t0 = time.time()
	ab46 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
	ab46.fit(X_train_46_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("AdaBoost (all 46 feat)", ab46, X_eval_46_imp, y_eval))
	print(f"      F1={results[-1]['f1']*100:.2f}%  FPR={results[-1]['fpr']*100:.2f}%  Acc={results[-1]['acc']*100:.2f}%  "
	      f"per-sample={results[-1]['per_sample_ns']:.1f} ns  size={fmt_bytes(results[-1]['joblib_bytes'])}")
	del ab46

	# --- Perceptron (top-20, median impute + StandardScaler) ---
	print("\n[5/5] Training Perceptron (sklearn defaults + StandardScaler, top-20)...", flush=True)
	t0 = time.time()
	perc = Pipeline([
		("imp", SimpleImputer(strategy="median")),
		("sc", StandardScaler()),
		("clf", Perceptron(random_state=42, n_jobs=-1)),
	])
	perc.fit(X_train_20, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("Perceptron", perc, X_eval_20, y_eval))
	print(f"      F1={results[-1]['f1']*100:.2f}%  FPR={results[-1]['fpr']*100:.2f}%  Acc={results[-1]['acc']*100:.2f}%  "
	      f"per-sample={results[-1]['per_sample_ns']:.1f} ns  size={fmt_bytes(results[-1]['joblib_bytes'])}")
	del perc

	# --- Final summary table (Table 5 columns) ---
	print("\n" + "=" * 100)
	print("Table 5 baseline (46M) — F1 / FPR / Acc / per-sample latency / model size")
	print(f"Eval set: {len(y_eval):,} samples (test+val merged from random_3way)")
	print("=" * 100)
	print(f"{'Model':<25} {'F1 (%)':>8} {'FPR (%)':>8} {'Acc (%)':>8} {'Latency':>12} {'Size':>12}")
	print("-" * 100)
	for r in results:
		print(f"{r['name']:<25} {r['f1']*100:>8.2f} {r['fpr']*100:>8.2f} {r['acc']*100:>8.2f} "
		      f"{r['per_sample_ns']:>9.1f} ns {fmt_bytes(r['joblib_bytes']):>12}")
	print()


if __name__ == "__main__":
	main()
