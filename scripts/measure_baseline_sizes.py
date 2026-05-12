"""Measure trained-model artifact sizes for the Table 5 baselines.

Same hyperparameters + random_state as the 28/04/2026 run that produced the
F1/FPR/Acc numbers in the paper, so the trained-model bytes match exactly.
Only writes the model-size column; does NOT touch F1/FPR/Acc (the previous
run's numbers stand).

Models measured:
- Random Forest (n_estimators=100)
- XGBoost (n_estimators=100, max_depth=6, lr=0.1)
- AdaBoost (top-20, n_estimators=50)
- AdaBoost (all 46 features, n_estimators=50)
- Perceptron (sklearn defaults + StandardScaler + median imputer)

For each model we report:
- joblib.dump byte size (deterministic given random_state + sklearn version)
- pickle byte size (in-memory serialization)
- A "Table 5" column suggestion using a human-friendly unit

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "wnn"))

TOP20_CICIOT = [
	"HTTPS", "Number", "Time_To_Live", "Max", "ack_flag_number",
	"Rate", "IAT", "ack_count", "Header_Length", "Min",
	"Variance", "psh_flag_number", "Tot sum", "Std", "Tot size",
	"syn_count", "AVG", "rst_flag_number", "DNS", "rst_count",
]


def fmt_bytes(n: int) -> str:
	"""Human-friendly size string."""
	if n < 10_000:
		return f"{n:,} B"
	if n < 10_000_000:
		return f"{n / 1024:.1f} KB"
	if n < 10_000_000_000:
		return f"{n / (1024 * 1024):.1f} MB"
	return f"{n / (1024 * 1024 * 1024):.2f} GB"


def measure(name: str, clf) -> dict:
	# Pickle size
	pkl_buf = pickle.dumps(clf)
	pkl_size = len(pkl_buf)
	# Joblib size (compressed=False by default; sklearn's reference)
	jl_buf = io.BytesIO()
	joblib.dump(clf, jl_buf)
	jl_size = len(jl_buf.getvalue())
	return {"name": name, "pickle_bytes": pkl_size, "joblib_bytes": jl_size}


def main():
	from datasets import load_dataset

	print("=" * 78)
	print("Table 5 baseline model-size measurement")
	print("Same hyperparameters + random_state=42 as the 28/04/2026 paper run.")
	print("Dataset: lacg030175/CIC-IoT-2023-neto-full (canonical 46.7M Neto)")
	print("=" * 78)

	repo = "lacg030175/CIC-IoT-2023-neto-full"
	print(f"\nLoading {repo} (random_3way)...", flush=True)
	t0 = time.time()
	ds = load_dataset(repo, "random_3way")
	df_train = ds["train"].to_pandas()
	print(f"  Loaded {len(df_train):,} train rows in {time.time() - t0:.1f}s")

	# Binary label — already int (0=Benign, 1=Attack) in lacg030175 HF dataset
	y_train = df_train["label"].astype(np.int32).values

	# Top-20 features for everything except all-46 AdaBoost
	features_20 = [f for f in TOP20_CICIOT if f in df_train.columns]
	X_train_20 = df_train[features_20].values.astype(np.float32)
	# All-46 (numeric-only columns; HF dataset includes string label/attack cols)
	all_features = [
		c for c in df_train.columns
		if c not in ("label", "attack_class", "is_benign") and pd.api.types.is_numeric_dtype(df_train[c])
	]
	print(f"  all-46 feature count: {len(all_features)}")
	X_train_46 = df_train[all_features].values.astype(np.float32)

	# Impute NaN/Inf for RF / Perceptron / AdaBoost; XGBoost handles natively
	# (RF on 37M with NaN raises errors)
	imp = SimpleImputer(strategy="median")
	print("  Median-imputing NaN/Inf for RF/AdaBoost/Perceptron...")
	X_train_20_imp = imp.fit_transform(X_train_20)
	X_train_46_imp = SimpleImputer(strategy="median").fit_transform(X_train_46)

	results = []

	# --- Random Forest (top-20) ---
	print("\n[1/5] Training Random Forest (n_estimators=100, top-20)...", flush=True)
	t0 = time.time()
	rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
	rf.fit(X_train_20_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("Random Forest", rf))
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
	results.append(measure("XGBoost", xgb))
	del xgb

	# --- AdaBoost (top-20) ---
	print("\n[3/5] Training AdaBoost (n_estimators=50, top-20)...", flush=True)
	t0 = time.time()
	ab20 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
	ab20.fit(X_train_20_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("AdaBoost (top-20)", ab20))
	del ab20

	# --- AdaBoost (all 46 features) ---
	print("\n[4/5] Training AdaBoost (n_estimators=50, all 46 feat)...", flush=True)
	t0 = time.time()
	ab46 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
	ab46.fit(X_train_46_imp, y_train)
	print(f"      Train time: {time.time() - t0:.1f}s")
	results.append(measure("AdaBoost (all 46 feat)", ab46))
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
	results.append(measure("Perceptron", perc))
	del perc

	# --- Report ---
	print("\n" + "=" * 78)
	print("Model artifact sizes (deterministic given seed + sklearn version)")
	print("=" * 78)
	print(f"{'Model':<25} {'pickle':>14} {'joblib':>14}   suggestion")
	print("-" * 78)
	for r in results:
		pkl = r["pickle_bytes"]
		jl = r["joblib_bytes"]
		sug = fmt_bytes(jl)
		print(f"{r['name']:<25} {fmt_bytes(pkl):>14} {fmt_bytes(jl):>14}   {sug}")
	print()
	print("Suggestions are based on joblib byte counts (the sklearn reference for")
	print("serialized estimator size); pickle is shown as a cross-check.")
	print()


if __name__ == "__main__":
	main()
