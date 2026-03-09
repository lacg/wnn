"""
Phase A1: Explore UNSW-NB15 dataset
- Feature distributions, types, missing values
- Class balance (binary and multi-class)
- Feature statistics for thermometer encoding design
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "unsw-nb15"

# ── Load both sources ──────────────────────────────────────────────────
print("=" * 70)
print("UNSW-NB15 Dataset Exploration")
print("=" * 70)

# CSV training set (original format)
csv_train = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
print(f"\nCSV training set: {csv_train.shape}")
print(f"  Columns: {list(csv_train.columns)}")

# Parquet train/test (HuggingFace mirror)
pq_train = pd.read_parquet(DATA_DIR / "train.parquet")
pq_test = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"\nParquet train: {pq_train.shape}")
print(f"Parquet test:  {pq_test.shape}")
print(f"  Columns: {list(pq_train.columns)}")

# ── Check if parquet columns match CSV ─────────────────────────────────
csv_cols = set(csv_train.columns)
pq_cols = set(pq_train.columns)
print(f"\n--- Column comparison ---")
print(f"In CSV only:     {csv_cols - pq_cols}")
print(f"In parquet only: {pq_cols - csv_cols}")

# ── Use CSV training set as reference for exploration ──────────────────
df = csv_train
print(f"\n{'=' * 70}")
print("FEATURE ANALYSIS (CSV training set)")
print(f"{'=' * 70}")

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Data types and missing values ──────────────────────────────────────
print(f"\n--- Data Types & Missing Values ---")
for col in df.columns:
	dtype = df[col].dtype
	n_missing = df[col].isna().sum()
	n_unique = df[col].nunique()
	missing_pct = 100 * n_missing / len(df) if n_missing > 0 else 0
	missing_str = f" ({missing_pct:.1f}% missing)" if n_missing > 0 else ""
	print(f"  {col:25s} {str(dtype):10s} {n_unique:8d} unique{missing_str}")

# ── Class distribution (binary) ────────────────────────────────────────
print(f"\n--- Binary Class Distribution ---")
if "label" in df.columns:
	label_col = "label"
elif "Label" in df.columns:
	label_col = "Label"
else:
	label_col = None
	print("  WARNING: No label column found!")

if label_col:
	vc = df[label_col].value_counts().sort_index()
	total = len(df)
	for val, count in vc.items():
		print(f"  {val} ({'Normal' if val == 0 else 'Attack'}): {count:>8,} ({100*count/total:.1f}%)")

# ── Class distribution (multi-class) ──────────────────────────────────
print(f"\n--- Multi-Class Distribution ---")
if "attack_cat" in df.columns:
	cat_col = "attack_cat"
elif "Attack_cat" in df.columns:
	cat_col = "Attack_cat"
else:
	# Try to find it
	cat_candidates = [c for c in df.columns if "attack" in c.lower() or "cat" in c.lower()]
	cat_col = cat_candidates[0] if cat_candidates else None

if cat_col:
	# Clean up: fill NaN with 'Normal' for normal traffic
	cats = df[cat_col].fillna("Normal").str.strip()
	cats = cats.replace("", "Normal")
	vc = cats.value_counts()
	total = len(df)
	print(f"  {'Category':20s} {'Count':>8s} {'Pct':>6s}")
	print(f"  {'-'*20} {'-'*8} {'-'*6}")
	for cat, count in vc.items():
		print(f"  {cat:20s} {count:>8,} {100*count/total:>5.1f}%")
else:
	print("  WARNING: No attack category column found!")

# ── Feature statistics for thermometer encoding ────────────────────────
print(f"\n{'=' * 70}")
print("FEATURE STATISTICS (for thermometer encoding design)")
print(f"{'=' * 70}")

# Identify feature types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove label/id columns from features
exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
numeric_features = [c for c in numeric_cols if c not in exclude]
categorical_features = [c for c in categorical_cols if c not in exclude]

print(f"\nNumeric features ({len(numeric_features)}):")
print(f"  {'Feature':20s} {'Min':>12s} {'Max':>12s} {'Mean':>12s} {'Std':>12s} {'Zeros%':>7s} {'Uniq':>8s}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*7} {'-'*8}")
for col in numeric_features:
	s = df[col]
	zeros_pct = 100 * (s == 0).sum() / len(s)
	print(f"  {col:20s} {s.min():>12.2f} {s.max():>12.2f} {s.mean():>12.2f} {s.std():>12.2f} {zeros_pct:>6.1f}% {s.nunique():>8d}")

print(f"\nCategorical features ({len(categorical_features)}):")
for col in categorical_features:
	vals = df[col].fillna("(null)")
	n_unique = vals.nunique()
	top3 = vals.value_counts().head(3)
	top3_str = ", ".join(f"{v}={c}" for v, c in top3.items())
	bits_needed = int(np.ceil(np.log2(max(n_unique, 2))))
	print(f"  {col:20s} {n_unique:4d} values ({bits_needed} bits)  Top: {top3_str}")

# ── Binary features (already 0/1) ─────────────────────────────────────
print(f"\n--- Binary Features (0/1 only) ---")
binary_features = []
for col in numeric_features:
	unique_vals = set(df[col].dropna().unique())
	if unique_vals <= {0, 1, 0.0, 1.0}:
		ones_pct = 100 * df[col].sum() / len(df)
		binary_features.append(col)
		print(f"  {col:25s}  1s: {ones_pct:.1f}%")

# ── Summary for encoding design ───────────────────────────────────────
non_binary_numeric = [c for c in numeric_features if c not in binary_features]
print(f"\n{'=' * 70}")
print("ENCODING SUMMARY")
print(f"{'=' * 70}")
print(f"  Numeric features (need thermometer): {len(non_binary_numeric)}")
print(f"  Binary features (pass-through):      {len(binary_features)}")
print(f"  Categorical features (one-hot/bin):   {len(categorical_features)}")
print(f"  Total features:                       {len(non_binary_numeric) + len(binary_features) + len(categorical_features)}")

# Estimate total bits
bits_thermo = len(non_binary_numeric) * 8  # 8 bits per numeric feature
bits_binary = len(binary_features)
bits_categorical = sum(int(np.ceil(np.log2(max(df[c].nunique(), 2)))) for c in categorical_features)
total_bits = bits_thermo + bits_binary + bits_categorical
print(f"\n  Estimated total binary input bits:")
print(f"    Thermometer ({len(non_binary_numeric)} features × 8 bits): {bits_thermo}")
print(f"    Binary ({len(binary_features)} features × 1 bit):     {bits_binary}")
print(f"    Categorical ({len(categorical_features)} features):         {bits_categorical}")
print(f"    TOTAL:                                    {total_bits} bits")

# ── Also check parquet test set distribution ───────────────────────────
print(f"\n{'=' * 70}")
print("PARQUET TEST SET CHECK")
print(f"{'=' * 70}")
print(f"Shape: {pq_test.shape}")
print(f"Columns: {list(pq_test.columns[:10])}...")

# Find label column in parquet
for col in pq_test.columns:
	if col.lower() == "label":
		vc = pq_test[col].value_counts().sort_index()
		total = len(pq_test)
		print(f"\nLabel distribution ('{col}'):")
		for val, count in vc.items():
			print(f"  {val}: {count:>8,} ({100*count/total:.1f}%)")
		break

# Find attack category in parquet
for col in pq_test.columns:
	if "attack" in col.lower():
		cats = pq_test[col].fillna("Normal").astype(str).str.strip()
		cats = cats.replace("", "Normal")
		vc = cats.value_counts()
		total = len(pq_test)
		print(f"\nAttack categories ('{col}'):")
		for cat, count in vc.items():
			print(f"  {cat:20s} {count:>8,} ({100*count/total:.1f}%)")
		break
