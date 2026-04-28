"""Build lacg030175/CIC-IoT-2023-neto-subsample HF dataset (1.3M from full 46.7M).

Same subsampling strategy as the original lacg030175/CIC-IoT-2023:
  - 200K benign + 50K × N attack subclasses
  - Stratified by attack_class (Label-level granularity)

Source: /Users/lacg/wnn/.cache/kaggle_ciciot_full/ciciot23.csv  (13.75 GB)
Target: lacg030175/CIC-IoT-2023-neto-subsample

Uses polars for memory-efficient processing (~3-5 GB peak).
"""

import json
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier

# Reuse loader functions from full build script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_ciciot2023_neto_full_dataset import (
	load_and_translate, coerce_numeric, report_stats,
	compute_rf_importance, stratified_split, SOURCE_CSV,
)

TARGET_REPO = "lacg030175/CIC-IoT-2023-neto-subsample"

N_BENIGN = 200_000
N_PER_ATTACK = 50_000


def subsample(df: pl.DataFrame) -> pl.DataFrame:
	"""Stratified subsample: N_BENIGN benign + N_PER_ATTACK per attack subclass (Label-level)."""
	parts = []
	# Benign first
	benign = df.filter(pl.col("attack_class") == "Benign")
	if benign.height > N_BENIGN:
		parts.append(benign.sample(n=N_BENIGN, seed=42))
	else:
		parts.append(benign)
	# Each attack subclass — sample by Label (specific subclass), not attack_class group
	attack_labels = df.filter(pl.col("attack_class") != "Benign")["Label"].unique().to_list()
	for sub_label in attack_labels:
		sub = df.filter(pl.col("Label") == sub_label)
		if sub.height > N_PER_ATTACK:
			parts.append(sub.sample(n=N_PER_ATTACK, seed=42))
		else:
			parts.append(sub)
	out = pl.concat(parts).sample(fraction=1.0, seed=42, shuffle=True)
	print(f"\nSubsampled: {out.height:,} rows from {df.height:,} (stratified)")
	dist = out.group_by("attack_class").agg(pl.len().alias("count")).sort("count", descending=True)
	for cls, count in dist.iter_rows():
		print(f"  {cls:<15}: {count:>9,}")
	return out


def main():
	print("=" * 78)
	print("Building lacg030175/CIC-IoT-2023-neto-subsample (1.3M from 46.7M Kaggle CSV)")
	print("=" * 78)
	t0 = time.time()

	df = load_and_translate(SOURCE_CSV)
	df, feature_cols = coerce_numeric(df)
	print(f"\nFeatures coerced to Float32: {len(feature_cols)}")

	df_sub = subsample(df)
	del df
	gc.collect()
	print(f"\nFreed full DataFrame, working with subsampled {df_sub.height:,} rows")

	report_stats(df_sub, feature_cols)
	top20, ranked = compute_rf_importance(df_sub, feature_cols)

	print(f"\nCreating 80/10/10 stratified split...")
	df_train, df_test, df_val = stratified_split(df_sub)
	print(f"  Train: {df_train.height:,} | Test: {df_test.height:,} | Val: {df_val.height:,}")

	df_rand_test = pl.concat([df_test, df_val])

	# Snapshot for README
	n_total = df_sub.height
	class_dist = df_sub.group_by("attack_class").agg(pl.len().alias("count")).sort("count", descending=True)
	del df_sub
	gc.collect()

	print(f"\nUploading to {TARGET_REPO}...")
	api = HfApi()
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo create: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		r3w = tmpdir / "random_3way"; r3w.mkdir()
		df_train.write_parquet(r3w / "train-00000-of-00001.parquet")
		df_test.write_parquet(r3w / "test-00000-of-00001.parquet")
		df_val.write_parquet(r3w / "validation-00000-of-00001.parquet")

		r = tmpdir / "random"; r.mkdir()
		df_train.write_parquet(r / "train-00000-of-00001.parquet")
		df_rand_test.write_parquet(r / "test-00000-of-00001.parquet")

		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump({"top20": top20, "all_ranked": [(ft, float(i)) for ft, i in ranked]}, f, indent=2)

		readme = f"""---
language: [en]
license: cc-by-4.0
tags: [network-security, intrusion-detection, iot, nids, ciciot, neto, subsample]
configs:
- config_name: random_3way
  data_files:
  - {{split: train, path: random_3way/train-*.parquet}}
  - {{split: test, path: random_3way/test-*.parquet}}
  - {{split: validation, path: random_3way/validation-*.parquet}}
- config_name: random
  data_files:
  - {{split: train, path: random/train-*.parquet}}
  - {{split: test, path: random/test-*.parquet}}
---

# CIC-IoT-2023 — Neto-Subsample (1.3M, 46-feature canonical)

Stratified subsample (~{n_total:,} rows) of the canonical Neto 46.7M
dataset (lacg030175/CIC-IoT-2023-neto-full). Same 46-feature schema as the
full version. Drop-in replacement for `lacg030175/CIC-IoT-2023` (1.3M
bencorn-derived, 39 features) for new experiments needing the canonical
feature set.

Subsample composition:
- Benign: {N_BENIGN:,} rows
- Each attack subclass: up to {N_PER_ATTACK:,} rows

NaN/Inf preserved (no dropna). Pair with
`ThermometerEncoder(invalid_encoding="single_bit")`.

## Splits
- `random_3way`: 80% train / 10% test / 10% validation (stratified on binary label, seed=42)
- `random`: 80% train / 20% test (test = test ∪ validation from random_3way)

## Provenance
- Subsampled from `lacg030175/CIC-IoT-2023-neto-full` (46.7M canonical)
- Originally from `akashdogra/ciciot23csv` on Kaggle, derived from CIC's
  official 169-file distribution (Neto et al., 2023).

## Class distribution
"""
		for cls, count in class_dist.iter_rows():
			readme += f"- {cls}: {count:,} ({count/n_total*100:.2f}%)\n"

		(tmpdir / "README.md").write_text(readme)

		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
			commit_message=f"Initial: 1.3M subsample of canonical Neto ({n_total:,} rows × {len(feature_cols)} features)",
		)

	print(f"\n✓ Done! Available at: https://huggingface.co/datasets/{TARGET_REPO}")
	print(f"  Subsampled rows: {n_total:,}, features: {len(feature_cols)}")
	print(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
