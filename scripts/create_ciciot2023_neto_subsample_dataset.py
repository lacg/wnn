"""Build lacg030175/CIC-IoT-2023-neto-subsample HF dataset (1.3M from full 46.7M).

Same subsampling strategy as the original lacg030175/CIC-IoT-2023:
  - 200K benign + 50K × N attack subclasses
  - Stratified by attack_class

But sourced from the full 46.7M Kaggle CSV (akashdogra/ciciot23csv) so it has
the canonical 46-feature schema (vs bencorn's 39-feature MERGED).

Drop-in replacement for `lacg030175/CIC-IoT-2023` for new PUB50-class flows
that want the canonical feature set.

Source: /Users/lacg/wnn/.cache/kaggle_ciciot_full/ciciot23.csv  (13.75 GB)
Target: lacg030175/CIC-IoT-2023-neto-subsample
"""

import json
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Reuse build script's loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_ciciot2023_neto_full_dataset import (
	load_and_translate, coerce_numeric, report_stats, compute_rf_importance,
	NETO_LABEL_MAP, SOURCE_CSV,
)

TARGET_REPO = "lacg030175/CIC-IoT-2023-neto-subsample"

# Same subsample sizes as original lacg030175/CIC-IoT-2023
N_BENIGN = 200_000
N_PER_ATTACK = 50_000


def subsample(df: pd.DataFrame) -> pd.DataFrame:
	"""Subsample stratified by attack_class: N_BENIGN benign + N_PER_ATTACK per attack subclass."""
	rng_seed = 42
	parts = []
	# Benign first
	benign = df[df["attack_class"] == "Benign"]
	if len(benign) > N_BENIGN:
		parts.append(benign.sample(N_BENIGN, random_state=rng_seed))
	else:
		parts.append(benign)
	# Each attack subclass — sample by Label (the specific subclass), not attack_class group
	for sub_label in df.loc[df["attack_class"] != "Benign", "Label"].unique():
		sub = df[df["Label"] == sub_label]
		if len(sub) > N_PER_ATTACK:
			parts.append(sub.sample(N_PER_ATTACK, random_state=rng_seed))
		else:
			parts.append(sub)
	out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=rng_seed)
	print(f"\nSubsampled: {len(out):,} rows from {len(df):,} (stratified)")
	for cls in sorted(out["attack_class"].unique()):
		count = (out["attack_class"] == cls).sum()
		print(f"  {cls:<15}: {count:>9,}")
	return out


def main():
	print("=" * 78)
	print("Building lacg030175/CIC-IoT-2023-neto-subsample (1.3M from 46.7M Kaggle CSV)")
	print("=" * 78)
	t0 = time.time()

	# Load full + translate (~30-40 GB peak)
	df = load_and_translate(SOURCE_CSV)
	feature_cols = coerce_numeric(df)
	print(f"\nFeatures coerced: {len(feature_cols)}")

	# Subsample to ~1.3M
	df_sub = subsample(df)
	del df
	gc.collect()
	print(f"\nFreed parent DataFrame, working with subsampled {len(df_sub):,} rows")

	# Reapply numeric coerce on subsample (in case sample dtype shifted)
	for col in feature_cols:
		df_sub[col] = pd.to_numeric(df_sub[col], errors="coerce")

	report_stats(df_sub, feature_cols)
	top20, ranked = compute_rf_importance(df_sub, feature_cols)

	# Splits: 80/10/10 stratified
	print(f"\nCreating 80/10/10 stratified split...")
	df_train, df_remaining = train_test_split(df_sub, test_size=0.2, random_state=42, stratify=df_sub["label"])
	df_test, df_val = train_test_split(df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"])
	del df_remaining
	gc.collect()
	print(f"  Train: {len(df_train):,} | Test: {len(df_test):,} | Val: {len(df_val):,}")

	df_rand_test = pd.concat([df_test, df_val], ignore_index=True)

	# Upload
	print(f"\nUploading to {TARGET_REPO}...")
	api = HfApi()
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo create: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		r3w = tmpdir / "random_3way"; r3w.mkdir()
		df_train.to_parquet(r3w / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(r3w / "test-00000-of-00001.parquet", index=False)
		df_val.to_parquet(r3w / "validation-00000-of-00001.parquet", index=False)

		r = tmpdir / "random"; r.mkdir()
		df_train.to_parquet(r / "train-00000-of-00001.parquet", index=False)
		df_rand_test.to_parquet(r / "test-00000-of-00001.parquet", index=False)

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

Stratified subsample (~{len(df_sub):,} rows) of the canonical Neto 46.7M
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
		for cls in sorted(df_sub["attack_class"].unique()):
			count = (df_sub["attack_class"] == cls).sum()
			readme += f"- {cls}: {count:,} ({count/len(df_sub)*100:.2f}%)\n"

		(tmpdir / "README.md").write_text(readme)

		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
			commit_message=f"Initial: 1.3M subsample of canonical Neto ({len(df_sub):,} rows × {len(feature_cols)} features)",
		)

	print(f"\n✓ Done! Available at: https://huggingface.co/datasets/{TARGET_REPO}")
	print(f"  Subsampled rows: {len(df_sub):,}, features: {len(feature_cols)}")
	print(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
