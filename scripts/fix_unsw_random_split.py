"""
Fix the UNSW-NB15 'random' HuggingFace config from 90/10 to 80/20.

The existing 'random' config has a 90/10 train/test split, while
CIC-IoT-2023 'random' uses 80/20. For paper consistency, we want
80/20 on both datasets.

Strategy: reuse 'random_3way' data (which is the correct 80/10/10
split) — the new 'random' becomes:
  - train = random_3way train (80%, identical rows)
  - test  = random_3way test + val merged (20%)

This guarantees identical training data between 'random' and
'random_3way', so experiments on either config are directly comparable.

Usage:
    source wnn/bin/activate
    python scripts/fix_unsw_random_split.py              # preview
    python scripts/fix_unsw_random_split.py --upload      # upload to HF
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
from datasets import load_dataset


TARGET_REPO = "lacg030175/UNSW-NB15"


def main():
	parser = argparse.ArgumentParser(description="Fix UNSW random split to 80/20")
	parser.add_argument("--upload", action="store_true", help="Actually upload to HF (default: preview only)")
	args = parser.parse_args()

	print("=" * 60)
	print("Fixing UNSW-NB15 'random' config: 90/10 → 80/20")
	print("=" * 60)

	# Show current state
	print("\n--- Current 'random' split (90/10) ---")
	ds_old = load_dataset(TARGET_REPO, "random")
	for split, data in ds_old.items():
		print(f"  {split}: {len(data):,} rows")
	total = sum(len(d) for d in ds_old.values())
	print(f"  Total: {total:,}")
	print(f"  Split: {len(ds_old['train'])/total*100:.0f}/{len(ds_old['test'])/total*100:.0f}")

	# Load random_3way
	print("\n--- Loading 'random_3way' (80/10/10) ---")
	ds_3way = load_dataset(TARGET_REPO, "random_3way")
	for split, data in ds_3way.items():
		print(f"  {split}: {len(data):,} rows")

	# New random = train from 3way + test+val merged from 3way
	df_train = ds_3way["train"].to_pandas()
	df_test = pd.concat([
		ds_3way["test"].to_pandas(),
		ds_3way["validation"].to_pandas(),
	], ignore_index=True)

	new_total = len(df_train) + len(df_test)
	print(f"\n--- New 'random' split (80/20) ---")
	print(f"  train: {len(df_train):,} ({len(df_train)/new_total*100:.1f}%)")
	print(f"  test:  {len(df_test):,} ({len(df_test)/new_total*100:.1f}%)")
	print(f"  Total: {new_total:,}")

	# Verify label distribution matches
	for label, df, name in [(0, df_train, "train"), (0, df_test, "test")]:
		n_normal = (df["label"] == 0).sum()
		n_attack = (df["label"] == 1).sum()
		print(f"  {name}: {n_normal:,} normal / {n_attack:,} attack ({n_attack/len(df)*100:.1f}% attack)")

	if not args.upload:
		print("\n[Preview only — pass --upload to actually upload to HuggingFace]")
		return

	# Upload
	from huggingface_hub import HfApi
	api = HfApi()

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		random_dir = tmpdir / "random"
		random_dir.mkdir()

		print("\n--- Saving parquet files ---")
		train_path = random_dir / "train-00000-of-00001.parquet"
		test_path = random_dir / "test-00000-of-00001.parquet"
		df_train.to_parquet(train_path, index=False)
		df_test.to_parquet(test_path, index=False)
		print(f"  train: {train_path.stat().st_size / 1024 / 1024:.1f} MB")
		print(f"  test:  {test_path.stat().st_size / 1024 / 1024:.1f} MB")

		print("\n--- Uploading to HuggingFace ---")
		api.upload_folder(
			folder_path=str(random_dir),
			path_in_repo="random",
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	print(f"\n✓ Done! 'random' config updated to 80/20 on: https://huggingface.co/datasets/{TARGET_REPO}")
	print("  Train data is identical to random_3way train.")
	print("  Test data is random_3way test + val merged.")


if __name__ == "__main__":
	main()
