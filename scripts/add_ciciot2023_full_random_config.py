"""
Add a `random` (80/20) config to the existing CIC-IoT-2023-full dataset.

The full 46M dataset (lacg030175/CIC-IoT-2023-full) currently only has a
random_3way (80/10/10) config. This script:

1. Downloads the existing random_3way parquets (train, test, validation)
2. Concatenates all three back into the full 46M dataset
3. Re-splits as stratified random 80/20
4. Uploads as a new `random` config to the same repo

This avoids re-downloading the source 13GB CSVs from bencorn/CIC-IoT-2023.

Why a separate 80/20 config: matches the 1.3M subsample dataset's `random`
config, and is the standard protocol used by the existing PUB50 ciciot
runs (K-fold on 80% train, 20% held out).

Run from project root:
    python scripts/add_ciciot2023_full_random_config.py
"""

import gc
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from sklearn.model_selection import train_test_split

REPO_ID = "lacg030175/CIC-IoT-2023-full"


def main():
	print("=" * 70)
	print(f"Adding `random` (80/20) config to {REPO_ID}")
	print("=" * 70)

	api = HfApi()

	# Verify repo exists
	try:
		api.dataset_info(REPO_ID)
	except Exception as e:
		print(f"ERROR: Cannot access {REPO_ID}: {e}")
		sys.exit(1)

	# 1. Download existing random_3way parquets
	print(f"\nDownloading existing random_3way parquets from {REPO_ID}...")
	cache_dir = snapshot_download(
		repo_id=REPO_ID,
		repo_type="dataset",
		allow_patterns=["random_3way/*.parquet", "feature_importance.json", "README.md"],
	)
	cache_path = Path(cache_dir)
	print(f"  Cached to: {cache_path}")

	# Find parquet files
	r3w_dir = cache_path / "random_3way"
	if not r3w_dir.exists():
		print(f"ERROR: random_3way directory not found at {r3w_dir}")
		sys.exit(1)

	train_files = sorted(r3w_dir.glob("train-*.parquet"))
	test_files = sorted(r3w_dir.glob("test-*.parquet"))
	val_files = sorted(r3w_dir.glob("validation-*.parquet"))
	print(f"  Found: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val parquet files")

	# 2. Load all three splits and concatenate
	print(f"\nLoading parquet files into memory...")
	dfs = []
	for label, files in (("train", train_files), ("test", test_files), ("validation", val_files)):
		for f in files:
			df = pd.read_parquet(f)
			print(f"  {label}: {f.name} -> {len(df):,} rows")
			dfs.append(df)

	print(f"\nConcatenating into full dataset...")
	df_all = pd.concat(dfs, ignore_index=True)
	del dfs
	gc.collect()
	print(f"  Total: {len(df_all):,} rows")

	# Sanity check the binary label distribution
	if "label" not in df_all.columns:
		print(f"ERROR: 'label' column missing from parquets. Columns: {list(df_all.columns)[:20]}")
		sys.exit(1)
	n_benign = int((df_all["label"] == 0).sum())
	n_attack = int((df_all["label"] == 1).sum())
	print(f"  Binary: {n_benign:,} benign ({n_benign/len(df_all)*100:.2f}%), "
		  f"{n_attack:,} attack ({n_attack/len(df_all)*100:.2f}%)")

	# 3. Stratified random 80/20 split
	print(f"\nCreating stratified random 80/20 split (seed=42)...")
	df_train, df_test = train_test_split(
		df_all,
		test_size=0.2,
		random_state=42,
		stratify=df_all["label"],
	)
	del df_all
	gc.collect()
	print(f"  Train: {len(df_train):,} rows ({len(df_train)/(len(df_train)+len(df_test))*100:.1f}%)")
	print(f"  Test:  {len(df_test):,} rows ({len(df_test)/(len(df_train)+len(df_test))*100:.1f}%)")

	# 4. Write parquets and upload
	print(f"\nUploading new `random` config to {REPO_ID}...")
	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		random_dir = tmpdir / "random"
		random_dir.mkdir()

		print(f"  Writing train parquet...")
		df_train.to_parquet(random_dir / "train-00000-of-00001.parquet", index=False)
		del df_train
		gc.collect()

		print(f"  Writing test parquet...")
		df_test.to_parquet(random_dir / "test-00000-of-00001.parquet", index=False)
		del df_test
		gc.collect()

		# Upload only the new random/ directory
		print(f"  Uploading to HuggingFace...")
		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=REPO_ID,
			repo_type="dataset",
			commit_message="Add random (80/20) config",
		)

	# 5. Patch the README to register the new config
	# The README currently declares only random_3way; we need to add random.
	print(f"\nUpdating README to register the new `random` config...")
	readme_path = cache_path / "README.md"
	if readme_path.exists():
		readme_text = readme_path.read_text()
		# Add `random` config block to the YAML frontmatter if not already present
		if "config_name: random\n" not in readme_text:
			# Find the configs: section and append the new config
			marker = "configs:\n"
			if marker in readme_text:
				# Insert the new config right after the configs: line
				new_config_block = (
					"  - config_name: random\n"
					"    data_files:\n"
					"      - split: train\n"
					"        path: random/train-*\n"
					"      - split: test\n"
					"        path: random/test-*\n"
				)
				readme_text = readme_text.replace(marker, marker + new_config_block, 1)
				with tempfile.TemporaryDirectory() as tmpdir:
					tmpdir = Path(tmpdir)
					(tmpdir / "README.md").write_text(readme_text)
					api.upload_folder(
						folder_path=str(tmpdir),
						repo_id=REPO_ID,
						repo_type="dataset",
						commit_message="Register random config in README",
					)
				print(f"  README updated.")
			else:
				print(f"  WARNING: 'configs:' marker not found in README, skipping update.")
		else:
			print(f"  README already has random config registered.")
	else:
		print(f"  WARNING: README.md not found in cache.")

	print(f"\n✓ Done! New config available at: https://huggingface.co/datasets/{REPO_ID}/tree/main/random")
	print(f"  Usage: load_dataset('{REPO_ID}', 'random')")


if __name__ == "__main__":
	main()
