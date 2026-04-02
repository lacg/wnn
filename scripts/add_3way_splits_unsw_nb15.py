"""
Add 80/10/10 three-way splits to the existing UNSW-NB15 HuggingFace dataset.

Adds two new configs:
- temporal_3way: Mon-Thu train, Friday split into test (50%) + validation (50%)
- random_3way: 80/10/10 from deduped data

Existing configs (temporal, standard, random) are preserved for backward compatibility.

Source: lacg030175/UNSW-NB15 (existing HF dataset)
"""

import pandas as pd
import numpy as np
from huggingface_hub import HfApi
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import tempfile

TARGET_REPO = "lacg030175/UNSW-NB15"


def main():
	print("=" * 70)
	print("Adding 3-way splits to UNSW-NB15 HuggingFace Dataset")
	print("=" * 70)

	api = HfApi()

	# Load existing temporal split
	print("\n--- Loading existing temporal split ---")
	ds_temporal = load_dataset(TARGET_REPO, "temporal")
	df_train_temporal = ds_temporal["train"].to_pandas()
	df_test_temporal = ds_temporal["test"].to_pandas()
	print(f"Temporal: {len(df_train_temporal):,} train, {len(df_test_temporal):,} test")

	# Load existing random split
	print("\n--- Loading existing random split ---")
	ds_random = load_dataset(TARGET_REPO, "random")
	df_train_random = ds_random["train"].to_pandas()
	df_test_random = ds_random["test"].to_pandas()
	print(f"Random: {len(df_train_random):,} train, {len(df_test_random):,} test")

	# Create temporal_3way: split test into test (50%) + validation (50%)
	print("\n--- Creating temporal_3way (test/val from Friday) ---")
	df_test_t3w, df_val_t3w = train_test_split(
		df_test_temporal, test_size=0.5, random_state=42, stratify=df_test_temporal["label"]
	)
	print(f"Temporal 3-way: {len(df_train_temporal):,} train, {len(df_test_t3w):,} test, {len(df_val_t3w):,} val")

	# Create random_3way: 80/10/10 from combined random data
	print("\n--- Creating random_3way (80/10/10) ---")
	df_all_random = pd.concat([df_train_random, df_test_random], ignore_index=True)
	df_r3_train, df_r3_remaining = train_test_split(
		df_all_random, test_size=0.2, random_state=42, stratify=df_all_random["label"]
	)
	df_r3_test, df_r3_val = train_test_split(
		df_r3_remaining, test_size=0.5, random_state=42, stratify=df_r3_remaining["label"]
	)
	print(f"Random 3-way: {len(df_r3_train):,} train, {len(df_r3_test):,} test, {len(df_r3_val):,} val")

	# Upload new splits only (don't overwrite existing)
	print("\n--- Uploading new splits to HuggingFace ---")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# Save temporal_3way
		t3w_dir = tmpdir / "temporal_3way"
		t3w_dir.mkdir()
		df_train_temporal.to_parquet(t3w_dir / "train-00000-of-00001.parquet", index=False)
		df_test_t3w.to_parquet(t3w_dir / "test-00000-of-00001.parquet", index=False)
		df_val_t3w.to_parquet(t3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Save random_3way
		r3w_dir = tmpdir / "random_3way"
		r3w_dir.mkdir()
		df_r3_train.to_parquet(r3w_dir / "train-00000-of-00001.parquet", index=False)
		df_r3_test.to_parquet(r3w_dir / "test-00000-of-00001.parquet", index=False)
		df_r3_val.to_parquet(r3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Upload data files (don't overwrite README — we'll update it separately)
		print("Uploading temporal_3way...")
		api.upload_folder(
			folder_path=str(t3w_dir),
			path_in_repo="temporal_3way",
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

		print("Uploading random_3way...")
		api.upload_folder(
			folder_path=str(r3w_dir),
			path_in_repo="random_3way",
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	# Update README to add new configs
	print("\nUpdating README with new configs...")
	readme = api.hf_hub_download(repo_id=TARGET_REPO, filename="README.md", repo_type="dataset")
	with open(readme) as f:
		content = f.read()

	# Find the configs section and add new configs before existing ones
	old_configs = """configs:
  - config_name: temporal
    data_files:
      - split: train
        path: temporal/train-*
      - split: test
        path: temporal/test-*
    default: true
  - config_name: standard
    data_files:
      - split: train
        path: temporal/train-*
      - split: test
        path: temporal/test-*
  - config_name: random
    data_files:
      - split: train
        path: random/train-*
      - split: test
        path: random/test-*"""

	new_configs = """configs:
  - config_name: temporal_3way
    data_files:
      - split: train
        path: temporal_3way/train-*
      - split: test
        path: temporal_3way/test-*
      - split: validation
        path: temporal_3way/validation-*
    default: true
  - config_name: random_3way
    data_files:
      - split: train
        path: random_3way/train-*
      - split: test
        path: random_3way/test-*
      - split: validation
        path: random_3way/validation-*
  - config_name: temporal
    data_files:
      - split: train
        path: temporal/train-*
      - split: test
        path: temporal/test-*
  - config_name: standard
    data_files:
      - split: train
        path: temporal/train-*
      - split: test
        path: temporal/test-*
  - config_name: random
    data_files:
      - split: train
        path: random/train-*
      - split: test
        path: random/test-*"""

	if old_configs in content:
		content = content.replace(old_configs, new_configs)
		with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
			f.write(content)
			tmp_readme = f.name
		api.upload_file(
			path_or_fileobj=tmp_readme,
			path_in_repo="README.md",
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)
		print("README updated with new configs")
	else:
		print("WARNING: Could not find configs section in README — manual update needed")

	print(f"\n✓ Done! 3-way splits added to: https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
	main()
