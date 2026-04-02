"""
Create CICIDS2017 HuggingFace dataset with temporal and random splits.

Temporal: Mon-Thu train, Friday test (realistic deployment scenario)
Random: 80/20 stratified random split (for literature comparison)

Source: c01dsnap/CIC-IDS2017 on HuggingFace (raw CSVs from UNB CIC)
Target: lacg030175/CICIDS2017 on HuggingFace
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import tempfile
import json

SOURCE_REPO = "c01dsnap/CIC-IDS2017"
TARGET_REPO = "lacg030175/CICIDS2017"

# Files organized by day
TRAIN_FILES = {
	"Monday": "Monday-WorkingHours.pcap_ISCX.csv",
	"Tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
	"Wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
	"Thursday-AM": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
	"Thursday-PM": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
}

TEST_FILES = {
	"Friday-AM": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
	"Friday-DDos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
	"Friday-PortScan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}

# Attack types for multi-class
ATTACK_CATEGORIES = [
	"BENIGN",
	"Bot",
	"DDoS",
	"DoS GoldenEye",
	"DoS Hulk",
	"DoS Slowhttptest",
	"DoS slowloris",
	"FTP-Patator",
	"Heartbleed",
	"Infiltration",
	"PortScan",
	"SSH-Patator",
	"Web Attack - Brute Force",
	"Web Attack - SQL Injection",
	"Web Attack - XSS",
]


def load_csv(filename: str) -> pd.DataFrame:
	"""Download and load a CICIDS2017 CSV file."""
	path = hf_hub_download(repo_id=SOURCE_REPO, filename=filename, repo_type="dataset")
	df = pd.read_csv(path, low_memory=False)
	# Strip whitespace from column names
	df.columns = df.columns.str.strip()
	# Strip whitespace from label
	df["Label"] = df["Label"].astype(str).str.strip()
	return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean CICIDS2017 data: handle inf, NaN, duplicates."""
	# Replace inf with NaN
	df = df.replace([np.inf, -np.inf], np.nan)

	# Drop rows with NaN in any feature column
	feature_cols = [c for c in df.columns if c != "Label"]
	n_before = len(df)
	df = df.dropna(subset=feature_cols)
	n_after = len(df)
	if n_before != n_after:
		print(f"  Dropped {n_before - n_after:,} rows with NaN/inf values")

	# Convert all feature columns to float64
	for col in feature_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=feature_cols)

	# Add binary label
	df["label"] = (df["Label"] != "BENIGN").astype(int)

	# Normalize attack labels (handle encoding issues with Web Attack labels)
	df["Label"] = df["Label"].str.replace("Â\xa0", " ", regex=False)
	df["Label"] = df["Label"].str.replace("\xa0", " ", regex=False)
	df["Label"] = df["Label"].str.replace("ï»¿", "", regex=False)

	return df


def compute_rf_importance(df_train: pd.DataFrame, feature_cols: list[str], top_n: int = 20):
	"""Compute Random Forest feature importance and return top-N features."""
	print(f"Computing RF feature importance on {len(df_train):,} samples...")

	# Subsample for speed if dataset is large
	if len(df_train) > 100_000:
		sample = df_train.sample(100_000, random_state=42)
	else:
		sample = df_train

	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values

	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)

	importances = rf.feature_importances_
	ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

	print(f"\nTop-{top_n} RF Features:")
	for i, (feat, imp) in enumerate(ranked[:top_n]):
		print(f"  {i+1:2d}. {feat:<35} {imp:.6f}")

	top_features = [feat for feat, _ in ranked[:top_n]]
	return top_features, ranked


def main():
	print("=" * 70)
	print("Creating CICIDS2017 HuggingFace Dataset")
	print("=" * 70)

	# Load all files
	print("\n--- Loading train files (Mon-Thu) ---")
	train_dfs = []
	for day, fname in TRAIN_FILES.items():
		print(f"Loading {day}...")
		df = load_csv(fname)
		df = clean_dataframe(df)
		print(f"  {day}: {len(df):,} rows")
		train_dfs.append(df)

	print("\n--- Loading test files (Friday) ---")
	test_dfs = []
	for day, fname in TEST_FILES.items():
		print(f"Loading {day}...")
		df = load_csv(fname)
		df = clean_dataframe(df)
		print(f"  {day}: {len(df):,} rows")
		test_dfs.append(df)

	df_train = pd.concat(train_dfs, ignore_index=True)
	df_test = pd.concat(test_dfs, ignore_index=True)

	print(f"\nTemporal split: {len(df_train):,} train (Mon-Thu), {len(df_test):,} test (Friday)")
	print(f"Train attacks: {df_train['label'].sum():,} / {len(df_train):,} ({df_train['label'].mean()*100:.1f}%)")
	print(f"Test attacks:  {df_test['label'].sum():,} / {len(df_test):,} ({df_test['label'].mean()*100:.1f}%)")

	# Feature columns (exclude labels)
	exclude = {"Label", "label"}
	feature_cols = sorted(set(df_train.columns) - exclude)
	print(f"Features: {len(feature_cols)}")

	# Compute RF importance
	top20, all_ranked = compute_rf_importance(df_train, feature_cols)

	# Save feature importance
	importance_data = {
		"top20": top20,
		"all_ranked": [(feat, float(imp)) for feat, imp in all_ranked],
	}

	# Create random split from combined data
	print("\n--- Creating random split ---")
	df_all = pd.concat([df_train, df_test], ignore_index=True)
	df_rand_train, df_rand_test = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	print(f"Random split: {len(df_rand_train):,} train, {len(df_rand_test):,} test")

	# Create 80/10/10 three-way splits
	print("\n--- Creating 80/10/10 three-way splits ---")
	# Temporal 3-way: split Friday test into test (50%) + validation (50%)
	df_test_t3w, df_val_t3w = train_test_split(
		df_test, test_size=0.5, random_state=42, stratify=df_test["label"]
	)
	print(f"Temporal 3-way: {len(df_train):,} train, {len(df_test_t3w):,} test, {len(df_val_t3w):,} val")

	# Random 3-way: 80/10/10 from all data
	df_rand3_train, df_rand3_remaining = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	df_rand3_test, df_rand3_val = train_test_split(
		df_rand3_remaining, test_size=0.5, random_state=42, stratify=df_rand3_remaining["label"]
	)
	print(f"Random 3-way: {len(df_rand3_train):,} train, {len(df_rand3_test):,} test, {len(df_rand3_val):,} val")

	# Upload to HuggingFace
	print("\n--- Uploading to HuggingFace ---")
	api = HfApi()

	# Create repo if needed
	try:
		api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e:
		print(f"  Repo creation: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# Save temporal split
		print("Saving temporal split...")
		temporal_dir = tmpdir / "temporal"
		temporal_dir.mkdir()
		df_train.to_parquet(temporal_dir / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(temporal_dir / "test-00000-of-00001.parquet", index=False)

		# Save random split (legacy 80/20)
		print("Saving random split...")
		random_dir = tmpdir / "random"
		random_dir.mkdir()
		df_rand_train.to_parquet(random_dir / "train-00000-of-00001.parquet", index=False)
		df_rand_test.to_parquet(random_dir / "test-00000-of-00001.parquet", index=False)

		# Save temporal_3way split (train=Mon-Thu, test+val=Friday 50/50)
		print("Saving temporal_3way split...")
		t3w_dir = tmpdir / "temporal_3way"
		t3w_dir.mkdir()
		df_train.to_parquet(t3w_dir / "train-00000-of-00001.parquet", index=False)
		df_test_t3w.to_parquet(t3w_dir / "test-00000-of-00001.parquet", index=False)
		df_val_t3w.to_parquet(t3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Save random_3way split (80/10/10)
		print("Saving random_3way split...")
		r3w_dir = tmpdir / "random_3way"
		r3w_dir.mkdir()
		df_rand3_train.to_parquet(r3w_dir / "train-00000-of-00001.parquet", index=False)
		df_rand3_test.to_parquet(r3w_dir / "test-00000-of-00001.parquet", index=False)
		df_rand3_val.to_parquet(r3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Save feature importance
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump(importance_data, f, indent=2)

		# Create README
		readme = f"""---
language:
- en
license: cc-by-4.0
size_categories:
- 1M<n<10M
task_categories:
- tabular-classification
tags:
- network-intrusion-detection
- cybersecurity
- CICIDS2017
- IDS
- binary-classification
pretty_name: CICIDS2017 Network Intrusion Detection
configs:
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
        path: random/test-*
---

# CICIDS2017 Network Intrusion Detection Dataset

The [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset from the Canadian Institute for Cybersecurity, provided with **temporal and random splits** for fair evaluation.

## Configurations

### `temporal` (default) — Day-Based Temporal Split

> **Note:** `standard` is an alias for `temporal` — both load the same data.

Train on Monday-Thursday, test on Friday. The model must generalize to unseen attack types (DDoS, Botnet, PortScan).

```python
from datasets import load_dataset
ds = load_dataset("lacg030175/CICIDS2017", "temporal")  # or "standard"
# ds["train"]: {len(df_train):,} rows (Mon-Thu)
# ds["test"]:  {len(df_test):,} rows (Friday)
```

**Train attacks:** {df_train['label'].sum():,} / {len(df_train):,} ({df_train['label'].mean()*100:.1f}%)
**Test attacks:** {df_test['label'].sum():,} / {len(df_test):,} ({df_test['label'].mean()*100:.1f}%)

### `random` — Stratified Random Split

80/20 stratified random split from all days combined.

```python
ds = load_dataset("lacg030175/CICIDS2017", "random")
# ds["train"]: {len(df_rand_train):,} rows
# ds["test"]:  {len(df_rand_test):,} rows
```

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Attack Types

| Day | Attack Types |
|---|---|
| Monday | Benign only |
| Tuesday | FTP-Patator, SSH-Patator |
| Wednesday | DoS Hulk, DoS GoldenEye, DoS Slowhttptest, DoS slowloris, Heartbleed |
| Thursday | Web Attack (Brute Force, XSS, SQL Injection), Infiltration |
| **Friday (test)** | **Bot, DDoS, PortScan** |

## Labels

- **Binary** (`label`): 0 = BENIGN, 1 = Attack
- **Multi-class** (`Label`): 15 categories (BENIGN + 14 attack types)

## Features

{len(feature_cols)} numeric flow-level features extracted by CICFlowMeter.

## Preprocessing

- Removed rows with NaN/infinity values
- Stripped whitespace from column names and labels
- All features converted to numeric (float64)
- Added binary `label` column (0=BENIGN, 1=Attack)

## Citation

```bibtex
@inproceedings{{sharafaldin2018toward,
  title={{Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization}},
  author={{Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A}},
  booktitle={{International Conference on Information Systems Security and Privacy}},
  year={{2018}}
}}
```

## License

CC BY 4.0 — original dataset by the Canadian Institute for Cybersecurity, University of New Brunswick.
"""
		with open(tmpdir / "README.md", "w") as f:
			f.write(readme)

		# Upload everything
		print("Uploading to HuggingFace...")
		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	print("\n✓ Done! Dataset available at: https://huggingface.co/datasets/" + TARGET_REPO)


if __name__ == "__main__":
	main()
