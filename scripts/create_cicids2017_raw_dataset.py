"""
Create CICIDS2017 HuggingFace dataset WITHOUT NaN/inf row dropping.

This is the "raw" companion to create_cicids2017_dataset.py. The original
drops rows with NaN or ±inf in any feature (via pd.DataFrame.dropna); this
version preserves those rows so the paired ThermometerEncoder
(invalid_encoding="single_bit") can encode NaN/±inf as a learnable
is_invalid flag bit rather than silently collapsing them to zero.

Source: c01dsnap/CIC-IDS2017 on HuggingFace (raw CSVs from UNB CIC)
Target: lacg030175/CICIDS2017-raw on HuggingFace
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
TARGET_REPO = "lacg030175/CICIDS2017-raw"

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


def load_csv(filename: str) -> pd.DataFrame:
	"""Download and load a CICIDS2017 CSV file."""
	path = hf_hub_download(repo_id=SOURCE_REPO, filename=filename, repo_type="dataset")
	df = pd.read_csv(path, low_memory=False)
	df.columns = df.columns.str.strip()
	df["Label"] = df["Label"].astype(str).str.strip()
	return df


def prepare_dataframe_raw(df: pd.DataFrame) -> pd.DataFrame:
	"""Prepare CICIDS2017 data WITHOUT dropping NaN/inf rows.

	Differences from clean_dataframe() in create_cicids2017_dataset.py:
	- Does NOT drop rows with NaN or ±inf values
	- Still coerces feature columns to numeric (strings → NaN) but keeps rows
	- Still handles label encoding (binary + multi-class)

	The invariant: every row present in the source CSV (that has a valid label)
	is preserved; "missing" / "undefined" values survive as NaN/±inf and are
	handled downstream by ThermometerEncoder(invalid_encoding="single_bit").
	"""
	feature_cols = [c for c in df.columns if c != "Label"]

	# Coerce to numeric; unparseable strings → NaN (but we KEEP the row)
	for col in feature_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	# NO dropna calls on feature columns — this is the whole point of the raw variant.

	# Add binary label
	df["label"] = (df["Label"] != "BENIGN").astype(int)

	# Normalize attack labels (handle encoding issues with Web Attack labels)
	df["Label"] = df["Label"].str.replace("Â\xa0", " ", regex=False)
	df["Label"] = df["Label"].str.replace("\xa0", " ", regex=False)
	df["Label"] = df["Label"].str.replace("ï»¿", "", regex=False)

	return df


def compute_rf_importance(df_train: pd.DataFrame, feature_cols: list[str], top_n: int = 20):
	"""RF importance, computed on a NaN-dropped subsample (RF can't handle NaN)."""
	print(f"Computing RF feature importance on {len(df_train):,} samples...")
	# RF requires finite values; subsample and drop NaN for this computation only.
	# Does NOT affect the final uploaded dataset.
	if len(df_train) > 100_000:
		sample = df_train.sample(min(200_000, len(df_train)), random_state=42)
	else:
		sample = df_train
	sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
	print(f"  (Using {len(sample):,} finite-valued samples for RF importance)")

	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values

	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)

	importances = rf.feature_importances_
	ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

	print(f"\nTop-{top_n} RF Features:")
	for i, (feat, imp) in enumerate(ranked[:top_n]):
		print(f"  {i+1:2d}. {feat:<35} {imp:.6f}")

	return [feat for feat, _ in ranked[:top_n]], ranked


def main():
	print("=" * 70)
	print("Creating CICIDS2017-raw HuggingFace Dataset (no NaN/inf filtering)")
	print("=" * 70)

	# Load all files with raw preservation
	print("\n--- Loading train files (Mon-Thu) ---")
	train_dfs = []
	for day, fname in TRAIN_FILES.items():
		print(f"Loading {day}...")
		df = load_csv(fname)
		df = prepare_dataframe_raw(df)
		print(f"  {day}: {len(df):,} rows (raw, no NaN filtering)")
		train_dfs.append(df)

	print("\n--- Loading test files (Friday) ---")
	test_dfs = []
	for day, fname in TEST_FILES.items():
		print(f"Loading {day}...")
		df = load_csv(fname)
		df = prepare_dataframe_raw(df)
		print(f"  {day}: {len(df):,} rows (raw, no NaN filtering)")
		test_dfs.append(df)

	df_train = pd.concat(train_dfs, ignore_index=True)
	df_test = pd.concat(test_dfs, ignore_index=True)

	# Report NaN rates (informational, not filtered)
	feat_cols_all = [c for c in df_train.columns if c not in ("Label", "label")]
	feat_cols_num = [c for c in feat_cols_all if df_train[c].dtype in (np.float32, np.float64, np.int32, np.int64)]
	n_nan_train = df_train[feat_cols_num].isna().any(axis=1).sum()
	n_nan_test = df_test[feat_cols_num].isna().any(axis=1).sum()
	n_inf_train = df_train[feat_cols_num].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
	print(f"\nRaw preservation report (informational):")
	print(f"  Train: {len(df_train):,} rows, {n_nan_train:,} have NaN ({n_nan_train/len(df_train)*100:.3f}%), "
		  f"{n_inf_train:,} have NaN+Inf combined ({n_inf_train/len(df_train)*100:.3f}%)")
	print(f"  Test:  {len(df_test):,} rows, {n_nan_test:,} have NaN ({n_nan_test/len(df_test)*100:.3f}%)")

	# Sanity on labels
	print(f"\nTemporal split: {len(df_train):,} train (Mon-Thu), {len(df_test):,} test (Friday)")
	print(f"Train attacks: {df_train['label'].sum():,} / {len(df_train):,} ({df_train['label'].mean()*100:.1f}%)")
	print(f"Test attacks:  {df_test['label'].sum():,} / {len(df_test):,} ({df_test['label'].mean()*100:.1f}%)")

	# Feature columns (exclude labels)
	feature_cols = sorted(set(df_train.columns) - {"Label", "label"})
	print(f"Features: {len(feature_cols)}")

	# Compute RF importance (on finite-valued subsample — not affecting output)
	top20, all_ranked = compute_rf_importance(df_train, feature_cols)
	importance_data = {
		"top20": top20,
		"all_ranked": [(feat, float(imp)) for feat, imp in all_ranked],
	}

	# Random 80/20 split
	print("\n--- Creating random split (80/20) ---")
	df_all = pd.concat([df_train, df_test], ignore_index=True)
	df_rand_train, df_rand_test = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	print(f"  {len(df_rand_train):,} train, {len(df_rand_test):,} test")

	# Random 80/10/10 three-way split
	print("\n--- Creating random_3way split (80/10/10) ---")
	df_rand3_train, df_rand3_remaining = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	df_rand3_test, df_rand3_val = train_test_split(
		df_rand3_remaining, test_size=0.5, random_state=42, stratify=df_rand3_remaining["label"]
	)
	print(f"  {len(df_rand3_train):,} train, {len(df_rand3_test):,} test, {len(df_rand3_val):,} val")

	# Temporal 3-way: split Friday test into test (50%) + validation (50%)
	print("\n--- Creating temporal_3way split ---")
	df_test_t3w, df_val_t3w = train_test_split(
		df_test, test_size=0.5, random_state=42, stratify=df_test["label"]
	)
	print(f"  {len(df_train):,} train (Mon-Thu), {len(df_test_t3w):,} test, {len(df_val_t3w):,} val (Friday split)")

	# Upload to HuggingFace
	print("\n--- Uploading to HuggingFace ---")
	api = HfApi()
	try:
		api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e:
		print(f"  Repo creation: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# temporal (80/20 via Mon-Thu / Fri)
		temporal_dir = tmpdir / "temporal"; temporal_dir.mkdir()
		df_train.to_parquet(temporal_dir / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(temporal_dir / "test-00000-of-00001.parquet", index=False)

		# random 80/20
		random_dir = tmpdir / "random"; random_dir.mkdir()
		df_rand_train.to_parquet(random_dir / "train-00000-of-00001.parquet", index=False)
		df_rand_test.to_parquet(random_dir / "test-00000-of-00001.parquet", index=False)

		# temporal_3way
		t3w_dir = tmpdir / "temporal_3way"; t3w_dir.mkdir()
		df_train.to_parquet(t3w_dir / "train-00000-of-00001.parquet", index=False)
		df_test_t3w.to_parquet(t3w_dir / "test-00000-of-00001.parquet", index=False)
		df_val_t3w.to_parquet(t3w_dir / "validation-00000-of-00001.parquet", index=False)

		# random_3way
		r3w_dir = tmpdir / "random_3way"; r3w_dir.mkdir()
		df_rand3_train.to_parquet(r3w_dir / "train-00000-of-00001.parquet", index=False)
		df_rand3_test.to_parquet(r3w_dir / "test-00000-of-00001.parquet", index=False)
		df_rand3_val.to_parquet(r3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Feature importance sidecar
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump(importance_data, f, indent=2)

		# README
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
- raw-preservation
pretty_name: CICIDS2017 (raw — NaN/inf preserved)
configs:
  - config_name: random_3way
    data_files:
      - split: train
        path: random_3way/train-*
      - split: test
        path: random_3way/test-*
      - split: validation
        path: random_3way/validation-*
    default: true
  - config_name: random
    data_files:
      - split: train
        path: random/train-*
      - split: test
        path: random/test-*
  - config_name: temporal_3way
    data_files:
      - split: train
        path: temporal_3way/train-*
      - split: test
        path: temporal_3way/test-*
      - split: validation
        path: temporal_3way/validation-*
  - config_name: temporal
    data_files:
      - split: train
        path: temporal/train-*
      - split: test
        path: temporal/test-*
---

# CICIDS2017 (raw variant)

Companion to `lacg030175/CICIDS2017`. This variant preserves rows with NaN or ±infinity values in any feature column (the original dataset drops them via `pd.dropna`). Intended for use with `ThermometerEncoder(invalid_encoding="single_bit")`, which treats missing / undefined values as a learnable is_invalid flag bit rather than silently encoding them as zero.

## Row counts

Full dataset (all days): {len(df_all):,} rows

Splits:
- `random` (80/20): {len(df_rand_train):,} train / {len(df_rand_test):,} test
- `random_3way` (80/10/10): {len(df_rand3_train):,} / {len(df_rand3_test):,} / {len(df_rand3_val):,}
- `temporal` (Mon-Thu / Friday): {len(df_train):,} train / {len(df_test):,} test
- `temporal_3way` (train + Friday split 50/50): {len(df_train):,} / {len(df_test_t3w):,} / {len(df_val_t3w):,}

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Labels

- **Binary** (`label`): 0 = BENIGN, 1 = Attack
- **Multi-class** (`Label`): 15 categories

## Features

{len(feature_cols)} numeric flow-level features extracted by CICFlowMeter.

## Preprocessing (raw variant)

- **Rows with NaN / ±inf are PRESERVED** (not dropped).
- Feature columns are still coerced to numeric via `pd.to_numeric(errors="coerce")`; unparseable strings become NaN.
- Label column preserved as multi-class string; binary `label` derived.
- Whitespace and encoding artifacts in `Label` are normalized.
- Use with `ThermometerEncoder(invalid_encoding="single_bit")` to encode NaN/±inf as a learnable state.

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

		print("Uploading to HuggingFace...")
		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	print("\n✓ Done! Dataset available at: https://huggingface.co/datasets/" + TARGET_REPO)


if __name__ == "__main__":
	main()
