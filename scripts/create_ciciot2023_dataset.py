"""
Create CIC-IoT-2023 HuggingFace dataset with random split.

No temporal split available (data is organized by attack type, not time).
We subsample to keep dataset manageable (~500K rows) and create 80/20 splits.

Source: bencorn/CIC-IoT-2023 on HuggingFace (CSVs organized by attack type)
Target: lacg030175/CIC-IoT-2023 on HuggingFace
"""

import pandas as pd
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import tempfile
import json

SOURCE_REPO = "bencorn/CIC-IoT-2023"
TARGET_REPO = "lacg030175/CIC-IoT-2023"

# Attack type mapping: folder name → (label, attack class)
ATTACK_FOLDERS = {
	"Benign_Final": ("BenignTraffic", "Benign"),
	"Backdoor_Malware": ("Backdoor_Malware", "Web-based"),
	"BrowserHijacking": ("BrowserHijacking", "Web-based"),
	"CommandInjection": ("CommandInjection", "Web-based"),
	"DDoS-ACK_Fragmentation": ("DDoS-ACK_Fragmentation", "DDoS"),
	"DDoS-HTTP_Flood": ("DDoS-HTTP_Flood", "DDoS"),
	"DDoS-ICMP_Flood": ("DDoS-ICMP_Flood", "DDoS"),
	"DDoS-ICMP_Fragmentation": ("DDoS-ICMP_Fragmentation", "DDoS"),
	"DDoS-PSHACK_Flood": ("DDoS-PSHACK_Flood", "DDoS"),
	"DDoS-RSTFINFlood": ("DDoS-RSTFINFlood", "DDoS"),
	"DDoS-SlowLoris": ("DDoS-SlowLoris", "DDoS"),
	"DDoS-SYN_Flood": ("DDoS-SYN_Flood", "DDoS"),
	"DDoS-SynonymousIP_Flood": ("DDoS-SynonymousIP_Flood", "DDoS"),
	"DDoS-TCP_Flood": ("DDoS-TCP_Flood", "DDoS"),
	"DDoS-UDP_Flood": ("DDoS-UDP_Flood", "DDoS"),
	"DDoS-UDP_Fragmentation": ("DDoS-UDP_Fragmentation", "DDoS"),
	"DictionaryBruteForce": ("DictionaryBruteForce", "BruteForce"),
	"DNS_Spoofing": ("DNS_Spoofing", "Spoofing"),
	"DoS-HTTP_Flood": ("DoS-HTTP_Flood", "DoS"),
	"DoS-SYN_Flood": ("DoS-SYN_Flood", "DoS"),
	"DoS-TCP_Flood": ("DoS-TCP_Flood", "DoS"),
	"DoS-UDP_Flood": ("DoS-UDP_Flood", "DoS"),
	"Mirai-greeth_flood": ("Mirai-greeth_flood", "Mirai"),
	"Mirai-greip_flood": ("Mirai-greip_flood", "Mirai"),
	"Mirai-udpplain": ("Mirai-udpplain", "Mirai"),
	"MITM-ArpSpoofing": ("MITM-ArpSpoofing", "Spoofing"),
	"Recon-HostDiscovery": ("Recon-HostDiscovery", "Recon"),
	"Recon-OSScan": ("Recon-OSScan", "Recon"),
	"Recon-PingSweep": ("Recon-PingSweep", "Recon"),
	"Recon-PortScan": ("Recon-PortScan", "Recon"),
	"SqlInjection": ("SqlInjection", "Web-based"),
	"Uploading_Attack": ("Uploading_Attack", "Web-based"),
	"VulnerabilityScan": ("VulnerabilityScan", "Recon"),
	"XSS": ("XSS", "Web-based"),
}

# 7 attack classes for grouped classification
ATTACK_CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]

# Max rows to sample per attack type (to keep dataset manageable)
MAX_PER_ATTACK = 50_000
MAX_BENIGN = 200_000


def main():
	print("=" * 70)
	print("Creating CIC-IoT-2023 HuggingFace Dataset")
	print("=" * 70)

	api = HfApi()

	# List all CSV files in the source repo
	all_files = list(api.list_repo_files(SOURCE_REPO, repo_type="dataset"))
	csv_files = [f for f in all_files if f.endswith(".csv")]
	print(f"Found {len(csv_files)} CSV files in source repo")

	# Group files by attack folder
	folder_files = {}
	for f in csv_files:
		# Path: CSV/CSV/<AttackFolder>/<file>.csv
		parts = f.split("/")
		if len(parts) >= 4:
			folder = parts[2]
			folder_files.setdefault(folder, []).append(f)

	print(f"Found {len(folder_files)} attack type folders")

	# Load and subsample each attack type
	all_dfs = []
	for folder, (label, attack_class) in ATTACK_FOLDERS.items():
		files = folder_files.get(folder, [])
		if not files:
			print(f"  WARNING: No files for {folder}")
			continue

		max_rows = MAX_BENIGN if attack_class == "Benign" else MAX_PER_ATTACK
		print(f"\n  Loading {folder} ({len(files)} files, max {max_rows:,} rows)...")

		dfs = []
		total_rows = 0
		for fname in sorted(files):
			if total_rows >= max_rows:
				break
			path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
			df = pd.read_csv(path, low_memory=False)
			dfs.append(df)
			total_rows += len(df)

		if not dfs:
			continue

		df_cat = pd.concat(dfs, ignore_index=True)

		# Subsample if over limit
		if len(df_cat) > max_rows:
			df_cat = df_cat.sample(max_rows, random_state=42)

		# Add labels
		df_cat["Label"] = label
		df_cat["attack_class"] = attack_class
		df_cat["label"] = 0 if attack_class == "Benign" else 1

		print(f"    {folder}: {len(df_cat):,} rows (label={label}, class={attack_class})")
		all_dfs.append(df_cat)

	# Combine
	df_all = pd.concat(all_dfs, ignore_index=True)
	print(f"\nTotal: {len(df_all):,} rows")

	# Clean
	feature_cols = [c for c in df_all.columns if c not in ("Label", "attack_class", "label")]
	print(f"Features: {len(feature_cols)}")

	# Replace inf with NaN, drop NaN rows
	df_all = df_all.replace([np.inf, -np.inf], np.nan)
	n_before = len(df_all)
	df_all = df_all.dropna(subset=feature_cols)
	n_after = len(df_all)
	if n_before != n_after:
		print(f"Dropped {n_before - n_after:,} rows with NaN/inf")

	# Convert features to numeric
	for col in feature_cols:
		df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
	df_all = df_all.dropna(subset=feature_cols)

	print(f"\nClass distribution:")
	for cls in sorted(df_all["attack_class"].unique()):
		count = (df_all["attack_class"] == cls).sum()
		print(f"  {cls:<15}: {count:>8,}")

	print(f"\nBinary: {(df_all['label']==0).sum():,} benign, {(df_all['label']==1).sum():,} attack")

	# RF feature importance
	print(f"\nComputing RF feature importance...")
	sample = df_all.sample(min(100_000, len(df_all)), random_state=42)
	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values

	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)
	importances = rf.feature_importances_
	ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])

	top20 = [feat for feat, _ in ranked[:20]]
	print(f"\nTop-20 RF Features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"  {i+1:2d}. {feat:<25} {imp:.6f}")

	# Create random split (no temporal split available)
	print(f"\nCreating 80/20 random split...")
	df_train, df_test = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	print(f"Train: {len(df_train):,}, Test: {len(df_test):,}")

	# Create 80/10/10 three-way split (train / test / validation)
	# test = threshold calibration, validation = final reported metrics (never touched during training)
	print(f"\nCreating 80/10/10 three-way split...")
	df_train_3w, df_remaining = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	df_test_3w, df_val_3w = train_test_split(
		df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"]
	)
	print(f"Train: {len(df_train_3w):,}, Test: {len(df_test_3w):,}, Val: {len(df_val_3w):,}")

	# Upload to HuggingFace
	print(f"\nUploading to HuggingFace...")
	try:
		api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e:
		print(f"  Repo: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# Save random split (legacy 80/20 for backward compatibility)
		random_dir = tmpdir / "random"
		random_dir.mkdir()
		df_train.to_parquet(random_dir / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(random_dir / "test-00000-of-00001.parquet", index=False)

		# Save random_3way split (80/10/10 — preferred for new experiments)
		r3w_dir = tmpdir / "random_3way"
		r3w_dir.mkdir()
		df_train_3w.to_parquet(r3w_dir / "train-00000-of-00001.parquet", index=False)
		df_test_3w.to_parquet(r3w_dir / "test-00000-of-00001.parquet", index=False)
		df_val_3w.to_parquet(r3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Save feature importance
		importance_data = {
			"top20": top20,
			"all_ranked": [(feat, float(imp)) for feat, imp in ranked],
		}
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump(importance_data, f, indent=2)

		# Create README
		n_benign = int((df_all["label"] == 0).sum())
		n_attack = int((df_all["label"] == 1).sum())
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
- CIC-IoT-2023
- IoT
- IDS
- binary-classification
pretty_name: CIC-IoT-2023 IoT Intrusion Detection
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
---

# CIC-IoT-2023 IoT Intrusion Detection Dataset

The [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset from the Canadian Institute for Cybersecurity, subsampled and preprocessed for machine learning evaluation.

## Configurations

### `random_3way` (default) — 80/10/10 Three-Way Split

Stratified random split with fully separated train/test/validation sets:
- **Train (80%)**: Model training and architecture search
- **Test (10%)**: Threshold calibration (held out from training)
- **Validation (10%)**: Final reported metrics (never touched during training or calibration)

```python
from datasets import load_dataset
ds = load_dataset("lacg030175/CIC-IoT-2023", "random_3way")
# ds["train"]:      {len(df_train_3w):,} rows
# ds["test"]:       {len(df_test_3w):,} rows
# ds["validation"]: {len(df_val_3w):,} rows
```

### `random` (legacy) — 80/20 Split

Original 80/20 split for backward compatibility with existing runs.

```python
ds = load_dataset("lacg030175/CIC-IoT-2023", "random")
# ds["train"]: {len(df_train):,} rows
# ds["test"]:  {len(df_test):,} rows
```

## Subsampling Strategy

The original dataset has 46.7M rows (97.6% attack traffic). To create a manageable benchmark:
- **Benign**: up to {MAX_BENIGN:,} rows
- **Each attack type**: up to {MAX_PER_ATTACK:,} rows
- **Total**: {len(df_all):,} rows ({n_benign:,} benign, {n_attack:,} attack)

This preserves all 33 attack types while balancing the dataset for binary classification.

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Attack Types (7 classes, 33 sub-types)

| Class | Sub-types |
|---|---|
| Benign | BenignTraffic |
| BruteForce | DictionaryBruteForce |
| DDoS | ACK_Fragmentation, HTTP_Flood, ICMP_Flood/Frag, PSHACK, RSTFINFlood, SlowLoris, SYN_Flood, SynonymousIP, TCP_Flood, UDP_Flood/Frag |
| DoS | HTTP_Flood, SYN_Flood, TCP_Flood, UDP_Flood |
| Mirai | greeth_flood, greip_flood, udpplain |
| Recon | HostDiscovery, OSScan, PingSweep, PortScan, VulnerabilityScan |
| Spoofing | DNS_Spoofing, MITM-ArpSpoofing |
| Web-based | Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, XSS |

## Labels

- **Binary** (`label`): 0 = Benign, 1 = Attack
- **Multi-class** (`Label`): 34 categories (fine-grained attack types)
- **Grouped** (`attack_class`): 8 classes (7 attack groups + Benign)

## Features

{len(feature_cols)} numeric flow-level features.

## Note on Temporal Split

Unlike UNSW-NB15 and CICIDS2017, CIC-IoT-2023 does not have a natural temporal ordering
(data is organized by attack type, not capture time). Only a random split is provided.

## Citation

```bibtex
@article{{neto2023ciciot,
  title={{CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment}},
  author={{Neto, Euclides Carlos Pinto and others}},
  journal={{Sensors}},
  volume={{23}},
  number={{13}},
  year={{2023}},
  publisher={{MDPI}}
}}
```

## License

CC BY 4.0 — original dataset by the Canadian Institute for Cybersecurity, University of New Brunswick.
"""
		with open(tmpdir / "README.md", "w") as f:
			f.write(readme)

		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	print(f"\n✓ Done! Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
	main()
