"""
Create FULL CIC-IoT-2023 HuggingFace dataset (46.7M rows) with 80/10/10 split.

Unlike the subsampled version (~1.3M rows), this loads ALL data from the source.
Only provides random_3way config (80/10/10 three-way split).

Source: bencorn/CIC-IoT-2023 on HuggingFace (CSVs organized by attack type)
Target: lacg030175/CIC-IoT-2023-full on HuggingFace

Space requirements:
- Download: ~13GB CSVs (cached on disk)
- RAM peak: ~20-30GB (pandas DataFrames)
- Upload: ~4-7GB parquet
"""

import pandas as pd
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import tempfile
import json
import gc

SOURCE_REPO = "bencorn/CIC-IoT-2023"
TARGET_REPO = "lacg030175/CIC-IoT-2023-full"

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

ATTACK_CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]


def main():
	print("=" * 70)
	print("Creating FULL CIC-IoT-2023 HuggingFace Dataset (46M+ rows)")
	print("=" * 70)

	api = HfApi()

	# List all CSV files in the source repo
	all_files = list(api.list_repo_files(SOURCE_REPO, repo_type="dataset"))
	csv_files = [f for f in all_files if f.endswith(".csv")]
	print(f"Found {len(csv_files)} CSV files in source repo")

	# Group files by attack folder
	folder_files = {}
	for f in csv_files:
		parts = f.split("/")
		if len(parts) >= 4:
			folder = parts[2]
			folder_files.setdefault(folder, []).append(f)

	print(f"Found {len(folder_files)} attack type folders")

	# Load ALL data (no subsampling)
	all_dfs = []
	total_loaded = 0
	for folder, (label, attack_class) in ATTACK_FOLDERS.items():
		files = folder_files.get(folder, [])
		if not files:
			print(f"  WARNING: No files for {folder}")
			continue

		print(f"\n  Loading {folder} ({len(files)} files, ALL rows)...")

		dfs = []
		for fname in sorted(files):
			path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
			df = pd.read_csv(path, low_memory=False)
			dfs.append(df)

		if not dfs:
			continue

		df_cat = pd.concat(dfs, ignore_index=True)
		del dfs
		gc.collect()

		# Add labels
		df_cat["Label"] = label
		df_cat["attack_class"] = attack_class
		df_cat["label"] = 0 if attack_class == "Benign" else 1

		total_loaded += len(df_cat)
		print(f"    {folder}: {len(df_cat):,} rows (total so far: {total_loaded:,})")
		all_dfs.append(df_cat)

	# Combine
	print(f"\nConcatenating {len(all_dfs)} attack types...")
	df_all = pd.concat(all_dfs, ignore_index=True)
	del all_dfs
	gc.collect()
	print(f"Total: {len(df_all):,} rows")

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

	print(f"\nFinal: {len(df_all):,} rows")
	print(f"\nClass distribution:")
	for cls in sorted(df_all["attack_class"].unique()):
		count = (df_all["attack_class"] == cls).sum()
		pct = count / len(df_all) * 100
		print(f"  {cls:<15}: {count:>10,} ({pct:.1f}%)")

	n_benign = int((df_all["label"] == 0).sum())
	n_attack = int((df_all["label"] == 1).sum())
	print(f"\nBinary: {n_benign:,} benign ({n_benign/len(df_all)*100:.1f}%), "
		  f"{n_attack:,} attack ({n_attack/len(df_all)*100:.1f}%)")

	# RF feature importance (on 100K sample to keep it fast)
	print(f"\nComputing RF feature importance (100K sample)...")
	sample = df_all.sample(min(100_000, len(df_all)), random_state=42)
	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values

	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)
	importances = rf.feature_importances_
	ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
	del X, y, sample
	gc.collect()

	top20 = [feat for feat, _ in ranked[:20]]
	print(f"\nTop-20 RF Features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"  {i+1:2d}. {feat:<25} {imp:.6f}")

	# Create 80/10/10 three-way split
	print(f"\nCreating 80/10/10 three-way split...")
	df_train, df_remaining = train_test_split(
		df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
	)
	df_test, df_val = train_test_split(
		df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"]
	)
	del df_remaining
	gc.collect()
	print(f"Train: {len(df_train):,}, Test: {len(df_test):,}, Val: {len(df_val):,}")

	# Upload to HuggingFace
	print(f"\nUploading to HuggingFace ({TARGET_REPO})...")
	try:
		api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e:
		print(f"  Repo: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# Save 80/10/10 split
		r3w_dir = tmpdir / "random_3way"
		r3w_dir.mkdir()

		print("Writing train parquet...")
		df_train.to_parquet(r3w_dir / "train-00000-of-00001.parquet", index=False)
		print("Writing test parquet...")
		df_test.to_parquet(r3w_dir / "test-00000-of-00001.parquet", index=False)
		print("Writing validation parquet...")
		df_val.to_parquet(r3w_dir / "validation-00000-of-00001.parquet", index=False)

		# Save feature importance
		importance_data = {
			"top20": top20,
			"all_ranked": [(feat, float(imp)) for feat, imp in ranked],
		}
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump(importance_data, f, indent=2)

		# Create README
		readme = f"""---
language:
- en
license: cc-by-4.0
size_categories:
- 10M<n<100M
task_categories:
- tabular-classification
tags:
- network-intrusion-detection
- cybersecurity
- CIC-IoT-2023
- IoT
- IDS
- binary-classification
pretty_name: CIC-IoT-2023 Full (46M) IoT Intrusion Detection
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
---

# CIC-IoT-2023 Full Dataset (46M+ rows)

The FULL [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset — all {len(df_all):,} rows, no subsampling.

## Configuration

### `random_3way` (default) — 80/10/10 Three-Way Split

Stratified random split with fully separated sets:
- **Train (80%)**: {len(df_train):,} rows — model training and architecture search
- **Test (10%)**: {len(df_test):,} rows — threshold calibration (held out from training)
- **Validation (10%)**: {len(df_val):,} rows — final reported metrics (never touched)

```python
from datasets import load_dataset
ds = load_dataset("{TARGET_REPO}", "random_3way")
```

## Class Distribution

| Class | Count | % |
|---|---:|---:|
| Benign | {n_benign:,} | {n_benign/len(df_all)*100:.1f}% |
| Attack | {n_attack:,} | {n_attack/len(df_all)*100:.1f}% |

## Comparison with Subsampled Version

| Version | Rows | Benign % | Purpose |
|---|---:|---:|---|
| `lacg030175/CIC-IoT-2023` | ~1.3M | ~15% | Fast experiments (112 runs) |
| **This dataset** | **{len(df_all):,}** | **{n_benign/len(df_all)*100:.1f}%** | **Literature comparison** |

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Attack Types (7 classes, 33 sub-types)

| Class | Sub-types |
|---|---|
| Benign | BenignTraffic |
| BruteForce | DictionaryBruteForce |
| DDoS | 12 sub-types |
| DoS | 4 sub-types |
| Mirai | 3 sub-types |
| Recon | 5 sub-types |
| Spoofing | 2 sub-types |
| Web-based | 6 sub-types |

## Labels

- **Binary** (`label`): 0 = Benign, 1 = Attack
- **Multi-class** (`Label`): 34 categories
- **Grouped** (`attack_class`): 8 classes

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

		print("Uploading to HuggingFace...")
		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
		)

	print(f"\n✓ Done! Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
	main()
