"""
Create FULL CIC-IoT-2023 HuggingFace dataset (~46.7M rows) WITHOUT NaN/inf filtering.

Companion to create_ciciot2023_full_dataset.py. The original drops rows with
NaN or ±inf (reducing 46.7M → 38.5M, -17.5%); this version preserves them so
the paired ThermometerEncoder(invalid_encoding="single_bit") can encode
NaN/±inf as a learnable is_invalid flag bit rather than silently collapsing
them to zero.

This variant matches Neto et al.'s raw record count (46,686,748) for fair
direct comparison with their baseline numbers.

Source: bencorn/CIC-IoT-2023 on HuggingFace (CSVs organized by attack type)
Target: lacg030175/CIC-IoT-2023-full-raw on HuggingFace

Space requirements:
- Download: ~13GB CSVs (cached on disk)
- RAM peak: ~25-35GB (pandas DataFrames, no NaN dropping)
- Upload: ~5-8GB parquet
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
TARGET_REPO = "lacg030175/CIC-IoT-2023-full-raw"

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
	print("Creating FULL CIC-IoT-2023-raw HuggingFace Dataset (~46.7M, no NaN filter)")
	print("=" * 70)

	api = HfApi()
	all_files = list(api.list_repo_files(SOURCE_REPO, repo_type="dataset"))
	csv_files = [f for f in all_files if f.endswith(".csv")]
	print(f"Found {len(csv_files)} CSV files in source repo")

	folder_files = {}
	for f in csv_files:
		parts = f.split("/")
		if len(parts) >= 4:
			folder = parts[2]
			folder_files.setdefault(folder, []).append(f)
	print(f"Found {len(folder_files)} attack type folders")

	all_dfs = []
	total_loaded = 0
	for folder, (label, attack_class) in ATTACK_FOLDERS.items():
		files = folder_files.get(folder, [])
		if not files:
			print(f"  WARNING: No files for {folder}"); continue
		print(f"\n  Loading {folder} ({len(files)} files, ALL rows)...")

		dfs = []
		for fname in sorted(files):
			path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
			df = pd.read_csv(path, low_memory=False)
			dfs.append(df)
		if not dfs: continue
		df_cat = pd.concat(dfs, ignore_index=True)
		del dfs; gc.collect()

		df_cat["Label"] = label
		df_cat["attack_class"] = attack_class
		df_cat["label"] = 0 if attack_class == "Benign" else 1
		total_loaded += len(df_cat)
		print(f"    {folder}: {len(df_cat):,} rows (total so far: {total_loaded:,})")
		all_dfs.append(df_cat)

	print(f"\nConcatenating {len(all_dfs)} attack types...")
	df_all = pd.concat(all_dfs, ignore_index=True)
	del all_dfs; gc.collect()
	print(f"Total: {len(df_all):,} rows (raw, pre-coercion)")

	# Coerce features to numeric — but do NOT drop NaN/inf rows.
	feature_cols = [c for c in df_all.columns if c not in ("Label", "attack_class", "label")]
	print(f"Features: {len(feature_cols)}")
	for col in feature_cols:
		df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
	# NO dropna — raw preservation is the whole point.

	print(f"\nFinal: {len(df_all):,} rows (raw, preserved)")

	# Informational NaN/inf report
	num_cols = [c for c in feature_cols if df_all[c].dtype in (np.float32, np.float64, np.int32, np.int64)]
	n_nan = df_all[num_cols].isna().any(axis=1).sum()
	n_inf = df_all[num_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
	print(f"  Rows with NaN: {n_nan:,} ({n_nan/len(df_all)*100:.3f}%)")
	print(f"  Rows with NaN or ±Inf: {n_inf:,} ({n_inf/len(df_all)*100:.3f}%)")

	print(f"\nClass distribution:")
	for cls in sorted(df_all["attack_class"].unique()):
		count = (df_all["attack_class"] == cls).sum()
		pct = count / len(df_all) * 100
		print(f"  {cls:<15}: {count:>12,} ({pct:.2f}%)")
	n_benign = int((df_all["label"] == 0).sum())
	n_attack = int((df_all["label"] == 1).sum())
	print(f"\nBinary: {n_benign:,} benign ({n_benign/len(df_all)*100:.1f}%), "
		  f"{n_attack:,} attack ({n_attack/len(df_all)*100:.1f}%)")

	# RF importance on finite-valued subsample
	print(f"\nComputing RF feature importance (on finite-valued 100K sample)...")
	sample = df_all.sample(min(200_000, len(df_all)), random_state=42)
	sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
	sample = sample.sample(min(100_000, len(sample)), random_state=42)
	print(f"  (Using {len(sample):,} finite-valued samples for importance calc)")
	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values
	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)
	importances = rf.feature_importances_
	ranked = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
	del X, y, sample; gc.collect()
	top20 = [feat for feat, _ in ranked[:20]]
	print(f"\nTop-20 RF Features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"  {i+1:2d}. {feat:<25} {imp:.6f}")

	# Create 80/10/10 three-way split
	print(f"\nCreating 80/10/10 three-way split...")
	df_train, df_remaining = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all["label"])
	df_test, df_val = train_test_split(df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"])
	del df_remaining; gc.collect()
	print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}, Val: {len(df_val):,}")

	# Create random 80/20 split (derived from 3-way: train is same, test+val merged)
	print(f"\nCreating 80/20 random split (test ∪ validation)...")
	df_rand_test = pd.concat([df_test, df_val], ignore_index=True)
	print(f"  Train: {len(df_train):,}, Test: {len(df_rand_test):,}")

	# Upload
	print(f"\nUploading to HuggingFace ({TARGET_REPO})...")
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		# random_3way (80/10/10)
		r3w = tmpdir / "random_3way"; r3w.mkdir()
		df_train.to_parquet(r3w / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(r3w / "test-00000-of-00001.parquet", index=False)
		df_val.to_parquet(r3w / "validation-00000-of-00001.parquet", index=False)

		# random (80/20, test+val merged)
		r = tmpdir / "random"; r.mkdir()
		df_train.to_parquet(r / "train-00000-of-00001.parquet", index=False)
		df_rand_test.to_parquet(r / "test-00000-of-00001.parquet", index=False)

		# Feature importance sidecar
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump({"top20": top20, "all_ranked": [(ft, float(i)) for ft, i in ranked]}, f, indent=2)

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
- raw-preservation
pretty_name: CIC-IoT-2023 full (~46.7M, raw — NaN/inf preserved)
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

# CIC-IoT-2023 full (~46.7M, raw variant)

Companion to `lacg030175/CIC-IoT-2023-full`. This variant preserves rows with NaN or ±infinity values (the filtered variant drops them, reducing 46.7M → 38.5M records). Intended for use with `ThermometerEncoder(invalid_encoding="single_bit")`, which treats missing / undefined values as a learnable `is_invalid` flag bit rather than silently encoding them as zero.

Row count matches Neto et al. 2023's reported 46,686,748 raw flow records for direct baseline comparison.

## Row counts

Total: {len(df_all):,} rows ({n_nan:,} with NaN, {n_inf:,} with NaN or ±Inf).

Splits:
- `random_3way` (80/10/10): {len(df_train):,} / {len(df_test):,} / {len(df_val):,}
- `random` (80/20): {len(df_train):,} / {len(df_rand_test):,}

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Preprocessing (raw variant)

- **Rows with NaN / ±inf are PRESERVED** (not dropped).
- Feature columns coerced to numeric via `pd.to_numeric(errors="coerce")`; unparseable strings become NaN.
- Use with `ThermometerEncoder(invalid_encoding="single_bit")` to encode NaN/±inf as a learnable state — the recommended encoder parameter for this dataset.

## Citation

```bibtex
@article{{neto2023ciciot,
  title={{CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment}},
  author={{Neto, Euclides Carlos Pinto and Dadkhah, Sajjad and Ferreira, Raphael and Zohourian, Alireza and Lu, Rongxing and Ghorbani, Ali A}},
  journal={{Sensors}}, volume={{23}}, number={{13}}, pages={{5941}}, year={{2023}}, publisher={{MDPI}}
}}
```

## License

CC BY 4.0 — original dataset by the Canadian Institute for Cybersecurity, University of New Brunswick.
"""
		with open(tmpdir / "README.md", "w") as f:
			f.write(readme)

		print("Uploading to HuggingFace (may take 15-60 min for ~5-8GB)...")
		api.upload_folder(folder_path=str(tmpdir), repo_id=TARGET_REPO, repo_type="dataset")

	print("\n✓ Done! Dataset available at: https://huggingface.co/datasets/" + TARGET_REPO)


if __name__ == "__main__":
	main()
