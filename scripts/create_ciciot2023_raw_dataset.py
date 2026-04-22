"""
Create CIC-IoT-2023 HuggingFace dataset (1.3M subsample) WITHOUT NaN/inf filtering.

Companion to create_ciciot2023_dataset.py. The original drops rows with NaN
or ±inf (via pd.DataFrame.dropna); this version preserves them so the paired
ThermometerEncoder(invalid_encoding="single_bit") can encode NaN/±inf as a
learnable is_invalid flag bit rather than silently collapsing them to zero.

Source: bencorn/CIC-IoT-2023 on HuggingFace (CSVs organized by attack type)
Target: lacg030175/CIC-IoT-2023-raw on HuggingFace
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
TARGET_REPO = "lacg030175/CIC-IoT-2023-raw"

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

# Subsample limits (same as original 1.3M script — for parity with original)
MAX_PER_ATTACK = 50_000
MAX_BENIGN = 200_000


def main():
	print("=" * 70)
	print("Creating CIC-IoT-2023-raw HuggingFace Dataset (1.3M, no NaN/inf filtering)")
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

	all_dfs = []
	for folder, (label, attack_class) in ATTACK_FOLDERS.items():
		files = folder_files.get(folder, [])
		if not files:
			print(f"  WARNING: No files for {folder}"); continue
		max_rows = MAX_BENIGN if attack_class == "Benign" else MAX_PER_ATTACK
		print(f"\n  Loading {folder} (max {max_rows:,} rows, raw)...")

		dfs = []
		total_rows = 0
		for fname in sorted(files):
			if total_rows >= max_rows: break
			path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
			df = pd.read_csv(path, low_memory=False)
			dfs.append(df)
			total_rows += len(df)
		if not dfs: continue
		df_cat = pd.concat(dfs, ignore_index=True)
		if len(df_cat) > max_rows:
			df_cat = df_cat.sample(max_rows, random_state=42)
		df_cat["Label"] = label
		df_cat["attack_class"] = attack_class
		df_cat["label"] = 0 if attack_class == "Benign" else 1
		print(f"    {folder}: {len(df_cat):,} rows")
		all_dfs.append(df_cat)

	df_all = pd.concat(all_dfs, ignore_index=True)
	print(f"\nTotal: {len(df_all):,} rows")

	# Coerce features to numeric — but do NOT drop NaN/inf rows.
	feature_cols = [c for c in df_all.columns if c not in ("Label", "attack_class", "label")]
	print(f"Features: {len(feature_cols)}")
	for col in feature_cols:
		df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
	# NO dropna — raw preservation is the whole point.

	# Informational NaN/inf report
	num_cols = [c for c in feature_cols if df_all[c].dtype in (np.float32, np.float64, np.int32, np.int64)]
	n_nan = df_all[num_cols].isna().any(axis=1).sum()
	n_inf = df_all[num_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
	print(f"\nRaw preservation report (informational):")
	print(f"  Rows with NaN: {n_nan:,} ({n_nan/len(df_all)*100:.3f}%)")
	print(f"  Rows with NaN or ±Inf: {n_inf:,} ({n_inf/len(df_all)*100:.3f}%)")

	print(f"\nClass distribution:")
	for cls in sorted(df_all["attack_class"].unique()):
		count = (df_all["attack_class"] == cls).sum()
		print(f"  {cls:<15}: {count:>8,}")
	print(f"\nBinary: {(df_all['label']==0).sum():,} benign, {(df_all['label']==1).sum():,} attack")

	# RF importance (on a finite-valued subsample only — doesn't affect output)
	print(f"\nComputing RF feature importance (on finite subsample)...")
	sample = df_all.sample(min(200_000, len(df_all)), random_state=42)
	sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
	print(f"  (Using {len(sample):,} finite-valued samples)")
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

	# Random 80/20
	print(f"\nCreating 80/20 random split...")
	df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all["label"])
	print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}")

	# Random 80/10/10
	print(f"\nCreating 80/10/10 three-way split...")
	df_train_3w, df_remaining = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all["label"])
	df_test_3w, df_val_3w = train_test_split(df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"])
	print(f"  Train: {len(df_train_3w):,}, Test: {len(df_test_3w):,}, Val: {len(df_val_3w):,}")

	print(f"\nUploading to HuggingFace...")
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)
		(tmpdir / "random").mkdir()
		df_train.to_parquet(tmpdir / "random" / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(tmpdir / "random" / "test-00000-of-00001.parquet", index=False)

		(tmpdir / "random_3way").mkdir()
		df_train_3w.to_parquet(tmpdir / "random_3way" / "train-00000-of-00001.parquet", index=False)
		df_test_3w.to_parquet(tmpdir / "random_3way" / "test-00000-of-00001.parquet", index=False)
		df_val_3w.to_parquet(tmpdir / "random_3way" / "validation-00000-of-00001.parquet", index=False)

		importance_data = {"top20": top20, "all_ranked": [(f, float(i)) for f, i in ranked]}
		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump(importance_data, f, indent=2)

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
- raw-preservation
pretty_name: CIC-IoT-2023 (1.3M subsample, raw — NaN/inf preserved)
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

# CIC-IoT-2023 (1.3M subsample, raw variant)

Companion to `lacg030175/CIC-IoT-2023`. This variant preserves rows with NaN or ±infinity values in any feature column (the original dataset drops them via `pd.dropna`). Intended for use with `ThermometerEncoder(invalid_encoding="single_bit")`, which treats missing / undefined values as a learnable is_invalid flag bit rather than silently encoding them as zero.

## Row counts

Full dataset: {len(df_all):,} rows, {n_nan:,} ({n_nan/len(df_all)*100:.3f}%) with NaN in numeric features.

Splits:
- `random` (80/20): {len(df_train):,} train / {len(df_test):,} test
- `random_3way` (80/10/10): {len(df_train_3w):,} / {len(df_test_3w):,} / {len(df_val_3w):,}

## Top-20 RF Features

{chr(10).join(f'{i+1:2d}. {feat}' for i, feat in enumerate(top20))}

## Preprocessing (raw variant)

- **Rows with NaN / ±inf are PRESERVED** (not dropped).
- Feature columns coerced to numeric via `pd.to_numeric(errors="coerce")`.
- Subsampled to ~1.3M rows via fixed caps per attack type (same as parent `lacg030175/CIC-IoT-2023`).
- Use with `ThermometerEncoder(invalid_encoding="single_bit")` to encode NaN/±inf as a learnable state.

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

		print("Uploading...")
		api.upload_folder(folder_path=str(tmpdir), repo_id=TARGET_REPO, repo_type="dataset")

	print("\n✓ Done! Dataset available at: https://huggingface.co/datasets/" + TARGET_REPO)


if __name__ == "__main__":
	main()
