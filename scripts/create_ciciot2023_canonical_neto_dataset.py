"""
Create CANONICAL CIC-IoT-2023 dataset (~45M rows) from Neto et al.'s authentic
MERGED_CSV files in bencorn's HF mirror.

Why this exists:
- Our previous datasets (`-full`, `-full-raw`) used bencorn/CIC-IoT-2023's
  CSV/CSV/<attack>/ folder — bencorn's RE-organization that lost ~6.5M rows
  in the process (38.5M kept out of ~45M Neto provided).
- bencorn's CSV/MERGED_CSV/Merged01-63.csv folder has the canonical Neto
  data (~712k rows × 63 files ≈ 45M rows) WITH the embedded `Label` column.
- This script reads those MERGED files instead, preserving NaN/inf, to give
  the GA the actual full Neto dataset.

Sources:
- bencorn/CIC-IoT-2023 → CSV/MERGED_CSV/Merged{01..63}.csv (canonical Neto)
- Neto et al. (2023): "CICIoT2023: A real-time dataset and benchmark for
  large-scale attacks in IoT environment", Sensors

Target: lacg030175/CIC-IoT-2023-canonical-neto

Space requirements:
- Download: ~9.3 GB (cached after first run)
- RAM peak: ~30-40 GB (pandas DataFrames, no dropna)
- Upload: ~5-7 GB parquet
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
import time

SOURCE_REPO = "bencorn/CIC-IoT-2023"
TARGET_REPO = "lacg030175/CIC-IoT-2023-canonical-neto"

# Translate UPPERCASE Neto labels → canonical (label_str, attack_class).
# Mirrors ATTACK_FOLDERS in create_ciciot2023_full_raw_dataset.py.
NETO_LABEL_MAP = {
	"BENIGN":                       ("BenignTraffic",          "Benign"),
	"BACKDOOR_MALWARE":             ("Backdoor_Malware",       "Web-based"),
	"BROWSERHIJACKING":             ("BrowserHijacking",       "Web-based"),
	"COMMANDINJECTION":             ("CommandInjection",       "Web-based"),
	"DDOS-ACK_FRAGMENTATION":       ("DDoS-ACK_Fragmentation", "DDoS"),
	"DDOS-HTTP_FLOOD":              ("DDoS-HTTP_Flood",        "DDoS"),
	"DDOS-ICMP_FLOOD":              ("DDoS-ICMP_Flood",        "DDoS"),
	"DDOS-ICMP_FRAGMENTATION":      ("DDoS-ICMP_Fragmentation","DDoS"),
	"DDOS-PSHACK_FLOOD":            ("DDoS-PSHACK_Flood",      "DDoS"),
	"DDOS-RSTFINFLOOD":             ("DDoS-RSTFINFlood",       "DDoS"),
	"DDOS-SLOWLORIS":               ("DDoS-SlowLoris",         "DDoS"),
	"DDOS-SYN_FLOOD":               ("DDoS-SYN_Flood",         "DDoS"),
	"DDOS-SYNONYMOUSIP_FLOOD":      ("DDoS-SynonymousIP_Flood","DDoS"),
	"DDOS-TCP_FLOOD":               ("DDoS-TCP_Flood",         "DDoS"),
	"DDOS-UDP_FLOOD":               ("DDoS-UDP_Flood",         "DDoS"),
	"DDOS-UDP_FRAGMENTATION":       ("DDoS-UDP_Fragmentation", "DDoS"),
	"DICTIONARYBRUTEFORCE":         ("DictionaryBruteForce",   "BruteForce"),
	"DNS_SPOOFING":                 ("DNS_Spoofing",           "Spoofing"),
	"DOS-HTTP_FLOOD":               ("DoS-HTTP_Flood",         "DoS"),
	"DOS-SYN_FLOOD":                ("DoS-SYN_Flood",          "DoS"),
	"DOS-TCP_FLOOD":                ("DoS-TCP_Flood",          "DoS"),
	"DOS-UDP_FLOOD":                ("DoS-UDP_Flood",          "DoS"),
	"MIRAI-GREETH_FLOOD":           ("Mirai-greeth_flood",     "Mirai"),
	"MIRAI-GREIP_FLOOD":            ("Mirai-greip_flood",      "Mirai"),
	"MIRAI-UDPPLAIN":               ("Mirai-udpplain",         "Mirai"),
	"MITM-ARPSPOOFING":             ("MITM-ArpSpoofing",       "Spoofing"),
	"RECON-HOSTDISCOVERY":          ("Recon-HostDiscovery",    "Recon"),
	"RECON-OSSCAN":                 ("Recon-OSScan",           "Recon"),
	"RECON-PINGSWEEP":              ("Recon-PingSweep",        "Recon"),
	"RECON-PORTSCAN":               ("Recon-PortScan",         "Recon"),
	"SQLINJECTION":                 ("SqlInjection",           "Web-based"),
	"UPLOADING_ATTACK":             ("Uploading_Attack",       "Web-based"),
	"VULNERABILITYSCAN":            ("VulnerabilityScan",      "Recon"),
	"XSS":                          ("XSS",                    "Web-based"),
}


def main():
	print("=" * 78)
	print("Creating CANONICAL CIC-IoT-2023-neto HF Dataset (~45M, no NaN filter)")
	print("Source: bencorn/CIC-IoT-2023 → CSV/MERGED_CSV/ (Neto's canonical files)")
	print("=" * 78)

	api = HfApi()
	all_files = sorted(api.list_repo_files(SOURCE_REPO, repo_type="dataset"))
	merged_files = [f for f in all_files if "MERGED_CSV/" in f and f.endswith(".csv")]
	print(f"\nFound {len(merged_files)} MERGED CSV files")

	# Load all MERGED files iteratively
	all_dfs = []
	t0 = time.time()
	total_loaded = 0
	unknown_labels = set()
	for i, fname in enumerate(merged_files):
		t_start = time.time()
		path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
		df = pd.read_csv(path, low_memory=False)
		# Translate labels
		df["Label_orig"] = df["Label"]  # keep raw upper-case for provenance
		mapped = df["Label"].map(NETO_LABEL_MAP)
		# Find unknown labels
		mask_unknown = mapped.isna()
		if mask_unknown.any():
			unknown_labels.update(df.loc[mask_unknown, "Label"].unique())
			# Default unknown labels to (Label_as_is, "Unknown")
			mapped = mapped.where(~mask_unknown, df.loc[mask_unknown, "Label"].apply(lambda x: (x, "Unknown")))
		df["Label"] = mapped.apply(lambda t: t[0])
		df["attack_class"] = mapped.apply(lambda t: t[1])
		df["label"] = (df["attack_class"] != "Benign").astype(np.int8)

		total_loaded += len(df)
		dt = time.time() - t_start
		all_dfs.append(df)
		print(f"  [{i+1:2d}/{len(merged_files)}] {fname.split('/')[-1]}: {len(df):>8,} rows  ({dt:5.1f}s)  [total: {total_loaded:,}]", flush=True)

	if unknown_labels:
		print(f"\n  WARNING: {len(unknown_labels)} unknown labels (mapped to 'Unknown' class):")
		for u in sorted(unknown_labels):
			print(f"    {u}")

	print(f"\nConcatenating {len(all_dfs)} dataframes ({(time.time()-t0)/60:.1f} min elapsed)...")
	df_all = pd.concat(all_dfs, ignore_index=True)
	del all_dfs
	gc.collect()
	print(f"Total: {len(df_all):,} rows (raw, pre-coercion)")

	# Coerce features to numeric — NO dropna (raw preservation)
	feature_cols = [c for c in df_all.columns if c not in ("Label", "Label_orig", "attack_class", "label")]
	print(f"Features: {len(feature_cols)}")
	for col in feature_cols:
		df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

	print(f"\nFinal: {len(df_all):,} rows (raw, preserved)")

	# Informational NaN/inf report
	num_cols = [c for c in feature_cols if df_all[c].dtype in (np.float32, np.float64, np.int32, np.int64)]
	n_nan = df_all[num_cols].isna().any(axis=1).sum()
	n_inf = df_all[num_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
	print(f"  Rows with NaN: {n_nan:,} ({n_nan/len(df_all)*100:.4f}%)")
	print(f"  Rows with NaN or ±Inf: {n_inf:,} ({n_inf/len(df_all)*100:.4f}%)")

	print(f"\nClass distribution:")
	for cls in sorted(df_all["attack_class"].unique()):
		count = (df_all["attack_class"] == cls).sum()
		pct = count / len(df_all) * 100
		print(f"  {cls:<15}: {count:>12,} ({pct:.2f}%)")
	n_benign = int((df_all["label"] == 0).sum())
	n_attack = int((df_all["label"] == 1).sum())
	print(f"\nBinary: {n_benign:,} benign ({n_benign/len(df_all)*100:.2f}%), "
		  f"{n_attack:,} attack ({n_attack/len(df_all)*100:.2f}%)")

	# RF feature importance on finite-valued subsample
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
	del X, y, sample
	gc.collect()
	top20 = [feat for feat, _ in ranked[:20]]
	print(f"\nTop-20 RF Features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"  {i+1:2d}. {feat:<25} {imp:.6f}")

	# Create 80/10/10 three-way split
	print(f"\nCreating 80/10/10 three-way split...")
	df_train, df_remaining = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all["label"])
	df_test, df_val = train_test_split(df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"])
	del df_remaining
	gc.collect()
	print(f"  Train: {len(df_train):,}, Test: {len(df_test):,}, Val: {len(df_val):,}")

	# Create random 80/20 split (test+val merged, same train)
	print(f"\nCreating 80/20 random split (test ∪ validation)...")
	df_rand_test = pd.concat([df_test, df_val], ignore_index=True)
	print(f"  Train: {len(df_train):,}, Test: {len(df_rand_test):,}")

	# Upload
	print(f"\nUploading to HuggingFace ({TARGET_REPO})...")
	try:
		api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e:
		print(f"  Repo: {e}")

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
			json.dump({"top20": top20,
					   "all_ranked": [(ft, float(i)) for ft, i in ranked]}, f, indent=2)

		readme = f"""---
language:
- en
license: cc-by-4.0
tags:
- network-security
- intrusion-detection
- iot
- nids
configs:
- config_name: random_3way
  data_files:
  - split: train
    path: random_3way/train-*.parquet
  - split: test
    path: random_3way/test-*.parquet
  - split: validation
    path: random_3way/validation-*.parquet
- config_name: random
  data_files:
  - split: train
    path: random/train-*.parquet
  - split: test
    path: random/test-*.parquet
---

# CIC-IoT-2023 — Canonical (Neto et al.) Variant

This is the **canonical** CIC-IoT-2023 dataset, sourced from
[bencorn/CIC-IoT-2023](https://huggingface.co/datasets/bencorn/CIC-IoT-2023)'s
`CSV/MERGED_CSV/` folder, which contains Neto et al.'s authentic merged CSVs
WITH embedded labels (vs. bencorn's other `CSV/CSV/<attack>/` re-organization
which lost ~6.5M rows during the folder-restructure).

**Why this exists**: prior `lacg030175/CIC-IoT-2023-full` and `-full-raw` were
built from `CSV/CSV/` and contained only 38.5M rows. This one contains
**~{len(df_all):,} rows** (the actual canonical Neto count).

**NaN/Inf preservation**: rows with NaN or ±Inf are kept (no dropna). Use the
paired `ThermometerEncoder(invalid_encoding="single_bit")` to encode those
states as a learnable per-feature `is_invalid` flag bit instead of silently
collapsing them to zero.

## Provenance
- Original CSV format from Neto et al. (2023): "CICIoT2023: A real-time dataset
  and benchmark for large-scale attacks in IoT environment", Sensors 2023.
- HF mirror: bencorn/CIC-IoT-2023, `CSV/MERGED_CSV/Merged{{01..63}}.csv`.
- This dataset preserves Neto's original labels in `Label_orig` (UPPERCASE)
  and provides our normalized `Label` + `attack_class` + `label` (binary 0/1).

## Splits
- `random_3way`: 80% train / 10% test / 10% validation (stratified on binary label, seed=42).
- `random`: 80% train / 20% test (test = test ∪ validation from `random_3way`).

## Class distribution
"""
		for cls in sorted(df_all["attack_class"].unique()):
			count = (df_all["attack_class"] == cls).sum()
			pct = count / len(df_all) * 100
			readme += f"- {cls}: {count:,} ({pct:.2f}%)\n"

		with open(tmpdir / "README.md", "w") as f:
			f.write(readme)

		# Upload everything
		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
			commit_message=f"Initial upload: canonical Neto CIC-IoT-2023 ({len(df_all):,} rows, NaN preserved)",
		)

	print(f"\n✓ Done! Dataset available at: https://huggingface.co/datasets/{TARGET_REPO}")
	print(f"  Total rows: {len(df_all):,}")
	print(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
