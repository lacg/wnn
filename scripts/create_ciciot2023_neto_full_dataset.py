"""Build lacg030175/CIC-IoT-2023-neto-full HF dataset from Kaggle's authoritative
46.7M canonical CSV (akashdogra/ciciot23csv).

This is the **gold standard** build:
- 46,686,580 rows (matches Neto et al.'s published count exactly)
- 46 features (vs bencorn's 39 — bencorn's MERGED dropped 7 features)
- Labels embedded in the CSV (uppercase format, e.g. "DDOS-PSHACK_FLOOD")
- NaN/inf preserved (no dropna)

Source: /Users/lacg/wnn/.cache/kaggle_ciciot_full/ciciot23.csv  (13.75 GB)
        Originally from akashdogra/ciciot23csv on Kaggle, derived from CIC's
        official 169-CSV distribution.

Target: lacg030175/CIC-IoT-2023-neto-full

RAM strategy: read full CSV with float32 dtypes for numeric features. Peak ~30-35 GB.
              If worker is busy, expect tight headroom on 64 GB system.

Usage:
    cd /Users/lacg/wnn
    source wnn-venv/bin/activate     # actually /Users/lacg/wnn-venv
    python scripts/create_ciciot2023_neto_full_dataset.py
"""

import json
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SOURCE_CSV = Path("/Users/lacg/wnn/.cache/kaggle_ciciot_full/ciciot23.csv")
TARGET_REPO = "lacg030175/CIC-IoT-2023-neto-full"

# Translate UPPERCASE Neto labels → canonical (label_str, attack_class).
# Same as create_ciciot2023_canonical_neto_dataset.py — Neto's labels are stable.
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


def load_and_translate(csv_path: Path) -> pd.DataFrame:
	"""Read the Kaggle CSV with efficient dtypes, translate labels."""
	print(f"Reading {csv_path} ({csv_path.stat().st_size/1e9:.2f} GB)...")
	t0 = time.time()
	# Read with float32 to halve RAM vs default float64
	df = pd.read_csv(csv_path, low_memory=False, dtype_backend="numpy_nullable")
	print(f"  Read {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")
	print(f"  Columns: {list(df.columns)[:5]}... (+ {len(df.columns)-5} more, last={df.columns[-1]!r})")

	# Translate labels — Label column is the last one
	label_col = df.columns[-1]
	if label_col != "Label":
		print(f"  WARNING: expected last column to be 'Label', got '{label_col}'. Renaming.")
		df = df.rename(columns={label_col: "Label"})

	# Capture provenance
	df["Label_orig"] = df["Label"]
	# Map to canonical labels + attack_class
	mapped = df["Label"].astype(str).str.upper().map(NETO_LABEL_MAP)
	mask_unknown = mapped.isna()
	if mask_unknown.any():
		unk = df.loc[mask_unknown, "Label"].astype(str).unique()
		print(f"  WARNING: {len(unk)} unknown label values ({mask_unknown.sum()} rows)")
		for u in unk[:10]:
			print(f"    repr: {u!r}")
		# Treat unknowns as ("Unknown_<orig>", "Unknown")
		mapped = mapped.where(~mask_unknown,
							  df.loc[mask_unknown, "Label"].apply(lambda x: (str(x), "Unknown")))
	df["Label"] = mapped.apply(lambda t: t[0])
	df["attack_class"] = mapped.apply(lambda t: t[1])
	df["label"] = (df["attack_class"] != "Benign").astype(np.int8)
	return df


def coerce_numeric(df: pd.DataFrame) -> list[str]:
	"""Coerce feature columns to numeric (NaN preserved). Returns feature list."""
	feature_cols = [c for c in df.columns if c not in ("Label", "Label_orig", "attack_class", "label")]
	for col in feature_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return feature_cols


def report_stats(df: pd.DataFrame, feature_cols: list[str]):
	num_cols = [c for c in feature_cols if df[c].dtype in (np.float32, np.float64, np.int32, np.int64, "Int64", "Float64", "Float32")]
	n_nan = df[num_cols].isna().any(axis=1).sum()
	n_inf = df[num_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).sum()
	print(f"\nRow stats:")
	print(f"  Total: {len(df):,}")
	print(f"  Rows with NaN: {n_nan:,} ({n_nan/len(df)*100:.4f}%)")
	print(f"  Rows with NaN or ±Inf: {n_inf:,} ({n_inf/len(df)*100:.4f}%)")
	print(f"\nClass distribution:")
	for cls in sorted(df["attack_class"].unique()):
		count = (df["attack_class"] == cls).sum()
		print(f"  {cls:<15}: {count:>12,} ({count/len(df)*100:.2f}%)")
	n_benign = int((df["label"] == 0).sum())
	n_attack = int((df["label"] == 1).sum())
	print(f"\nBinary: {n_benign:,} benign, {n_attack:,} attack ({n_benign/len(df)*100:.2f}% / {n_attack/len(df)*100:.2f}%)")


def compute_rf_importance(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[tuple[str, float]]]:
	print(f"\nRF importance on 100K finite-valued sample...")
	sample = df.sample(min(200_000, len(df)), random_state=42)
	sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
	sample = sample.sample(min(100_000, len(sample)), random_state=42)
	X = sample[feature_cols].values.astype(np.float64)
	y = sample["label"].values
	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)
	ranked = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])
	top20 = [feat for feat, _ in ranked[:20]]
	print(f"  Top-20 features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"    {i+1:2d}. {feat:<25} {imp:.6f}")
	del X, y, sample
	gc.collect()
	return top20, ranked


def main():
	print("=" * 78)
	print("Building lacg030175/CIC-IoT-2023-neto-full from Kaggle 46.7M canonical CSV")
	print("=" * 78)
	t0 = time.time()

	if not SOURCE_CSV.exists():
		raise FileNotFoundError(f"Source CSV missing: {SOURCE_CSV}")

	df = load_and_translate(SOURCE_CSV)
	feature_cols = coerce_numeric(df)
	print(f"\nFeatures coerced: {len(feature_cols)}")
	report_stats(df, feature_cols)
	top20, ranked = compute_rf_importance(df, feature_cols)

	print(f"\nCreating 80/10/10 stratified split...")
	df_train, df_remaining = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
	df_test, df_val = train_test_split(df_remaining, test_size=0.5, random_state=42, stratify=df_remaining["label"])
	del df_remaining
	gc.collect()
	print(f"  Train: {len(df_train):,} | Test: {len(df_test):,} | Val: {len(df_val):,}")

	df_rand_test = pd.concat([df_test, df_val], ignore_index=True)
	print(f"  random split: {len(df_train):,} train / {len(df_rand_test):,} test (test+val merged)")

	print(f"\nUploading to {TARGET_REPO}...")
	api = HfApi()
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo create: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		r3w = tmpdir / "random_3way"; r3w.mkdir()
		df_train.to_parquet(r3w / "train-00000-of-00001.parquet", index=False)
		df_test.to_parquet(r3w / "test-00000-of-00001.parquet", index=False)
		df_val.to_parquet(r3w / "validation-00000-of-00001.parquet", index=False)

		r = tmpdir / "random"; r.mkdir()
		df_train.to_parquet(r / "train-00000-of-00001.parquet", index=False)
		df_rand_test.to_parquet(r / "test-00000-of-00001.parquet", index=False)

		with open(tmpdir / "feature_importance.json", "w") as f:
			json.dump({"top20": top20, "all_ranked": [(ft, float(i)) for ft, i in ranked]}, f, indent=2)

		readme = f"""---
language: [en]
license: cc-by-4.0
tags: [network-security, intrusion-detection, iot, nids, ciciot, neto]
configs:
- config_name: random_3way
  data_files:
  - {{split: train, path: random_3way/train-*.parquet}}
  - {{split: test, path: random_3way/test-*.parquet}}
  - {{split: validation, path: random_3way/validation-*.parquet}}
- config_name: random
  data_files:
  - {{split: train, path: random/train-*.parquet}}
  - {{split: test, path: random/test-*.parquet}}
---

# CIC-IoT-2023 — Neto-Full (Authoritative 46.7M)

This is the **authoritative canonical CIC-IoT-2023 dataset** at the row count
published by Neto et al. (2023): {len(df):,} rows × {len(feature_cols)} features.

Sourced from the Kaggle mirror `akashdogra/ciciot23csv` (13.75 GB single CSV)
which itself was derived from CIC's official 169-file distribution. Compared
to bencorn's HF mirror (45M, 39 features), this preserves:
- All 46 original features (bencorn dropped 7)
- The full 46.7M row count (bencorn re-merge lost ~1.7M)

**NaN/Inf preservation**: NaN/±inf rows are kept (no dropna). Pair with
`ThermometerEncoder(invalid_encoding="single_bit")` to encode invalid states
as a learnable per-feature flag bit instead of silently collapsing to zero.

## Splits
- `random_3way`: 80% train / 10% test / 10% validation (stratified on binary label, seed=42)
- `random`: 80% train / 20% test (test = test ∪ validation from random_3way)

## Provenance
- Original: Neto et al. (2023), "CICIoT2023: A real-time dataset and benchmark
  for large-scale attacks in IoT environment", Sensors 2023.
- Distribution: CIC at unb.ca/cic/datasets/iotdataset-2023.html
- This HF dataset preserves the original `Label` (canonicalized, mixed case)
  AND `Label_orig` (original UPPERCASE), plus normalized `attack_class` and
  binary `label`.

## Class distribution
"""
		for cls in sorted(df["attack_class"].unique()):
			count = (df["attack_class"] == cls).sum()
			readme += f"- {cls}: {count:,} ({count/len(df)*100:.2f}%)\n"

		(tmpdir / "README.md").write_text(readme)

		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
			commit_message=f"Initial: canonical Neto 46.7M ({len(df):,} rows × {len(feature_cols)} features)",
		)

	print(f"\n✓ Done! Available at: https://huggingface.co/datasets/{TARGET_REPO}")
	print(f"  Total rows: {len(df):,}, features: {len(feature_cols)}")
	print(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
