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

RAM strategy: uses Polars for the heavy operations (CSV read, label transform,
              stratified split, parquet write). Peak ~6-10 GB. Safe to run
              concurrently with the worker.

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
import polars as pl
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier

SOURCE_CSV = Path("/Users/lacg/wnn/.cache/kaggle_ciciot_full/ciciot23.csv")
TARGET_REPO = "lacg030175/CIC-IoT-2023-neto-full"

NETO_LABEL_MAP = {
	# Kaggle CSV uses "BenignTraffic" (mixed case) → uppercased "BENIGNTRAFFIC"
	# bencorn MERGED uses "BENIGN" — both kept here for cross-source compatibility
	"BENIGNTRAFFIC":                ("BenignTraffic",          "Benign"),
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


def load_and_translate(csv_path: Path) -> pl.DataFrame:
	"""Read CSV with polars, translate labels. Returns polars DataFrame."""
	print(f"Reading {csv_path.name} ({csv_path.stat().st_size/1e9:.2f} GB) with polars...")
	t0 = time.time()
	# Polars reads CSV very efficiently, much lower RAM than pandas
	df = pl.read_csv(csv_path, infer_schema_length=10000)
	print(f"  Read {len(df):,} rows × {len(df.columns)} cols in {time.time()-t0:.1f}s")
	print(f"  Columns: {list(df.columns)[:5]}... + {len(df.columns)-5} more (last={df.columns[-1]!r})")

	# Last column should be Label
	label_col = df.columns[-1]
	if label_col != "Label":
		print(f"  Renaming last col {label_col!r} → 'Label'")
		df = df.rename({label_col: "Label"})

	# Build mapping dicts for polars replace_strict
	label_canonical_map = {k: v[0] for k, v in NETO_LABEL_MAP.items()}
	class_map = {k: v[1] for k, v in NETO_LABEL_MAP.items()}

	# Translate via uppercased intermediate
	df = df.with_columns([
		pl.col("Label").alias("Label_orig"),
		pl.col("Label").cast(pl.Utf8).str.to_uppercase().alias("_label_upper"),
	])

	# Apply mappings — replace_strict raises on missing keys, so use replace + handle Unknowns
	df = df.with_columns([
		pl.col("_label_upper").replace_strict(label_canonical_map, default=pl.col("Label_orig")).alias("Label"),
		pl.col("_label_upper").replace_strict(class_map, default=pl.lit("Unknown")).alias("attack_class"),
	]).drop("_label_upper")

	df = df.with_columns([
		(pl.col("attack_class") != "Benign").cast(pl.Int8).alias("label")
	])

	# Report unknowns
	n_unknown = df.filter(pl.col("attack_class") == "Unknown").height
	if n_unknown > 0:
		unknown_labels = df.filter(pl.col("attack_class") == "Unknown")["Label_orig"].unique().to_list()
		print(f"  WARNING: {n_unknown} rows with Unknown attack_class (Label_orig values: {unknown_labels[:5]}...)")

	return df


def coerce_numeric(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
	"""Coerce feature columns to Float32 (NaN preserved). Returns (df, feature_cols)."""
	feature_cols = [c for c in df.columns if c not in ("Label", "Label_orig", "attack_class", "label")]
	df = df.with_columns([
		pl.col(c).cast(pl.Float32, strict=False) for c in feature_cols
	])
	return df, feature_cols


def report_stats(df: pl.DataFrame, feature_cols: list[str]):
	# Polars NaN-or-inf detection (after Float32 cast, inf is preserved, NaN is preserved)
	# A row is "invalid" if any feature is NaN or +/-inf
	print(f"\nRow stats:")
	n_total = df.height
	# Check NaN
	n_nan_per_row = df.select([
		pl.sum_horizontal([pl.col(c).is_nan().cast(pl.Int32) for c in feature_cols]).alias("n_nan_in_row")
	])["n_nan_in_row"]
	n_with_nan = (n_nan_per_row > 0).sum()
	# Check inf
	n_inf_per_row = df.select([
		pl.sum_horizontal([pl.col(c).is_infinite().cast(pl.Int32) for c in feature_cols]).alias("n_inf_in_row")
	])["n_inf_in_row"]
	n_with_inf = (n_inf_per_row > 0).sum()
	n_with_invalid = ((n_nan_per_row > 0) | (n_inf_per_row > 0)).sum()
	print(f"  Total: {n_total:,}")
	print(f"  Rows with NaN: {n_with_nan:,} ({n_with_nan/n_total*100:.4f}%)")
	print(f"  Rows with NaN or ±Inf: {n_with_invalid:,} ({n_with_invalid/n_total*100:.4f}%)")
	print(f"\nClass distribution:")
	dist = df.group_by("attack_class").agg(pl.len().alias("count")).sort("count", descending=True)
	for row in dist.iter_rows():
		cls, count = row
		print(f"  {cls:<15}: {count:>12,} ({count/n_total*100:.2f}%)")
	n_benign = df.filter(pl.col("label") == 0).height
	n_attack = df.filter(pl.col("label") == 1).height
	print(f"\nBinary: {n_benign:,} benign, {n_attack:,} attack ({n_benign/n_total*100:.2f}% / {n_attack/n_total*100:.2f}%)")


def compute_rf_importance(df: pl.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[tuple[str, float]]]:
	"""RF importance on a 100K finite-valued sample (converted to pandas, low RAM)."""
	print(f"\nRF importance on 100K finite-valued sample...")
	# Sample 200K, drop nan/inf, take 100K
	n_sample = min(200_000, df.height)
	sub = df.sample(n=n_sample, seed=42)
	# Drop rows with NaN or inf in any feature column
	sub = sub.filter(~pl.any_horizontal([pl.col(c).is_nan() | pl.col(c).is_infinite() for c in feature_cols]))
	if len(sub) > 100_000:
		sub = sub.sample(n=100_000, seed=42)
	print(f"  Using {len(sub):,} finite samples")
	# Convert just this small sample to pandas
	pdf = sub.to_pandas()
	X = pdf[feature_cols].values.astype(np.float64)
	y = pdf["label"].values
	rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	rf.fit(X, y)
	ranked = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])
	top20 = [feat for feat, _ in ranked[:20]]
	print(f"  Top-20 features:")
	for i, (feat, imp) in enumerate(ranked[:20]):
		print(f"    {i+1:2d}. {feat:<25} {imp:.6f}")
	del X, y, pdf, sub
	gc.collect()
	return top20, ranked


def stratified_split(df: pl.DataFrame, test_frac: float = 0.2, val_frac_of_test: float = 0.5) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
	"""80/10/10 split stratified by binary label. Returns (train, test, val)."""
	parts_train, parts_test, parts_val = [], [], []
	for lbl in [0, 1]:
		sub = df.filter(pl.col("label") == lbl)
		shuffled = sub.sample(fraction=1.0, seed=42, shuffle=True)
		n = shuffled.height
		n_train = int(n * (1 - test_frac))
		n_test_only = int((n - n_train) * (1 - val_frac_of_test))
		parts_train.append(shuffled[:n_train])
		parts_test.append(shuffled[n_train:n_train + n_test_only])
		parts_val.append(shuffled[n_train + n_test_only:])
	df_train = pl.concat(parts_train).sample(fraction=1.0, seed=42, shuffle=True)
	df_test = pl.concat(parts_test).sample(fraction=1.0, seed=42, shuffle=True)
	df_val = pl.concat(parts_val).sample(fraction=1.0, seed=42, shuffle=True)
	return df_train, df_test, df_val


def main():
	print("=" * 78)
	print("Building lacg030175/CIC-IoT-2023-neto-full from Kaggle 46.7M canonical CSV")
	print("=" * 78)
	t0 = time.time()

	if not SOURCE_CSV.exists():
		raise FileNotFoundError(f"Source CSV missing: {SOURCE_CSV}")

	df = load_and_translate(SOURCE_CSV)
	df, feature_cols = coerce_numeric(df)
	print(f"\nFeatures coerced to Float32: {len(feature_cols)}")
	report_stats(df, feature_cols)

	top20, ranked = compute_rf_importance(df, feature_cols)

	print(f"\nCreating 80/10/10 stratified split via polars...")
	df_train, df_test, df_val = stratified_split(df)
	print(f"  Train: {df_train.height:,} | Test: {df_test.height:,} | Val: {df_val.height:,}")

	df_rand_test = pl.concat([df_test, df_val])
	print(f"  random split: {df_train.height:,} train / {df_rand_test.height:,} test (test+val merged)")

	# Snapshot for README
	n_total = df.height
	class_dist = df.group_by("attack_class").agg(pl.len().alias("count")).sort("count", descending=True)
	# free df early — we still have the splits
	del df
	gc.collect()

	print(f"\nUploading to {TARGET_REPO}...")
	api = HfApi()
	try: api.create_repo(TARGET_REPO, repo_type="dataset", exist_ok=True)
	except Exception as e: print(f"  Repo create: {e}")

	with tempfile.TemporaryDirectory() as tmpdir:
		tmpdir = Path(tmpdir)

		r3w = tmpdir / "random_3way"; r3w.mkdir()
		df_train.write_parquet(r3w / "train-00000-of-00001.parquet")
		df_test.write_parquet(r3w / "test-00000-of-00001.parquet")
		df_val.write_parquet(r3w / "validation-00000-of-00001.parquet")

		r = tmpdir / "random"; r.mkdir()
		df_train.write_parquet(r / "train-00000-of-00001.parquet")
		df_rand_test.write_parquet(r / "test-00000-of-00001.parquet")

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
published by Neto et al. (2023): {n_total:,} rows × {len(feature_cols)} features.

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
		for cls, count in class_dist.iter_rows():
			readme += f"- {cls}: {count:,} ({count/n_total*100:.2f}%)\n"

		(tmpdir / "README.md").write_text(readme)

		api.upload_folder(
			folder_path=str(tmpdir),
			repo_id=TARGET_REPO,
			repo_type="dataset",
			commit_message=f"Initial: canonical Neto 46.7M ({n_total:,} rows × {len(feature_cols)} features)",
		)

	print(f"\n✓ Done! Available at: https://huggingface.co/datasets/{TARGET_REPO}")
	print(f"  Total rows: {n_total:,}, features: {len(feature_cols)}")
	print(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
