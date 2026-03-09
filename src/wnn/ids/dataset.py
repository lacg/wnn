"""
UNSW-NB15 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .encoder import ThermometerEncoder, ThermometerType


# Standard attack categories in UNSW-NB15 (using exact dataset labels)
ATTACK_CATEGORIES = [
	"Normal",
	"Analysis",
	"Backdoor",
	"DoS",
	"Exploits",
	"Fuzzers",
	"Generic",
	"Reconnaissance",
	"Shellcode",
	"Worms",
]

# Map variant spellings to canonical names
_CATEGORY_ALIASES = {
	"backdoors": "Backdoor",
	"backdoor": "Backdoor",
	"exploit": "Exploits",
	"exploits": "Exploits",
	"fuzzer": "Fuzzers",
	"fuzzers": "Fuzzers",
	"dos": "DoS",
	"worm": "Worms",
	"worms": "Worms",
	"shellcode": "Shellcode",
	"generic": "Generic",
	"reconnaissance": "Reconnaissance",
	"analysis": "Analysis",
	"normal": "Normal",
	" fuzzers": "Fuzzers",
	" shellcode": "Shellcode",
	" reconnaissance": "Reconnaissance",
	" backdoors": "Backdoor",
	" backdoor": "Backdoor",
}


@dataclass
class IDSDataset:
	"""Preprocessed IDS dataset ready for RAM neuron training."""
	X_train: np.ndarray  # (n_train, total_bits) bool
	y_train_binary: np.ndarray  # (n_train,) int: 0=normal, 1=attack
	y_train_multi: np.ndarray  # (n_train,) int: 0-9 attack category index
	X_test: np.ndarray  # (n_test, total_bits) bool
	y_test_binary: np.ndarray  # (n_test,) int
	y_test_multi: np.ndarray  # (n_test,) int
	encoder: ThermometerEncoder
	category_names: list[str]  # index → category name
	feature_names: list[str]  # feature names in order


HF_DATASET_ID = "lacg030175/UNSW-NB15"


def _load_from_huggingface(config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
	"""Load train/test from our published HuggingFace dataset.

	Configs: "standard" (175K/82K temporal) or "random" (1.4M/158K deduped).
	"""
	from datasets import load_dataset

	print(f"  Loading from HuggingFace: {HF_DATASET_ID} ({config})...")
	ds = load_dataset(HF_DATASET_ID, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()

	# Features: everything except labels and IDs
	exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
	if config == "random":
		# Random config has IP/port columns — drop nominal features
		exclude |= {"srcip", "dstip", "sport", "dsport"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	# Encode any string columns as integers
	from sklearn.preprocessing import LabelEncoder
	for col in common_features:
		if df_train[col].dtype == object:
			le = LabelEncoder()
			le.fit(pd.concat([df_train[col], df_test[col]]).astype(str).fillna("?"))
			df_train[col] = le.transform(df_train[col].astype(str).fillna("?"))
			df_test[col] = le.transform(df_test[col].astype(str).fillna("?"))

	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features


def _load_standard_split_local(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
	"""Fallback: load standard split from local CSV files."""
	df_train = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv")
	test_csv = data_dir / "UNSW_NB15_testing-set.csv"
	if test_csv.exists():
		df_test = pd.read_csv(test_csv, encoding="utf-8-sig")
	else:
		print("  WARNING: Test CSV not found, falling back to parquet (34 features)")
		df_test = pd.read_parquet(data_dir / "train.parquet")

	exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))
	print(f"  Standard split (local): {len(df_train):,} train, {len(df_test):,} test")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features


def load_unsw_nb15(
	data_dir: str | Path | None = None,
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "standard",
) -> IDSDataset:
	"""Load UNSW-NB15 with thermometer encoding.

	Primary source: our published HuggingFace dataset (lacg030175/UNSW-NB15).
	Fallback: local CSV files (standard split only).

	Two evaluation protocols:
	- "standard": Original temporal train/test (175K/82K, ~87% RF baseline)
	- "random": Deduped 90/10 random split (1.4M/158K, ~99.6% RF baseline)

	Args:
		data_dir: path to local data directory (fallback only). Auto-detected if None.
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "standard" or "random" evaluation protocol.

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	if split not in ("standard", "random"):
		raise ValueError(f"split must be 'standard' or 'random', got '{split}'")

	# ── Load raw data ──────────────────────────────────────────────────
	print(f"Loading UNSW-NB15 (split={split})...")

	try:
		df_train, df_test, common_features = _load_from_huggingface(split)
	except Exception as e:
		if split == "random":
			raise RuntimeError(
				f"Random split requires HuggingFace dataset ({HF_DATASET_ID}). "
				f"Install: pip install datasets\nError: {e}"
			)
		# Fallback to local CSVs for standard split only
		print(f"  HuggingFace unavailable ({e}), trying local CSV fallback...")
		if data_dir is None:
			candidates = [
				Path(__file__).parents[4] / "data" / "unsw-nb15",
				Path.cwd() / "data" / "unsw-nb15",
			]
			for c in candidates:
				if c.exists():
					data_dir = c
					break
			if data_dir is None:
				raise FileNotFoundError(
					f"UNSW-NB15 data not found. Install 'datasets' for HuggingFace "
					f"or place CSVs at data/unsw-nb15/. Original error: {e}"
				)
		df_train, df_test, common_features = _load_standard_split_local(Path(data_dir))

	# ── Extract labels ─────────────────────────────────────────────────
	# Binary labels
	y_train_binary = df_train["label"].values.astype(np.int32)
	y_test_binary = df_test["label"].values.astype(np.int32)

	# Multi-class labels
	cat_to_idx = {cat: i for i, cat in enumerate(ATTACK_CATEGORIES)}

	def encode_categories(series):
		cats = series.fillna("Normal").str.strip().replace("", "Normal")
		# Normalize via alias map, then look up index
		def _normalize(x):
			x_lower = x.strip().lower()
			canonical = _CATEGORY_ALIASES.get(x_lower, x.strip())
			return cat_to_idx.get(canonical, 0)
		return cats.map(_normalize).values.astype(np.int32)

	y_train_multi = encode_categories(df_train["attack_cat"])
	y_test_multi = encode_categories(df_test["attack_cat"])

	# ── Fit encoder on training data ───────────────────────────────────
	encoder = ThermometerEncoder(n_bits=n_bits, method=method)
	encoder.fit(df_train[common_features])

	print(f"  Encoder: {encoder.total_bits} total bits "
		  f"({method.value}, {n_bits} bits/feature)")

	# ── Transform ──────────────────────────────────────────────────────
	X_train = encoder.transform(df_train[common_features])
	X_test = encoder.transform(df_test[common_features])

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
	print(f"  Train: {(y_train_binary == 0).sum():,} normal, "
		  f"{(y_train_binary == 1).sum():,} attack")
	print(f"  Test:  {(y_test_binary == 0).sum():,} normal, "
		  f"{(y_test_binary == 1).sum():,} attack")

	return IDSDataset(
		X_train=X_train,
		y_train_binary=y_train_binary,
		y_train_multi=y_train_multi,
		X_test=X_test,
		y_test_binary=y_test_binary,
		y_test_multi=y_test_multi,
		encoder=encoder,
		category_names=ATTACK_CATEGORIES,
		feature_names=common_features,
	)
