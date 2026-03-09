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


def load_unsw_nb15(
	data_dir: str | Path | None = None,
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
) -> IDSDataset:
	"""Load UNSW-NB15 with thermometer encoding.

	Uses the CSV training set (175K rows, 42 features) and parquet test set
	(82K rows, 34 features). Only the 34 common features are used.

	Args:
		data_dir: path to data/unsw-nb15/ directory. Auto-detected if None.
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	if data_dir is None:
		# Auto-detect from project root
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
				"UNSW-NB15 data not found. Expected at data/unsw-nb15/. "
				"Run explore_unsw_nb15.py to download."
			)
	data_dir = Path(data_dir)

	# ── Load raw data ──────────────────────────────────────────────────
	print(f"Loading UNSW-NB15 from {data_dir}...")

	# CSV training set (175K rows, all features)
	df_train = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv")

	# Parquet "train" is actually the test set (82K rows, see exploration)
	df_test = pd.read_parquet(data_dir / "train.parquet")

	# ── Use common features only ───────────────────────────────────────
	exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
	train_features = set(df_train.columns) - exclude
	test_features = set(df_test.columns) - exclude
	common_features = sorted(train_features & test_features)

	print(f"  Train: {len(df_train):,} rows, Test: {len(df_test):,} rows")
	print(f"  Using {len(common_features)} common features")

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
