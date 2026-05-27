"""
CICIDS2017 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
Temporal split: Mon-Thu train, Friday test.
Random split: 80/20 stratified.
"""

import numpy as np
import pandas as pd
from typing import Optional

from wnn.representations.thermometer import ThermometerEncoder, ThermometerType
from .dataset import IDSDataset, encode_features, VALID_FEATURE_SELECTIONS
from .dataset import IDSDataset, split_train_validation

# Attack categories in CICIDS2017
ATTACK_CATEGORIES = [
	"BENIGN",
	"Bot",
	"DDoS",
	"DoS GoldenEye",
	"DoS Hulk",
	"DoS Slowhttptest",
	"DoS slowloris",
	"FTP-Patator",
	"Heartbleed",
	"Infiltration",
	"PortScan",
	"SSH-Patator",
	"Web Attack - Brute Force",
	"Web Attack - SQL Injection",
	"Web Attack - XSS",
]

# Top-20 features by Random Forest importance on CICIDS2017 temporal split.
TOP20_RF_FEATURES = [
	"Bwd Packet Length Std",
	"Destination Port",
	"Packet Length Std",
	"Bwd Packet Length Max",
	"Avg Bwd Segment Size",
	"Bwd Packet Length Mean",
	"Fwd IAT Std",
	"Average Packet Size",
	"Packet Length Variance",
	"Flow IAT Max",
	"Packet Length Mean",
	"Init_Win_bytes_forward",
	"Idle Min",
	"Idle Mean",
	"Fwd IAT Max",
	"Flow IAT Std",
	"Flow Packets/s",
	"Flow IAT Mean",
	"Fwd Header Length",
	"Bwd Header Length",
]

# Use shared VALID_FEATURE_SELECTIONS from dataset.py

HF_DATASET_ID = "lacg030175/CICIDS2017"
HF_DATASET_RAW_ID = "lacg030175/CICIDS2017-raw"


def _load_from_huggingface(config: str, raw: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame | None]:
	"""Load train/test(/validation) from our published HuggingFace dataset.

	Args:
		config: HF dataset config name (e.g., "random", "temporal_3way").
		raw: when True, load NaN/inf-preserving variant (-raw).
	"""
	from datasets import load_dataset

	repo_id = HF_DATASET_RAW_ID if raw else HF_DATASET_ID
	print(f"  Loading from HuggingFace: {repo_id} ({config})...")
	ds = load_dataset(repo_id, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas() if "validation" in ds else None

	# Features: everything except labels
	exclude = {"Label", "label"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	val_str = f", {len(df_val):,} val" if df_val is not None else ""
	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test{val_str}")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features, df_val


def load_cicids2017(
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "temporal",
	feature_selection: str = "all",
	rest_bits: Optional[int] = None,
	raw: bool = False,
	invalid_encoding: str = "none",
	encoded_storage: str = "memory",
	storage_dir=None,
	auto_max_bits: int = 32,
) -> IDSDataset:
	"""Load CICIDS2017 dataset with thermometer encoding.

	Splits:
	- "temporal" (default): Mon-Thu train, Friday test (realistic deployment)
	- "standard": alias for "temporal"
	- "random": 80/20 stratified random split (literature comparison)

	Feature selection:
	- "all": All 78 features at uniform n_bits
	- "top20": Top-20 RF features at n_bits
	- "top20_split": Top-20 at 16b + rest at rest_bits

	Args:
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "temporal", "standard", or "random".
		feature_selection: "all", "top20", or "top20_split".
		rest_bits: bits for non-top features in top20_split (defaults to n_bits).

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	# Thin wrapper over the unified CICIDSLoader (single source of truth).
	if split == "standard":
		split = "temporal"
	from .loaders import CICIDSLoader, LoadSpec
	return CICIDSLoader().load(LoadSpec(
		split=split, n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, auto_max_bits=auto_max_bits, invalid_encoding=invalid_encoding,
		encoded_storage=encoded_storage, storage_dir=storage_dir, raw=raw,
	))
