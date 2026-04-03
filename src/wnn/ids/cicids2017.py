"""
CICIDS2017 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
Temporal split: Mon-Thu train, Friday test.
Random split: 80/20 stratified.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .encoder import ThermometerEncoder, ThermometerType
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


def _load_from_huggingface(config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame | None]:
	"""Load train/test(/validation) from our published HuggingFace dataset."""
	from datasets import load_dataset

	print(f"  Loading from HuggingFace: {HF_DATASET_ID} ({config})...")
	ds = load_dataset(HF_DATASET_ID, config)
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
	if split == "standard":
		split = "temporal"
	if split not in ("temporal", "random", "temporal_3way", "random_3way"):
		raise ValueError(f"split must be 'temporal', 'standard', 'random', 'temporal_3way', or 'random_3way', got '{split}'")

	print(f"Loading CICIDS2017 (split={split})...")
	df_train, df_test, common_features, df_val = _load_from_huggingface(split)

	X_train, X_test, encoder, used_features, X_val = encode_features(
		df_train, df_test, common_features, TOP20_RF_FEATURES,
		n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, df_val=df_val,
	)

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

	# Binary labels (already in the dataset as 'label' column)
	y_train_binary = df_train["label"].values.astype(np.int64)
	y_test_binary = df_test["label"].values.astype(np.int64)

	# Multi-class labels
	cat_to_idx = {cat: i for i, cat in enumerate(ATTACK_CATEGORIES)}
	y_train_multi = df_train["Label"].map(lambda x: cat_to_idx.get(x, 0)).values.astype(np.int64)
	y_test_multi = df_test["Label"].map(lambda x: cat_to_idx.get(x, 0)).values.astype(np.int64)

	print(f"  Train: {(y_train_binary == 0).sum():,} normal, {(y_train_binary == 1).sum():,} attack")
	print(f"  Test:  {(y_test_binary == 0).sum():,} normal, {(y_test_binary == 1).sum():,} attack")

	# Validation labels if present (3-way splits)
	y_val_binary = None
	y_val_multi = None
	if df_val is not None:
		y_val_binary = df_val["label"].values.astype(np.int64)
		y_val_multi = df_val["Label"].map(lambda x: cat_to_idx.get(x, 0)).values.astype(np.int64)
		print(f"  Val:   {(y_val_binary == 0).sum():,} normal, {(y_val_binary == 1).sum():,} attack")

	return IDSDataset(
		X_train=X_train,
		y_train_binary=y_train_binary,
		y_train_multi=y_train_multi,
		X_test=X_test,
		y_test_binary=y_test_binary,
		y_test_multi=y_test_multi,
		encoder=encoder,
		category_names=ATTACK_CATEGORIES,
		feature_names=used_features,
		X_val=X_val,
		y_val_binary=y_val_binary,
		y_val_multi=y_val_multi,
	)
