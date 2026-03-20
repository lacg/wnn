"""
CICIDS2017 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
Temporal split: Mon-Thu train, Friday test.
Random split: 80/20 stratified.
"""

import numpy as np
import pandas as pd

from .encoder import ThermometerEncoder, ThermometerType
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

VALID_FEATURE_SELECTIONS = ("all", "top20")

HF_DATASET_ID = "lacg030175/CICIDS2017"


def _load_from_huggingface(config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
	"""Load train/test from our published HuggingFace dataset."""
	from datasets import load_dataset

	print(f"  Loading from HuggingFace: {HF_DATASET_ID} ({config})...")
	ds = load_dataset(HF_DATASET_ID, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()

	# Features: everything except labels
	exclude = {"Label", "label"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features


def load_cicids2017(
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "temporal",
	feature_selection: str = "all",
) -> IDSDataset:
	"""Load CICIDS2017 dataset with thermometer encoding.

	Splits:
	- "temporal" (default): Mon-Thu train, Friday test (realistic deployment)
	- "standard": alias for "temporal"
	- "random": 80/20 stratified random split (literature comparison)

	Feature selection:
	- "all": All 78 features
	- "top20": Top-20 RF features

	Args:
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "temporal", "standard", or "random".
		feature_selection: "all" or "top20".

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	# Alias: "standard" maps to "temporal"
	if split == "standard":
		split = "temporal"
	if split not in ("temporal", "random"):
		raise ValueError(f"split must be 'temporal', 'standard', or 'random', got '{split}'")
	if feature_selection not in VALID_FEATURE_SELECTIONS:
		raise ValueError(f"feature_selection must be one of {VALID_FEATURE_SELECTIONS}, got '{feature_selection}'")

	# Load raw data
	print(f"Loading CICIDS2017 (split={split})...")
	df_train, df_test, common_features = _load_from_huggingface(split)

	# Apply feature selection
	if feature_selection == "top20":
		selected = [f for f in TOP20_RF_FEATURES if f in common_features]
		if len(selected) < len(TOP20_RF_FEATURES):
			missing = set(TOP20_RF_FEATURES) - set(common_features)
			print(f"  WARNING: {len(missing)} top-20 features not in dataset: {missing}")
		encoder = ThermometerEncoder(n_bits=n_bits, method=method)
		encoder.fit(df_train[selected])
		X_train = encoder.transform(df_train[selected])
		X_test = encoder.transform(df_test[selected])
		used_features = selected
		print(f"  Encoder: {encoder.total_bits} total bits "
			  f"({method.value}, {n_bits} bits/feature, feature_selection=top20, {len(selected)} features)")
	else:
		encoder = ThermometerEncoder(n_bits=n_bits, method=method)
		encoder.fit(df_train[common_features])
		X_train = encoder.transform(df_train[common_features])
		X_test = encoder.transform(df_test[common_features])
		used_features = common_features
		print(f"  Encoder: {encoder.total_bits} total bits "
			  f"({method.value}, {n_bits} bits/feature, feature_selection=all)")

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

	# Binary labels (already in the dataset as 'label' column)
	y_train_binary = df_train["label"].values.astype(np.int64)
	y_test_binary = df_test["label"].values.astype(np.int64)

	# Multi-class labels
	cat_to_idx = {cat: i for i, cat in enumerate(ATTACK_CATEGORIES)}
	y_train_multi = df_train["Label"].map(lambda x: cat_to_idx.get(x, 0)).values.astype(np.int64)
	y_test_multi = df_test["Label"].map(lambda x: cat_to_idx.get(x, 0)).values.astype(np.int64)

	n_train_normal = int((y_train_binary == 0).sum())
	n_train_attack = int((y_train_binary == 1).sum())
	n_test_normal = int((y_test_binary == 0).sum())
	n_test_attack = int((y_test_binary == 1).sum())
	print(f"  Train: {n_train_normal:,} normal, {n_train_attack:,} attack")
	print(f"  Test:  {n_test_normal:,} normal, {n_test_attack:,} attack")

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
	)
