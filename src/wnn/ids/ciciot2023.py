"""
CIC-IoT-2023 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
Only random split available (no temporal ordering in source data).
"""

import numpy as np
import pandas as pd

from typing import Optional
from .encoder import ThermometerEncoder, ThermometerType
from .dataset import IDSDataset, encode_features, VALID_FEATURE_SELECTIONS

# Attack classes (grouped) in CIC-IoT-2023
ATTACK_CLASSES = [
	"Benign",
	"BruteForce",
	"DDoS",
	"DoS",
	"Mirai",
	"Recon",
	"Spoofing",
	"Web-based",
]

# Top-20 features by Random Forest importance on CIC-IoT-2023.
TOP20_RF_FEATURES = [
	"HTTPS",
	"Number",
	"Time_To_Live",
	"Max",
	"ack_flag_number",
	"Rate",
	"IAT",
	"ack_count",
	"Header_Length",
	"Min",
	"Variance",
	"psh_flag_number",
	"Tot sum",
	"Std",
	"Tot size",
	"syn_count",
	"AVG",
	"rst_flag_number",
	"DNS",
	"rst_count",
]

# Use shared VALID_FEATURE_SELECTIONS from dataset.py

HF_DATASET_ID = "lacg030175/CIC-IoT-2023"


def _load_from_huggingface(config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
	"""Load train/test from our published HuggingFace dataset."""
	from datasets import load_dataset

	print(f"  Loading from HuggingFace: {HF_DATASET_ID} ({config})...")
	ds = load_dataset(HF_DATASET_ID, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()

	# Features: everything except labels
	exclude = {"Label", "label", "attack_class"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features


def _load_top20_features() -> list[str]:
	"""Load top-20 features from the HuggingFace dataset."""
	global TOP20_RF_FEATURES
	if TOP20_RF_FEATURES:
		return TOP20_RF_FEATURES

	try:
		from huggingface_hub import hf_hub_download
		import json
		path = hf_hub_download(repo_id=HF_DATASET_ID, filename="feature_importance.json", repo_type="dataset")
		with open(path) as f:
			data = json.load(f)
		TOP20_RF_FEATURES = data["top20"]
		return TOP20_RF_FEATURES
	except Exception as e:
		print(f"  WARNING: Could not load top-20 features: {e}")
		return []


def load_ciciot2023(
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "random",
	feature_selection: str = "all",
	rest_bits: Optional[int] = None,
) -> IDSDataset:
	"""Load CIC-IoT-2023 dataset with thermometer encoding.

	Splits:
	- "random" (default): 80/20 stratified random split

	Note: No temporal split available (data organized by attack type, not time).

	Feature selection:
	- "all": All features at uniform n_bits
	- "top20": Top-20 RF features at n_bits
	- "top20_split": Top-20 at 16b + rest at rest_bits

	Args:
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "random" (only option for this dataset).
		feature_selection: "all", "top20", or "top20_split".
		rest_bits: bits for non-top features in top20_split (defaults to n_bits).

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	if split not in ("random",):
		raise ValueError(f"CIC-IoT-2023 only supports 'random' split (no temporal ordering), got '{split}'")

	print(f"Loading CIC-IoT-2023 (split={split})...")
	df_train, df_test, common_features = _load_from_huggingface(split)

	# Load top-20 features (from HuggingFace for this dataset)
	top20 = _load_top20_features()
	if not top20 and feature_selection in ("top20", "top20_split"):
		raise ValueError("Could not load top-20 features from HuggingFace")

	X_train, X_test, encoder, used_features = encode_features(
		df_train, df_test, common_features, top20 or [],
		n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits,
	)

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

	# Binary labels
	y_train_binary = df_train["label"].values.astype(np.int64)
	y_test_binary = df_test["label"].values.astype(np.int64)

	# Multi-class labels (grouped attack classes)
	class_to_idx = {cls: i for i, cls in enumerate(ATTACK_CLASSES)}
	y_train_multi = df_train["attack_class"].map(lambda x: class_to_idx.get(x, 0)).values.astype(np.int64)
	y_test_multi = df_test["attack_class"].map(lambda x: class_to_idx.get(x, 0)).values.astype(np.int64)

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
		category_names=ATTACK_CLASSES,
		feature_names=used_features,
	)
