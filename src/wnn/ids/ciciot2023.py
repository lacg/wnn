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
HF_DATASET_FULL_ID = "lacg030175/CIC-IoT-2023-full"
HF_DATASET_RAW_ID = "lacg030175/CIC-IoT-2023-raw"
HF_DATASET_FULL_RAW_ID = "lacg030175/CIC-IoT-2023-full-raw"


def _load_from_huggingface(config: str, dataset_size: str = "subsample", raw: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame | None]:
	"""Load train/test(/validation) from our published HuggingFace dataset.

	Args:
		config: HF dataset config name (e.g., "random", "random_3way").
		dataset_size: "subsample" (default, ~1.3M rows) or "full" (~46M rows).
		raw: when True, load NaN/inf-preserving variant (-raw or -full-raw).
	"""
	from datasets import load_dataset

	if raw:
		repo_id = HF_DATASET_FULL_RAW_ID if dataset_size == "full" else HF_DATASET_RAW_ID
	else:
		repo_id = HF_DATASET_FULL_ID if dataset_size == "full" else HF_DATASET_ID
	print(f"  Loading from HuggingFace: {repo_id} ({config})...")
	ds = load_dataset(repo_id, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas() if "validation" in ds else None

	# Features: everything except labels
	exclude = {"Label", "label", "attack_class"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	val_str = f", {len(df_val):,} val" if df_val is not None else ""
	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test{val_str}")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features, df_val


def _load_ranked_features() -> list[str]:
	"""Load all RF-ranked features from the HuggingFace dataset."""
	global TOP20_RF_FEATURES
	if TOP20_RF_FEATURES:
		return TOP20_RF_FEATURES

	try:
		from huggingface_hub import hf_hub_download
		import json
		path = hf_hub_download(repo_id=HF_DATASET_ID, filename="feature_importance.json", repo_type="dataset")
		with open(path) as f:
			data = json.load(f)
		# Load ALL ranked features (not just top 20) for top15/top25 support
		TOP20_RF_FEATURES = [feat for feat, _ in data["all_ranked"]]
		return TOP20_RF_FEATURES
	except Exception as e:
		print(f"  WARNING: Could not load ranked features: {e}")
		return []


def load_ciciot2023(
	n_bits: int = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "random",
	feature_selection: str = "all",
	rest_bits: Optional[int] = None,
	dataset_size: str = "subsample",
	raw: bool = False,
	invalid_encoding: str = "none",
) -> IDSDataset:
	"""Load CIC-IoT-2023 dataset with thermometer encoding.

	Splits:
	- "random" (default): 80/20 stratified random split
	- "random_3way": 80/10/10 stratified split (train/test/validation)

	Note: No temporal split available (data organized by attack type, not time).

	Feature selection:
	- "all": All features at uniform n_bits
	- "top20": Top-20 RF features at n_bits
	- "top20_split": Top-20 at 16b + rest at rest_bits

	Args:
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "random" or "random_3way".
		feature_selection: "all", "top20", or "top20_split".
		rest_bits: bits for non-top features in top20_split (defaults to n_bits).
		dataset_size: "subsample" (default, ~1.3M rows) or "full" (~46M rows from
			lacg030175/CIC-IoT-2023-full).

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	if split not in ("random", "random_3way"):
		raise ValueError(f"CIC-IoT-2023 only supports 'random' or 'random_3way' split, got '{split}'")
	if dataset_size not in ("subsample", "full"):
		raise ValueError(f"dataset_size must be 'subsample' or 'full', got '{dataset_size}'")

	size_label = "FULL 46M" if dataset_size == "full" else "1.3M subsample"
	raw_label = " RAW" if raw else ""
	print(f"Loading CIC-IoT-2023 ({size_label}{raw_label}, split={split}, invalid_encoding={invalid_encoding})...")
	df_train, df_test, common_features, df_val = _load_from_huggingface(split, dataset_size, raw=raw)

	# Load ranked features (from HuggingFace for this dataset)
	top20 = _load_ranked_features()
	if not top20 and feature_selection in ("top15", "top20", "top25", "top20_split"):
		raise ValueError("Could not load top-20 features from HuggingFace")

	X_train, X_test, encoder, used_features, X_val = encode_features(
		df_train, df_test, common_features, top20 or [],
		n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, df_val=df_val,
		invalid_encoding=invalid_encoding,
	)

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

	# Binary labels
	y_train_binary = df_train["label"].values.astype(np.int64)
	y_test_binary = df_test["label"].values.astype(np.int64)

	# Multi-class labels (grouped attack classes)
	class_to_idx = {cls: i for i, cls in enumerate(ATTACK_CLASSES)}
	y_train_multi = df_train["attack_class"].map(lambda x: class_to_idx.get(x, 0)).values.astype(np.int64)
	y_test_multi = df_test["attack_class"].map(lambda x: class_to_idx.get(x, 0)).values.astype(np.int64)

	print(f"  Train: {(y_train_binary == 0).sum():,} normal, {(y_train_binary == 1).sum():,} attack")
	print(f"  Test:  {(y_test_binary == 0).sum():,} normal, {(y_test_binary == 1).sum():,} attack")

	# Validation labels if present (3-way splits)
	y_val_binary = None
	y_val_multi = None
	if df_val is not None:
		y_val_binary = df_val["label"].values.astype(np.int64)
		y_val_multi = df_val["attack_class"].map(lambda x: class_to_idx.get(x, 0)).values.astype(np.int64)
		print(f"  Val:   {(y_val_binary == 0).sum():,} normal, {(y_val_binary == 1).sum():,} attack")

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
		X_val=X_val,
		y_val_binary=y_val_binary,
		y_val_multi=y_val_multi,
	)
