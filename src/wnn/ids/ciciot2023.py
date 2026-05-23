"""
CIC-IoT-2023 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
Only random split available (no temporal ordering in source data).
"""

import numpy as np
import pandas as pd

from typing import Optional
from wnn.representations.thermometer import ThermometerEncoder, ThermometerType
from .dataset import (
	IDSDataset,
	encode_features,
	encode_and_build_dataset,
	encode_and_build_dataset_streaming,
	VALID_FEATURE_SELECTIONS,
)

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

# Top-20 features by Random Forest importance on canonical CIC-IoT-2023.
# Derived on lacg030175/CIC-IoT-2023-neto-full (46.7M canonical Neto, sourced
# from Kaggle akashdogra/ciciot23csv) via RandomForestClassifier(
# n_estimators=100, n_jobs=-1, random_state=42).feature_importances_ on the
# full 37.3M train set.
#
# IMPORTANT: This list was re-derived on 13/05/2026 to remove a fabricated
# `Time_To_Live` feature that existed in bencorn's HF mirror but is NOT in
# Neto et al.'s published 46-feature distribution. The prior bencorn-derived
# TOP20 had only 13 features in common with this canonical TOP20.
#
# See scripts/derive_top20_neto_full.py and data/top20_canonical_neto_full.json
# for the derivation. Ranking saved to JSON; this list is the top-20 ordered
# by descending RF importance.
TOP20_RF_FEATURES = [
	"IAT",
	"rst_count",
	"urg_count",
	"flow_duration",
	"Duration",
	"Number",
	"Weight",
	"Header_Length",
	"Variance",
	"Rate",
	"Srate",
	"Covariance",
	"HTTPS",
	"Tot size",
	"Max",
	"AVG",
	"Radius",
	"Min",
	"Std",
	"Tot sum",
]

# Post-quantization MI ranking at 8-bit thermometer.
# Derived 13/05/2026 by scripts/derive_top20_wnn_thermo.py using
# mutual_info_classif on KBinsDiscretizer(n_bins=9, quantile) over a
# 2M-row subsample of neto-full. Overlaps 18/20 with TOP20_RF_FEATURES;
# substitutions: drop HTTPS+Min, add Magnitue+Protocol Type.
TOP20_MI_8B_FEATURES = [
	"urg_count",
	"rst_count",
	"Std",
	"Radius",
	"Variance",
	"flow_duration",
	"Covariance",
	"Header_Length",
	"Max",
	"AVG",
	"Magnitue",
	"IAT",
	"Tot size",
	"Duration",
	"Tot sum",
	"Number",
	"Weight",
	"Rate",
	"Srate",
	"Protocol Type",
]

# Post-quantization MI ranking at 96-bit thermometer (wide-encoding cohort).
# Overlaps 17/20 with TOP20_RF_FEATURES; substitutions: drop HTTPS+Rate+Srate,
# add Magnitue+Protocol Type+syn_count.
TOP20_MI_96B_FEATURES = [
	"IAT",
	"rst_count",
	"Number",
	"urg_count",
	"flow_duration",
	"AVG",
	"Tot size",
	"Variance",
	"Magnitue",
	"Duration",
	"Max",
	"Std",
	"Radius",
	"Tot sum",
	"Header_Length",
	"Covariance",
	"Min",
	"Protocol Type",
	"Weight",
	"syn_count",
]

# Mapping of feature_selection string → feature list. Used by
# load_ciciot2023() to route between RF-importance, MI-8b, and MI-96b rankings.
FEATURE_LIST_BY_NAME = {
	"top20": TOP20_RF_FEATURES,
	"top20_mi8b": TOP20_MI_8B_FEATURES,
	"top20_mi96b": TOP20_MI_96B_FEATURES,
}

# Use shared VALID_FEATURE_SELECTIONS from dataset.py

HF_DATASET_ID = "lacg030175/CIC-IoT-2023"
HF_DATASET_FULL_ID = "lacg030175/CIC-IoT-2023-full"
HF_DATASET_RAW_ID = "lacg030175/CIC-IoT-2023-raw"
HF_DATASET_FULL_RAW_ID = "lacg030175/CIC-IoT-2023-full-raw"
HF_DATASET_CANONICAL_NETO_ID = "lacg030175/CIC-IoT-2023-canonical-neto"
HF_DATASET_NETO_FULL_ID = "lacg030175/CIC-IoT-2023-neto-full"
HF_DATASET_NETO_SUBSAMPLE_ID = "lacg030175/CIC-IoT-2023-neto-subsample"


def _load_from_huggingface_streaming(
	config: str,
	dataset_size: str = "subsample",
	raw: bool = False,
	chunk_size: int = 1_000_000,
):
	"""Load HF dataset in streaming mode.

	Returns:
		train_factory: callable () → Iterator[pd.DataFrame]. Re-iterable —
			each call creates a fresh HF streaming connection.
		test_factory: same for the test split.
		val_factory: same for validation (or None if absent).
		n_train, n_test, n_val: row counts from HF metadata.
		common_features: list of feature column names (excludes label cols).
	"""
	from datasets import load_dataset

	if dataset_size == "neto_full":
		repo_id = HF_DATASET_NETO_FULL_ID
	elif dataset_size == "neto_subsample":
		repo_id = HF_DATASET_NETO_SUBSAMPLE_ID
	elif dataset_size == "canonical":
		repo_id = HF_DATASET_CANONICAL_NETO_ID
	elif raw:
		repo_id = HF_DATASET_FULL_RAW_ID if dataset_size == "full" else HF_DATASET_RAW_ID
	else:
		repo_id = HF_DATASET_FULL_ID if dataset_size == "full" else HF_DATASET_ID

	# One-time metadata fetch (small) to get row counts + column names.
	# Streaming=True doesn't materialize data; the call returns an
	# IterableDataset whose info has split sizes from the parquet metadata.
	info_ds = load_dataset(repo_id, config, streaming=True)
	has_val = "validation" in info_ds

	def _split_n(split: str) -> int:
		splits = info_ds[split].info.splits
		if splits is not None and split in splits and splits[split].num_examples is not None:
			return int(splits[split].num_examples)
		# Fallback: HF metadata not populated. Count manually (slow — one
		# pass over the stream just to count). In practice the metadata
		# is always present for our published datasets.
		count = 0
		for _ in info_ds[split]:
			count += 1
		return count

	n_train = _split_n("train")
	n_test = _split_n("test")
	n_val = _split_n("validation") if has_val else 0

	# Determine columns by peeking the first row from each split.
	sample = next(iter(info_ds["train"]))
	all_cols = list(sample.keys())
	exclude = {"Label", "Label_orig", "label", "attack_class"}
	common_features = sorted(set(all_cols) - exclude)

	def _make_factory(split: str):
		"""Build a re-iterable factory for one HF split.

		Each invocation creates a NEW load_dataset call → new streaming
		connection → fresh iteration. This is the load-bearing property:
		consumers (StreamingEncoded.iter_chunks, IDSGenomeStreamer
		train_chunk loop) can iterate multiple times.
		"""
		def factory():
			ds = load_dataset(repo_id, config, streaming=True)[split]
			for batch in ds.iter(batch_size=chunk_size):
				# batch is a dict of column → list-of-values; pd.DataFrame
				# constructs from this efficiently.
				yield pd.DataFrame(batch)
		return factory

	train_factory = _make_factory("train")
	test_factory = _make_factory("test")
	val_factory = _make_factory("validation") if has_val else None

	val_str = f", {n_val:,} val" if has_val else ""
	print(f"  {config.capitalize()} split (streaming): {n_train:,} train, {n_test:,} test{val_str}")
	print(f"  Using {len(common_features)} features (HF metadata; data not materialized)")

	return train_factory, test_factory, val_factory, n_train, n_test, n_val, common_features


def _load_from_huggingface(config: str, dataset_size: str = "subsample", raw: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame | None]:
	"""Load train/test(/validation) from our published HuggingFace dataset.

	Args:
		config: HF dataset config name (e.g., "random", "random_3way").
		dataset_size: "subsample" (~1.3M), "full" (~38.5M, lossy reorg), or
			"canonical" (~45M, Neto's authentic data with NaN preserved).
		raw: when True, load NaN/inf-preserving variant (-raw or -full-raw).
			Ignored when dataset_size="canonical" (canonical IS raw by definition).
	"""
	from datasets import load_dataset

	if dataset_size == "neto_full":
		repo_id = HF_DATASET_NETO_FULL_ID  # 46.7M, 46 features, canonical Neto
	elif dataset_size == "neto_subsample":
		repo_id = HF_DATASET_NETO_SUBSAMPLE_ID  # 1.43M sample of neto_full, 46 features
	elif dataset_size == "canonical":
		repo_id = HF_DATASET_CANONICAL_NETO_ID  # raw flag ignored — canonical IS raw
	elif raw:
		repo_id = HF_DATASET_FULL_RAW_ID if dataset_size == "full" else HF_DATASET_RAW_ID
	else:
		repo_id = HF_DATASET_FULL_ID if dataset_size == "full" else HF_DATASET_ID
	print(f"  Loading from HuggingFace: {repo_id} ({config})...")
	ds = load_dataset(repo_id, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas() if "validation" in ds else None

	# Features: everything except labels
	exclude = {"Label", "Label_orig", "label", "attack_class"}
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
	n_bits=8,  # int OR "auto"
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "random",
	feature_selection: str = "all",
	rest_bits: Optional[int] = None,
	dataset_size: str = "subsample",
	raw: bool = False,
	invalid_encoding: Optional[str] = None,
	auto_max_bits: int = 32,
	encoded_storage: str = "memory",
	storage_dir=None,
	streaming: bool = False,
	streaming_chunk_size: int = 1_000_000,
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
	if dataset_size not in ("subsample", "full", "canonical", "neto_full", "neto_subsample"):
		raise ValueError(f"dataset_size must be 'subsample', 'full', 'canonical', 'neto_full', or 'neto_subsample', got '{dataset_size}'")

	# canonical-neto + neto_full + neto_subsample are intrinsically raw (NaN preserved by build)
	is_raw_data = dataset_size in ("canonical", "neto_full", "neto_subsample") or raw
	if invalid_encoding is None:
		invalid_encoding = "single_bit" if is_raw_data else "none"

	size_label = {
		"full": "FULL 38.5M",
		"canonical": "CANONICAL 45M (bencorn-MERGED)",
		"neto_full": "NETO-FULL 46.7M (canonical)",
		"neto_subsample": "NETO-SUBSAMPLE 1.43M (canonical)",
		"subsample": "1.3M subsample (bencorn)",
	}[dataset_size]
	raw_label = " RAW" if (raw and dataset_size not in ("canonical", "neto_full", "neto_subsample")) else ""
	print(f"Loading CIC-IoT-2023 ({size_label}{raw_label}, split={split}, invalid_encoding={invalid_encoding})...")
	# Pick the right TOP20 list based on feature_selection name.
	# top20 / top20_mi8b / top20_mi96b → canonical lists in this module.
	# top15 / top25 / top20_split → fall back to the canonical RF ranking
	# (HuggingFace-loaded all_ranked list with slicing in encode_features).
	if feature_selection in FEATURE_LIST_BY_NAME:
		top20 = FEATURE_LIST_BY_NAME[feature_selection]
	else:
		top20 = _load_ranked_features()
		if not top20 and feature_selection in ("top15", "top25", "top20_split"):
			raise ValueError("Could not load top-N features from HuggingFace")

	# Phase F: streaming path — never materializes full DataFrames.
	if streaming:
		if encoded_storage != "memory":
			raise NotImplementedError(
				f"streaming=True is incompatible with encoded_storage={encoded_storage!r}; "
				f"streaming-encoded data is consumed chunk-by-chunk, not memmap-backed."
			)
		(train_factory, test_factory, val_factory,
		 n_train, n_test, n_val, common_features) = _load_from_huggingface_streaming(
			split, dataset_size, raw=raw, chunk_size=streaming_chunk_size,
		)
		return encode_and_build_dataset_streaming(
			train_factory, test_factory, val_factory,
			n_train, n_test, n_val,
			common_features=common_features,
			top_features=top20 or [],
			category_names=ATTACK_CLASSES,
			label_binary_col="label",
			label_multi_col="attack_class",
			n_bits=n_bits, method=method,
			feature_selection=feature_selection,
			rest_bits=rest_bits,
			invalid_encoding=invalid_encoding,
			auto_max_bits=auto_max_bits,
		)

	# Phase 3 (Option B): chain the load → encode → build pipeline via an
	# inner closure so the DataFrames live ONLY as encode_and_build_dataset()
	# parameters. This loader function never binds df_train/df_test/df_val to
	# locals, which lets the helper's `del df_*` actually free memory (refcount
	# → 0). On 46M datasets this is a ~10 GB drop before IDSDataset is built.
	def _build(df_train, df_test, common_features, df_val):
		return encode_and_build_dataset(
			df_train, df_test, df_val,
			common_features=common_features,
			top_features=top20 or [],
			category_names=ATTACK_CLASSES,
			label_binary_col="label",
			label_multi_col="attack_class",
			n_bits=n_bits, method=method,
			feature_selection=feature_selection,
			rest_bits=rest_bits,
			invalid_encoding=invalid_encoding,
			auto_max_bits=auto_max_bits,
			encoded_storage=encoded_storage,
			storage_dir=storage_dir,
		)

	return _build(*_load_from_huggingface(split, dataset_size, raw=raw))
