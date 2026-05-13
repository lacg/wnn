"""
UNSW-NB15 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from .encoder import ThermometerEncoder, ThermometerType
from .encoded_array import LazyEncodedArray, InMemoryEncoded


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
	"""Preprocessed IDS dataset ready for RAM neuron training.

	X_train/X_test/X_val are LazyEncodedArray wrappers (Phase 1+). Phase 1 holds
	a bool numpy array inside InMemoryEncoded (no behavior change vs the legacy
	np.ndarray contract). Phase 2 will switch the inner storage to bit-packed
	uint8 (8x memory reduction). Phase 4 adds MemmapEncoded for disk-backed
	matrices on heavy configs.

	Consumers access bit-pattern data via:
	- dataset.X_train.as_packed_uint8()  → (n_rows, ceil(total_bits/8)) uint8
	- dataset.X_train.total_bits         → bits per row (logical width)
	- dataset.X_train[idx]               → row(s) at idx (numpy, raw layout)
	- dataset.X_train.shape              → (n_rows, total_bits) logical shape
	- len(dataset.X_train)               → n_rows
	"""
	X_train: "LazyEncodedArray"
	y_train_binary: np.ndarray  # (n_train,) int: 0=normal, 1=attack
	y_train_multi: np.ndarray  # (n_train,) int: 0-9 attack category index
	X_test: "LazyEncodedArray"
	y_test_binary: np.ndarray  # (n_test,) int
	y_test_multi: np.ndarray  # (n_test,) int
	encoder: ThermometerEncoder
	category_names: list[str]  # index → category name
	feature_names: list[str]  # feature names in order
	# Optional validation split (80/10/10). When present, test is for threshold
	# calibration and validation is for final reported metrics.
	X_val: "LazyEncodedArray | None" = None
	y_val_binary: np.ndarray | None = None
	y_val_multi: np.ndarray | None = None

	def release_source_data(self):
		"""F-prep hook: release any cached source DataFrames after encoding.

		Default no-op. Loaders that hold references to pandas DataFrames may
		override or pass an explicit closure. Used by Option B (Phase 3) to
		drop ~10 GB of pandas memory after the encoder consumes it.
		"""
		pass


VALID_FEATURE_SELECTIONS = ("all", "top15", "top20", "top25", "top20_split", "top20_mi8b", "top20_mi96b")


def encode_features(
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	common_features: list[str],
	top_features: list[str],
	n_bits: int | str = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	feature_selection: str = "all",
	rest_bits: int | None = None,
	df_val: pd.DataFrame | None = None,
	auto_max_bits: int = 32,
	invalid_encoding: str = "none",
) -> tuple[np.ndarray, np.ndarray, ThermometerEncoder, list[str], np.ndarray | None]:
	"""Shared thermometer encoding logic for all IDS datasets.

	Feature selection modes:
	- "all": All features at uniform n_bits
	- "top15": Top-15 RF features only at n_bits
	- "top20": Top-20 RF features only at n_bits
	- "top25": Top-25 RF features only at n_bits
	- "top20_split": Top-20 at 16 bits + rest at rest_bits

	Args:
		df_train, df_test: DataFrames with numeric feature columns
		common_features: all available features
		top_features: top-N RF features for this dataset
		n_bits: bits per feature (or for rest features in top20_split)
		method: thermometer encoding strategy
		feature_selection: "all", "top20", or "top20_split"
		rest_bits: bits for non-top features in top20_split (defaults to n_bits)

	Returns:
		(X_train, X_test, encoder, used_features, X_val) — X_val is None if df_val not provided
	"""
	if feature_selection not in VALID_FEATURE_SELECTIONS:
		raise ValueError(f"feature_selection must be one of {VALID_FEATURE_SELECTIONS}, got '{feature_selection}'")

	X_val = None

	if feature_selection == "all":
		encoder = ThermometerEncoder(n_bits=n_bits, method=method, auto_max_bits=auto_max_bits,
									 invalid_encoding=invalid_encoding)
		encoder.fit(df_train[common_features])
		X_train_raw, total_bits = encoder.transform(df_train[common_features])
		X_test_raw, _ = encoder.transform(df_test[common_features])
		if df_val is not None:
			X_val_raw, _ = encoder.transform(df_val[common_features])
		else:
			X_val_raw = None
		used_features = common_features
		print(f"  Encoder: {encoder.total_bits} total bits "
			  f"({method.value}, {n_bits} bits/feature, feature_selection=all, {len(common_features)} features)")

	elif feature_selection in ("top15", "top20", "top25", "top20_mi8b", "top20_mi96b"):
		top_n = {"top15": 15, "top20": 20, "top25": 25,
		         "top20_mi8b": 20, "top20_mi96b": 20}[feature_selection]
		selected = [f for f in top_features[:top_n] if f in common_features]
		if len(selected) < top_n:
			available = [f for f in top_features if f in common_features]
			selected = available[:top_n]
			if len(selected) < top_n:
				print(f"  WARNING: Only {len(selected)} of {top_n} top features found in dataset")
		encoder = ThermometerEncoder(n_bits=n_bits, method=method, auto_max_bits=auto_max_bits,
									 invalid_encoding=invalid_encoding)
		encoder.fit(df_train[selected])
		X_train_raw, total_bits = encoder.transform(df_train[selected])
		X_test_raw, _ = encoder.transform(df_test[selected])
		if df_val is not None:
			X_val_raw, _ = encoder.transform(df_val[selected])
		else:
			X_val_raw = None
		used_features = selected
		print(f"  Encoder: {encoder.total_bits} total bits "
			  f"({method.value}, {n_bits} bits/feature, feature_selection={feature_selection}, {len(selected)} features)")

	elif feature_selection == "top20_split":
		rb = rest_bits if rest_bits is not None else n_bits
		top = [f for f in top_features if f in common_features]
		rest = [f for f in common_features if f not in top_features]
		if len(top) < len(top_features):
			missing = set(top_features) - set(common_features)
			print(f"  WARNING: {len(missing)} top features not in dataset: {missing}")

		enc_top = ThermometerEncoder(n_bits=16, method=method, invalid_encoding=invalid_encoding)
		enc_top.fit(df_train[top])
		# top20_split joins two encoders' outputs at the bit-level — easiest to do
		# in bool form, then re-pack at the end.
		X_train_top_packed, top_bits = enc_top.transform(df_train[top])
		X_test_top_packed, _ = enc_top.transform(df_test[top])

		enc_rest = ThermometerEncoder(n_bits=rb, method=method, invalid_encoding=invalid_encoding)
		enc_rest.fit(df_train[rest])
		X_train_rest_packed, rest_bits_count = enc_rest.transform(df_train[rest])
		X_test_rest_packed, _ = enc_rest.transform(df_test[rest])

		def _unpack(packed, n_rows, n_bits_logical):
			return np.unpackbits(packed, axis=1, count=n_bits_logical, bitorder='little').astype(bool)

		n_train = len(df_train)
		n_test = len(df_test)
		top_bool_train = _unpack(X_train_top_packed, n_train, top_bits)
		rest_bool_train = _unpack(X_train_rest_packed, n_train, rest_bits_count)
		top_bool_test = _unpack(X_test_top_packed, n_test, top_bits)
		rest_bool_test = _unpack(X_test_rest_packed, n_test, rest_bits_count)

		joined_train = np.hstack([top_bool_train, rest_bool_train])
		joined_test = np.hstack([top_bool_test, rest_bool_test])
		total_bits = joined_train.shape[1]
		X_train_raw = np.packbits(joined_train.astype(np.uint8), axis=1, bitorder='little')
		X_test_raw = np.packbits(joined_test.astype(np.uint8), axis=1, bitorder='little')
		if df_val is not None:
			X_val_top_packed, _ = enc_top.transform(df_val[top])
			X_val_rest_packed, _ = enc_rest.transform(df_val[rest])
			n_val = len(df_val)
			top_bool_val = _unpack(X_val_top_packed, n_val, top_bits)
			rest_bool_val = _unpack(X_val_rest_packed, n_val, rest_bits_count)
			joined_val = np.hstack([top_bool_val, rest_bool_val])
			X_val_raw = np.packbits(joined_val.astype(np.uint8), axis=1, bitorder='little')
		else:
			X_val_raw = None
		encoder = enc_top
		used_features = top + rest
		print(f"  Encoder: {total_bits} total bits "
			  f"({method.value}, top-{len(top)}@16b + {len(rest)} rest@{rb}b, feature_selection=top20_split)")

	# Wrap packed numpy arrays in InMemoryEncoded (Phase 2: bit-packed uint8).
	X_train = InMemoryEncoded(X_train_raw, total_bits=total_bits)
	X_test = InMemoryEncoded(X_test_raw, total_bits=total_bits)
	X_val = InMemoryEncoded(X_val_raw, total_bits=total_bits) if X_val_raw is not None else None

	return X_train, X_test, encoder, used_features, X_val


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — Option B (centralized): encode + build dataset in one call
# ────────────────────────────────────────────────────────────────────────────


def encode_and_build_dataset(
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	df_val: pd.DataFrame | None,
	*,
	common_features: list[str],
	top_features: list[str],
	category_names: list[str],
	# Label-column metadata (per-dataset)
	label_binary_col: str = "label",
	label_multi_col: str | None = "attack_class",
	# Encoder config
	n_bits: int | str = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	feature_selection: str = "all",
	rest_bits: int | None = None,
	invalid_encoding: str = "none",
	auto_max_bits: int = 32,
) -> IDSDataset:
	"""Build an IDSDataset from DataFrames in one call, releasing them inline.

	This is the canonical entry point — loaders should compose:

	    return encode_and_build_dataset(
	        *_load_from_huggingface(...),  # chained, no caller-local df binding
	        common_features=...,
	        top_features=...,
	        ...,
	    )

	The chained call pattern is load-bearing: this function holds the ONLY
	references to df_train/df_test/df_val. After labels are extracted and
	features encoded, the explicit `del` actually frees the DataFrames
	(refcount → 0). On 46M canonical-neto datasets this releases ~10 GB before
	the IDSDataset is constructed.

	Labels are extracted with `.copy()` to break any views into the DataFrame's
	underlying buffer — otherwise the `del` wouldn't free the buffer.

	Args:
	    df_train, df_test: training and test DataFrames (consumed; deleted before return)
	    df_val: optional validation DataFrame for 80/10/10 splits (consumed if not None)
	    common_features: numeric feature columns present in all dfs
	    top_features: ranked feature list (used by feature_selection variants)
	    category_names: multi-class label names (index → name)
	    label_binary_col: column name for 0/1 binary label (default "label")
	    label_multi_col: column name for multi-class string label (default
	        "attack_class"); if None, multi_class labels mirror binary
	    n_bits, method, feature_selection, rest_bits, invalid_encoding,
	    auto_max_bits: passed through to encode_features().

	Returns:
	    IDSDataset with LazyEncodedArray-wrapped X matrices and int64 labels.
	"""
	# 1. Extract labels first (cheap; .copy() breaks pandas views so del works)
	y_train_binary = df_train[label_binary_col].values.astype(np.int64).copy()
	y_test_binary = df_test[label_binary_col].values.astype(np.int64).copy()
	y_val_binary = (df_val[label_binary_col].values.astype(np.int64).copy()
	                if df_val is not None else None)

	if label_multi_col and label_multi_col in df_train.columns:
		class_to_idx = {cls: i for i, cls in enumerate(category_names)}
		y_train_multi = df_train[label_multi_col].map(
			lambda x: class_to_idx.get(x, 0)
		).values.astype(np.int64).copy()
		y_test_multi = df_test[label_multi_col].map(
			lambda x: class_to_idx.get(x, 0)
		).values.astype(np.int64).copy()
		y_val_multi = (df_val[label_multi_col].map(
			lambda x: class_to_idx.get(x, 0)
		).values.astype(np.int64).copy()
		               if df_val is not None else None)
	else:
		# No multi-class column — duplicate binary so downstream code has a value
		y_train_multi = y_train_binary.copy()
		y_test_multi = y_test_binary.copy()
		y_val_multi = y_val_binary.copy() if y_val_binary is not None else None

	# 2. Encode features (memory-heavy step — X matrices materialized here)
	X_train, X_test, encoder, used_features, X_val = encode_features(
		df_train, df_test, common_features, top_features,
		n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, df_val=df_val,
		invalid_encoding=invalid_encoding,
		auto_max_bits=auto_max_bits,
	)

	# 3. Release source DataFrames. This is the load-bearing memory drop —
	# requires the caller to have used a chained call (no caller-local df
	# bindings). Otherwise these `del` statements only decrement our refcount,
	# not the caller's, and the DataFrames live until the caller returns.
	del df_train, df_test
	if df_val is not None:
		del df_val

	# 4. Print balance stats (legacy info from loaders)
	print(f"  Train: {(y_train_binary == 0).sum():,} normal, "
	      f"{(y_train_binary == 1).sum():,} attack")
	print(f"  Test:  {(y_test_binary == 0).sum():,} normal, "
	      f"{(y_test_binary == 1).sum():,} attack")
	if y_val_binary is not None:
		print(f"  Val:   {(y_val_binary == 0).sum():,} normal, "
		      f"{(y_val_binary == 1).sum():,} attack")

	# 5. Construct and return
	return IDSDataset(
		X_train=X_train,
		y_train_binary=y_train_binary,
		y_train_multi=y_train_multi,
		X_test=X_test,
		y_test_binary=y_test_binary,
		y_test_multi=y_test_multi,
		encoder=encoder,
		category_names=list(category_names),
		feature_names=used_features,
		X_val=X_val,
		y_val_binary=y_val_binary,
		y_val_multi=y_val_multi,
	)


def split_train_validation(
	dataset: IDSDataset,
	val_fraction: float = 0.25,
	seed: int = 42,
) -> tuple[IDSDataset, IDSDataset]:
	"""Split training data into train + validation, keeping original test set separate.

	Uses stratified sampling on multi-class labels to preserve class distribution.
	Returns two IDSDataset objects:
	  - train_dataset: reduced training set (for optimization), validation as "test" slot
	  - test_dataset: original full training set for train, original test set for eval
	                   (used only for final reporting)

	The train_dataset puts validation data in the X_test/y_test slots so the existing
	IDSEvaluator/IDSCache can use it without changes — the Rust evaluator treats
	whatever is in the "test" slot as the eval set.

	Args:
		dataset: Original dataset with full training + test data
		val_fraction: Fraction of training data to hold out (default 0.25 = 25%)
		seed: Random seed for reproducible splits

	Returns:
		(train_val_dataset, test_dataset) — first for optimization, second for final eval
	"""
	n_train = len(dataset.X_train)
	indices = np.arange(n_train)

	train_idx, val_idx = train_test_split(
		indices,
		test_size=val_fraction,
		random_state=seed,
		stratify=dataset.y_train_multi,  # stratify on multi-class for best balance
	)

	# Train-val dataset: reduced train, validation in "test" slot.
	# Use row_subset() to preserve LazyEncodedArray contract (Phase 1+).
	train_val_dataset = IDSDataset(
		X_train=dataset.X_train.row_subset(train_idx),
		y_train_binary=dataset.y_train_binary[train_idx],
		y_train_multi=dataset.y_train_multi[train_idx],
		X_test=dataset.X_train.row_subset(val_idx),
		y_test_binary=dataset.y_train_binary[val_idx],
		y_test_multi=dataset.y_train_multi[val_idx],
		encoder=dataset.encoder,
		category_names=dataset.category_names,
		feature_names=dataset.feature_names,
	)

	# Test dataset: full original train → test (for final eval after optimization)
	test_dataset = IDSDataset(
		X_train=dataset.X_train,
		y_train_binary=dataset.y_train_binary,
		y_train_multi=dataset.y_train_multi,
		X_test=dataset.X_test,
		y_test_binary=dataset.y_test_binary,
		y_test_multi=dataset.y_test_multi,
		encoder=dataset.encoder,
		category_names=dataset.category_names,
		feature_names=dataset.feature_names,
	)

	print(f"Validation split: {len(train_idx):,} train, {len(val_idx):,} val "
		  f"({val_fraction*100:.0f}% holdout, seed={seed})")

	# Show per-class distribution in validation
	for c in range(len(dataset.category_names)):
		n_t = int((dataset.y_train_multi[train_idx] == c).sum())
		n_v = int((dataset.y_train_multi[val_idx] == c).sum())
		pct = n_v / (n_t + n_v) * 100 if (n_t + n_v) > 0 else 0
		print(f"  {dataset.category_names[c]:15s}: {n_t:>7,} train, {n_v:>5,} val ({pct:.1f}%)")

	return train_val_dataset, test_dataset


def create_attack_only_dataset(dataset: IDSDataset) -> IDSDataset:
	"""Create a dataset containing only attack examples for Stage 1 classification.

	Filters to y_binary == 1, remaps multi-class labels from 1-9 → 0-8
	(dropping Normal=0). Returns a new IDSDataset with 9 attack classes.
	"""
	# Filter to attack examples
	train_mask = dataset.y_train_binary == 1
	test_mask = dataset.y_test_binary == 1

	X_train = dataset.X_train.row_subset(train_mask)
	X_test = dataset.X_test.row_subset(test_mask)

	# Remap multi-class labels: 1-9 → 0-8 (drop Normal)
	y_train_multi = dataset.y_train_multi[train_mask] - 1
	y_test_multi = dataset.y_test_multi[test_mask] - 1

	# Clamp any stray Normal labels (y_multi=0 → remapped to -1) to 0
	y_train_multi = np.clip(y_train_multi, 0, 8)
	y_test_multi = np.clip(y_test_multi, 0, 8)

	# All examples are attacks, so binary labels are all 1
	y_train_binary = np.ones(len(X_train), dtype=np.int32)
	y_test_binary = np.ones(len(X_test), dtype=np.int32)

	# Attack category names (excluding Normal)
	attack_names = dataset.category_names[1:]  # ["Analysis", "Backdoor", ..., "Worms"]

	print(f"Attack-only dataset: {len(X_train):,} train, {len(X_test):,} test, "
		  f"{len(attack_names)} classes")
	for i, name in enumerate(attack_names):
		n_train = int((y_train_multi == i).sum())
		n_test = int((y_test_multi == i).sum())
		print(f"  [{i}] {name:15s}: {n_train:>7,} train, {n_test:>6,} test")

	return IDSDataset(
		X_train=X_train,
		y_train_binary=y_train_binary,
		y_train_multi=y_train_multi,
		X_test=X_test,
		y_test_binary=y_test_binary,
		y_test_multi=y_test_multi,
		encoder=dataset.encoder,
		category_names=attack_names,
		feature_names=dataset.feature_names,
	)


# Top-20 features by Random Forest importance (captures ~87% of total importance).
# Based on RF analysis of UNSW-NB15 standard split with 100 trees.
# Names match the HuggingFace dataset columns (lacg030175/UNSW-NB15).
# Note: the original UNSW CSV uses slightly different column names
# (e.g. "dinpkt" vs "Dintpkt", "dmean" vs "dmeansz"). These names
# were updated on 2026-04-09 to match HF — prior runs with top20
# feature selection on HF data silently dropped 7/20 features.
TOP20_RF_FEATURES = [
	"ct_dst_sport_ltm",
	"ct_src_dport_ltm",
	"ct_srv_dst",
	"ct_state_ttl",
	"Dintpkt",       # was "dinpkt" in original CSV
	"dmeansz",       # was "dmean" in original CSV
	"Dpkts",         # was "dpkts" in original CSV (case)
	"dttl",
	"dur",
	"sbytes",
	"Sintpkt",       # was "sinpkt" in original CSV
	"Sjit",          # was "sjit" in original CSV (case)
	"smeansz",       # was "smean" in original CSV
	"Spkts",         # was "spkts" in original CSV (case)
	"sttl",
	"swin",
	"tcprtt",
	"proto",
	"service",
	"state",
]

VALID_FEATURE_SELECTIONS = ("all", "top15", "top20", "top25", "top20_split", "top20_mi8b", "top20_mi96b")

HF_DATASET_ID = "lacg030175/UNSW-NB15"


def _load_from_huggingface(config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame | None]:
	"""Load train/test(/validation) from our published HuggingFace dataset.

	Configs: "temporal"/"standard" (175K/82K), "random" (90/10 deduped),
	"temporal_3way"/"random_3way" (80/10/10 with separate validation).
	"""
	from datasets import load_dataset

	print(f"  Loading from HuggingFace: {HF_DATASET_ID} ({config})...")
	ds = load_dataset(HF_DATASET_ID, config)
	df_train = ds["train"].to_pandas()
	df_test = ds["test"].to_pandas()
	df_val = ds["validation"].to_pandas() if "validation" in ds else None

	# Features: everything except labels and IDs
	exclude = {"id", "label", "Label", "attack_cat", "Attack_cat"}
	if config in ("random", "random_3way"):
		# Random configs have IP/port columns — drop nominal features
		exclude |= {"srcip", "dstip", "sport", "dsport"}
	common_features = sorted((set(df_train.columns) - exclude) & (set(df_test.columns) - exclude))

	# Note: string columns (proto, service, state) are kept as-is.
	# ThermometerEncoder handles them as categoricals with binary coding,
	# which is correct (no false ordering imposed by LabelEncoder).

	val_str = f", {len(df_val):,} val" if df_val is not None else ""
	print(f"  {config.capitalize()} split: {len(df_train):,} train, {len(df_test):,} test{val_str}")
	print(f"  Using {len(common_features)} features")
	return df_train, df_test, common_features, df_val


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
	n_bits: int | str = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "standard",
	feature_selection: str = "all",
	rest_bits: int | None = None,
	auto_max_bits: int = 32,
	invalid_encoding: str = "none",
) -> IDSDataset:
	"""Load UNSW-NB15 with thermometer encoding.

	Primary source: our published HuggingFace dataset (lacg030175/UNSW-NB15).
	Fallback: local CSV files (standard split only).

	Two evaluation protocols:
	- "standard": Original temporal train/test (175K/82K, ~87% RF baseline)
	- "random": Deduped 90/10 random split (1.4M/158K, ~99.6% RF baseline)

	Feature selection modes:
	- "all": All features at uniform n_bits (~321 bits at 8b)
	- "top20": Top-20 RF features only, all at n_bits (e.g. 8b→~148, 16b→~288)
	- "top20_split": All features, top-20 at 16 bits + rest at rest_bits (~varies)

	Args:
		data_dir: path to local data directory (fallback only). Auto-detected if None.
		n_bits: bits per numeric feature for thermometer encoding.
		method: thermometer encoding strategy.
		split: "standard" or "random" evaluation protocol.
		feature_selection: feature selection mode ("all", "top20", "top20_split").
		rest_bits: bits for non-top-20 features in "top20_split" mode. Defaults to n_bits.

	Returns:
		IDSDataset with binary-encoded features and labels.
	"""
	# Alias: "standard" maps to "temporal" (both are the temporal split)
	if split == "standard":
		split = "temporal"
	if split not in ("temporal", "random", "temporal_3way", "random_3way"):
		raise ValueError(f"split must be 'temporal', 'standard', 'random', 'temporal_3way', or 'random_3way', got '{split}'")
	if feature_selection not in VALID_FEATURE_SELECTIONS:
		raise ValueError(f"feature_selection must be one of {VALID_FEATURE_SELECTIONS}, got '{feature_selection}'")

	# ── Load raw data ──────────────────────────────────────────────────
	print(f"Loading UNSW-NB15 (split={split})...")

	df_val = None
	try:
		df_train, df_test, common_features, df_val = _load_from_huggingface(split)
	except Exception as e:
		if split in ("random", "random_3way", "temporal_3way"):
			raise RuntimeError(
				f"{split} split requires HuggingFace dataset ({HF_DATASET_ID}). "
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

	# ── Encode features using shared logic ───────────────
	X_train, X_test, encoder, used_features, X_val = encode_features(
		df_train, df_test, common_features, TOP20_RF_FEATURES,
		n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, df_val=df_val, auto_max_bits=auto_max_bits,
		invalid_encoding=invalid_encoding,
	)

	print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
	print(f"  Train: {(y_train_binary == 0).sum():,} normal, "
		  f"{(y_train_binary == 1).sum():,} attack")
	print(f"  Test:  {(y_test_binary == 0).sum():,} normal, "
		  f"{(y_test_binary == 1).sum():,} attack")

	# Validation labels if present (3-way splits)
	y_val_binary = None
	y_val_multi = None
	if df_val is not None:
		y_val_binary = df_val["label"].values.astype(np.int32)
		y_val_multi = encode_categories(df_val["attack_cat"])
		print(f"  Val:   {(y_val_binary == 0).sum():,} normal, "
			  f"{(y_val_binary == 1).sum():,} attack")

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
