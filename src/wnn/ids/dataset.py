"""
UNSW-NB15 dataset loader with thermometer encoding.

Provides train/test splits as binary numpy arrays ready for RAM neurons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from wnn.representations.thermometer import ThermometerEncoder, ThermometerType
from .encoded_array import (
	LazyEncodedArray,
	InMemoryEncoded,
	MemmapEncoded,
	StreamingEncoded,
	encode_df_to_memmap,
	write_packed_to_memmap,
)


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


VALID_ENCODED_STORAGE = ("memory", "memmap")


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
	encoded_storage: str = "memory",
	storage_dir: "Path | str | None" = None,
) -> tuple[LazyEncodedArray, LazyEncodedArray, ThermometerEncoder, list[str], LazyEncodedArray | None]:
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
		if encoded_storage == "memmap":
			# Chunked encode straight to disk — the one-shot transform() would
			# materialize ~n_rows × total_bits bool intermediates in RAM
			# (~150 GB at 96b × 20f × 37M rows; SIGKILLed flows 4299/4300 03/07).
			X_train = encode_df_to_memmap(encoder, df_train[common_features], storage_dir)
			X_test = encode_df_to_memmap(encoder, df_test[common_features], storage_dir)
			if df_val is not None:
				X_val = encode_df_to_memmap(encoder, df_val[common_features], storage_dir)
			X_train_raw = X_test_raw = X_val_raw = None
			total_bits = X_train.total_bits
		else:
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
		if encoded_storage == "memmap":
			# Chunked encode straight to disk (see feature_selection="all" note).
			X_train = encode_df_to_memmap(encoder, df_train[selected], storage_dir)
			X_test = encode_df_to_memmap(encoder, df_test[selected], storage_dir)
			if df_val is not None:
				X_val = encode_df_to_memmap(encoder, df_val[selected], storage_dir)
			X_train_raw = X_test_raw = X_val_raw = None
			total_bits = X_train.total_bits
		else:
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

	# Wrap packed numpy arrays in the chosen LazyEncodedArray storage.
	# Phase 2: bit-packed uint8 in RAM.
	# Phase 4: opt-in to disk-backed memmap via encoded_storage="memmap".
	if encoded_storage not in VALID_ENCODED_STORAGE:
		raise ValueError(
			f"encoded_storage must be one of {VALID_ENCODED_STORAGE}, got '{encoded_storage}'"
		)
	if encoded_storage == "memmap":
		if X_train_raw is not None:
			# One-shot fallback: only top20_split reaches here (its bit-level
			# two-encoder join isn't chunked yet — do NOT use it with memmap
			# on 10M+-row datasets; see encode_df_to_memmap for the safe path).
			X_train = write_packed_to_memmap(X_train_raw, total_bits=total_bits, storage_dir=storage_dir, suffix=".tmp")
			del X_train_raw
			X_test = write_packed_to_memmap(X_test_raw, total_bits=total_bits, storage_dir=storage_dir, suffix=".tmp")
			del X_test_raw
			if X_val_raw is not None:
				X_val = write_packed_to_memmap(X_val_raw, total_bits=total_bits, storage_dir=storage_dir, suffix=".tmp")
				del X_val_raw
			else:
				X_val = None
		# else: the all/topN branches already encoded chunk-wise into memmaps.
	else:
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
	encoded_storage: str = "memory",
	storage_dir: "Path | str | None" = None,
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
		encoded_storage=encoded_storage,
		storage_dir=storage_dir,
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


def _select_top_features(feature_selection: str, common_features, top_features, rest_bits):
	"""Same feature-selection logic as encode_features(), factored for reuse."""
	if feature_selection == "all":
		return list(common_features), None
	if feature_selection in ("top15", "top20", "top25", "top20_mi8b", "top20_mi96b"):
		top_n = {"top15": 15, "top20": 20, "top25": 25,
		         "top20_mi8b": 20, "top20_mi96b": 20}[feature_selection]
		selected = [f for f in top_features[:top_n] if f in common_features]
		if len(selected) < top_n:
			available = [f for f in top_features if f in common_features]
			selected = available[:top_n]
		return selected, None
	if feature_selection == "top20_split":
		top = [f for f in top_features if f in common_features]
		rest = [f for f in common_features if f not in top_features]
		return top + rest, (top, rest)
	raise ValueError(f"unsupported feature_selection for streaming: {feature_selection}")


def _peek_feature_types(df_first_chunk: pd.DataFrame, columns: list[str]) -> dict[str, str]:
	"""Detect feature types from the first chunk for streaming-fit.

	Mirrors ThermometerEncoder._detect_type() logic. Streaming mode requires
	a single-chunk classification because we can't iterate the full stream
	to count uniques across the whole dataset.
	"""
	types: dict[str, str] = {}
	for col in columns:
		s = df_first_chunk[col].dropna()
		if pd.api.types.is_bool_dtype(s):
			types[col] = "binary"
		elif pd.api.types.is_numeric_dtype(s):
			u = s.unique()
			if len(u) <= 2 and set(u.astype(int).tolist()).issubset({0, 1}):
				types[col] = "binary"
			else:
				types[col] = "numeric"
		else:
			types[col] = "categorical"
	return types


def encode_and_build_dataset_streaming(
	train_factory,
	test_factory,
	val_factory,  # callable or None
	n_train: int,
	n_test: int,
	n_val: int,
	*,
	common_features: list[str],
	top_features: list[str],
	category_names: list[str],
	label_binary_col: str = "label",
	label_multi_col: "str | None" = "attack_class",
	n_bits: "int | str" = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	feature_selection: str = "all",
	rest_bits: "int | None" = None,
	invalid_encoding: str = "none",
	auto_max_bits: int = 32,
) -> IDSDataset:
	"""Streaming counterpart to `encode_and_build_dataset` (Phase F).

	The DataFrame factories yield chunks lazily; the encoder is fitted via
	streaming t-digest quantile estimation; X_train/X_test/X_val become
	`StreamingEncoded` instances that re-fetch from the source on each
	iter_chunks() call.

	Labels (y_*_binary and y_*_multi) ARE materialized — they're small
	relative to features (8 bytes per row for binary int64; 4.6M × 8 ≈
	~370 MB for the largest current dataset). For a 1B-row dataset
	labels are still ~8 GB, which fits comfortably on disk and in RAM.
	True label streaming would require a re-architected IDSEvaluator;
	deferred to a v2 enhancement.

	Args mirror encode_and_build_dataset, with factories replacing dfs.
	"""
	# 1. Select features and detect types from a single peek into train stream
	selected, split_info = _select_top_features(feature_selection, common_features, top_features, rest_bits)
	if feature_selection == "top20_split":
		raise NotImplementedError(
			"feature_selection='top20_split' not yet supported in streaming mode "
			"(requires two encoders fitted on different feature subsets). Use "
			"'top20' or 'all' instead."
		)

	# Peek first chunk for type detection (one network fetch, ~few seconds for
	# a remote HF dataset)
	first_chunk = next(train_factory())
	feature_types = _peek_feature_types(first_chunk, selected)

	# 2. Streaming-fit the encoder on the train factory (one full pass)
	print(f"  Streaming-fit encoder over {n_train:,} train rows...")
	encoder = ThermometerEncoder(
		n_bits=n_bits, method=method, auto_max_bits=auto_max_bits,
		invalid_encoding=invalid_encoding,
	)
	encoder.begin_streaming_fit(feature_types)
	for df_chunk in train_factory():
		encoder.partial_fit(df_chunk[selected])
	encoder.finalize_fit()
	# Capture total_bits AFTER finalize; encoder doesn't expose this directly
	# so we infer from the per-feature widths + invalid_encoding flag bit.
	flag_bits = 1 if invalid_encoding == "single_bit" else 0
	total_bits = 0
	for col in selected:
		ftype = feature_types[col]
		if ftype == "binary":
			total_bits += 1
		elif ftype == "categorical":
			cats = encoder.categories_[col]
			total_bits += max(int(np.ceil(np.log2(max(len(cats), 2)))), 1)
		else:
			total_bits += encoder.per_feature_bits_[col]
		total_bits += flag_bits
	print(f"  Encoder: {total_bits} total bits, {len(selected)} features (streaming, feature_selection={feature_selection})")

	# 3. Materialize labels in one pass over each factory
	def _materialize_labels(factory, n_rows, binary_col, multi_col):
		y_bin = np.empty(n_rows, dtype=np.int64)
		y_multi = np.empty(n_rows, dtype=np.int64) if multi_col is not None else None
		class_to_idx = {cls: i for i, cls in enumerate(category_names)}
		pos = 0
		for df_chunk in factory():
			n = len(df_chunk)
			y_bin[pos:pos + n] = df_chunk[binary_col].values.astype(np.int64)
			if y_multi is not None and multi_col in df_chunk.columns:
				y_multi[pos:pos + n] = df_chunk[multi_col].map(
					lambda x: class_to_idx.get(x, 0)
				).values.astype(np.int64)
			pos += n
		if y_multi is None:
			y_multi = y_bin.copy()
		return y_bin[:pos], y_multi[:pos]

	print(f"  Materializing labels (small): train={n_train:,}, test={n_test:,}...")
	y_train_binary, y_train_multi = _materialize_labels(train_factory, n_train, label_binary_col, label_multi_col)
	y_test_binary, y_test_multi = _materialize_labels(test_factory, n_test, label_binary_col, label_multi_col)
	if val_factory is not None and n_val > 0:
		y_val_binary, y_val_multi = _materialize_labels(val_factory, n_val, label_binary_col, label_multi_col)
	else:
		y_val_binary, y_val_multi = None, None

	# 4. Build StreamingEncoded factories that yield (packed_chunk, labels_chunk)
	# tuples. Each iter_chunks() call invokes the underlying DataFrame factory
	# AGAIN, producing a fresh HF streaming connection.
	def _make_packed_factory(df_factory, binary_col):
		def factory():
			for df_chunk in df_factory():
				packed, _ = encoder.transform(df_chunk[selected])
				labels = df_chunk[binary_col].values.astype(np.int64)
				yield (packed, labels)
		return factory

	X_train = StreamingEncoded(
		_make_packed_factory(train_factory, label_binary_col),
		n_rows=n_train, total_bits=total_bits,
	)
	X_test = StreamingEncoded(
		_make_packed_factory(test_factory, label_binary_col),
		n_rows=n_test, total_bits=total_bits,
	)
	X_val = StreamingEncoded(
		_make_packed_factory(val_factory, label_binary_col),
		n_rows=n_val, total_bits=total_bits,
	) if val_factory is not None and n_val > 0 else None

	return IDSDataset(
		X_train=X_train,
		y_train_binary=y_train_binary,
		y_train_multi=y_train_multi,
		X_test=X_test,
		y_test_binary=y_test_binary,
		y_test_multi=y_test_multi,
		encoder=encoder,
		category_names=list(category_names),
		feature_names=selected,
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


def load_unsw_nb15(
	n_bits: int | str = 8,
	method: ThermometerType = ThermometerType.DISTRIBUTIVE,
	split: str = "standard",
	feature_selection: str = "all",
	rest_bits: int | None = None,
	auto_max_bits: int = 32,
	invalid_encoding: str = "none",
	encoded_storage: str = "memory",
	storage_dir=None,
	raw: bool = False,
) -> IDSDataset:
	"""Load UNSW-NB15 — thin wrapper over the unified UNSWLoader (HuggingFace-only).

	Splits: "standard"/"temporal" (175K/82K), "random" (deduped ~1.27M/158K), plus
	"temporal_3way"/"random_3way" (80/10/10 train/test/val). Feature selection:
	"all"/"top20"/"top20_split"/top20_mi*. No raw variant (the HF copy is pre-cleaned);
	raw=True raises NotImplementedError. The old local-CSV fallback (data_dir) is gone.
	"""
	if split == "standard":
		split = "temporal"
	from .loaders import UNSWLoader, LoadSpec
	return UNSWLoader().load(LoadSpec(
		split=split, n_bits=n_bits, method=method, feature_selection=feature_selection,
		rest_bits=rest_bits, auto_max_bits=auto_max_bits, invalid_encoding=invalid_encoding,
		encoded_storage=encoded_storage, storage_dir=storage_dir, raw=raw,
	))
