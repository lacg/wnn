"""
Thermometer encoding for converting continuous/categorical features to binary vectors.

Three encoding strategies:
- Linear: uniform threshold spacing (baseline)
- Gaussian: thresholds placed via Gaussian CDF (BTHOWeN)
- Distributive: thresholds placed via empirical CDF (best for skewed data)
"""

import numpy as np
import pandas as pd
from enum import Enum


class ThermometerType(Enum):
	LINEAR = "linear"
	GAUSSIAN = "gaussian"
	DISTRIBUTIVE = "distributive"


class InvalidEncoding(Enum):
	"""How the encoder handles NaN / +Inf / -Inf values.

	NONE        -- (back-compat default) replace invalid with 0 / first category;
	               invalid values are indistinguishable from real 0 / first category.
	SINGLE_BIT  -- prepend a 1-bit "is_invalid" flag to each feature's output.
	               flag=1 clears the feature's value bits (they become 0);
	               flag=0 emits the normal thermometer / binary / categorical bits.
	               Cost: +1 bit per feature. Keeps NaN/+Inf/-Inf as a learnable state.
	THREE_BIT   -- (future) three flags: is_nan, is_posinf, is_neginf.
	               Cost: +3 bits per feature. Distinguishes invalid causes.
	"""
	NONE = "none"
	SINGLE_BIT = "single_bit"
	# THREE_BIT = "three_bit"  # reserved — implement if analysis shows predictive value


class ThermometerEncoder:
	"""Encode continuous features as binary thermometer vectors.

	Each feature is compared against n_bits thresholds, producing a unary code
	where all bits below the value's quantile position are 1.

	Example (4 bits, linear on [0, 10]):
		thresholds = [2.0, 4.0, 6.0, 8.0]
		value=5 → [1, 1, 1, 0]  (5 > 2, 5 > 4, 5 < 6, 5 < 8)

	Wait — standard thermometer: bit_i = 1 if value >= threshold_i.
		value=5 → [1, 1, 0, 0]  (5 >= 2, 5 >= 4, 5 < 6, 5 < 8)
	"""

	def __init__(self, n_bits: int | str = 8, method: ThermometerType = ThermometerType.DISTRIBUTIVE,
				 auto_max_bits: int = 32,
				 invalid_encoding: InvalidEncoding | str = InvalidEncoding.NONE):
		"""
		Args:
			n_bits: int for uniform width, or "auto" for per-feature adaptive width.
			method: threshold placement strategy.
			auto_max_bits: maximum bits per feature when n_bits="auto".
			invalid_encoding: how to represent NaN/+Inf/-Inf inputs.
				NONE (default)       -- replace with 0 / first category (back-compat).
				SINGLE_BIT           -- prepend 1 is_invalid flag per feature (+1 bit/feature);
				                        when flag=1, the feature's value bits are cleared.
		"""
		self.n_bits = n_bits
		self.method = method
		self.auto_max_bits = auto_max_bits
		self.invalid_encoding = (invalid_encoding if isinstance(invalid_encoding, InvalidEncoding)
								 else InvalidEncoding(invalid_encoding))
		self.per_feature_bits_: dict[str, int] = {}  # feature_name → actual bits used (excluding invalid flag)
		self.thresholds_: dict[str, np.ndarray] = {}  # feature_name → thresholds
		self.categories_: dict[str, list] = {}  # feature_name → sorted unique values
		self.feature_names_: list[str] = []
		self.feature_types_: dict[str, str] = {}  # "numeric", "binary", "categorical"

	def fit(self, df, feature_config: dict[str, str] | None = None):
		"""Learn thresholds from training data.

		Args:
			df: pandas DataFrame with training data
			feature_config: optional dict mapping feature_name → type override
				("numeric", "binary", "categorical"). If None, auto-detected.
		"""
		self.feature_names_ = []
		self.thresholds_ = {}
		self.categories_ = {}
		self.feature_types_ = {}

		for col in df.columns:
			if col in ("id", "label", "Label", "attack_cat", "Attack_cat"):
				continue

			# Determine feature type
			if feature_config and col in feature_config:
				ftype = feature_config[col]
			else:
				ftype = self._detect_type(df[col])

			self.feature_names_.append(col)
			self.feature_types_[col] = ftype

			if ftype == "binary":
				# No thresholds needed — pass through as single bit
				pass
			elif ftype == "categorical":
				# Store sorted unique values for one-hot-ish encoding
				self.categories_[col] = sorted(df[col].dropna().unique())
			else:
				# Numeric — compute thresholds
				values = df[col].dropna().values.astype(np.float64)
				n_unique = len(np.unique(values))
				if self.n_bits == "auto":
					# Adaptive: use min(unique_values - 1, max_cap)
					feat_bits = min(max(n_unique - 1, 1), self.auto_max_bits)
				else:
					feat_bits = self.n_bits
				self.per_feature_bits_[col] = feat_bits
				self.thresholds_[col] = self._compute_thresholds(values, feat_bits)

		if self.n_bits == "auto" and self.per_feature_bits_:
			bits_list = list(self.per_feature_bits_.values())
			print(f"  Auto thermometer: {len(bits_list)} features, "
				  f"{min(bits_list)}-{max(bits_list)} bits/feature "
				  f"(total {sum(bits_list)} bits, max={self.auto_max_bits})")
			for col, nb in self.per_feature_bits_.items():
				n_unique = len(np.unique(df[col].dropna().values))
				lossless = "lossless" if nb >= n_unique - 1 else f"lossy ({n_unique} unique)"
				print(f"    {col:<22s}: {nb:>3} bits  ({lossless})")

		return self

	def transform(self, df) -> tuple[np.ndarray, int]:
		"""Transform DataFrame to bit-packed binary matrix.

		Returns:
			(packed_matrix, total_bits) where packed_matrix is a uint8
			np.ndarray of shape (n_samples, ceil(total_bits / 8)) — the
			output of np.packbits(bool_matrix, axis=1, bitorder='little').
			total_bits is the logical width (per row) before packing.

			If invalid_encoding=SINGLE_BIT, each feature's output is prepended
			with a 1-bit is_invalid flag; when the flag is 1, the value bits
			are cleared (0). This keeps NaN/+Inf/-Inf as a learnable state
			rather than silently collapsing to the zero encoding.

		Phase 2: returns packed bytes instead of bool matrix to avoid the
		~8x memory blowup of np.ndarray(dtype=bool). Consumers must use
		InMemoryEncoded which auto-detects packed vs bool input.

		For chunked iteration (Option F streaming, post-paper), see
		`iter_chunks()` which yields packed slabs without materializing
		the full matrix.
		"""
		parts = []
		use_flag = (self.invalid_encoding == InvalidEncoding.SINGLE_BIT)
		n_rows = len(df)

		for col in self.feature_names_:
			ftype = self.feature_types_[col]

			# Detect invalid (NaN for all types; +Inf/-Inf also invalid for numeric)
			if ftype == "numeric":
				raw = df[col].values.astype(np.float64)  # np.float64 represents NaN/±Inf natively
				invalid_mask = ~np.isfinite(raw)
			else:
				invalid_mask = df[col].isna().values  # binary & categorical: only NaN is invalid

			if ftype == "binary":
				bits = df[col].fillna(0).values.astype(bool).reshape(-1, 1)
				value_bits = bits

			elif ftype == "categorical":
				cats = self.categories_[col]
				n_cat_bits = max(int(np.ceil(np.log2(max(len(cats), 2)))), 1)
				cat_to_idx = {c: i for i, c in enumerate(cats)}
				indices = df[col].fillna(cats[0]).map(
					lambda x, m=cat_to_idx: m.get(x, 0)
				).values.astype(int)
				bit_matrix = np.zeros((n_rows, n_cat_bits), dtype=bool)
				for b in range(n_cat_bits):
					bit_matrix[:, b] = (indices >> b) & 1
				value_bits = bit_matrix

			else:
				# Numeric — thermometer encoding
				thresholds = self.thresholds_[col]
				# Replace NaN/±Inf with 0 for the dot-product comparison; flag (if used)
				# carries the is_invalid signal. This prevents comparisons against NaN
				# from producing undefined behavior.
				values = np.where(invalid_mask, 0.0, raw)
				bit_matrix = values[:, np.newaxis] >= thresholds[np.newaxis, :]
				value_bits = bit_matrix

			if use_flag:
				# Clear value bits for invalid rows — flag=1 is the only non-zero output
				value_bits = value_bits & ~invalid_mask[:, np.newaxis]
				flag_col = invalid_mask.astype(bool).reshape(-1, 1)
				# Layout: [is_invalid, value_bits...]
				parts.append(np.hstack([flag_col, value_bits]))
			else:
				parts.append(value_bits)

		bool_matrix = np.hstack(parts)
		total_bits = int(bool_matrix.shape[1])
		# np.packbits with bitorder='little' matches Rust PackedBits layout
		# (LSB-first within byte). Each row packs to ceil(total_bits/8) bytes.
		packed = np.packbits(bool_matrix.astype(np.uint8), axis=1, bitorder='little')
		return packed, total_bits

	def iter_chunks(self, df, chunk_size: int):
		"""Yield (packed_chunk, total_bits) tuples by encoding `df` in row slabs.

		Phase 5 F-prep: lets `partial_fit()` (Rust accelerator) consume the
		encoded data without ever materializing the full packed matrix.
		For Phase 2-onward this is unused — callers use `transform()` which
		produces the whole matrix. Option F (streaming, post-paper) wires
		this up via a worker-level chunk loop.

		Behavioral contract:
		- The yielded chunks span all rows of `df` in order; concatenating
		  them along axis=0 is byte-exact equivalent to a single
		  `transform()` call on the full `df`.
		- All chunks report the same `total_bits` (encoder schema is fixed
		  after `fit()`).
		- The last chunk may be smaller than `chunk_size`.

		Args:
		    df: pandas DataFrame with the same feature columns used in fit().
		    chunk_size: rows per chunk. Must be >= 1.

		Yields:
		    (packed_chunk_uint8, total_bits) for each slab.
		"""
		if chunk_size < 1:
			raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
		n_rows = len(df)
		for start in range(0, n_rows, chunk_size):
			end = min(start + chunk_size, n_rows)
			chunk_df = df.iloc[start:end]
			yield self.transform(chunk_df)

	def feature_bit_ranges(self) -> dict[str, tuple[int, int]]:
		"""Return (start_bit, end_bit) for each feature.

		When invalid_encoding=SINGLE_BIT, each feature contributes one extra
		bit for the is_invalid flag (prepended to the value bits).
		"""
		ranges = {}
		offset = 0
		extra_per_feature = 1 if self.invalid_encoding == InvalidEncoding.SINGLE_BIT else 0
		for col in self.feature_names_:
			ftype = self.feature_types_[col]
			if ftype == "binary":
				n = 1
			elif ftype == "categorical":
				n = max(int(np.ceil(np.log2(max(len(self.categories_[col]), 2)))), 1)
			else:
				n = self.per_feature_bits_.get(col, self.n_bits if isinstance(self.n_bits, int) else 8)
			n += extra_per_feature
			ranges[col] = (offset, offset + n)
			offset += n
		return ranges

	@property
	def total_bits(self) -> int:
		"""Total number of output bits."""
		ranges = self.feature_bit_ranges()
		if not ranges:
			return 0
		last = max(ranges.values(), key=lambda x: x[1])
		return last[1]

	def _detect_type(self, series) -> str:
		"""Auto-detect feature type."""
		if series.dtype == object or pd.api.types.is_string_dtype(series):
			return "categorical"
		unique = set(series.dropna().unique())
		if unique <= {0, 1, 0.0, 1.0}:
			return "binary"
		return "numeric"

	def _compute_thresholds(self, values: np.ndarray, feat_bits: int | None = None) -> np.ndarray:
		"""Compute thresholds for a numeric feature.

		Args:
			values: training data for this feature.
			feat_bits: number of bits for this feature (overrides self.n_bits).
		"""
		nb = feat_bits if feat_bits is not None else (self.n_bits if isinstance(self.n_bits, int) else 8)
		if self.method == ThermometerType.LINEAR:
			return self._linear_thresholds(values, nb)
		elif self.method == ThermometerType.GAUSSIAN:
			return self._gaussian_thresholds(values, nb)
		else:
			return self._distributive_thresholds(values, nb)

	def _linear_thresholds(self, values: np.ndarray, nb: int) -> np.ndarray:
		"""Uniform spacing between min and max."""
		vmin, vmax = values.min(), values.max()
		if vmin == vmax:
			return np.full(nb, vmin)
		return np.linspace(vmin, vmax, nb + 2)[1:-1]  # exclude endpoints

	def _gaussian_thresholds(self, values: np.ndarray, nb: int) -> np.ndarray:
		"""Thresholds placed via Gaussian CDF (more bits in the center)."""
		from scipy.stats import norm
		mu, sigma = values.mean(), values.std()
		if sigma < 1e-10:
			return np.full(nb, mu)
		# Quantiles of standard normal, mapped to data distribution
		quantiles = np.linspace(0, 1, nb + 2)[1:-1]
		return norm.ppf(quantiles, loc=mu, scale=sigma)

	def _distributive_thresholds(self, values: np.ndarray, nb: int) -> np.ndarray:
		"""Thresholds placed via empirical CDF (adapts to any distribution)."""
		# Place thresholds at evenly-spaced quantiles of the data
		quantiles = np.linspace(0, 100, nb + 2)[1:-1]
		thresholds = np.percentile(values, quantiles)
		# Deduplicate (can happen for features with few unique values)
		# If duplicates exist, spread them slightly
		for i in range(1, len(thresholds)):
			if thresholds[i] <= thresholds[i - 1]:
				thresholds[i] = thresholds[i - 1] + 1e-10
		return thresholds

	# ──────────────────────────────────────────────────────────────────────
	# Phase F: streaming fit (online quantile estimation via t-digest)
	# ──────────────────────────────────────────────────────────────────────

	def begin_streaming_fit(self, feature_types: "dict[str, str]"):
		"""Start a streaming fit pass with explicit per-feature types.

		Unlike `fit()`, which auto-detects feature types from the full
		DataFrame, streaming fit requires the caller to declare types
		up front (we can't reliably auto-detect from a single chunk).
		The dict maps feature name → "numeric" | "binary" | "categorical".

		After `begin_streaming_fit()`, call `partial_fit(chunk_df)`
		repeatedly with each chunk, then `finalize_fit()` to convert
		accumulated statistics into thresholds/categories.

		Args:
		    feature_types: explicit type assignment for every feature
		        the encoder will see. Must be consistent across chunks.
		"""
		self.feature_names_ = [c for c in feature_types if c not in
			("id", "label", "Label", "attack_cat", "Attack_cat")]
		self.feature_types_ = {c: feature_types[c] for c in self.feature_names_}
		self.thresholds_ = {}
		self.categories_ = {}
		self.per_feature_bits_ = {}
		# Per-column streaming statistics accumulator
		self._streaming_stats_ = {
			col: _make_streaming_stats(ftype) for col, ftype in self.feature_types_.items()
		}
		# Tracks total rows seen across partial_fit calls
		self._streaming_rows_ = 0
		return self

	def partial_fit(self, df_chunk):
		"""Update streaming statistics with one chunk.

		Call after `begin_streaming_fit()`, then again per chunk, then
		`finalize_fit()`. Each chunk's per-column data updates the
		appropriate streaming accumulator (t-digest for numeric quantiles,
		Welford for mean/std, set for categorical uniques, running
		min/max for linear thresholds).
		"""
		if not hasattr(self, "_streaming_stats_") or self._streaming_stats_ is None:
			raise RuntimeError(
				"partial_fit called without begin_streaming_fit (or after finalize_fit)"
			)
		for col in self.feature_names_:
			if col not in df_chunk.columns:
				continue
			self._streaming_stats_[col].update(df_chunk[col])
		self._streaming_rows_ += len(df_chunk)
		return self

	def finalize_fit(self):
		"""Convert accumulated streaming statistics to thresholds/categories.

		After this call the encoder is fully fitted and ready for `transform()`
		or `iter_chunks()`. Streaming state is cleared.
		"""
		if not hasattr(self, "_streaming_stats_") or self._streaming_stats_ is None:
			raise RuntimeError("finalize_fit called without begin_streaming_fit")
		if self._streaming_rows_ == 0:
			raise RuntimeError("finalize_fit called before any partial_fit data was seen")

		for col, stats in self._streaming_stats_.items():
			ftype = self.feature_types_[col]
			if ftype == "binary":
				# Nothing to fit — binary uses single bit
				pass
			elif ftype == "categorical":
				self.categories_[col] = sorted(stats.uniques())
			else:  # numeric
				# Determine bit width
				if self.n_bits == "auto":
					n_unique = stats.approx_unique_count()
					feat_bits = min(max(n_unique - 1, 1), self.auto_max_bits)
				else:
					feat_bits = self.n_bits
				self.per_feature_bits_[col] = feat_bits
				# Compute thresholds via the streaming-aware method dispatch
				self.thresholds_[col] = self._compute_streaming_thresholds(stats, feat_bits)

		# Done — clear streaming state to release t-digest memory
		self._streaming_stats_ = None
		return self

	def _compute_streaming_thresholds(self, stats, feat_bits: int) -> np.ndarray:
		"""Method-specific threshold extraction from streaming statistics."""
		if self.method == ThermometerType.LINEAR:
			return self._linear_thresholds_streaming(stats, feat_bits)
		elif self.method == ThermometerType.GAUSSIAN:
			return self._gaussian_thresholds_streaming(stats, feat_bits)
		else:
			return self._distributive_thresholds_streaming(stats, feat_bits)

	def _linear_thresholds_streaming(self, stats, nb: int) -> np.ndarray:
		vmin, vmax = stats.min_value(), stats.max_value()
		if vmin is None or vmax is None or vmin == vmax:
			return np.full(nb, vmin if vmin is not None else 0.0)
		return np.linspace(vmin, vmax, nb + 2)[1:-1]

	def _gaussian_thresholds_streaming(self, stats, nb: int) -> np.ndarray:
		from scipy.stats import norm
		mu, sigma = stats.mean(), stats.std()
		if sigma is None or sigma < 1e-10:
			return np.full(nb, mu if mu is not None else 0.0)
		quantiles = np.linspace(0, 1, nb + 2)[1:-1]
		return norm.ppf(quantiles, loc=mu, scale=sigma)

	def _distributive_thresholds_streaming(self, stats, nb: int) -> np.ndarray:
		# t-digest-based quantile extraction
		quantiles = np.linspace(0, 1, nb + 2)[1:-1]  # [0, 1] scale, exclude endpoints
		thresholds = np.array([stats.quantile(q) for q in quantiles])
		# Same dedup as the in-memory path: nudge tied thresholds apart slightly
		for i in range(1, len(thresholds)):
			if thresholds[i] <= thresholds[i - 1]:
				thresholds[i] = thresholds[i - 1] + 1e-10
		return thresholds


# ──────────────────────────────────────────────────────────────────────────
# Streaming statistics collectors (one per feature column during partial_fit)
# ──────────────────────────────────────────────────────────────────────────


def _make_streaming_stats(ftype: str):
	"""Build a per-column streaming stats accumulator for the given feature type."""
	if ftype == "numeric":
		return _NumericStreamingStats()
	elif ftype == "categorical":
		return _CategoricalStreamingStats()
	elif ftype == "binary":
		return _BinaryStreamingStats()
	raise ValueError(f"unknown feature type for streaming stats: {ftype}")


class _NumericStreamingStats:
	"""Online statistics for a numeric column: running min/max, Welford
	mean/variance, t-digest for quantiles, and a sample set for approximate
	unique-count (for auto-thermometer bit width).

	Memory: O(1) for moments + O(centroids) for t-digest (~50-100 centroids
	per default compression) + O(min(N, sample_cap)) for unique sample.
	Independent of total stream size N.
	"""

	# Cap on the unique-value sample (auto-thermometer needs n_unique count,
	# bounded so a 1B-row stream doesn't blow this up; for cardinality >
	# auto_max_bits the threshold is clamped anyway).
	_UNIQUE_SAMPLE_CAP = 100_000

	def __init__(self):
		from pytdigest import TDigest
		self._digest = TDigest()
		self._min = None
		self._max = None
		# Welford for mean/std
		self._n = 0
		self._mean = 0.0
		self._m2 = 0.0
		# Bounded unique sample (set of float values seen) — for auto-bits only
		self._unique_sample: "set[float]" = set()
		self._unique_capped = False

	def update(self, series):
		"""Update with a pandas Series or numpy array of values."""
		# Drop NaN/Inf (matches encoder fit logic)
		import pandas as pd
		if isinstance(series, pd.Series):
			values = series.values
		else:
			values = np.asarray(series)
		values = values.astype(np.float64, copy=False)
		finite_mask = np.isfinite(values)
		values = values[finite_mask]
		if values.size == 0:
			return

		# t-digest
		self._digest.update(values)
		# min/max
		v_min, v_max = float(values.min()), float(values.max())
		self._min = v_min if self._min is None else min(self._min, v_min)
		self._max = v_max if self._max is None else max(self._max, v_max)
		# Welford (batch update via Chan's parallel algorithm)
		n_chunk = values.size
		chunk_mean = float(values.mean())
		chunk_m2 = float(((values - chunk_mean) ** 2).sum())
		delta = chunk_mean - self._mean
		new_n = self._n + n_chunk
		new_mean = (self._n * self._mean + n_chunk * chunk_mean) / new_n
		new_m2 = self._m2 + chunk_m2 + (delta ** 2) * self._n * n_chunk / new_n
		self._n, self._mean, self._m2 = new_n, new_mean, new_m2
		# Unique sample (bounded — only adds new values until cap)
		if not self._unique_capped:
			for v in values:
				if len(self._unique_sample) >= self._UNIQUE_SAMPLE_CAP:
					self._unique_capped = True
					break
				self._unique_sample.add(float(v))

	def quantile(self, q: float) -> float:
		"""Approximate quantile via t-digest. q in [0, 1]."""
		return float(self._digest.inverse_cdf(q))

	def min_value(self):
		return self._min

	def max_value(self):
		return self._max

	def mean(self):
		return self._mean if self._n > 0 else None

	def std(self):
		if self._n < 2:
			return 0.0
		return float(np.sqrt(self._m2 / (self._n - 1)))

	def approx_unique_count(self) -> int:
		"""Approximate count of unique values; capped at _UNIQUE_SAMPLE_CAP."""
		if self._unique_capped:
			return self._UNIQUE_SAMPLE_CAP
		return len(self._unique_sample)


class _CategoricalStreamingStats:
	"""Streaming set of unique values for a categorical column."""

	def __init__(self):
		self._uniques: set = set()

	def update(self, series):
		import pandas as pd
		if isinstance(series, pd.Series):
			values = series.dropna().unique()
		else:
			arr = np.asarray(series)
			values = np.unique(arr[~_is_nan_array(arr)]) if arr.size else arr
		for v in values:
			self._uniques.add(v)

	def uniques(self) -> list:
		return list(self._uniques)


class _BinaryStreamingStats:
	"""No-op streaming stats for binary columns (no thresholds needed)."""

	def update(self, series):
		pass


def _is_nan_array(arr):
	"""Best-effort NaN mask for arbitrary dtypes (object arrays included)."""
	try:
		return np.isnan(arr.astype(float))
	except (ValueError, TypeError):
		# Object dtype with non-numeric strings — assume nothing is NaN
		return np.zeros(arr.shape, dtype=bool)
