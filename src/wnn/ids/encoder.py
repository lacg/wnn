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
