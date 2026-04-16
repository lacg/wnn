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
				 auto_max_bits: int = 32):
		"""
		Args:
			n_bits: int for uniform width, or "auto" for per-feature adaptive width.
			method: threshold placement strategy.
			auto_max_bits: maximum bits per feature when n_bits="auto".
		"""
		self.n_bits = n_bits
		self.method = method
		self.auto_max_bits = auto_max_bits
		self.per_feature_bits_: dict[str, int] = {}  # feature_name → actual bits used
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

	def transform(self, df) -> np.ndarray:
		"""Transform DataFrame to binary matrix.

		Returns:
			np.ndarray of shape (n_samples, total_bits) with dtype bool
		"""
		parts = []
		for col in self.feature_names_:
			ftype = self.feature_types_[col]

			if ftype == "binary":
				bits = df[col].fillna(0).values.astype(bool).reshape(-1, 1)
				parts.append(bits)

			elif ftype == "categorical":
				cats = self.categories_[col]
				n_cat_bits = max(int(np.ceil(np.log2(max(len(cats), 2)))), 1)
				# Map each category to an integer, then to binary
				cat_to_idx = {c: i for i, c in enumerate(cats)}
				indices = df[col].fillna(cats[0]).map(
					lambda x, m=cat_to_idx: m.get(x, 0)
				).values.astype(int)
				# Binary encoding (not one-hot — saves bits)
				bit_matrix = np.zeros((len(df), n_cat_bits), dtype=bool)
				for b in range(n_cat_bits):
					bit_matrix[:, b] = (indices >> b) & 1
				parts.append(bit_matrix)

			else:
				# Numeric — thermometer encoding
				thresholds = self.thresholds_[col]
				values = df[col].fillna(0).values.astype(np.float64)
				# bit_i = 1 if value >= threshold_i
				bit_matrix = values[:, np.newaxis] >= thresholds[np.newaxis, :]
				parts.append(bit_matrix)

		return np.hstack(parts)

	def feature_bit_ranges(self) -> dict[str, tuple[int, int]]:
		"""Return (start_bit, end_bit) for each feature."""
		ranges = {}
		offset = 0
		for col in self.feature_names_:
			ftype = self.feature_types_[col]
			if ftype == "binary":
				n = 1
			elif ftype == "categorical":
				n = max(int(np.ceil(np.log2(max(len(self.categories_[col]), 2)))), 1)
			else:
				n = self.per_feature_bits_.get(col, self.n_bits if isinstance(self.n_bits, int) else 8)
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
