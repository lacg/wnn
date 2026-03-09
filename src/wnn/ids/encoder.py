"""
Thermometer encoding for converting continuous/categorical features to binary vectors.

Three encoding strategies:
- Linear: uniform threshold spacing (baseline)
- Gaussian: thresholds placed via Gaussian CDF (BTHOWeN)
- Distributive: thresholds placed via empirical CDF (best for skewed data)
"""

import numpy as np
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

	def __init__(self, n_bits: int = 8, method: ThermometerType = ThermometerType.DISTRIBUTIVE):
		self.n_bits = n_bits
		self.method = method
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
				self.thresholds_[col] = self._compute_thresholds(values)

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
				n = self.n_bits
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
		if series.dtype == object:
			return "categorical"
		unique = set(series.dropna().unique())
		if unique <= {0, 1, 0.0, 1.0}:
			return "binary"
		return "numeric"

	def _compute_thresholds(self, values: np.ndarray) -> np.ndarray:
		"""Compute n_bits thresholds for a numeric feature."""
		if self.method == ThermometerType.LINEAR:
			return self._linear_thresholds(values)
		elif self.method == ThermometerType.GAUSSIAN:
			return self._gaussian_thresholds(values)
		else:
			return self._distributive_thresholds(values)

	def _linear_thresholds(self, values: np.ndarray) -> np.ndarray:
		"""Uniform spacing between min and max."""
		vmin, vmax = values.min(), values.max()
		if vmin == vmax:
			return np.full(self.n_bits, vmin)
		return np.linspace(vmin, vmax, self.n_bits + 2)[1:-1]  # exclude endpoints

	def _gaussian_thresholds(self, values: np.ndarray) -> np.ndarray:
		"""Thresholds placed via Gaussian CDF (more bits in the center)."""
		from scipy.stats import norm
		mu, sigma = values.mean(), values.std()
		if sigma < 1e-10:
			return np.full(self.n_bits, mu)
		# Quantiles of standard normal, mapped to data distribution
		quantiles = np.linspace(0, 1, self.n_bits + 2)[1:-1]
		return norm.ppf(quantiles, loc=mu, scale=sigma)

	def _distributive_thresholds(self, values: np.ndarray) -> np.ndarray:
		"""Thresholds placed via empirical CDF (adapts to any distribution)."""
		# Place thresholds at evenly-spaced quantiles of the data
		quantiles = np.linspace(0, 100, self.n_bits + 2)[1:-1]
		thresholds = np.percentile(values, quantiles)
		# Deduplicate (can happen for features with few unique values)
		# If duplicates exist, spread them slightly
		for i in range(1, len(thresholds)):
			if thresholds[i] <= thresholds[i - 1]:
				thresholds[i] = thresholds[i - 1] + 1e-10
		return thresholds
