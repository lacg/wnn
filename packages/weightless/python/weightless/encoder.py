"""`ThermometerEncoder` — real-valued features → thermometer (unary) bits.

Each feature gets `n_bits` thresholds; bit_i = 1 iff value >= threshold_i, so a
value lights the bits below its position like a thermometer column. Thresholds
are placed by `method`:

	"distributive"  evenly spaced quantiles of the training data (default)
	"linear"        evenly spaced between min and max
	"gaussian"      quantiles of a normal fitted to the data (needs scipy)

NaN / ±inf encode as all-zero bits. Output is a uint8 0/1 matrix ready for
`WiSARDClassifier` (which packs it). This is the numpy/scikit-learn form of the
research encoder in `wnn.representations.thermometer`; the DataFrame-aware
one (categoricals, invalid-flag bits, streaming quantiles) will become an
adapter over this class.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ThermometerMethod(str, Enum):
	DISTRIBUTIVE = "distributive"
	LINEAR = "linear"
	GAUSSIAN = "gaussian"


def _spread_duplicates(t: np.ndarray) -> np.ndarray:
	"""Strictly increasing thresholds: a repeated quantile (few unique values)
	is nudged to the next representable float so no two bits are identical."""
	t = t.astype(np.float64).copy()
	for i in range(1, len(t)):
		if t[i] <= t[i - 1]:
			t[i] = np.nextafter(t[i - 1], np.inf)
	return t


class ThermometerEncoder(TransformerMixin, BaseEstimator):
	"""Thermometer-encode every column of a numeric matrix.

	Parameters
	----------
	n_bits : int, default 8
		Bits (thresholds) per feature.
	method : {"distributive", "linear", "gaussian"}, default "distributive"
	"""

	def __init__(self, n_bits: int = 8, method: ThermometerMethod | str = ThermometerMethod.DISTRIBUTIVE):
		self.n_bits = n_bits
		self.method = method

	def _thresholds(self, values: np.ndarray) -> np.ndarray:
		nb = int(self.n_bits)
		values = values[np.isfinite(values)]
		if values.size == 0:
			return np.zeros(nb)
		method = ThermometerMethod(self.method)
		if method is ThermometerMethod.LINEAR:
			lo, hi = values.min(), values.max()
			if lo == hi:
				return np.full(nb, lo)
			return np.linspace(lo, hi, nb + 2)[1:-1]
		if method is ThermometerMethod.GAUSSIAN:
			from scipy.stats import norm  # optional dependency: weightless[gaussian]

			mu, sigma = values.mean(), values.std()
			if sigma < 1e-10:
				return np.full(nb, mu)
			return norm.ppf(np.linspace(0, 1, nb + 2)[1:-1], loc=mu, scale=sigma)
		q = np.linspace(0, 100, nb + 2)[1:-1]
		return _spread_duplicates(np.percentile(values, q))

	def fit(self, X, y=None):
		X = np.asarray(X, dtype=np.float64)
		if X.ndim != 2:
			raise ValueError(f"X must be 2-D, got shape {X.shape}")
		if int(self.n_bits) < 1:
			raise ValueError("n_bits must be >= 1")
		self.n_features_in_ = X.shape[1]
		self.thresholds_ = np.stack([self._thresholds(X[:, j]) for j in range(X.shape[1])])
		self.n_bits_out_ = X.shape[1] * int(self.n_bits)
		return self

	def transform(self, X) -> np.ndarray:
		check_is_fitted(self, "thresholds_")
		X = np.asarray(X, dtype=np.float64)
		if X.ndim != 2 or X.shape[1] != self.n_features_in_:
			raise ValueError(f"X must have shape (n, {self.n_features_in_}), got {X.shape}")
		# (n, d, 1) >= (d, nb) → (n, d, nb); NaN compares False → all-zero bits.
		bits = X[:, :, None] >= self.thresholds_[None, :, :]
		return bits.reshape(X.shape[0], self.n_bits_out_).astype(np.uint8)

	def get_feature_names_out(self, input_features=None):
		d, nb = self.thresholds_.shape
		names = input_features if input_features is not None else [f"x{j}" for j in range(d)]
		return np.array([f"{names[j]}>={self.thresholds_[j, i]:.6g}" for j in range(d) for i in range(nb)])
