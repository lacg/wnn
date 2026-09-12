"""`WiSARDClassifier` — the scikit-learn estimator over `weightless._core.SparseModel`.

X is a BIT matrix (0/1, any integer or bool dtype, shape (n, total_bits)); the
estimator never encodes. One discriminator (cluster) per class, `neurons_per_class`
RAM neurons each, every neuron observing `bits_per_neuron` input positions drawn
without replacement from `random_state`. Training is order-independent; scoring is
the mean vote of a class's neurons.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._cell_mode import CellMode
from ._core import SparseModel

_BACKENDS = ("auto", "cpu", "gpu")


def _as_bits(X) -> np.ndarray:
	"""Validate a 0/1 matrix and return it as uint8 (n, total_bits)."""
	X = np.asarray(X)
	if X.ndim != 2:
		raise ValueError(f"X must be 2-D (n_samples, n_bits); got shape {X.shape}")
	if X.dtype == bool:
		return X.astype(np.uint8)
	if not np.issubdtype(X.dtype, np.integer):
		if np.issubdtype(X.dtype, np.floating) and np.all((X == 0) | (X == 1)):
			return X.astype(np.uint8)
		raise TypeError(
			"X must be bits (bool or 0/1 integers). Encode real-valued features first, "
			"e.g. weightless.ThermometerEncoder in a Pipeline."
		)
	if X.size and (X.min() < 0 or X.max() > 1):
		raise ValueError("X must contain only 0 and 1")
	return X.astype(np.uint8)


def _pack(bits: np.ndarray) -> np.ndarray:
	"""(n, total_bits) 0/1 → (n, ceil(total_bits/8)) uint8, LSB-first — the core's layout."""
	return np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))


class WiSARDClassifier(ClassifierMixin, BaseEstimator):
	"""Weightless (RAM) neural-network classifier.

	Parameters
	----------
	neurons_per_class : int, default 50
	bits_per_neuron : int, default 16
		Input positions each neuron observes (its address width). Above 64 the
		address is a hash of the observed bits.
	cell_mode : CellMode or int, default CellMode.QUAD_WEIGHTED
	empty_value : float, default 0.5
		Weight of an untrained cell — TERNARY only (ignored by every other mode).
	connections : ndarray of shape (n_classes, neurons_per_class, bits_per_neuron), optional
		Explicit connectivity. None draws it from `random_state`.
	coverage_aware : bool, default False
		Score a sparse miss as "no vote" (cell 0) instead of the mode's untrained cell.
	backend : {"auto", "cpu", "gpu"}, default "auto"
		Scoring device. "gpu" is Metal on Apple silicon; "auto" falls back to CPU.
	random_state : int, optional
		Seeds the connectivity draw and the PLN/QSR read coins.
	"""

	def __init__(
		self,
		neurons_per_class: int = 50,
		bits_per_neuron: int = 16,
		cell_mode: CellMode | int = CellMode.QUAD_WEIGHTED,
		empty_value: float = 0.5,
		connections: np.ndarray | None = None,
		coverage_aware: bool = False,
		backend: str = "auto",
		random_state: int | None = None,
	):
		self.neurons_per_class = neurons_per_class
		self.bits_per_neuron = bits_per_neuron
		self.cell_mode = cell_mode
		self.empty_value = empty_value
		self.connections = connections
		self.coverage_aware = coverage_aware
		self.backend = backend
		self.random_state = random_state

	# ---- fitting -------------------------------------------------------------

	def _validate_params_(self) -> CellMode:
		if self.neurons_per_class < 1 or self.bits_per_neuron < 1:
			raise ValueError("neurons_per_class and bits_per_neuron must be >= 1")
		if self.backend not in _BACKENDS:
			raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
		return CellMode(int(self.cell_mode))

	def _draw_connections(self, n_classes: int, total_bits: int) -> np.ndarray:
		if self.connections is not None:
			c = np.asarray(self.connections, dtype=np.int64)
			want = (n_classes, self.neurons_per_class, self.bits_per_neuron)
			if c.shape != want:
				raise ValueError(f"connections must have shape {want}, got {c.shape}")
			if c.min() < 0 or c.max() >= total_bits:
				raise ValueError(f"connections must index 0..{total_bits - 1}")
			return c
		if self.bits_per_neuron > total_bits:
			raise ValueError(
				f"bits_per_neuron={self.bits_per_neuron} exceeds the input width {total_bits}; "
				"pass explicit connections to sample with replacement"
			)
		rng = np.random.default_rng(self.random_state)
		n = n_classes * self.neurons_per_class
		c = np.empty((n, self.bits_per_neuron), dtype=np.int64)
		for i in range(n):
			c[i] = rng.choice(total_bits, size=self.bits_per_neuron, replace=False)
		return c.reshape(n_classes, self.neurons_per_class, self.bits_per_neuron)

	def _init_model(self, classes: np.ndarray, total_bits: int) -> None:
		mode = self._validate_params_()
		self.classes_ = classes
		self.n_features_in_ = total_bits
		self.connections_ = self._draw_connections(len(classes), total_bits)
		self._model = SparseModel(
			len(classes),
			self.neurons_per_class,
			self.bits_per_neuron,
			total_bits,
			int(mode),
			np.ascontiguousarray(self.connections_.reshape(-1)),
		)
		self._mode = mode

	def _encode_y(self, y) -> np.ndarray:
		y = np.asarray(y).reshape(-1)
		idx = np.searchsorted(self.classes_, y)
		bad = (idx >= len(self.classes_)) | (self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y)
		if bad.any():
			raise ValueError(f"y contains labels unseen at init: {np.unique(y[bad])[:5]}")
		return idx.astype(np.int64)

	def fit(self, X, y, sample_weight=None):
		"""Reset and train on (X, y). `sample_weight` is rounded to an integer vote weight >= 1:
		it scales an example's vote (TERNARY sums, QUAD net); in the QUAD modes the observation
		count still counts examples, so a weight of 2 is NOT two copies of the row."""
		bits = _as_bits(X)
		y = np.asarray(y).reshape(-1)
		if len(y) != bits.shape[0]:
			raise ValueError("X and y have different numbers of rows")
		self._init_model(np.unique(y), bits.shape[1])
		return self._train(bits, y, sample_weight)

	def partial_fit(self, X, y, classes=None, sample_weight=None):
		"""Fold (X, y) into the memory. RAM writes accumulate, so
		`partial_fit(A); partial_fit(B)` equals `fit(A + B)` exactly.
		`classes` is required on the first call."""
		bits = _as_bits(X)
		y = np.asarray(y).reshape(-1)
		if not hasattr(self, "_model"):
			if classes is None:
				raise ValueError("classes must be passed on the first partial_fit call")
			self._init_model(np.unique(np.asarray(classes)), bits.shape[1])
		elif bits.shape[1] != self.n_features_in_:
			raise ValueError(f"X has {bits.shape[1]} bits, the model was built for {self.n_features_in_}")
		return self._train(bits, y, sample_weight)

	def _train(self, bits: np.ndarray, y: np.ndarray, sample_weight):
		labels = self._encode_y(y)
		weights = None
		if sample_weight is not None:
			w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
			if len(w) != len(labels):
				raise ValueError("sample_weight has the wrong length")
			weights = np.ascontiguousarray(np.maximum(np.rint(w), 1).astype(np.uint32))
		self.n_cells_ = self._model.train(_pack(bits), np.ascontiguousarray(labels), weights)
		return self

	# ---- scoring -------------------------------------------------------------

	def _scores(self, X, expected: bool) -> np.ndarray:
		check_is_fitted(self, "_model")
		bits = _as_bits(X)
		if bits.shape[1] != self.n_features_in_:
			raise ValueError(f"X has {bits.shape[1]} bits, the model was built for {self.n_features_in_}")
		read_mode = int(self._mode.expected_read_mode) if expected else None
		flat = self._model.forward(
			_pack(bits),
			float(self.empty_value),
			bool(self.coverage_aware),
			int(self.random_state or 0),
			self.backend,
			read_mode,
		)
		return np.asarray(flat, dtype=np.float32).reshape(bits.shape[0], len(self.classes_))

	def decision_function(self, X):
		"""Raw per-class vote in [0, 1], shape (n, n_classes); (n,) margin for two classes."""
		s = self._scores(X, expected=False)
		if len(self.classes_) == 2:
			return s[:, 1] - s[:, 0]
		return s

	def predict(self, X):
		s = self._scores(X, expected=False)
		return self.classes_[np.argmax(s, axis=1)]

	def predict_proba(self, X):
		"""Row-normalised votes. For PLN/QSR this is the EXPECTED read
		(deterministic), so probabilities are stable across calls."""
		s = self._scores(X, expected=True).astype(np.float64)
		tot = s.sum(axis=1, keepdims=True)
		out = np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / s.shape[1])
		return out

	# ---- export --------------------------------------------------------------

	def export_keys(self) -> dict[str, np.ndarray]:
		"""The trained memory as sorted keys: offsets/counts per neuron (neuron-major,
		class-major), the keys (addresses) and their cell values."""
		check_is_fitted(self, "_model")
		offsets, counts, keys, values = self._model.export_keys()
		return {
			"offsets": np.asarray(offsets),
			"counts": np.asarray(counts),
			"keys": np.asarray(keys),
			"values": np.asarray(values),
		}

	@property
	def n_cells_(self) -> int:
		return self._n_cells

	@n_cells_.setter
	def n_cells_(self, v: int) -> None:
		self._n_cells = int(v)

	def __sklearn_is_fitted__(self) -> bool:
		return hasattr(self, "_model")

	def _more_tags(self):
		return {"binary_only": False, "requires_positive_X": True, "X_types": ["2darray"], "poor_score": True}
