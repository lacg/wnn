"""scikit-learn's own estimator checks.

The classifier consumes BITS by design (the encoder is an explicit pipeline step,
never a hidden binarisation), so every check that fits it on sklearn's random
FLOAT matrices is declared an expected failure with that one reason. Everything
else must pass.
"""

import pickle

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from weightless import CellMode, ThermometerEncoder, WiSARDClassifier

NOT_BITS = "must be bits"


def test_classifier_passes_sklearn_checks_except_float_input():
	"""Every failure must be the bits-only refusal (sklearn feeds random floats);
	the check NAMES differ across sklearn versions, so we classify by cause, not
	by name — any other failure is a real bug."""
	results = check_estimator(WiSARDClassifier(neurons_per_class=4, bits_per_neuron=4, random_state=0), on_fail=None)
	statuses = {r["status"] for r in results}
	assert "passed" in statuses
	def cause_chain(e):
		seen = []
		while e is not None and e not in seen:
			seen.append(e)
			yield str(e)
			e = e.__cause__ or e.__context__

	real = [
		(r["check_name"], str(r["exception"])[:200])
		for r in results
		if r["status"] == "failed" and not any(NOT_BITS in m for m in cause_chain(r["exception"]))
	]
	assert not real, f"non-bits failures: {real}"
	assert sum(r["status"] == "passed" for r in results) >= 20


def test_encoder_passes_sklearn_checks():
	check_estimator(ThermometerEncoder(n_bits=3))


def _bits(n=300, seed=0):
	rng = np.random.default_rng(seed)
	protos = rng.integers(0, 2, (3, 48), dtype=np.uint8)
	y = rng.integers(0, 3, n)
	return protos[y] ^ (rng.random((n, 48)) < 0.05).astype(np.uint8), y


@pytest.mark.parametrize("mode", list(CellMode))
def test_pickle_keeps_partial_fit_exact(mode):
	X, y = _bits()
	a = WiSARDClassifier(neurons_per_class=6, bits_per_neuron=10, cell_mode=mode, random_state=1)
	a.partial_fit(X[:150], y[:150], classes=[0, 1, 2])
	b = pickle.loads(pickle.dumps(a))
	a.partial_fit(X[150:], y[150:])
	b.partial_fit(X[150:], y[150:])
	whole = WiSARDClassifier(neurons_per_class=6, bits_per_neuron=10, cell_mode=mode, random_state=1).fit(X, y)
	np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
	np.testing.assert_array_equal(whole.decision_function(X), b.decision_function(X))
	for k, v in whole.export_keys().items():
		np.testing.assert_array_equal(v, b.export_keys()[k])


def test_cell_mode_accepts_names_ints_and_members():
	X, y = _bits(100)
	for cm in ("qsr", "QSR", 4, CellMode.QSR):
		clf = WiSARDClassifier(neurons_per_class=2, bits_per_neuron=4, cell_mode=cm).fit(X, y)
		assert clf._mode is CellMode.QSR
