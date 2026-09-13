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

BITS_ONLY = "WiSARDClassifier consumes bits (0/1) only; sklearn's generic checks fit it on random floats"

FLOAT_X_CHECKS = [
	"check_fit_score_takes_y", "check_estimators_overwrite_params", "check_dont_overwrite_parameters",
	"check_estimators_fit_returns_self", "check_readonly_memmap_input", "check_n_features_in_after_fitting",
	"check_estimators_dtypes", "check_sample_weights_pandas_series", "check_sample_weights_not_an_array",
	"check_sample_weights_list", "check_sample_weights_shape", "check_sample_weights_not_overwritten",
	"check_sample_weight_equivalence_on_dense_data", "check_dtype_object", "check_pipeline_consistency",
	"check_estimators_nan_inf", "check_estimators_pickle", "check_f_contiguous_array_estimator",
	"check_classifier_data_not_an_array", "check_classifiers_one_label", "check_classifiers_one_label_sample_weights",
	"check_classifiers_classes", "check_estimators_partial_fit_n_features", "check_classifiers_train",
	"check_supervised_y_2d", "check_decision_proba_consistency", "check_methods_sample_order_invariance",
	"check_methods_subset_invariance", "check_fit2d_1sample", "check_fit2d_1feature", "check_dict_unchanged",
	"check_fit_idempotent", "check_fit_check_is_fitted", "check_n_features_in", "check_fit2d_predict1d",
]


def test_classifier_passes_sklearn_checks_except_float_input():
	check_estimator(
		WiSARDClassifier(neurons_per_class=4, bits_per_neuron=4, random_state=0),
		expected_failed_checks={name: BITS_ONLY for name in FLOAT_X_CHECKS},
	)


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
