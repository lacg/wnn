import numpy as np
import pytest

from weightless import ThermometerEncoder, ThermometerMethod


def test_shapes_and_monotone_columns():
	rng = np.random.default_rng(0)
	X = rng.normal(size=(200, 3))
	enc = ThermometerEncoder(n_bits=6).fit(X)
	B = enc.transform(X)
	assert B.shape == (200, 18) and B.dtype == np.uint8
	assert enc.n_bits_out_ == 18 and enc.thresholds_.shape == (3, 6)
	# thermometer property: within a feature, bits are non-increasing left→right
	cols = B.reshape(200, 3, 6)
	assert np.all(np.diff(cols.astype(int), axis=2) <= 0)
	# thresholds strictly increasing
	assert np.all(np.diff(enc.thresholds_, axis=1) > 0)


def test_distributive_balances_bit_load():
	X = np.random.default_rng(1).exponential(size=(5000, 1))
	B = ThermometerEncoder(n_bits=9).fit_transform(X)
	load = B.mean(axis=0)
	# quantile thresholds → bit i lit for ~(9-i)/10 of rows
	np.testing.assert_allclose(load, np.linspace(0.9, 0.1, 9), atol=0.03)


def test_linear_thresholds_and_constant_feature():
	X = np.array([[0.0, 5.0], [10.0, 5.0], [5.0, 5.0]])
	enc = ThermometerEncoder(n_bits=4, method=ThermometerMethod.LINEAR).fit(X)
	np.testing.assert_allclose(enc.thresholds_[0], [2, 4, 6, 8])
	np.testing.assert_allclose(enc.thresholds_[1], [5, 5, 5, 5])
	B = enc.transform(X)
	assert B[1, :4].tolist() == [1, 1, 1, 1] and B[0, :4].tolist() == [0, 0, 0, 0]


def test_few_unique_values_do_not_collapse_bits():
	X = np.array([[0.0], [0.0], [0.0], [1.0]])
	enc = ThermometerEncoder(n_bits=4).fit(X)
	assert np.all(np.diff(enc.thresholds_[0]) > 0)


def test_nan_encodes_as_zero_bits():
	X = np.array([[1.0], [2.0], [np.nan], [np.inf]])
	B = ThermometerEncoder(n_bits=3).fit_transform(X)
	assert B[2].tolist() == [0, 0, 0]


def test_feature_names_out():
	enc = ThermometerEncoder(n_bits=2).fit(np.array([[0.0, 1.0], [2.0, 3.0]]))
	names = enc.get_feature_names_out(["a", "b"])
	assert len(names) == 4 and names[0].startswith("a>=")


def test_rejects_bad_input():
	with pytest.raises(ValueError):
		ThermometerEncoder(n_bits=0).fit(np.zeros((3, 2)))
	enc = ThermometerEncoder().fit(np.zeros((3, 2)))
	with pytest.raises(ValueError):
		enc.transform(np.zeros((3, 5)))
