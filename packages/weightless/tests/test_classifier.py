import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from weightless import CellMode, ThermometerEncoder, WiSARDClassifier, backends
from weightless._cell_mode import LADDER
from weightless._core import cell_mode_names


def make_bits(n=600, total_bits=64, n_classes=3, seed=0, noise=0.05):
	"""Three class prototypes with independent bit-flip noise: separable, not trivial."""
	rng = np.random.default_rng(seed)
	protos = rng.integers(0, 2, size=(n_classes, total_bits), dtype=np.uint8)
	y = rng.integers(0, n_classes, size=n)
	X = protos[y] ^ (rng.random((n, total_bits)) < noise).astype(np.uint8)
	return X, y


def test_python_enum_matches_rust_codes():
	assert sorted(cell_mode_names()) == sorted((int(m), m.name.lower()) for m in CellMode)
	assert len(LADDER) == 6


@pytest.mark.parametrize("mode", LADDER)
def test_every_mode_learns_the_prototypes(mode):
	X, y = make_bits()
	clf = WiSARDClassifier(neurons_per_class=20, bits_per_neuron=8, cell_mode=mode, random_state=1)
	clf.fit(X[:400], y[:400])
	acc = (clf.predict(X[400:]) == y[400:]).mean()
	assert acc > 0.9, f"{mode.name}: held-out accuracy {acc:.3f}"
	proba = clf.predict_proba(X[400:])
	assert proba.shape == (200, 3)
	assert np.allclose(proba.sum(axis=1), 1.0)
	assert clf.n_cells_ > 0


def test_training_is_order_independent():
	X, y = make_bits(seed=3)
	a = WiSARDClassifier(neurons_per_class=10, bits_per_neuron=12, random_state=7).fit(X, y)
	perm = np.random.default_rng(9).permutation(len(y))
	b = WiSARDClassifier(neurons_per_class=10, bits_per_neuron=12, random_state=7).fit(X[perm], y[perm])
	np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))
	ea, eb = a.export_keys(), b.export_keys()
	for k in ea:
		np.testing.assert_array_equal(ea[k], eb[k])


def test_partial_fit_accumulates_exactly():
	X, y = make_bits(seed=5)
	whole = WiSARDClassifier(neurons_per_class=10, bits_per_neuron=12, random_state=2).fit(X, y)
	parts = WiSARDClassifier(neurons_per_class=10, bits_per_neuron=12, random_state=2)
	parts.partial_fit(X[:200], y[:200], classes=np.unique(y))
	parts.partial_fit(X[200:], y[200:])
	np.testing.assert_array_equal(whole.decision_function(X), parts.decision_function(X))


def test_sample_weight_scales_the_vote_not_the_observation_count():
	X, y = make_bits(seed=8, n=200)
	w = np.full(len(y), 2.0)
	dup_X, dup_y = np.vstack([X, X]), np.concatenate([y, y])
	# TERNARY / BINARY: a weight of 2 IS two copies (vote sums / set membership).
	for mode in (CellMode.TERNARY, CellMode.BINARY):
		weighted = WiSARDClassifier(neurons_per_class=5, bits_per_neuron=10, cell_mode=mode, random_state=4).fit(X, y, sample_weight=w)
		doubled = WiSARDClassifier(neurons_per_class=5, bits_per_neuron=10, cell_mode=mode, random_state=4).fit(dup_X, dup_y)
		np.testing.assert_array_equal(weighted.decision_function(X), doubled.decision_function(X))
	# QUAD: the weight scales net, but obs counts EXAMPLES, so one weighted
	# example stays in the WEAK bins where two copies would reach TRUE/FALSE.
	weighted = WiSARDClassifier(neurons_per_class=5, bits_per_neuron=10, random_state=4).fit(X, y, sample_weight=w)
	doubled = WiSARDClassifier(neurons_per_class=5, bits_per_neuron=10, random_state=4).fit(dup_X, dup_y)
	assert not np.array_equal(weighted.decision_function(X), doubled.decision_function(X))
	assert (weighted.predict(X) == y).mean() > 0.9


@pytest.mark.parametrize("gpu_backend", ["metal", "wgpu"])
@pytest.mark.parametrize("mode", LADDER)
def test_cpu_and_gpu_agree(mode, gpu_backend):
	if not any(b.startswith(gpu_backend) for b in backends()):
		pytest.skip(f"no {gpu_backend} backend here")
	X, y = make_bits(seed=11)
	cpu = WiSARDClassifier(neurons_per_class=8, bits_per_neuron=16, cell_mode=mode, backend="cpu", random_state=3).fit(X, y)
	gpu = clone(cpu).set_params(backend=gpu_backend).fit(X, y)
	np.testing.assert_allclose(cpu._scores(X, False), gpu._scores(X, False), atol=1e-6)
	np.testing.assert_allclose(cpu.predict_proba(X), gpu.predict_proba(X), atol=1e-6)


def test_stochastic_reads_are_seeded_and_expected_is_deterministic():
	X, y = make_bits(seed=13)
	a = WiSARDClassifier(cell_mode=CellMode.QSR, neurons_per_class=8, bits_per_neuron=8, random_state=5).fit(X, y)
	np.testing.assert_array_equal(a.decision_function(X), a.decision_function(X))
	b = clone(a).set_params(random_state=6).fit(X, y)
	# different seed → different connectivity AND coins; probabilities stay well-formed
	assert np.allclose(b.predict_proba(X).sum(axis=1), 1.0)
	q = clone(a).set_params(cell_mode=CellMode.QUAD_WEIGHTED).fit(X, y)
	# expected read of QSR == QUAD_WEIGHTED read on the same memory
	np.testing.assert_allclose(a.predict_proba(X), q.predict_proba(X), atol=1e-6)


def test_rejects_non_bits_and_bad_shapes():
	X, y = make_bits(n=50)
	with pytest.raises(ValueError, match="must be bits"):
		WiSARDClassifier().fit(X.astype(float) * 0.5, y)
	with pytest.raises(ValueError):
		WiSARDClassifier(bits_per_neuron=999).fit(X, y)
	clf = WiSARDClassifier(neurons_per_class=2, bits_per_neuron=4).fit(X, y)
	with pytest.raises(ValueError):
		clf.predict(X[:, :10])


def test_explicit_connections_round_trip():
	X, y = make_bits(n=100, seed=2)
	conns = np.random.default_rng(0).integers(0, 64, size=(3, 4, 6))
	clf = WiSARDClassifier(neurons_per_class=4, bits_per_neuron=6, connections=conns).fit(X, y)
	np.testing.assert_array_equal(clf.connections_, conns)


def test_pipeline_with_encoder_and_cv():
	rng = np.random.default_rng(0)
	n = 300
	y = rng.integers(0, 2, size=n)
	Xf = rng.normal(size=(n, 5)) + y[:, None] * 2.0
	clf = make_pipeline(ThermometerEncoder(n_bits=8), WiSARDClassifier(neurons_per_class=10, bits_per_neuron=8, random_state=0))
	scores = cross_val_score(clf, Xf, y, cv=3)
	assert scores.mean() > 0.85, scores


def test_binary_decision_function_is_a_margin():
	X, y = make_bits(n_classes=2, seed=4)
	clf = WiSARDClassifier(neurons_per_class=6, bits_per_neuron=8, random_state=0).fit(X, y)
	d = clf.decision_function(X)
	assert d.shape == (len(y),)
	assert ((d > 0).astype(int) == clf.predict(X)).mean() > 0.95
