"""Tests for F7: auto-detect dispatch + streaming K-fold.

Two paths are exercised:
1. Small streaming dataset → auto-materialized via memmap → in-memory K-fold.
2. Streaming K-fold (Option B): re-stream per fold, filter rows by _kfold_perm.

The first is correctness-equivalent to in-memory (memmap is byte-identical),
so we test that route mechanically. The second is the new code path —
streaming K-fold should produce metrics within parallel-non-determinism
tolerance of the corresponding in-memory K-fold on the same data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))


class _SkipException(Exception):
	pass


def _importorskip(name: str):
	try:
		__import__(name)
	except ImportError as e:
		raise _SkipException(f"{name} not installed: {e}")


def _make_streaming_dataset(n_train=500, n_test=100, total_bits=16, seed=7, chunk_size=128):
	"""Build matched streaming + in-memory datasets for cross-path verification.

	Same underlying data (label = bit 0), wrapped two ways.
	"""
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded, StreamingEncoded
	from wnn.ids.encoder import ThermometerEncoder

	rng = np.random.default_rng(seed)
	df_full = pd.DataFrame({
		f"bin_{i}": rng.integers(0, 2, size=n_train + n_test)
		for i in range(total_bits)
	})
	df_train = df_full.iloc[:n_train].copy()
	df_test = df_full.iloc[n_train:].copy()

	enc = ThermometerEncoder(n_bits=1)
	enc.fit(df_train)
	X_train_packed, total_bits_emitted = enc.transform(df_train)
	X_test_packed, _ = enc.transform(df_test)

	X_train_inmem = InMemoryEncoded(X_train_packed, total_bits=total_bits_emitted)
	X_test_inmem = InMemoryEncoded(X_test_packed, total_bits=total_bits_emitted)

	y_train = df_train["bin_0"].values.astype(np.int64)
	y_test = df_test["bin_0"].values.astype(np.int64)

	ds_inmem = IDSDataset(
		X_train=X_train_inmem, y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=X_test_inmem, y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"],
		feature_names=list(df_train.columns),
	)

	# Streaming wrapper over the SAME packed bytes
	def make_factory(packed, labels, cs):
		def factory():
			for start in range(0, packed.shape[0], cs):
				yield (packed[start:start + cs], labels[start:start + cs])
		return factory

	X_train_stream = StreamingEncoded(
		make_factory(X_train_packed, y_train, chunk_size),
		n_rows=n_train, total_bits=total_bits_emitted,
	)
	X_test_stream = StreamingEncoded(
		make_factory(X_test_packed, y_test, chunk_size),
		n_rows=n_test, total_bits=total_bits_emitted,
	)
	ds_stream = IDSDataset(
		X_train=X_train_stream, y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=X_test_stream, y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"],
		feature_names=list(df_train.columns),
	)
	return ds_inmem, ds_stream


def _make_genome():
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	g = ClusterGenome.__new__(ClusterGenome)
	g.bits_per_neuron = [4, 4, 4, 4]
	g.neurons_per_cluster = [4]
	g.connections = [
		0, 1, 2, 3,
		0, 5, 7, 9,
		0, 2, 4, 8,
		1, 3, 5, 11,
	]
	g.threshold = None
	g.metrics = None
	return g


def test_auto_detect_small_streaming_materializes_to_memmap():
	"""A small streaming dataset (< 8 GB) gets auto-materialized to memmap;
	IDSEvaluator ends up with _streaming_mode=False and a valid in-memory cache."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	from wnn.ids.encoded_array import MemmapEncoded

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)

	# K-fold > 1 + streaming + small data → should materialize to memmap
	evaluator = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		single_cluster=True, seed=42,
	)
	# Auto-detect should have flipped to in-memory mode
	assert evaluator._streaming_mode is False, \
		"small streaming dataset should have been materialized for K-fold"
	assert evaluator._cache is not None, "in-memory path should have built IDSCacheWrapper"


def test_streaming_kfold_runs_when_threshold_low():
	"""Force the streaming K-fold path by setting streaming_materialize_threshold_gb=0."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)
	# Inject the threshold attribute on the dataset; IDSEvaluator reads it
	ds_stream.streaming_materialize_threshold_gb = 0.0  # force streaming K-fold

	evaluator = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		kfold_per_gen=1, single_cluster=True, seed=42,
	)
	assert evaluator._streaming_mode is True, \
		"threshold=0 should force streaming K-fold (no materialization)"
	assert evaluator._cache is None

	# Run one evaluation — uses _evaluate_batch_streaming_kfold internally
	results = evaluator.evaluate_batch([_make_genome()], train_subset_idx=0)
	assert len(results) == 1
	m = results[0]
	# label = bit 0, genome's first 3 neurons see bit 0 → should achieve > 0.85 acc
	# even with K-fold approximation; threshold=0 path tested for executability.
	assert m.acc > 0.8, f"streaming K-fold acc too low: {m.acc}"
	assert 0.0 <= m.fpr <= 1.0
	assert 0.0 <= m.f1 <= 1.0


def test_streaming_kfold_vs_in_memory_kfold():
	"""Streaming K-fold on the same data should produce metrics close to
	in-memory K-fold (within parallel-non-determinism tolerance, same as
	the Phase 2 smoke deltas)."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	ds_inmem, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)

	# In-memory K-fold (default path)
	ev_inmem = IDSEvaluator(
		ds_inmem, classification="binary", num_parts=5, k_folds=5,
		kfold_per_gen=1, single_cluster=True, seed=42,
	)
	# Force streaming K-fold path
	ds_stream.streaming_materialize_threshold_gb = 0.0
	ev_stream = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		kfold_per_gen=1, single_cluster=True, seed=42,
	)

	g1 = _make_genome(); g2 = _make_genome()
	m_inmem = ev_inmem.evaluate_batch([g1], train_subset_idx=0)[0]
	m_stream = ev_stream.evaluate_batch([g2], train_subset_idx=0)[0]

	# Both paths achieve high accuracy on the label=bit_0 task
	assert m_inmem.acc > 0.85
	assert m_stream.acc > 0.85
	# Within "Phase 2 parallel non-determinism" tolerance
	# (the streaming K-fold partitions slightly differently due to no
	# stratification, so we accept a slightly wider band on small data)
	assert abs(m_inmem.acc - m_stream.acc) < 0.10, \
		f"acc diverged too much: inmem={m_inmem.acc} stream={m_stream.acc}"


def test_streaming_kfold_multi_fold_per_gen():
	"""kfold_per_gen=3 rotates through 3 folds per call; result is averaged."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)
	ds_stream.streaming_materialize_threshold_gb = 0.0
	evaluator = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		kfold_per_gen=3, single_cluster=True, seed=42,
	)

	results = evaluator.evaluate_batch([_make_genome()])
	assert len(results) == 1
	assert results[0].acc > 0.8, f"3-fold-per-gen streaming acc too low: {results[0].acc}"


def test_streaming_balance_classes_supported():
	"""F9: balance_classes=True should now work in streaming mode.

	Class weights are computed from materialized y_train (already in RAM)
	and passed to IDSGenomeStreamer. Verifies no NotImplementedError and
	that metrics are sensible.
	"""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)
	ds_stream.streaming_materialize_threshold_gb = 0.0  # force streaming path

	# This used to raise; now should succeed.
	evaluator = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		kfold_per_gen=1, single_cluster=True, seed=42,
		balance_classes=True, class_weight_multiplier=1.0,
	)
	assert evaluator._streaming_mode is True
	# Class weights computed from y_train
	assert evaluator._streaming_class_weights is not None
	assert len(evaluator._streaming_class_weights) == 2
	assert all(w >= 1 for w in evaluator._streaming_class_weights)

	results = evaluator.evaluate_batch([_make_genome()], train_subset_idx=0)
	assert len(results) == 1
	# With balance_classes, the genome should still classify reasonably well
	# on the label=bit_0 task (different convergence point but same general band)
	assert results[0].acc > 0.7, f"streaming + balance_classes acc too low: {results[0].acc}"


def test_streaming_undersample_majority_still_raises():
	"""undersample_majority remains NotImplementedError in streaming mode v1."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)
	ds_stream.streaming_materialize_threshold_gb = 0.0
	try:
		IDSEvaluator(
			ds_stream, classification="binary", num_parts=5, k_folds=5,
			single_cluster=True, seed=42,
			undersample_majority=True,
		)
	except NotImplementedError as e:
		assert "undersample_majority" in str(e)
		return
	raise AssertionError("undersample_majority in streaming mode should raise")


def test_compute_class_weights_matches_intuition():
	"""Sanity check the _compute_class_weights helper against hand-computed expected values."""
	from wnn.ram.architecture.ids_evaluator import _compute_class_weights

	# 3 classes, counts = [10, 5, 1]
	labels = np.array([0]*10 + [1]*5 + [2]*1, dtype=np.int64)
	w = _compute_class_weights(labels, num_classes=3, multiplier=1.0)
	# max_count=10; weights = [10//10, 10//5, 10//1] = [1, 2, 10]
	assert w == [1, 2, 10], f"expected [1,2,10], got {w}"

	# multiplier=2.0 doubles the weights
	w2 = _compute_class_weights(labels, num_classes=3, multiplier=2.0)
	assert w2 == [2, 4, 20], f"multiplier=2: expected [2,4,20], got {w2}"

	# Empty class gets weight=1
	labels_empty = np.array([0, 0, 0], dtype=np.int64)
	w3 = _compute_class_weights(labels_empty, num_classes=3, multiplier=1.0)
	# class 0: max=3, 3//3=1. classes 1,2: empty → weight=1.
	assert w3 == [1, 1, 1], f"empty-class case: expected [1,1,1], got {w3}"


def test_streaming_kfold_threshold_override_default():
	"""Default threshold (8 GB) materializes the small test dataset."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, ds_stream = _make_streaming_dataset(n_train=500, n_test=100)
	# Don't set threshold attribute — should use 8 GB default
	evaluator = IDSEvaluator(
		ds_stream, classification="binary", num_parts=5, k_folds=5,
		single_cluster=True, seed=42,
	)
	# 500 + 100 rows × few bytes/row is way below 8 GB → materialized
	assert evaluator._streaming_mode is False


if __name__ == "__main__":
	import traceback

	tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
	passed, failed, skipped = 0, 0, 0
	for t in tests:
		try:
			t()
			print(f"  ✓ {t.__name__}")
			passed += 1
		except _SkipException as e:
			print(f"  ⊘ {t.__name__}: {e}")
			skipped += 1
		except Exception as e:
			print(f"  ✗ {t.__name__}: {e}")
			traceback.print_exc()
			failed += 1
	print(f"\n{passed} passed, {failed} failed, {skipped} skipped of {len(tests)}")
	sys.exit(0 if failed == 0 else 1)
