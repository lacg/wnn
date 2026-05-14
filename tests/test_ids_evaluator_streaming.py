"""End-to-end test: IDSEvaluator with a StreamingEncoded dataset.

This is the integration proof for Option F items 1-4 combined:
- Build a synthetic IDS-like dataset
- Encode it the conventional way (in-memory) for the reference path
- Wrap the same data as a StreamingEncoded factory for the streaming path
- Build an IDSEvaluator for each, call evaluate_batch_full(genome)
- Assert metrics are statistically close (within parallel-non-determinism
  tolerance, similar to the Phase 2 smoke comparison)
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


def _make_dataset(n_train=400, n_test=100, total_bits=24, seed=7):
	"""Synthetic IDS-like dataset.

	The label is correlated with bit 0 of the encoded matrix (since the
	binary encoder pass-through). Genome's first neurons can see bit 0,
	so we expect high accuracy on both in-memory and streaming paths.
	"""
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded, StreamingEncoded
	from wnn.ids.encoder import ThermometerEncoder

	rng = np.random.default_rng(seed)
	# Use binary features for predictable encoding (1 bit each).
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

	# Labels = bit 0 of the bool input (which is bin_0 column)
	y_train = df_train["bin_0"].values.astype(np.int64)
	y_test = df_test["bin_0"].values.astype(np.int64)

	dataset_inmem = IDSDataset(
		X_train=X_train_inmem, y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=X_test_inmem, y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"],
		feature_names=list(df_train.columns),
	)

	# Streaming wrapper over the SAME packed data, chunked into 64-row slabs
	def make_factory(packed, labels, chunk_size=64):
		def factory():
			for start in range(0, packed.shape[0], chunk_size):
				yield (packed[start:start + chunk_size], labels[start:start + chunk_size])
		return factory

	X_train_stream = StreamingEncoded(
		make_factory(X_train_packed, y_train),
		n_rows=n_train, total_bits=total_bits_emitted,
	)
	X_test_stream = StreamingEncoded(
		make_factory(X_test_packed, y_test),
		n_rows=n_test, total_bits=total_bits_emitted,
	)

	dataset_stream = IDSDataset(
		X_train=X_train_stream, y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=X_test_stream, y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"],
		feature_names=list(df_train.columns),
	)

	return dataset_inmem, dataset_stream, total_bits_emitted


def test_streaming_evaluator_constructs_without_cache():
	"""IDSEvaluator with StreamingEncoded skips the IDSCacheWrapper build."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, dataset_stream, _ = _make_dataset()
	evaluator = IDSEvaluator(
		dataset_stream, classification="binary", num_parts=5,
		single_cluster=True, seed=42,
	)
	assert evaluator._streaming_mode is True
	assert evaluator._cache is None
	assert evaluator._train_stream is not None
	assert evaluator._eval_stream is not None


def test_streaming_evaluator_balance_classes_supported_undersample_raises():
	"""Phase F9: balance_classes works (materialized-labels pre-pass);
	undersample_majority still raises (would require streaming row rejection)."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	_, dataset_stream, _ = _make_dataset()
	# Phase F9: balance_classes is now SUPPORTED in streaming mode via
	# materialized-labels pre-pass. Constructor should succeed and stash
	# class_weights for IDSGenomeStreamer to consume.
	evaluator = IDSEvaluator(
		dataset_stream, classification="binary", num_parts=5,
		single_cluster=True, seed=42, balance_classes=True,
	)
	assert evaluator._streaming_mode is True
	assert evaluator._streaming_class_weights is not None
	assert len(evaluator._streaming_class_weights) == 2

	# undersample_majority remains unsupported in streaming v1
	try:
		IDSEvaluator(
			dataset_stream, classification="binary", num_parts=5,
			single_cluster=True, seed=42, undersample_majority=True,
		)
	except NotImplementedError as e:
		assert "undersample_majority" in str(e)
		return
	raise AssertionError("undersample_majority in streaming mode should still raise")


def test_streaming_evaluator_evaluate_batch_full():
	"""evaluate_batch_full routes through IDSGenomeStreamer for streaming dataset.

	Compared to the in-memory path on the same data, accuracy should be
	in the same ballpark (within parallel-non-determinism tolerance).
	The label = bit 0, and the genome has neurons that see bit 0, so
	both paths should achieve > 0.9 acc.
	"""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	dataset_inmem, dataset_stream, total_bits = _make_dataset()

	# Genome that sees bit 0
	def make_genome():
		g = ClusterGenome.__new__(ClusterGenome)
		g.bits_per_neuron = [4, 4, 4, 4]
		g.neurons_per_cluster = [4]
		g.connections = [
			0, 1, 2, 3,   # neuron 0 sees bit 0 (label)
			0, 5, 7, 9,   # neuron 1 also sees bit 0
			0, 2, 4, 8,   # neuron 2 sees bit 0
			1, 3, 5, 11,
		]
		g.threshold = None
		g.metrics = None
		return g

	ev_inmem = IDSEvaluator(
		dataset_inmem, classification="binary", num_parts=5,
		single_cluster=True, seed=42,
	)
	ev_stream = IDSEvaluator(
		dataset_stream, classification="binary", num_parts=5,
		single_cluster=True, seed=42,
	)

	m_inmem = ev_inmem.evaluate_batch_full([make_genome()])[0]
	m_stream = ev_stream.evaluate_batch_full([make_genome()])[0]

	# Both should achieve high accuracy on the label=bit_0 task
	assert m_inmem.acc > 0.9, f"in-memory acc too low: {m_inmem.acc}"
	assert m_stream.acc > 0.9, f"streaming acc too low: {m_stream.acc}"

	# Within parallel-non-determinism tolerance (same as Phase 2 smoke deltas)
	assert abs(m_inmem.acc - m_stream.acc) < 0.05, \
		f"acc diverged too much: inmem={m_inmem.acc} stream={m_stream.acc}"
	assert abs(m_inmem.ce - m_stream.ce) < 0.15, \
		f"ce diverged too much: inmem={m_inmem.ce} stream={m_stream.ce}"


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
