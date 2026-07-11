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
	from wnn.representations.thermometer import ThermometerEncoder

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


def _make_multiclass_dataset(n_train=400, n_test=120, n_val=80, total_bits=24, seed=11):
	"""Synthetic K=4 dataset: class = bin_0 + 2*bin_1 (bits 0-1 of the input).

	Mirrors production streaming semantics: the chunk factories yield BINARY
	labels (class > 0 → 1, like _make_packed_factory), while y_*_multi carry
	the class indices — the streaming multiclass path must ignore the chunk
	labels and slice the materialized multi arrays by offset.

	Includes a val partition (Protocol v2) so margin_val_cal is exercised.
	X_val is in-memory on both variants (the design: only TRAIN needs true
	streaming; a StreamingEncoded val is materialized lazily anyway).
	"""
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded, StreamingEncoded
	from wnn.representations.thermometer import ThermometerEncoder

	rng = np.random.default_rng(seed)
	n_total = n_train + n_test + n_val
	df_full = pd.DataFrame({
		f"bin_{i}": rng.integers(0, 2, size=n_total)
		for i in range(total_bits)
	})
	y_multi_full = (df_full["bin_0"].values + 2 * df_full["bin_1"].values).astype(np.int64)
	y_bin_full = (y_multi_full > 0).astype(np.int64)

	enc = ThermometerEncoder(n_bits=1)
	enc.fit(df_full.iloc[:n_train])

	def _slice(lo, hi):
		packed, bits = enc.transform(df_full.iloc[lo:hi])
		return packed, y_bin_full[lo:hi], y_multi_full[lo:hi], bits

	Xtr, ytr_b, ytr_m, bits_emitted = _slice(0, n_train)
	Xte, yte_b, yte_m, _ = _slice(n_train, n_train + n_test)
	Xva, yva_b, yva_m, _ = _slice(n_train + n_test, n_total)
	names = ["Benign", "AttackA", "AttackB", "AttackC"]

	dataset_inmem = IDSDataset(
		X_train=InMemoryEncoded(Xtr, total_bits=bits_emitted),
		y_train_binary=ytr_b, y_train_multi=ytr_m,
		X_test=InMemoryEncoded(Xte, total_bits=bits_emitted),
		y_test_binary=yte_b, y_test_multi=yte_m,
		encoder=enc, category_names=names,
		feature_names=list(df_full.columns),
		X_val=InMemoryEncoded(Xva, total_bits=bits_emitted),
		y_val_binary=yva_b, y_val_multi=yva_m,
	)

	def make_factory(packed, labels_binary, chunk_size=64):
		def factory():
			for start in range(0, packed.shape[0], chunk_size):
				yield (packed[start:start + chunk_size], labels_binary[start:start + chunk_size])
		return factory

	dataset_stream = IDSDataset(
		X_train=StreamingEncoded(make_factory(Xtr, ytr_b), n_rows=n_train, total_bits=bits_emitted),
		y_train_binary=ytr_b, y_train_multi=ytr_m,
		X_test=StreamingEncoded(make_factory(Xte, yte_b), n_rows=n_test, total_bits=bits_emitted),
		y_test_binary=yte_b, y_test_multi=yte_m,
		encoder=enc, category_names=names,
		feature_names=list(df_full.columns),
		X_val=InMemoryEncoded(Xva, total_bits=bits_emitted),
		y_val_binary=yva_b, y_val_multi=yva_m,
	)
	return dataset_inmem, dataset_stream


def _make_multiclass_genome():
	"""4 clusters (one per class) × 2 neurons × 4 bits, every neuron sees
	bits 0-1 (the class bits) → the task is perfectly learnable."""
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	g = ClusterGenome.__new__(ClusterGenome)
	g.neurons_per_cluster = [2, 2, 2, 2]
	g.bits_per_neuron = [4] * 8
	g.connections = [0, 1, 2, 3, 0, 1, 5, 7] * 4
	g.threshold = None
	g.metrics = None
	return g


def test_streaming_multiclass_evaluate_batch_full():
	"""Streaming GA-search path with K clusters: finalize_metrics returns
	K-class metrics (argmax acc, macro-F1, benign-FPR) close to in-memory."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset_inmem, dataset_stream = _make_multiclass_dataset()
	ev_inmem = IDSEvaluator(dataset_inmem, classification="multi", num_parts=5, seed=42)
	ev_stream = IDSEvaluator(dataset_stream, classification="multi", num_parts=5, seed=42)
	assert ev_stream._streaming_mode is True

	m_inmem = ev_inmem.evaluate_batch_full([_make_multiclass_genome()])[0]
	m_stream = ev_stream.evaluate_batch_full([_make_multiclass_genome()])[0]

	assert m_inmem.acc > 0.9, f"in-memory multiclass acc too low: {m_inmem.acc}"
	assert m_stream.acc > 0.9, f"streaming multiclass acc too low: {m_stream.acc}"
	assert abs(m_inmem.acc - m_stream.acc) < 0.05, \
		f"acc diverged: inmem={m_inmem.acc} stream={m_stream.acc}"
	assert abs(m_inmem.f1 - m_stream.f1) < 0.05, \
		f"macro-F1 diverged: inmem={m_inmem.f1} stream={m_stream.f1}"


def test_streaming_multiclass_at_thresholds_parity():
	"""Protocol v2 streaming multiclass: evaluate_multiclass_at_thresholds
	returns the same mode structure as the in-memory cache path, with
	margin_val_cal calibrated on the val partition, and close metrics."""
	_importorskip("ram_accelerator")
	import math
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset_inmem, dataset_stream = _make_multiclass_dataset()
	ev_inmem = IDSEvaluator(dataset_inmem, classification="multi", num_parts=5, seed=42)
	ev_stream = IDSEvaluator(dataset_stream, classification="multi", num_parts=5, seed=42)

	r_inmem = ev_inmem.evaluate_multiclass_at_thresholds(_make_multiclass_genome())
	r_stream = ev_stream.evaluate_multiclass_at_thresholds(_make_multiclass_genome())

	assert r_stream["num_classes"] == 4
	assert set(r_stream["modes"]) == set(r_inmem["modes"]), \
		f"mode sets differ: {set(r_stream['modes'])} vs {set(r_inmem['modes'])}"
	assert "margin_val_cal" in r_stream["modes"], "val partition must produce margin_val_cal"

	for mode, entry in r_stream["modes"].items():
		ref = r_inmem["modes"][mode]
		assert abs(entry["macro_f1"] - ref["macro_f1"]) < 0.05, \
			f"{mode} macro_f1 diverged: {entry['macro_f1']} vs {ref['macro_f1']}"
		assert abs(entry["benign_fpr"] - ref["benign_fpr"]) < 0.05, \
			f"{mode} benign_fpr diverged: {entry['benign_fpr']} vs {ref['benign_fpr']}"
		# structure: aliases + confusion + per_class present, K-sized
		assert entry["f1"] == entry["macro_f1"] and entry["fpr"] == entry["benign_fpr"]
		assert len(entry["confusion"]) == 4 and len(entry["confusion"][0]) == 4
		assert set(entry["per_class"]) == {"Benign", "AttackA", "AttackB", "AttackC"}
		if mode.startswith("margin_"):
			assert math.isfinite(entry["tau"]), f"{mode} tau must be finite"
		else:
			assert "tau" not in entry

	# argmax on this separable task should be near-perfect on both paths
	assert r_stream["modes"]["argmax"]["acc"] > 0.9


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
