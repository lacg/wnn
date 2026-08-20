"""Tests for IDSCacheBuilder (Phase 5 F-prep item 3).

Verifies that:
- Single-chunk builder path produces an IDSCache equivalent to the
  one-shot IDSCacheWrapper.new_from_numpy path.
- Multi-chunk builder accumulation (across N partitions of the same data)
  produces the same IDSCache as a single-chunk path.
- finalize() is one-shot (raises on double-finalize).
- add_*_chunk after finalize raises.

We don't have a cheap way to check IDSCache equality directly (it's
opaque from Python), but we can:
- Evaluate the same genome on both caches and check metrics are identical.
- Compare num_train_subsets / total_features accessor outputs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))


class _SkipException(Exception):
	pass


def _importorskip(name: str):
	try:
		__import__(name)
	except ImportError as e:
		raise _SkipException(f"{name} not installed: {e}")


def _build_via_new_from_numpy(train_bool, train_labels, eval_bool, eval_labels, **cfg):
	import ram_accelerator

	total_features = cfg["total_features"]
	train_packed = np.packbits(train_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
	eval_packed = np.packbits(eval_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
	return ram_accelerator.IDSCacheWrapper.new_from_numpy(
		train_features=np.ascontiguousarray(train_packed),
		train_labels=train_labels,
		eval_features=np.ascontiguousarray(eval_packed),
		eval_labels=eval_labels,
		**cfg,
	)


def _build_via_builder_single_chunk(train_bool, train_labels, eval_bool, eval_labels, **cfg):
	import ram_accelerator

	total_features = cfg["total_features"]
	b = ram_accelerator.IDSCacheBuilderWrapper(
		num_classes=cfg["num_classes"],
		total_features=total_features,
		num_parts=cfg["num_parts"],
		num_negatives=cfg["num_negatives"],
		seed=cfg["seed"],
		balance_classes=cfg.get("balance_classes", False),
		single_cluster=cfg.get("single_cluster", False),
		undersample_majority=cfg.get("undersample_majority", False),
		class_weight_multiplier=cfg.get("class_weight_multiplier", 1.0),
	)
	train_packed = np.ascontiguousarray(
		np.packbits(train_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
	)
	eval_packed = np.ascontiguousarray(
		np.packbits(eval_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
	)
	b.add_train_chunk(train_packed, train_labels)
	b.add_eval_chunk(eval_packed, eval_labels)
	return b.finalize()


def _build_via_builder_multi_chunk(train_bool, train_labels, eval_bool, eval_labels, n_chunks, **cfg):
	import ram_accelerator

	total_features = cfg["total_features"]
	b = ram_accelerator.IDSCacheBuilderWrapper(
		num_classes=cfg["num_classes"],
		total_features=total_features,
		num_parts=cfg["num_parts"],
		num_negatives=cfg["num_negatives"],
		seed=cfg["seed"],
		balance_classes=cfg.get("balance_classes", False),
		single_cluster=cfg.get("single_cluster", False),
		undersample_majority=cfg.get("undersample_majority", False),
		class_weight_multiplier=cfg.get("class_weight_multiplier", 1.0),
	)
	# Split training data into n_chunks pieces along axis 0.
	for chunk_bool, chunk_labels in zip(
		np.array_split(train_bool, n_chunks, axis=0),
		np.array_split(np.asarray(train_labels), n_chunks),
	):
		chunk_packed = np.ascontiguousarray(
			np.packbits(chunk_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
		)
		b.add_train_chunk(chunk_packed, chunk_labels.tolist())
	for chunk_bool, chunk_labels in zip(
		np.array_split(eval_bool, n_chunks, axis=0),
		np.array_split(np.asarray(eval_labels), n_chunks),
	):
		chunk_packed = np.ascontiguousarray(
			np.packbits(chunk_bool.astype(np.uint8), axis=1, bitorder="little").ravel()
		)
		b.add_eval_chunk(chunk_packed, chunk_labels.tolist())
	return b.finalize()


def _evaluate_genome(cache):
	"""Run a deterministic single-genome evaluation on the cache for metric comparison."""
	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [
		0, 1, 2, 3,
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
	]
	return cache.evaluate_genomes_full_hybrid(
		genomes_bits_flat=bits_flat,
		genomes_neurons_flat=neurons_flat,
		genomes_connections_flat=connections,
		num_genomes=1,
		empty_value=0.5,
		neuron_sample_rate=1.0,
		rng_seed=42,
	)[0]


def _make_dataset(n_train=200, n_eval=50, total_features=16, seed=7):
	rng = np.random.default_rng(seed)
	train_bool = rng.integers(0, 2, size=(n_train, total_features), dtype=np.uint8).astype(bool)
	eval_bool = rng.integers(0, 2, size=(n_eval, total_features), dtype=np.uint8).astype(bool)
	# Label = bit 0 (deterministic, learnable)
	train_labels = train_bool[:, 0].astype(np.int64).tolist()
	eval_labels = eval_bool[:, 0].astype(np.int64).tolist()
	return train_bool, train_labels, eval_bool, eval_labels


# evaluate_genomes_full_hybrid returns (ce, acc, f1, fpr, threshold, eval_time_ms).
# The last field is WALL-CLOCK MILLISECONDS, so comparing whole tuples compares
# timings: these tests failed on 24ms vs 2ms while every result field matched to
# full precision. Determinism claims belong to the results only.
_RESULT_FIELDS = slice(0, 5)


def _results_only(m):
	return tuple(m)[_RESULT_FIELDS]


def test_builder_single_chunk_matches_new_from_numpy():
	"""Builder with one chunk produces the same IDSCache as new_from_numpy."""
	_importorskip("ram_accelerator")

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset()
	cfg = dict(num_classes=2, total_features=16, num_parts=5, num_negatives=1,
	           seed=42, single_cluster=True)

	cache_a = _build_via_new_from_numpy(train_bool, train_labels, eval_bool, eval_labels, **cfg)
	cache_b = _build_via_builder_single_chunk(train_bool, train_labels, eval_bool, eval_labels, **cfg)

	metrics_a = _evaluate_genome(cache_a)
	metrics_b = _evaluate_genome(cache_b)
	# Identical metrics — both paths exercised the same IDSCache::new internally
	assert _results_only(metrics_a) == _results_only(metrics_b), \
		f"single-chunk builder diverged: a={metrics_a} b={metrics_b}"


def test_builder_multi_chunk_matches_single_chunk():
	"""Builder fed via N chunks produces same IDSCache as one chunk (deterministic accumulation)."""
	_importorskip("ram_accelerator")

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset(n_train=200, n_eval=50)
	cfg = dict(num_classes=2, total_features=16, num_parts=5, num_negatives=1,
	           seed=42, single_cluster=True)

	cache_one = _build_via_builder_single_chunk(train_bool, train_labels, eval_bool, eval_labels, **cfg)
	for n_chunks in (2, 5, 7):
		cache_n = _build_via_builder_multi_chunk(train_bool, train_labels, eval_bool, eval_labels, n_chunks=n_chunks, **cfg)
		metrics_one = _evaluate_genome(cache_one)
		metrics_n = _evaluate_genome(cache_n)
		assert _results_only(metrics_one) == _results_only(metrics_n), \
			f"n_chunks={n_chunks} diverged: one={metrics_one} n={metrics_n}"


def test_builder_finalize_is_one_shot():
	"""Calling finalize twice raises RuntimeError."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	b = ram_accelerator.IDSCacheBuilderWrapper(
		num_classes=2, total_features=8, num_parts=2, num_negatives=1, seed=1,
	)
	# Empty finalize: still produces a cache (degenerate)
	b.finalize()
	try:
		b.finalize()
		raise AssertionError("second finalize() should raise")
	except RuntimeError:
		pass


def test_builder_add_after_finalize_raises():
	"""add_train_chunk after finalize raises RuntimeError."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	b = ram_accelerator.IDSCacheBuilderWrapper(
		num_classes=2, total_features=8, num_parts=2, num_negatives=1, seed=1,
	)
	b.finalize()
	packed = np.zeros(2, dtype=np.uint8)  # 2 rows * 1 byte/row (total_features=8)
	try:
		b.add_train_chunk(packed, [0, 1])
		raise AssertionError("add_train_chunk after finalize should raise")
	except RuntimeError:
		pass


def test_builder_num_train_num_eval_track_chunks():
	"""num_train()/num_eval() reflect accumulated row counts."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	b = ram_accelerator.IDSCacheBuilderWrapper(
		num_classes=2, total_features=8, num_parts=2, num_negatives=1, seed=1,
	)
	assert b.num_train() == 0
	assert b.num_eval() == 0

	# Add 5 train rows
	packed = np.zeros(5, dtype=np.uint8)  # 5 rows × 1 byte (total_features=8)
	b.add_train_chunk(packed, [0, 1, 0, 1, 0])
	assert b.num_train() == 5
	# Add 3 more train rows
	packed = np.zeros(3, dtype=np.uint8)
	b.add_train_chunk(packed, [1, 1, 1])
	assert b.num_train() == 8
	# Add 4 eval rows
	packed = np.zeros(4, dtype=np.uint8)
	b.add_eval_chunk(packed, [0, 0, 1, 1])
	assert b.num_eval() == 4


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
