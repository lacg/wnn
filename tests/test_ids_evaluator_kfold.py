"""Tests for IDSEvaluator.get_fold_indices (Phase 5 F-prep).

The pre-shuffled K-fold permutation must:
- Cover every training row exactly once across the K validation folds.
- Be deterministic given the evaluator's seed.
- Yield contiguous slabs of `_perm` order (so memmap/streaming sources
  see sequential row access).
- Last fold absorbs the remainder when n_train is not divisible by k_folds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))


# Lightweight skip mechanism — avoids pytest dependency for ad-hoc runs.
class _SkipException(Exception):
	pass


def _importorskip(name: str):
	try:
		__import__(name)
	except ImportError as e:
		raise _SkipException(f"{name} not installed: {e}")


def _make_tiny_dataset(n_train: int = 100, n_test: int = 20, n_bits: int = 8, seed: int = 42):
	"""Build a minimal IDSDataset with binary labels for evaluator tests.

	Avoids the full Rust accelerator path — we only need the Python-side
	fold indexing math, not actual genome evaluation.
	"""
	from wnn.ids.dataset import IDSDataset
	from wnn.ids.encoded_array import InMemoryEncoded
	from wnn.representations.thermometer import ThermometerEncoder

	rng = np.random.default_rng(seed)
	# Fake numeric features, just enough for the encoder to fit
	df_train = pd.DataFrame({
		"num_a": rng.normal(size=n_train),
		"num_b": rng.uniform(size=n_train),
	})
	df_test = pd.DataFrame({
		"num_a": rng.normal(size=n_test),
		"num_b": rng.uniform(size=n_test),
	})
	enc = ThermometerEncoder(n_bits=n_bits)
	enc.fit(df_train)
	X_train_packed, total_bits = enc.transform(df_train)
	X_test_packed, _ = enc.transform(df_test)

	X_train = InMemoryEncoded(X_train_packed, total_bits=total_bits)
	X_test = InMemoryEncoded(X_test_packed, total_bits=total_bits)

	y_train = rng.integers(0, 2, size=n_train, dtype=np.int64)
	y_test = rng.integers(0, 2, size=n_test, dtype=np.int64)

	return IDSDataset(
		X_train=X_train, y_train_binary=y_train, y_train_multi=y_train.copy(),
		X_test=X_test, y_test_binary=y_test, y_test_multi=y_test.copy(),
		encoder=enc, category_names=["Normal", "Attack"], feature_names=["num_a", "num_b"],
	)


def test_kfold_indices_cover_all_rows():
	"""Across K folds, every original training row must land in exactly one val_idx."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset = _make_tiny_dataset(n_train=100)
	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=5, k_folds=5, seed=123)

	covered = np.zeros(100, dtype=bool)
	for fold in range(5):
		_train_idx, val_idx = evaluator.get_fold_indices(fold)
		# No double-coverage
		assert not covered[val_idx].any(), f"fold {fold} val_idx overlaps prior folds"
		covered[val_idx] = True
	assert covered.all(), "some rows never appear in any val fold"


def test_kfold_train_and_val_disjoint():
	"""For each fold, train_idx and val_idx must be disjoint and together = all rows."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset = _make_tiny_dataset(n_train=100)
	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=5, k_folds=5, seed=42)

	for fold in range(5):
		train_idx, val_idx = evaluator.get_fold_indices(fold)
		assert len(np.intersect1d(train_idx, val_idx)) == 0, f"fold {fold} train/val overlap"
		union = np.union1d(train_idx, val_idx)
		assert len(union) == 100, f"fold {fold} train ∪ val != all rows"


def test_kfold_deterministic_for_same_seed():
	"""Same seed → same permutation → identical fold indices across construction."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset = _make_tiny_dataset(n_train=80)
	ev1 = IDSEvaluator(dataset, classification="binary", num_parts=4, k_folds=4, seed=7)
	ev2 = IDSEvaluator(dataset, classification="binary", num_parts=4, k_folds=4, seed=7)

	for fold in range(4):
		t1, v1 = ev1.get_fold_indices(fold)
		t2, v2 = ev2.get_fold_indices(fold)
		assert np.array_equal(t1, t2)
		assert np.array_equal(v1, v2)


def test_kfold_val_is_contiguous_slab_of_perm():
	"""val_idx for fold k equals `_perm[k*fold_size:(k+1)*fold_size]` (contiguous)."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset = _make_tiny_dataset(n_train=100)
	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=5, k_folds=5, seed=99)

	fold_size = evaluator._kfold_fold_size
	for fold in range(5):
		_train_idx, val_idx = evaluator.get_fold_indices(fold)
		expected_start = fold * fold_size
		expected_stop = 100 if fold == 4 else expected_start + fold_size
		expected = evaluator._kfold_perm[expected_start:expected_stop]
		assert np.array_equal(val_idx, expected), f"fold {fold} val_idx not contiguous slab of _perm"


def test_kfold_last_fold_absorbs_remainder():
	"""When n_train % k_folds != 0, the last fold gets the leftover rows."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	# 103 train rows, 5 folds → fold_size=20, last fold should have 23 rows.
	dataset = _make_tiny_dataset(n_train=103)
	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=5, k_folds=5, seed=1)

	sizes = [len(evaluator.get_fold_indices(f)[1]) for f in range(5)]
	assert sizes == [20, 20, 20, 20, 23], f"fold val sizes: {sizes}"
	# Total coverage = 103
	total = sum(sizes)
	assert total == 103


def test_kfold_invalid_fold_index():
	"""Out-of-range fold_idx raises ValueError."""
	_importorskip("ram_accelerator")
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator

	dataset = _make_tiny_dataset(n_train=50)
	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=5, k_folds=5, seed=1)

	for bad in (-1, 5, 100):
		try:
			evaluator.get_fold_indices(bad)
		except ValueError:
			continue
		raise AssertionError(f"fold_idx={bad} should raise ValueError")


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
