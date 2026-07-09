"""Parity: M2 cache-sharing (test_eval reuse_cache=optimizer._cache) produces
IDENTICAL validation results vs test_eval building its own cache.

The two evaluators wrap the same data; test_eval only reads full_train/full_eval.
For single_cluster (the paper cohort, 4299), full_train/full_eval carry 0 negatives
so they're seed-independent → sharing is provably exact. This test confirms the
per-example train scores, eval scores, and predictions match cell-for-cell.

Python-only change → runs against the currently installed ram_accelerator wheel."""
import sys
import numpy as np

from wnn.ids.dataset import IDSDataset
from wnn.ids.encoded_array import InMemoryEncoded
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.genome import ClusterGenome

SEED = 20260709


def make_dataset(n_train=400, n_test=80, total_bits=32, seed=SEED):
	rng = np.random.default_rng(seed)
	# Random bool feature matrices — InMemoryEncoded packs them internally.
	Xtr = rng.integers(0, 2, size=(n_train, total_bits)).astype(bool)
	Xte = rng.integers(0, 2, size=(n_test, total_bits)).astype(bool)
	y_tr = rng.integers(0, 2, size=n_train, dtype=np.int64)
	y_te = rng.integers(0, 2, size=n_test, dtype=np.int64)
	return IDSDataset(
		X_train=InMemoryEncoded(Xtr, total_bits=total_bits),
		y_train_binary=y_tr, y_train_multi=y_tr.copy(),
		X_test=InMemoryEncoded(Xte, total_bits=total_bits),
		y_test_binary=y_te, y_test_multi=y_te.copy(),
		encoder=None,
		category_names=["Normal", "Attack"],
		feature_names=[f"f{i}" for i in range(total_bits)],
	), total_bits


def build_evaluators(dataset):
	kw = dict(classification="binary", single_cluster=True, balance_classes=True,
	          neuron_sample_rate=0.25, seed=SEED)
	optimizer = IDSEvaluator(dataset=dataset, num_parts=5, k_folds=5, kfold_per_gen=5, **kw)
	test_own = IDSEvaluator(dataset=dataset, num_parts=1, **kw)
	test_shared = IDSEvaluator(dataset=dataset, num_parts=1, reuse_cache=optimizer._cache, **kw)
	return optimizer, test_own, test_shared


def main():
	dataset, total_bits = make_dataset()
	optimizer, test_own, test_shared = build_evaluators(dataset)

	shared_ok = test_shared._cache is optimizer._cache
	own_ok = test_own._cache is not optimizer._cache and test_own._cache is not None
	print(f"[parity] wiring: shared_is_optimizer_cache={shared_ok}  own_has_independent_cache={own_ok}")

	genome = ClusterGenome.create_uniform(num_clusters=1, bits=6, neurons=8,
	                                       total_input_bits=total_bits, rng=1)

	checks = []
	for name, fn in [
		("score_train_examples", lambda e: e.score_train_examples(genome)),
		("score_examples", lambda e: e.score_examples(genome)),
	]:
		a = np.asarray(fn(test_own), dtype=np.float64)
		b = np.asarray(fn(test_shared), dtype=np.float64)
		same = a.shape == b.shape and np.array_equal(a, b)
		mad = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float("nan")
		checks.append(same)
		print(f"[parity] {name:22s} own={a.shape} shared={b.shape} "
		      f"max_abs_diff={mad:.3e}  {'OK' if same else 'MISMATCH'}")

	ok = shared_ok and own_ok and all(checks)
	print(f"[parity] {'PASS ✅' if ok else 'FAIL ❌'}")
	sys.exit(0 if ok else 1)


if __name__ == "__main__":
	main()
