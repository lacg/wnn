"""
TestCacheScope — what a cross-flow test-cache hit must agree on.

The dashboard serves a cached final/init test evaluation row ONLY when genome_hash,
cache_key AND worker_abi all match a row stamped by the same scope (rows with a
NULL key or ABI — every row written before 10/10/2026 — are never served).
See test_cache_key.build_test_cache_key for the key contents.
"""

from dataclasses import dataclass

from wnn.ram.experiments.test_cache_key import canon_value


@dataclass(frozen=True)
class TestCacheScope:
	"""Cache scope: flow-level key + the worker accelerator ABI (trainer version)."""
	__test__ = False  # not a pytest class, despite the Test prefix
	key: str
	worker_abi: int

	def with_threshold_weights(self, w_ce: float, w_f1: float, w_fpr: float, w_acc: float) -> "TestCacheScope":
		"""Experiment-level scope: empirical_cumulative is fitted with THIS
		experiment's fitness weights, so they are part of what was validated."""
		weights = [float(w_ce), float(w_f1), float(w_fpr), float(w_acc)]
		return TestCacheScope(f"{self.key}|ew={canon_value(weights)}", self.worker_abi)
