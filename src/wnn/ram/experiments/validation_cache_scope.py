"""
ValidationCacheScope — what a cross-flow validation-cache hit must agree on.

The dashboard serves a cached final/init validation row ONLY when genome_hash,
cache_key AND worker_abi all match a row stamped by the same scope (rows with a
NULL key or ABI — every row written before 10/10/2026 — are never served).
See validation_cache_key.build_validation_cache_key for the key contents.
"""

from dataclasses import dataclass

from wnn.ram.experiments.validation_cache_key import canon_value


@dataclass(frozen=True)
class ValidationCacheScope:
	"""Cache scope: flow-level key + the worker accelerator ABI (trainer version)."""
	key: str
	worker_abi: int

	def with_threshold_weights(self, w_ce: float, w_f1: float, w_fpr: float, w_acc: float) -> "ValidationCacheScope":
		"""Experiment-level scope: empirical_cumulative is fitted with THIS
		experiment's fitness weights, so they are part of what was validated."""
		weights = [float(w_ce), float(w_f1), float(w_fpr), float(w_acc)]
		return ValidationCacheScope(f"{self.key}|ew={canon_value(weights)}", self.worker_abi)
