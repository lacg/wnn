"""FidelityGate: the exported model must REPRODUCE the recorded validation row, exactly.

The exporter re-trains the genome (exactly as final validation does) and gets the
production scorer's held-out scores. For every recorded threshold mode it
recomputes the metrics with the accelerator's own metric function at the stored
threshold (fixed_05 at 0.5) and requires equality with the DB row. Any
mismatch means the deployable tables are NOT the reported model: refuse to write.

Equality is bit-exact up to MAX_ULPS: rows that went through the dashboard's
validation cache before serde_json `float_roundtrip` was enabled (IDS-37) carry
a 1-ULP parse drift. A real model difference flips at least one prediction,
i.e. moves a metric by >= 1/n (~1e-5 on these sets) — orders of magnitude
beyond any ULP, so the tolerance cannot hide one.
"""

from __future__ import annotations

import math


class FidelityMismatch(RuntimeError):
	pass


class FidelityGate:
	METRICS = ("f1", "fpr", "acc")
	MAX_ULPS = 4

	def __init__(self, labels: list[int], normal_class: int):
		self._labels = labels
		self._normal_class = normal_class

	def check(self, eval_scores: list[float], threshold_metadata: dict) -> list[str]:
		"""Mismatch descriptions (empty = faithful). Modes without a threshold
		(other than fixed_05) carry nothing to reproduce and are skipped."""
		problems = []
		for mode, recorded in sorted(threshold_metadata.items()):
			threshold = self._threshold(mode, recorded)
			if threshold is None:
				continue
			problems += self._compare(mode, recorded, self._metrics(eval_scores, threshold))
		return problems

	def require(self, eval_scores: list[float], threshold_metadata: dict) -> int:
		"""Raise FidelityMismatch on any difference; return the number of modes checked."""
		if not threshold_metadata:
			raise FidelityMismatch("no recorded threshold_metadata to verify against")
		problems = self.check(eval_scores, threshold_metadata)
		if problems:
			raise FidelityMismatch("exported model does not reproduce the recorded row:\n  " + "\n  ".join(problems))
		return sum(1 for m, r in threshold_metadata.items() if self._threshold(m, r) is not None)

	@staticmethod
	def _threshold(mode: str, recorded) -> float | None:
		if not isinstance(recorded, dict):
			return None
		if mode == "fixed_05":
			return 0.5
		return recorded.get("threshold")

	def _metrics(self, eval_scores: list[float], threshold: float) -> dict:
		from wnn.accel import require_accel
		_, acc, f1, fpr = require_accel().compute_binary_metrics_at_threshold_py(
			eval_scores, self._labels, float(threshold), self._normal_class)
		return {"f1": f1, "fpr": fpr, "acc": acc}

	def _compare(self, mode: str, recorded: dict, now: dict) -> list[str]:
		return [f"{mode}.{k}: recorded {recorded[k]!r} != exported {now[k]!r}"
			for k in self.METRICS if k in recorded and not self._same(recorded[k], now[k])]

	@classmethod
	def _same(cls, recorded: float, exported: float) -> bool:
		return abs(recorded - exported) <= cls.MAX_ULPS * math.ulp(max(abs(recorded), abs(exported)))
