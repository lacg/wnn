"""
Integration of smoothing with RAM-based language models.

This module provides a hybrid approach that combines:
1. Exact RAM lookups (count-based probability when available)
2. Generalized RAM predictions (with confidence weighting)
3. Kneser-Ney smoothed fallback (for unseen contexts)

The key insight is that RAMs excel at high-frequency patterns but need
smoothing for the long tail of rare/unseen contexts.

Usage:
	from wnn.smoothing.ram_integration import SmoothedRAMPredictor

	predictor = SmoothedRAMPredictor(
		exact_rams=model.exact_rams,
		smoothing=kneser_ney,
		confidence_threshold=0.1,
	)

	prob = predictor.probability("fox", ["the", "quick", "brown"])
"""

from typing import Optional, Callable
from collections import Counter

from wnn.smoothing.base import SmoothingStrategy


class SmoothedRAMPredictor:
	"""
	Hybrid predictor combining RAM lookups with n-gram smoothing.

	Priority order:
	1. Exact RAM match with sufficient confidence → use count-based P(target)
	2. Generalized RAM with high confidence → interpolate RAM + smoothing
	3. Fallback → use smoothing probability

	This gives the best of both worlds:
	- High accuracy on frequent patterns (RAM memorization)
	- Reasonable probability on rare patterns (smoothing)
	"""

	def __init__(
		self,
		exact_rams: dict[int, dict[tuple, Counter]],
		smoothing: SmoothingStrategy,
		confidence_threshold: float = 0.1,
		exact_coverage_threshold: float = 0.2,
		interpolation_weight: float = 0.8,
	):
		"""
		Initialize smoothed RAM predictor.

		Args:
			exact_rams: Dict mapping n -> {context_tuple -> Counter of next words}
			smoothing: Trained smoothing strategy for fallback
			confidence_threshold: Minimum confidence to trust generalized RAM
			exact_coverage_threshold: Min confidence for exact RAM to be used
			interpolation_weight: Weight for RAM prob when interpolating (1-w for smoothing)
		"""
		self._exact_rams = exact_rams
		self._smoothing = smoothing
		self._confidence_threshold = confidence_threshold
		self._exact_coverage_threshold = exact_coverage_threshold
		self._interpolation_weight = interpolation_weight

		# Statistics for analysis
		self._stats = {
			"exact_hits": 0,
			"generalized_hits": 0,
			"smoothing_fallbacks": 0,
		}

	def probability(
		self,
		target: str,
		context: list[str],
		generalized_predict_fn: Optional[Callable] = None,
	) -> float:
		"""
		Get probability of target word given context.

		Args:
			target: Word to predict
			context: List of preceding context tokens
			generalized_predict_fn: Optional function(n, context) -> (pred, conf)
				for generalized RAM predictions

		Returns:
			Probability estimate P(target|context)
		"""
		# 1. Try exact RAM match (highest priority)
		exact_prob = self._try_exact_ram(target, context)
		if exact_prob is not None:
			self._stats["exact_hits"] += 1
			return exact_prob

		# 2. Try generalized RAM with smoothing interpolation
		if generalized_predict_fn is not None:
			gen_prob = self._try_generalized_ram(
				target, context, generalized_predict_fn
			)
			if gen_prob is not None:
				self._stats["generalized_hits"] += 1
				return gen_prob

		# 3. Fall back to smoothing
		self._stats["smoothing_fallbacks"] += 1
		return self._smoothing_probability(target, context)

	def _try_exact_ram(self, target: str, context: list[str]) -> Optional[float]:
		"""
		Try to get probability from exact RAM.

		Returns count-based P(target|context) if context is found
		with sufficient coverage.
		"""
		# Try each n-gram order from highest to lowest
		for n in sorted(self._exact_rams.keys(), reverse=True):
			if len(context) >= n - 1:
				ctx = tuple(context[-(n - 1):]) if n > 1 else ()

				if ctx in self._exact_rams[n]:
					counts = self._exact_rams[n][ctx]
					total = sum(counts.values())

					if total > 0:
						best_word, best_count = counts.most_common(1)[0]
						confidence = best_count / total

						# Only use if confident enough
						if confidence >= self._exact_coverage_threshold or total >= 3:
							target_count = counts.get(target, 0)
							# Interpolate with smoothing for non-zero floor
							if target_count > 0:
								ram_prob = target_count / total
								smooth_prob = self._smoothing_probability(target, context)
								# Heavy weight on RAM when we have counts
								return 0.9 * ram_prob + 0.1 * smooth_prob
							else:
								# Target not in exact counts - return None to try other methods
								return None

		return None

	def _try_generalized_ram(
		self,
		target: str,
		context: list[str],
		predict_fn: Callable,
	) -> Optional[float]:
		"""
		Try generalized RAM with smoothing interpolation.

		If generalized RAM is confident, interpolate its implicit probability
		with smoothing for better calibration.
		"""
		for n in sorted(self._exact_rams.keys(), reverse=True):
			if len(context) >= n - 1:
				try:
					pred, conf = predict_fn(n, context)
				except (KeyError, TypeError):
					continue

				if pred and conf > self._confidence_threshold:
					smooth_prob = self._smoothing_probability(target, context)

					if pred == target:
						# RAM predicts target - high probability
						# Interpolate RAM confidence with smoothing
						ram_prob = float(conf)
						return self._interpolation_weight * ram_prob + \
							   (1 - self._interpolation_weight) * smooth_prob
					else:
						# RAM predicts different word
						# Use smoothing but reduce weight slightly
						return smooth_prob * 0.9

		return None

	def _smoothing_probability(self, target: str, context: list[str]) -> float:
		"""Get probability from smoothing strategy."""
		# Convert context to tuple as required by smoothing
		ctx = tuple(context) if context else ()

		# Truncate context to smoothing's max order
		max_ctx = self._smoothing.max_order - 1
		if len(ctx) > max_ctx:
			ctx = ctx[-max_ctx:]

		return self._smoothing.probability(target, ctx)

	def reset_stats(self) -> None:
		"""Reset hit statistics."""
		self._stats = {
			"exact_hits": 0,
			"generalized_hits": 0,
			"smoothing_fallbacks": 0,
		}

	def get_stats(self) -> dict:
		"""Get hit statistics."""
		total = sum(self._stats.values())
		stats = self._stats.copy()
		if total > 0:
			stats["exact_hit_rate"] = self._stats["exact_hits"] / total
			stats["generalized_hit_rate"] = self._stats["generalized_hits"] / total
			stats["fallback_rate"] = self._stats["smoothing_fallbacks"] / total
		return stats


def create_smoothed_evaluator(
	exact_rams: dict[int, dict[tuple, Counter]],
	tokens: list[str],
	max_order: int = 5,
	smoothing_type: str = "kneser_ney",
) -> SmoothedRAMPredictor:
	"""
	Create a smoothed RAM predictor trained on tokens.

	Convenience function that creates and trains the smoothing model.

	Args:
		exact_rams: Dict of exact RAM n-gram counts
		tokens: Training tokens for smoothing
		max_order: Maximum n-gram order for smoothing
		smoothing_type: "kneser_ney", "backoff", or "add_k"

	Returns:
		Configured SmoothedRAMPredictor
	"""
	from wnn.smoothing import KneserNeySmoothing, SimpleBackoffSmoothing, AddKSmoothing

	# Create smoothing strategy
	if smoothing_type == "kneser_ney":
		smoothing = KneserNeySmoothing(max_order=max_order)
	elif smoothing_type == "backoff":
		smoothing = SimpleBackoffSmoothing(max_order=max_order)
	elif smoothing_type == "add_k":
		smoothing = AddKSmoothing(max_order=max_order, k=0.5)
	else:
		raise ValueError(f"Unknown smoothing type: {smoothing_type}")

	# Train on tokens
	smoothing.train(tokens)

	return SmoothedRAMPredictor(
		exact_rams=exact_rams,
		smoothing=smoothing,
	)
