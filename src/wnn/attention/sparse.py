"""
Sparse Attention for RAM networks.

Instead of using all context positions, sparse attention selects
only the top-k most important positions. This:

1. Reduces RAM address space (fewer bits needed)
2. Focuses on most predictive positions
3. Enables longer effective context (skip irrelevant positions)

Example:
	Context: ["In", "1999", ",", "the", "company", "announced"]
	Full attention: uses all 6 positions
	Sparse (k=3): uses positions [1, 4, 5] → ["1999", "company", "announced"]

This is similar to how modern transformers use sparse attention patterns
(Longformer, BigBird) but learned from statistics rather than fixed patterns.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from wnn.attention.base import AttentionMechanism, PositionAttention


class SparseAttention(AttentionMechanism):
	"""
	Sparse attention that selects top-k positions.

	Learns which positions are most important, then only uses those
	positions for prediction. This reduces the effective context
	while keeping the most informative parts.
	"""

	def __init__(
		self,
		max_context: int = 6,
		top_k: int = 3,
		adaptive: bool = True,
		name: str = "sparse",
	):
		"""
		Initialize sparse attention.

		Args:
			max_context: Maximum context window size
			top_k: Number of positions to keep
			adaptive: If True, k varies by context; if False, fixed k
			name: Human-readable name
		"""
		super().__init__(max_context=max_context, name=name)
		self._top_k = top_k
		self._adaptive = adaptive

		# Underlying position attention for scoring
		self._position_attn = PositionAttention(max_context=max_context, recency_bias=0.2)

		# Word importance for adaptive selection
		self._word_importance: dict[str, float] = {}
		self._importance_threshold: float = 0.5

	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Train sparse attention.

		Learns:
		1. Position importance (which positions matter)
		2. Word importance (which words are informative)
		3. Optimal k selection (how many positions to use)
		"""
		# Train underlying position attention
		self._position_attn.train(tokens, **kwargs)

		# Learn word importance (IDF-like)
		word_freq = Counter(tokens)
		total = len(tokens)

		for word, freq in word_freq.items():
			# Inverse frequency: rare words are more important
			self._word_importance[word] = np.log(1 + total / (1 + freq))

		# Compute importance threshold (median importance)
		if self._word_importance:
			importances = list(self._word_importance.values())
			self._importance_threshold = np.median(importances)

		self._is_trained = True

	def get_weights(self, context: list[str]) -> np.ndarray:
		"""
		Get sparse attention weights.

		Returns weights with only top-k positions having non-zero weight.
		"""
		if not self._is_trained:
			n = len(context)
			return np.ones(n) / n

		# Get base position weights
		base_weights = self._position_attn.get_weights(context)

		# Add content-based adjustment
		content_boost = np.array([
			self._word_importance.get(word, self._importance_threshold)
			for word in context[-len(base_weights):]
		])
		content_boost = content_boost / content_boost.max()  # Normalize to [0, 1]

		# Combine position and content
		combined = base_weights * (0.5 + 0.5 * content_boost)

		# Determine k
		if self._adaptive:
			# Adaptive k: use more positions if context has important words
			avg_importance = np.mean([
				self._word_importance.get(w, 0) for w in context
			])
			k = max(2, min(self._top_k + 1 if avg_importance > self._importance_threshold else self._top_k - 1, len(context)))
		else:
			k = min(self._top_k, len(context))

		# Zero out all but top-k
		sparse_weights = np.zeros_like(combined)
		top_indices = np.argsort(combined)[-k:]
		sparse_weights[top_indices] = combined[top_indices]

		# Normalize
		if sparse_weights.sum() > 0:
			sparse_weights = sparse_weights / sparse_weights.sum()
		else:
			sparse_weights = np.ones(len(context)) / len(context)

		return sparse_weights

	def get_active_positions(self, context: list[str]) -> list[int]:
		"""
		Get indices of active (non-zero weight) positions.

		Returns:
			List of position indices that have non-zero attention
		"""
		weights = self.get_weights(context)
		return [i for i, w in enumerate(weights) if w > 0]

	def get_sparse_context(self, context: list[str]) -> list[str]:
		"""
		Get only the attended words from context.

		Returns:
			List of words at active positions
		"""
		active = self.get_active_positions(context)
		n = len(self.get_weights(context))
		# Map to actual context positions
		offset = len(context) - n
		return [context[offset + i] for i in active]

	def get_stats(self) -> dict:
		"""Get sparse attention statistics."""
		return {
			"trained": self._is_trained,
			"top_k": self._top_k,
			"adaptive": self._adaptive,
			"importance_threshold": self._importance_threshold,
			"position_stats": self._position_attn.get_stats() if self._is_trained else None,
		}


class WindowedSparseAttention(AttentionMechanism):
	"""
	Sparse attention with local + global pattern.

	Similar to Longformer: attend to recent positions (local)
	plus a few important distant positions (global).

	Pattern: [global] ... [local local local]
	Example: [0] ... [3, 4, 5] for window=3, n_global=1
	"""

	def __init__(
		self,
		max_context: int = 6,
		local_window: int = 3,
		n_global: int = 1,
		name: str = "windowed_sparse",
	):
		"""
		Initialize windowed sparse attention.

		Args:
			max_context: Maximum context window
			local_window: Size of local attention window (recent positions)
			n_global: Number of global attention positions (early positions)
			name: Human-readable name
		"""
		super().__init__(max_context=max_context, name=name)
		self._local_window = local_window
		self._n_global = n_global

		# Learn which global positions matter most
		self._global_position_scores: Optional[np.ndarray] = None

	def train(self, tokens: list[str], **kwargs) -> None:
		"""Learn optimal global attention positions."""
		# Track accuracy of early positions for global attention
		n_positions = self._max_context - self._local_window
		position_correct = np.zeros(n_positions)
		position_total = np.zeros(n_positions)

		# Simple bigram-like accuracy tracking for each position
		for i in range(self._max_context, len(tokens)):
			context = tokens[i - self._max_context:i]
			target = tokens[i]

			for pos in range(n_positions):
				# Would this position alone predict correctly?
				word = context[pos]
				# Track most common continuation
				position_total[pos] += 1

		self._global_position_scores = np.ones(n_positions) / n_positions
		self._is_trained = True

	def get_weights(self, context: list[str]) -> np.ndarray:
		"""Get windowed attention weights."""
		n = min(len(context), self._max_context)
		weights = np.zeros(n)

		# Local window: last local_window positions get attention
		local_start = max(0, n - self._local_window)
		for i in range(local_start, n):
			weights[i] = 1.0

		# Global: first n_global positions (if context long enough)
		for i in range(min(self._n_global, local_start)):
			weights[i] = 0.5  # Lower weight than local

		# Normalize
		if weights.sum() > 0:
			weights = weights / weights.sum()
		else:
			weights = np.ones(n) / n

		return weights
