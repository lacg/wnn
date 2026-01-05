"""
Dynamic Attention for RAM-based language models.

Traditional n-gram models use fixed context windows where all positions
have equal importance. Dynamic attention learns which positions matter
for prediction, enabling:

1. Variable-length context: Skip irrelevant positions
2. Position weighting: Important positions contribute more
3. Content-based selection: Choose positions based on content

For RAM networks (no backprop), we learn attention weights from:
- Frequency statistics: Which positions correlate with correct predictions?
- Mutual information: Which positions have highest MI with target?
- Recency bias: Recent positions typically matter more

Usage:
    attention = PositionAttention(max_context=6)
    attention.train(tokens)

    weights = attention.get_weights(context)  # [0.1, 0.2, 0.05, 0.15, 0.3, 0.2]
    # Use weights to focus RAM lookup on important positions
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Optional
import numpy as np


class AttentionMechanism(ABC):
	"""
	Abstract base class for attention mechanisms in RAM networks.

	Attention assigns importance weights to context positions,
	allowing the model to focus on relevant parts of the context.
	"""

	def __init__(self, max_context: int = 6, name: str = "base"):
		"""
		Initialize attention mechanism.

		Args:
			max_context: Maximum context window size
			name: Human-readable name
		"""
		self._max_context = max_context
		self._name = name
		self._is_trained = False

	@property
	def max_context(self) -> int:
		return self._max_context

	@property
	def name(self) -> str:
		return self._name

	@property
	def is_trained(self) -> bool:
		return self._is_trained

	@abstractmethod
	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Train attention weights from corpus.

		Args:
			tokens: Training corpus tokens
			**kwargs: Strategy-specific parameters
		"""
		...

	@abstractmethod
	def get_weights(self, context: list[str]) -> np.ndarray:
		"""
		Get attention weights for a context.

		Args:
			context: List of context tokens

		Returns:
			Array of weights for each position (sum to 1)
		"""
		...

	def get_top_positions(self, context: list[str], k: int = 3) -> list[int]:
		"""
		Get the top-k most important positions.

		Args:
			context: List of context tokens
			k: Number of positions to return

		Returns:
			List of position indices (0 = oldest, -1 = most recent)
		"""
		weights = self.get_weights(context)
		# Get indices of top k weights
		top_indices = np.argsort(weights)[-k:][::-1]
		return top_indices.tolist()


class PositionAttention(AttentionMechanism):
	"""
	Learn position importance from training statistics.

	For each position, track how often it contributes to correct predictions.
	Positions with higher accuracy correlation get higher weights.

	This is a simple but effective approach that doesn't require backprop.
	"""

	def __init__(
		self,
		max_context: int = 6,
		recency_bias: float = 0.3,
		name: str = "position",
	):
		"""
		Initialize position-based attention.

		Args:
			max_context: Maximum context window size
			recency_bias: Weight for recency prior (0 = pure learned, 1 = pure recency)
			name: Human-readable name
		"""
		super().__init__(max_context=max_context, name=name)
		self._recency_bias = recency_bias

		# Learned position weights (populated during training)
		self._position_weights: Optional[np.ndarray] = None

		# Statistics for learning
		self._position_correct: Optional[np.ndarray] = None
		self._position_total: Optional[np.ndarray] = None

	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Learn position importance from corpus.

		For each position, compute mutual information with target word.
		Positions with higher MI get higher attention weights.
		"""
		# Initialize statistics
		self._position_correct = np.zeros(self._max_context)
		self._position_total = np.zeros(self._max_context)

		# Build word-position co-occurrence statistics
		# For each position, track: P(target | word at position)
		position_word_target = [defaultdict(Counter) for _ in range(self._max_context)]

		for i in range(self._max_context, len(tokens)):
			context = tokens[i - self._max_context:i]
			target = tokens[i]

			for pos, word in enumerate(context):
				position_word_target[pos][word][target] += 1

		# Compute position importance based on predictive power
		# Higher entropy reduction = more important position
		for pos in range(self._max_context):
			total_correct = 0
			total_samples = 0

			for word, targets in position_word_target[pos].items():
				# How well does knowing word at pos predict target?
				most_common_count = targets.most_common(1)[0][1] if targets else 0
				total_count = sum(targets.values())

				# Accuracy if we predict most common target
				total_correct += most_common_count
				total_samples += total_count

			if total_samples > 0:
				self._position_correct[pos] = total_correct
				self._position_total[pos] = total_samples

		# Convert to weights
		self._compute_weights()
		self._is_trained = True

	def _compute_weights(self) -> None:
		"""Compute final attention weights from statistics."""
		# Base weights from accuracy
		if self._position_total is not None and self._position_total.sum() > 0:
			accuracy = self._position_correct / np.maximum(self._position_total, 1)
		else:
			accuracy = np.ones(self._max_context) / self._max_context

		# Recency prior: more recent positions typically matter more
		recency = np.array([i + 1 for i in range(self._max_context)], dtype=np.float32)
		recency = recency / recency.sum()

		# Combine learned weights with recency prior
		combined = (1 - self._recency_bias) * accuracy + self._recency_bias * recency

		# Normalize to sum to 1
		self._position_weights = combined / combined.sum()

	def get_weights(self, context: list[str]) -> np.ndarray:
		"""Get attention weights for positions."""
		if not self._is_trained or self._position_weights is None:
			# Default: uniform weights
			n = min(len(context), self._max_context)
			return np.ones(n) / n

		n = min(len(context), self._max_context)
		# Return weights for the positions we have
		weights = self._position_weights[-n:]
		return weights / weights.sum()

	def get_stats(self) -> dict:
		"""Get learned position statistics."""
		if not self._is_trained:
			return {"trained": False}

		accuracy = self._position_correct / np.maximum(self._position_total, 1)
		return {
			"trained": True,
			"position_accuracy": accuracy.tolist(),
			"position_weights": self._position_weights.tolist(),
			"recency_bias": self._recency_bias,
		}


class ContentAttention(AttentionMechanism):
	"""
	Content-based attention that weights positions by word type.

	Some word types are more predictive than others:
	- Function words ("the", "a", "is"): Low weight
	- Content words ("cat", "running", "quickly"): Higher weight
	- Rare words: Often very predictive

	This uses TF-IDF-like weighting without neural networks.
	"""

	def __init__(
		self,
		max_context: int = 6,
		name: str = "content",
	):
		super().__init__(max_context=max_context, name=name)

		# Word importance scores (IDF-like)
		self._word_importance: dict[str, float] = {}
		self._default_importance: float = 1.0

	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Learn word importance from corpus.

		Uses inverse document frequency (IDF) idea:
		- Common words (appear in many contexts) → low importance
		- Rare words (appear in few contexts) → high importance
		"""
		# Count document frequency (how many n-grams contain each word)
		word_doc_freq = Counter()
		total_docs = 0

		for i in range(self._max_context, len(tokens)):
			context = tokens[i - self._max_context:i]
			context_set = set(context)
			for word in context_set:
				word_doc_freq[word] += 1
			total_docs += 1

		# Compute IDF-like importance
		for word, df in word_doc_freq.items():
			# IDF = log(N / df), but we use a smoother version
			self._word_importance[word] = np.log(1 + total_docs / (1 + df))

		# Default for unseen words (high importance - they're rare!)
		self._default_importance = np.log(1 + total_docs)

		self._is_trained = True

	def get_weights(self, context: list[str]) -> np.ndarray:
		"""Get attention weights based on word importance."""
		if not self._is_trained:
			n = len(context)
			return np.ones(n) / n

		# Get importance for each word
		importance = np.array([
			self._word_importance.get(word, self._default_importance)
			for word in context
		])

		# Also add position bias (recent words matter more)
		position_bias = np.array([i + 1 for i in range(len(context))], dtype=np.float32)

		# Combine
		weights = importance * position_bias

		# Normalize
		return weights / weights.sum()


class HybridAttention(AttentionMechanism):
	"""
	Combines position and content attention.

	Final weight = α * position_weight + (1-α) * content_weight
	"""

	def __init__(
		self,
		max_context: int = 6,
		position_weight: float = 0.5,
		name: str = "hybrid",
	):
		super().__init__(max_context=max_context, name=name)
		self._alpha = position_weight

		self._position_attn = PositionAttention(max_context=max_context)
		self._content_attn = ContentAttention(max_context=max_context)

	def train(self, tokens: list[str], **kwargs) -> None:
		"""Train both attention mechanisms."""
		self._position_attn.train(tokens, **kwargs)
		self._content_attn.train(tokens, **kwargs)
		self._is_trained = True

	def get_weights(self, context: list[str]) -> np.ndarray:
		"""Combine position and content weights."""
		pos_weights = self._position_attn.get_weights(context)
		content_weights = self._content_attn.get_weights(context)

		# Ensure same length
		n = min(len(pos_weights), len(content_weights))
		combined = self._alpha * pos_weights[-n:] + (1 - self._alpha) * content_weights[-n:]

		return combined / combined.sum()
