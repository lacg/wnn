"""
Base classes for probability smoothing in language models.

Smoothing techniques handle the fundamental problem of unseen n-grams:
how to assign non-zero probability to events not seen in training.

This module provides an abstract interface for smoothing strategies,
enabling consistent handling across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Optional


class SmoothingStrategy(ABC):
	"""
	Abstract base class for n-gram smoothing strategies.

	Smoothing converts raw n-gram counts into probability estimates,
	handling the "zero frequency problem" for unseen n-grams.

	Key methods:
	- train(): Build statistics from corpus
	- probability(): Get P(word|context) with smoothing
	- log_probability(): Get log P(word|context) for numerical stability
	"""

	def __init__(self, name: str = "base", max_order: int = 5):
		"""
		Initialize smoothing strategy.

		Args:
			name: Human-readable name
			max_order: Maximum n-gram order (e.g., 5 for 5-grams)
		"""
		self._name = name
		self._max_order = max_order
		self._is_trained = False
		self._vocab_size = 0

	@property
	def name(self) -> str:
		return self._name

	@property
	def max_order(self) -> int:
		return self._max_order

	@property
	def is_trained(self) -> bool:
		return self._is_trained

	@property
	def vocab_size(self) -> int:
		return self._vocab_size

	@abstractmethod
	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Build n-gram statistics from token sequence.

		Args:
			tokens: List of tokens (strings)
			**kwargs: Strategy-specific parameters
		"""
		...

	@abstractmethod
	def probability(self, word: str, context: tuple[str, ...]) -> float:
		"""
		Get smoothed probability P(word|context).

		Args:
			word: Target word to predict
			context: Preceding context as tuple of tokens

		Returns:
			Smoothed probability estimate (0 < p <= 1)
		"""
		...

	def log_probability(self, word: str, context: tuple[str, ...]) -> float:
		"""
		Get log probability for numerical stability.

		Args:
			word: Target word
			context: Preceding context

		Returns:
			Log probability (negative value)
		"""
		from math import log
		prob = self.probability(word, context)
		return log(max(prob, 1e-10))

	def perplexity(self, tokens: list[str], context_size: Optional[int] = None) -> float:
		"""
		Calculate perplexity on a token sequence.

		PPL = exp(-1/N * sum(log P(w_i | context)))

		Args:
			tokens: Token sequence to evaluate
			context_size: Context window size (defaults to max_order - 1)

		Returns:
			Perplexity (lower is better)
		"""
		from math import exp

		if context_size is None:
			context_size = self._max_order - 1

		total_log_prob = 0.0
		count = 0

		for i in range(context_size, len(tokens)):
			context = tuple(tokens[i - context_size:i])
			word = tokens[i]
			total_log_prob += self.log_probability(word, context)
			count += 1

		if count == 0:
			return float('inf')

		avg_log_prob = total_log_prob / count
		return exp(-avg_log_prob)

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(name={self._name!r}, max_order={self._max_order})"
