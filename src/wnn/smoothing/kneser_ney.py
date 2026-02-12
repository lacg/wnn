"""
Kneser-Ney Smoothing for n-gram language models.

Kneser-Ney is considered the gold standard for n-gram smoothing.
Key innovations over simpler methods (Add-k, Good-Turing):

1. **Absolute Discounting**: Subtract fixed D from each count
   - More principled than relative discounting
   - D is estimated from held-out data or analytically

2. **Continuation Probability**: For backoff, use how many different
   contexts a word appears in, not raw frequency
   - "Francisco" is rare but always follows "San"
   - Raw frequency overestimates P(Francisco)
   - Continuation count gives more accurate estimate

3. **Interpolation**: Combine orders smoothly, not just backoff
   - Always mix in lower-order probability
   - Better than backing off only on zero counts

Formula (Interpolated Kneser-Ney):

P_KN(w|context) = max(c(context,w) - D, 0) / c(context)
                + λ(context) * P_KN(w|context[1:])

Where:
- D = discount (typically 0.75, can be estimated)
- λ(context) = D * N1+(context,•) / c(context)
- N1+(context,•) = number of unique words following context

For lowest order (unigram continuation):
P_cont(w) = N1+(•,w) / N1+(•,•)
Where N1+(•,w) = number of unique contexts preceding w

References:
- Kneser & Ney (1995): "Improved backing-off for M-gram language modeling"
- Chen & Goodman (1999): "An empirical study of smoothing techniques"
"""

from collections import Counter, defaultdict
from math import log
from typing import Optional

from wnn.smoothing.base import SmoothingStrategy


class KneserNeySmoothing(SmoothingStrategy):
	"""
	Interpolated Modified Kneser-Ney smoothing.

	This is the "modified" variant which uses different discounts
	for counts of 1, 2, and 3+ (D1, D2, D3+).

	Usage:
		kn = KneserNeySmoothing(max_order=5)
		kn.train(tokens)

		prob = kn.probability("fox", ("the", "quick", "brown"))
		ppl = kn.perplexity(test_tokens)
	"""

	def __init__(
		self,
		name: str = "kneser_ney",
		max_order: int = 5,
		discount: Optional[float] = None,
	):
		"""
		Initialize Kneser-Ney smoothing.

		Args:
			name: Strategy name
			max_order: Maximum n-gram order (5 = use up to 5-grams)
			discount: Fixed discount D. If None, estimate from data.
		"""
		super().__init__(name=name, max_order=max_order)
		self._fixed_discount = discount

		# Statistics (populated during training)
		# counts[n][context] = Counter of words following context
		self._counts: dict[int, dict[tuple, Counter]] = {}

		# Continuation counts for lower orders
		# continuation_counts[n][word] = number of unique contexts preceding word
		self._continuation_counts: dict[int, Counter] = {}

		# Total continuation count per order
		self._total_continuations: dict[int, int] = {}

		# Discount parameters (D1, D2, D3+) per order
		self._discounts: dict[int, tuple[float, float, float]] = {}

		# Cache for lambda values
		self._lambda_cache: dict[tuple, float] = {}

	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Build n-gram statistics from token sequence.

		Args:
			tokens: List of tokens
		"""
		# Build vocabulary
		vocab = set(tokens)
		self._vocab_size = len(vocab)

		# Initialize count structures
		self._counts = {n: defaultdict(Counter) for n in range(1, self._max_order + 1)}
		self._continuation_counts = {n: Counter() for n in range(1, self._max_order)}
		self._total_continuations = {}
		self._lambda_cache = {}

		# Count n-grams
		for i in range(len(tokens)):
			for n in range(1, self._max_order + 1):
				if i >= n - 1:
					# Context is tokens[i-n+1:i], word is tokens[i]
					context = tuple(tokens[i - n + 1:i]) if n > 1 else ()
					word = tokens[i]
					self._counts[n][context][word] += 1

					# Track continuation counts for orders 1 to max_order-1
					# (used for lower-order models in interpolation)
					if n > 1:
						self._continuation_counts[n - 1][word] += 1

		# Compute total continuations per order
		for n in range(1, self._max_order):
			# N1+(•,•) = total number of unique (context, word) pairs
			total = sum(len(counter) for counter in self._counts[n + 1].values())
			self._total_continuations[n] = total

		# Estimate discounts per order
		self._estimate_discounts()

		self._is_trained = True

	def _estimate_discounts(self) -> None:
		"""
		Estimate discount parameters from data.

		Uses the formula from Chen & Goodman (1999):
		D = n1 / (n1 + 2*n2)

		For modified KN, we estimate D1, D2, D3+ separately.
		"""
		for n in range(1, self._max_order + 1):
			# Count n-grams with count 1, 2, 3, 4
			n1, n2, n3, n4 = 0, 0, 0, 0
			for context_counts in self._counts[n].values():
				for count in context_counts.values():
					if count == 1:
						n1 += 1
					elif count == 2:
						n2 += 1
					elif count == 3:
						n3 += 1
					elif count == 4:
						n4 += 1

			# Estimate Y (common factor)
			if n1 + 2 * n2 > 0:
				Y = n1 / (n1 + 2 * n2)
			else:
				Y = 0.5  # Default

			# Estimate discounts
			if self._fixed_discount is not None:
				D1 = D2 = D3 = self._fixed_discount
			else:
				# Modified KN discounts
				if n1 > 0 and n2 > 0:
					D1 = 1 - 2 * Y * n2 / n1 if n1 > 0 else 0.5
				else:
					D1 = 0.5
				if n2 > 0 and n3 > 0:
					D2 = 2 - 3 * Y * n3 / n2 if n2 > 0 else 0.75
				else:
					D2 = 0.75
				if n3 > 0 and n4 > 0:
					D3 = 3 - 4 * Y * n4 / n3 if n3 > 0 else 0.9
				else:
					D3 = 0.9

				# Clamp to valid range
				D1 = max(0.0, min(1.0, D1))
				D2 = max(0.0, min(1.0, D2))
				D3 = max(0.0, min(1.0, D3))

			self._discounts[n] = (D1, D2, D3)

	def _get_discount(self, n: int, count: int) -> float:
		"""Get appropriate discount for a count at order n."""
		D1, D2, D3 = self._discounts.get(n, (0.75, 0.75, 0.75))
		if count == 1:
			return D1
		elif count == 2:
			return D2
		else:
			return D3

	def _get_lambda(self, n: int, context: tuple) -> float:
		"""
		Compute interpolation weight λ(context).

		λ(context) = D * N1+(context,•) / c(context)

		Where N1+(context,•) = number of unique words following context
		"""
		cache_key = (n, context)
		if cache_key in self._lambda_cache:
			return self._lambda_cache[cache_key]

		if context not in self._counts[n]:
			# No counts for this context, full weight to lower order
			return 1.0

		context_counts = self._counts[n][context]
		total_count = sum(context_counts.values())
		unique_following = len(context_counts)  # N1+(context,•)

		if total_count == 0:
			return 1.0

		# Use average discount weighted by count frequencies
		D1, D2, D3 = self._discounts.get(n, (0.75, 0.75, 0.75))
		n1 = sum(1 for c in context_counts.values() if c == 1)
		n2 = sum(1 for c in context_counts.values() if c == 2)
		n3 = sum(1 for c in context_counts.values() if c >= 3)

		# λ = (D1*n1 + D2*n2 + D3*n3) / total_count
		lambda_val = (D1 * n1 + D2 * n2 + D3 * n3) / total_count

		self._lambda_cache[cache_key] = lambda_val
		return lambda_val

	def probability(self, word: str, context: tuple[str, ...]) -> float:
		"""
		Get interpolated Kneser-Ney probability P(word|context).

		Uses recursive interpolation down to unigram level.

		Args:
			word: Target word
			context: Preceding context tuple

		Returns:
			Smoothed probability
		"""
		if not self._is_trained:
			raise RuntimeError("Must call train() before probability()")

		# Truncate context to max order - 1
		if len(context) >= self._max_order:
			context = context[-(self._max_order - 1):]

		return self._kn_probability(word, context, len(context) + 1)

	def _kn_probability(self, word: str, context: tuple, order: int) -> float:
		"""
		Recursive Kneser-Ney probability computation.

		Args:
			word: Target word
			context: Context tuple (length = order - 1)
			order: Current n-gram order
		"""
		if order == 1:
			# Unigram: use continuation probability
			return self._continuation_probability(word)

		# Get counts at this order
		if context in self._counts[order]:
			context_counts = self._counts[order][context]
			total_count = sum(context_counts.values())
			word_count = context_counts.get(word, 0)
		else:
			total_count = 0
			word_count = 0

		# Compute discounted probability
		if total_count > 0 and word_count > 0:
			discount = self._get_discount(order, word_count)
			discounted_prob = max(word_count - discount, 0) / total_count
		else:
			discounted_prob = 0.0

		# Compute interpolation weight
		lambda_val = self._get_lambda(order, context)

		# Recursive call to lower order
		lower_context = context[1:] if len(context) > 0 else ()
		lower_prob = self._kn_probability(word, lower_context, order - 1)

		# Interpolate
		return discounted_prob + lambda_val * lower_prob

	def _continuation_probability(self, word: str) -> float:
		"""
		Compute continuation probability for a word (unigram level).

		P_cont(w) = N1+(•,w) / N1+(•,•)

		Where:
		- N1+(•,w) = number of unique contexts preceding w
		- N1+(•,•) = total number of unique bigram types
		"""
		# N1+(•,w) = how many unique words precede this word
		# We stored this in continuation_counts[1]
		n1_word = self._continuation_counts[1].get(word, 0)

		# N1+(•,•) = total unique bigram types
		n1_total = self._total_continuations.get(1, 1)

		if n1_total == 0:
			# Fall back to uniform
			return 1.0 / max(self._vocab_size, 1)

		# Add-1 smoothing for unseen words
		if n1_word == 0:
			return 1.0 / (n1_total + self._vocab_size)

		return n1_word / n1_total

	def get_stats(self) -> dict:
		"""Get statistics about the trained model."""
		stats = {
			"vocab_size": self._vocab_size,
			"max_order": self._max_order,
			"discounts": self._discounts,
		}

		# Count n-grams per order
		for n in range(1, self._max_order + 1):
			total_ngrams = sum(len(c) for c in self._counts[n].values())
			unique_contexts = len(self._counts[n])
			stats[f"order_{n}_ngrams"] = total_ngrams
			stats[f"order_{n}_contexts"] = unique_contexts

		return stats


class SimpleBackoffSmoothing(SmoothingStrategy):
	"""
	Simple backoff smoothing (Katz-style).

	Less sophisticated than Kneser-Ney but faster and simpler.
	Falls back to lower-order model when count is zero.
	"""

	def __init__(
		self,
		name: str = "simple_backoff",
		max_order: int = 5,
		alpha: float = 0.4,
	):
		"""
		Initialize simple backoff.

		Args:
			name: Strategy name
			max_order: Maximum n-gram order
			alpha: Backoff weight (0 < alpha < 1)
		"""
		super().__init__(name=name, max_order=max_order)
		self._alpha = alpha
		self._counts: dict[int, dict[tuple, Counter]] = {}

	def train(self, tokens: list[str], **kwargs) -> None:
		"""Build n-gram counts."""
		vocab = set(tokens)
		self._vocab_size = len(vocab)

		self._counts = {n: defaultdict(Counter) for n in range(1, self._max_order + 1)}

		for i in range(len(tokens)):
			for n in range(1, self._max_order + 1):
				if i >= n - 1:
					context = tuple(tokens[i - n + 1:i]) if n > 1 else ()
					word = tokens[i]
					self._counts[n][context][word] += 1

		self._is_trained = True

	def probability(self, word: str, context: tuple[str, ...]) -> float:
		"""Get probability with simple backoff."""
		if not self._is_trained:
			raise RuntimeError("Must call train() before probability()")

		if len(context) >= self._max_order:
			context = context[-(self._max_order - 1):]

		return self._backoff_probability(word, context, len(context) + 1)

	def _backoff_probability(self, word: str, context: tuple, order: int) -> float:
		"""Recursive backoff probability."""
		if order == 0:
			# Uniform fallback
			return 1.0 / max(self._vocab_size, 1)

		if context in self._counts[order]:
			context_counts = self._counts[order][context]
			total_count = sum(context_counts.values())
			word_count = context_counts.get(word, 0)

			if word_count > 0:
				return word_count / total_count

		# Backoff to lower order
		lower_context = context[1:] if len(context) > 0 else ()
		return self._alpha * self._backoff_probability(word, lower_context, order - 1)


class AddKSmoothing(SmoothingStrategy):
	"""
	Add-k (Laplace) smoothing - simple baseline.

	Adds k to every count, including unseen n-grams.
	Simple but known to be suboptimal for language modeling.
	"""

	def __init__(
		self,
		name: str = "add_k",
		max_order: int = 5,
		k: float = 1.0,
	):
		super().__init__(name=name, max_order=max_order)
		self._k = k
		self._counts: dict[int, dict[tuple, Counter]] = {}

	def train(self, tokens: list[str], **kwargs) -> None:
		vocab = set(tokens)
		self._vocab_size = len(vocab)

		self._counts = {n: defaultdict(Counter) for n in range(1, self._max_order + 1)}

		for i in range(len(tokens)):
			for n in range(1, self._max_order + 1):
				if i >= n - 1:
					context = tuple(tokens[i - n + 1:i]) if n > 1 else ()
					word = tokens[i]
					self._counts[n][context][word] += 1

		self._is_trained = True

	def probability(self, word: str, context: tuple[str, ...]) -> float:
		if not self._is_trained:
			raise RuntimeError("Must call train() before probability()")

		if len(context) >= self._max_order:
			context = context[-(self._max_order - 1):]

		order = len(context) + 1

		if context in self._counts[order]:
			context_counts = self._counts[order][context]
			total_count = sum(context_counts.values())
			word_count = context_counts.get(word, 0)
		else:
			total_count = 0
			word_count = 0

		# Add-k smoothing
		return (word_count + self._k) / (total_count + self._k * self._vocab_size)
