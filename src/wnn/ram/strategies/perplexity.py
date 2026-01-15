"""
Perplexity Calculator - Unified logic for consistent PPL calculation.

This module provides a centralized perplexity calculation class to ensure
consistency across the codebase. Use this instead of scattered PPL logic.

Features:
- Consistent PPL/CE/accuracy calculation
- Optional per-category tracking (e.g., per-tier, per-frequency-bucket)
- Batch processing support for efficiency
"""

from math import exp, log
from typing import Optional, Callable, Union


class PerplexityCalculator:
	"""
	Unified perplexity calculation to ensure consistency across the codebase.

	PPL = exp(-avg(log(P(target))))

	Key insight: For language modeling, we want TRUE perplexity which measures
	the probability mass assigned to the correct target, NOT just whether
	the argmax prediction was correct.

	Two modes:
	- TRUE PPL: Uses actual P(target) from distribution (partial credit)
	- ACCURACY PPL: Binary - correct gets confidence, wrong gets 1/vocab (strict)

	TRUE PPL is the correct metric for language modeling. ACCURACY PPL
	heavily penalizes wrong predictions and gives misleadingly high values.

	Usage:
		# Create calculator
		calc = PerplexityCalculator(vocab_size=10000)

		# For each prediction, add observation
		calc.add_from_probability(target_prob=0.3, is_correct=True)
		# OR
		calc.add_from_vote_distribution(votes={'the': 0.5, 'a': 0.3}, target='a')
		# OR (less accurate)
		calc.add_from_prediction(prediction='the', target='a', confidence=0.5)

		# Get results
		ppl = calc.get_perplexity()
		acc = calc.get_accuracy()
	"""

	def __init__(
		self,
		vocab_size: int,
		num_categories: Optional[int] = None,
		category_names: Optional[list[str]] = None,
		empty_value: float = 0.0,
	):
		"""
		Initialize the calculator.

		Args:
			vocab_size: Size of vocabulary (used for min_prob = 1/vocab_size)
			num_categories: Optional number of categories for per-category tracking.
			                When set, enables per-category stats (e.g., per-tier metrics).
			category_names: Optional names for categories (for better reporting).
			                If None, uses ["cat_0", "cat_1", ...].
			empty_value: Value for EMPTY cells when computing probabilities from
			            RAM memory counts. 0.0 = abstain (recommended), 0.5 = uncertain.
		"""
		self.vocab_size = vocab_size
		self.min_prob = 1.0 / vocab_size
		self._log_min_prob = log(self.min_prob)
		self.empty_value = empty_value

		# Category tracking setup
		self.num_categories = num_categories
		if num_categories is not None:
			if category_names is not None:
				assert len(category_names) == num_categories
				self.category_names = category_names
			else:
				self.category_names = [f"cat_{i}" for i in range(num_categories)]
		else:
			self.category_names = None

		self.reset()

	def reset(self):
		"""Reset accumulator for new evaluation."""
		self._log_probs: list[float] = []
		self._correct: int = 0
		self._total: int = 0

		# Per-category accumulators (if enabled)
		if self.num_categories is not None:
			self._cat_log_probs: list[list[float]] = [[] for _ in range(self.num_categories)]
			self._cat_correct: list[int] = [0] * self.num_categories
			self._cat_total: list[int] = [0] * self.num_categories
		else:
			self._cat_log_probs = None
			self._cat_correct = None
			self._cat_total = None

	def add_from_probability(self, target_prob: float, is_correct: Optional[bool] = None):
		"""
		Add observation from direct probability estimate (e.g., exact RAM counts).

		This is the PREFERRED method when you have P(target) available.

		Args:
			target_prob: P(target) from the distribution
			is_correct: Whether argmax == target (optional, for accuracy tracking)
		"""
		prob = max(target_prob, self.min_prob)
		self._log_probs.append(log(prob))
		self._total += 1
		if is_correct is True:
			self._correct += 1
		elif is_correct is None and target_prob > self.min_prob:
			# Infer correctness from probability (likely correct if high prob)
			self._correct += 1

	def add_from_vote_distribution(self, votes: dict[str, float], target: str):
		"""
		Add observation from voting distribution (PREFERRED for voting strategies).

		Uses P(target) from vote distribution - gives partial credit even
		if target didn't win the vote. This is the CORRECT approach for
		language modeling perplexity.

		Args:
			votes: dict mapping predictions to vote weights
			target: the actual target word
		"""
		total_weight = sum(votes.values())
		if total_weight > 0:
			target_prob = votes.get(target, 0.0) / total_weight
			prob = max(target_prob, self.min_prob)
		else:
			prob = self.min_prob
		self._log_probs.append(log(prob))
		self._total += 1

		# Track accuracy: correct if target got highest votes
		if votes:
			winner = max(votes, key=votes.get)
			if winner == target:
				self._correct += 1

	def add_from_vote_with_smoothing(
		self,
		votes: dict[str, float],
		target: str,
		smoothing_prob: Optional[float] = None,
		vote_lambda: float = 0.8,
	):
		"""
		Add observation from voting distribution interpolated with smoothing.

		Interpolates vote-based probability with smoothing fallback to prevent
		PPL explosion when target is not in the vote distribution:

			P(target) = vote_lambda * P_vote(target) + (1-vote_lambda) * P_smooth(target)

		Args:
			votes: dict mapping predictions to vote weights
			target: the actual target word
			smoothing_prob: probability from smoothing model (e.g., Kneser-Ney).
			                If None, uses min_prob as fallback.
			vote_lambda: interpolation weight for votes (default 0.8 = 80% vote, 20% smooth)
		"""
		# Get smoothing fallback
		smooth_p = max(smoothing_prob, self.min_prob) if smoothing_prob else self.min_prob

		# Calculate vote probability
		total_weight = sum(votes.values()) if votes else 0
		if total_weight > 0:
			vote_p = votes.get(target, 0.0) / total_weight
		else:
			vote_p = 0.0

		# Interpolate: vote_lambda * vote + (1 - vote_lambda) * smoothing
		final_prob = vote_lambda * vote_p + (1 - vote_lambda) * smooth_p
		final_prob = max(final_prob, self.min_prob)

		self._log_probs.append(log(final_prob))
		self._total += 1

		# Track accuracy: correct if target got highest votes
		if votes:
			winner = max(votes, key=votes.get)
			if winner == target:
				self._correct += 1

	def add_from_prediction(self, prediction: str, target: str, confidence: float):
		"""
		Add observation from argmax prediction + confidence.

		WARNING: This is LESS ACCURATE than add_from_vote_distribution because
		when prediction != target, we don't know P(target) - we use min_prob.

		Use this only when vote distribution is not available.

		Args:
			prediction: the model's prediction (argmax)
			target: the actual target word
			confidence: model's confidence in its prediction
		"""
		if prediction == target:
			prob = max(confidence, self.min_prob)
			self._correct += 1
		else:
			# We don't know P(target), use min_prob as conservative estimate
			prob = self.min_prob
		self._log_probs.append(log(prob))
		self._total += 1

	def get_cross_entropy(self) -> float:
		"""
		Calculate average cross-entropy from accumulated observations.

		Cross-entropy = -avg(log(P(target)))

		This is the preferred metric for optimization (GA/TS) as it provides
		a more granular signal than perplexity. Lower is better.

		Relationship to perplexity: PPL = exp(cross_entropy)

		Returns:
			Average cross-entropy (negative log probability). Lower is better.
			Returns inf if no observations.
		"""
		if not self._log_probs:
			return float('inf')
		return -sum(self._log_probs) / len(self._log_probs)

	def get_perplexity(self) -> float:
		"""
		Calculate perplexity from accumulated observations.

		PPL = exp(-avg(log(P(target)))) = exp(cross_entropy)

		Returns:
			Perplexity. Lower is better. 1.0 = perfect, vocab_size = random.
		"""
		if not self._log_probs:
			return float('inf')
		avg_log_prob = sum(self._log_probs) / len(self._log_probs)
		return exp(-avg_log_prob)

	def get_accuracy(self) -> float:
		"""Get accuracy from accumulated observations."""
		return self._correct / self._total if self._total > 0 else 0.0

	def get_category_stats(self, category: int) -> dict:
		"""
		Get statistics for a specific category.

		Args:
			category: Category index (0 to num_categories-1)

		Returns:
			Dict with cross_entropy, perplexity, accuracy, correct, total for this category.
			Returns empty dict if category tracking is not enabled or category is invalid.
		"""
		if self.num_categories is None or category < 0 or category >= self.num_categories:
			return {}

		log_probs = self._cat_log_probs[category]
		correct = self._cat_correct[category]
		total = self._cat_total[category]

		if not log_probs:
			return {
				'cross_entropy': float('inf'),
				'perplexity': float('inf'),
				'accuracy': 0.0,
				'correct': 0,
				'total': 0,
			}

		avg_log_prob = sum(log_probs) / len(log_probs)
		return {
			'cross_entropy': -avg_log_prob,
			'perplexity': exp(-avg_log_prob),
			'accuracy': correct / total if total > 0 else 0.0,
			'correct': correct,
			'total': total,
		}

	def get_all_category_stats(self) -> Optional[list[dict]]:
		"""
		Get statistics for all categories.

		Returns:
			List of category stat dicts (each with name, cross_entropy, perplexity, etc.)
			Returns None if category tracking is not enabled.
		"""
		if self.num_categories is None:
			return None

		results = []
		for i in range(self.num_categories):
			stats = self.get_category_stats(i)
			stats['name'] = self.category_names[i] if self.category_names else f"cat_{i}"
			stats['category'] = i
			results.append(stats)
		return results

	def get_stats(self) -> dict:
		"""
		Get all statistics.

		Returns:
			Dict with cross_entropy, perplexity, accuracy, correct, total.
			If category tracking is enabled, also includes 'by_category' list.
		"""
		stats = {
			'cross_entropy': self.get_cross_entropy(),
			'perplexity': self.get_perplexity(),
			'accuracy': self.get_accuracy(),
			'correct': self._correct,
			'total': self._total,
		}

		# Include per-category stats if enabled
		if self.num_categories is not None:
			stats['by_category'] = self.get_all_category_stats()

		return stats

	@staticmethod
	def calculate_ppl_from_log_probs(log_probs: list[float]) -> float:
		"""Static helper to calculate PPL from list of log probabilities."""
		if not log_probs:
			return float('inf')
		avg_log_prob = sum(log_probs) / len(log_probs)
		return exp(-avg_log_prob)

	@staticmethod
	def interpolated_vote_prob(
		votes: dict[str, float],
		target: str,
		smoothing_prob: float,
		min_prob: float = 1e-4,
		vote_lambda: float = 0.8,
	) -> float:
		"""
		Compute interpolated probability without adding to the calculator.

		Use this when you need the probability value but don't want to accumulate.

		Args:
			votes: dict mapping predictions to vote weights
			target: the target word
			smoothing_prob: probability from smoothing model
			min_prob: minimum probability floor (default 1e-4)
			vote_lambda: interpolation weight (default 0.8)

		Returns:
			Interpolated probability: vote_lambda * vote_prob + (1-vote_lambda) * smoothing_prob
		"""
		smooth_p = max(smoothing_prob, min_prob) if smoothing_prob else min_prob

		total_weight = sum(votes.values()) if votes else 0
		if total_weight > 0:
			vote_p = votes.get(target, 0.0) / total_weight
		else:
			vote_p = 0.0

		final_prob = vote_lambda * vote_p + (1 - vote_lambda) * smooth_p
		return max(final_prob, min_prob)

	@staticmethod
	def log_prob(prob: float, min_prob: float = 1e-10) -> float:
		"""Static helper to safely compute log probability."""
		return log(max(prob, min_prob))

	def add_from_scores_batch(
		self,
		scores: 'Tensor',
		targets: 'Tensor',
		normalize: bool = True,
		categories: Optional['Tensor'] = None,
	) -> None:
		"""
		Add observations from batch of unnormalized scores (e.g., from RAMClusterLayer).

		CRITICAL: RAMClusterLayer outputs independent scores per cluster that don't
		sum to 1. This method applies softmax normalization to get true probabilities
		for perplexity calculation.

		Args:
			scores: [batch, vocab_size] tensor of unnormalized scores
			targets: [batch] tensor of target indices
			normalize: Whether to apply softmax normalization (default True).
			          Set False if scores are already proper probabilities.
			categories: Optional [batch] tensor of category indices for per-category
			           tracking. Each value should be in [0, num_categories).
			           Only used if num_categories was set in constructor.
		"""
		from torch import arange
		from torch.nn.functional import softmax

		batch_size = scores.shape[0]

		if normalize:
			# Apply softmax to convert scores to proper probability distribution
			probs = softmax(scores, dim=-1)  # [batch, vocab_size], sums to 1
		else:
			probs = scores

		# Get target probabilities
		batch_indices = arange(batch_size, device=scores.device)
		target_probs = probs[batch_indices, targets]  # [batch]

		# Clamp and compute log probabilities
		target_probs = target_probs.clamp(min=self.min_prob)
		log_probs = target_probs.log()

		# Track accuracy: correct if argmax == target
		predicted = probs.argmax(dim=-1)  # [batch]
		correct_mask = (predicted == targets)

		# Add to overall accumulator
		log_probs_list = log_probs.tolist()
		correct_list = correct_mask.tolist()

		for lp in log_probs_list:
			self._log_probs.append(lp)
		self._correct += correct_mask.sum().item()
		self._total += batch_size

		# Add to per-category accumulators if enabled
		if self.num_categories is not None and categories is not None:
			cat_list = categories.tolist()
			for i, (lp, is_correct, cat) in enumerate(zip(log_probs_list, correct_list, cat_list)):
				if 0 <= cat < self.num_categories:
					self._cat_log_probs[cat].append(lp)
					if is_correct:
						self._cat_correct[cat] += 1
					self._cat_total[cat] += 1

	def ram_counts_to_scores(
		self,
		true_counts: 'Tensor',
		empty_counts: 'Tensor',
		neurons_per_cluster: int,
	) -> 'Tensor':
		"""
		Convert RAM memory TRUE/EMPTY counts to scores using this calculator's empty_value.

		This centralizes the empty_value logic so RAM layers don't need to hardcode it.

		Score = (TRUE + empty_value * EMPTY) / neurons_per_cluster

		Args:
			true_counts: [batch, num_clusters] count of TRUE cells per cluster
			empty_counts: [batch, num_clusters] count of EMPTY cells per cluster
			neurons_per_cluster: Number of neurons per cluster (for normalization)

		Returns:
			[batch, num_clusters] scores (not yet softmax-normalized)
		"""
		return (true_counts.float() + self.empty_value * empty_counts.float()) / neurons_per_cluster

	@staticmethod
	def normalize_scores(scores: 'Tensor') -> 'Tensor':
		"""
		Apply softmax normalization to convert scores to probabilities.

		Use this when you need normalized probabilities but don't want to
		accumulate statistics.

		Args:
			scores: [batch, vocab_size] tensor of unnormalized scores

		Returns:
			[batch, vocab_size] tensor of probabilities (sums to 1 per row)
		"""
		from torch.nn.functional import softmax
		return softmax(scores, dim=-1)

	@staticmethod
	def ce_to_ppl(ce: float) -> float:
		"""Convert cross-entropy to perplexity. PPL = exp(CE)."""
		return exp(ce)

	@staticmethod
	def ppl_to_ce(ppl: float) -> float:
		"""Convert perplexity to cross-entropy. CE = log(PPL)."""
		return log(ppl)

	@staticmethod
	def ppl_improvement_pct(prev_ce: float, curr_ce: float) -> float:
		"""
		Calculate PPL improvement percentage from CE values.

		Since PPL = exp(CE), a decrease in CE means a decrease in PPL.
		Returns positive value if improved (curr < prev), negative if worse.

		Args:
			prev_ce: Previous cross-entropy value
			curr_ce: Current cross-entropy value

		Returns:
			Percentage improvement in PPL: (prev_ppl - curr_ppl) / prev_ppl * 100
		"""
		prev_ppl = exp(prev_ce)
		curr_ppl = exp(curr_ce)
		if prev_ppl == 0:
			return 0.0
		return (prev_ppl - curr_ppl) / prev_ppl * 100
