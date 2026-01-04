"""
Perplexity Calculator - Unified logic for consistent PPL calculation.

This module provides a centralized perplexity calculation class to ensure
consistency across the codebase. Use this instead of scattered PPL logic.
"""

from math import exp, log
from typing import Optional


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

	def __init__(self, vocab_size: int):
		"""
		Initialize the calculator.

		Args:
			vocab_size: Size of vocabulary (used for min_prob = 1/vocab_size)
		"""
		self.vocab_size = vocab_size
		self.min_prob = 1.0 / vocab_size
		self._log_min_prob = log(self.min_prob)
		self.reset()

	def reset(self):
		"""Reset accumulator for new evaluation."""
		self._log_probs: list[float] = []
		self._correct: int = 0
		self._total: int = 0

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

	def get_perplexity(self) -> float:
		"""Calculate perplexity from accumulated observations."""
		if not self._log_probs:
			return float('inf')
		avg_log_prob = sum(self._log_probs) / len(self._log_probs)
		return exp(-avg_log_prob)

	def get_accuracy(self) -> float:
		"""Get accuracy from accumulated observations."""
		return self._correct / self._total if self._total > 0 else 0.0

	def get_stats(self) -> dict:
		"""Get all statistics."""
		return {
			'perplexity': self.get_perplexity(),
			'accuracy': self.get_accuracy(),
			'correct': self._correct,
			'total': self._total,
		}

	@staticmethod
	def calculate_ppl_from_log_probs(log_probs: list[float]) -> float:
		"""Static helper to calculate PPL from list of log probabilities."""
		if not log_probs:
			return float('inf')
		avg_log_prob = sum(log_probs) / len(log_probs)
		return exp(-avg_log_prob)

	@staticmethod
	def log_prob(prob: float, min_prob: float = 1e-10) -> float:
		"""Static helper to safely compute log probability."""
		return log(max(prob, min_prob))
