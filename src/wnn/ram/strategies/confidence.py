"""
Confidence Analysis for RAM Language Models.

Measures prediction confidence and cache hit rate to determine
how useful RAM is as a "fast path" in a hybrid architecture.

Key metrics:
- Entropy: H = -sum(p*log(p)) measures uncertainty in predictions
- Confidence: max(p) measures how dominant the top prediction is
- Threshold sweep: finds optimal confidence cutoff for RAM vs fallback

Usage:
	from wnn.ram.strategies.confidence import ConfidenceAnalyzer

	analyzer = ConfidenceAnalyzer()
	report = analyzer.analyze_sequence(model, token_ids)
	results = analyzer.sweep_thresholds(report)

	# Find threshold where RAM covers >= 30% with good accuracy
	for r in results:
		if r.coverage >= 0.3 and r.accuracy > 0.10:
			print(f"Threshold {r.threshold:.2f}: {r.coverage:.1%} coverage, {r.accuracy:.1%} acc")
"""

from dataclasses import dataclass, field
from math import log, exp
from typing import Optional

from torch import Tensor, tensor, long, arange, float32, zeros
from torch.nn.functional import softmax


@dataclass
class ConfidenceReport:
	"""Results from confidence analysis on a dataset.

	Contains per-example entropy, confidence, correctness, and target probability
	for analyzing RAM prediction quality.

	Attributes:
		entropies: H = -sum(p*log(p)) per example (nats). Lower = more confident.
		confidences: max(p) per example. Higher = more confident.
		is_correct: Whether argmax == target per example.
		target_probs: P(target) per example (after softmax normalization).
		total: Number of examples analyzed.
	"""
	entropies: list[float] = field(default_factory=list)
	confidences: list[float] = field(default_factory=list)
	is_correct: list[bool] = field(default_factory=list)
	target_probs: list[float] = field(default_factory=list)
	total: int = 0


@dataclass
class ThresholdResult:
	"""Results for a specific confidence threshold.

	Partitions examples into "confident" (entropy < threshold) and
	"uncertain" (entropy >= threshold) groups and measures each.

	Attributes:
		threshold: Entropy threshold value.
		coverage: Fraction of examples below threshold (RAM would handle these).
		accuracy: Accuracy on confident predictions.
		avg_ce: Cross-entropy on confident predictions.
		fallback_accuracy: Accuracy on uncertain predictions (transformer would handle).
		fallback_ce: Cross-entropy on uncertain predictions.
	"""
	threshold: float
	coverage: float
	accuracy: float
	avg_ce: float
	fallback_accuracy: float
	fallback_ce: float


class ConfidenceAnalyzer:
	"""Analyzes RAM prediction confidence for hybrid architecture design.

	Computes per-example entropy and confidence from RAMLM forward passes,
	then sweeps thresholds to find the optimal partition between RAM fast
	path and transformer fallback.
	"""

	@staticmethod
	def compute_entropy(probs: Tensor) -> Tensor:
		"""Compute entropy of probability distributions.

		H = -sum(p * log(p + eps)) computed per row.

		Args:
			probs: [batch, vocab_size] probability tensor (should sum to ~1 per row)

		Returns:
			[batch] entropy tensor (in nats)
		"""
		eps = 1e-10
		log_probs = (probs + eps).log()
		return -(probs * log_probs).sum(dim=-1)

	@staticmethod
	def analyze_sequence(
		model: 'RAMLM',
		token_ids: list[int],
		batch_size: int = 5000,
		backend: 'AccelerationMode | None' = None,
		verbose: bool = True,
	) -> ConfidenceReport:
		"""Analyze confidence of RAM predictions on a token sequence.

		Runs the model on all sliding windows and records per-example
		entropy, confidence, correctness, and target probability.

		Args:
			model: Trained RAMLM model.
			token_ids: Token sequence to analyze.
			batch_size: Examples per batch (same as evaluate_fast).
			backend: Acceleration backend (None = AUTO).
			verbose: Print progress.

		Returns:
			ConfidenceReport with per-example metrics.
		"""
		from wnn.ram.core import AccelerationMode

		if backend is None:
			backend = AccelerationMode.AUTO

		total_examples = len(token_ids) - model.context_size

		if verbose:
			print(f"Confidence analysis on {total_examples:,} examples (batch_size={batch_size})...")

		# Pre-encode all contexts (same pattern as evaluate_fast)
		if verbose:
			print("  Encoding contexts...")
		all_bits = model.encode_sequence(token_ids)
		targets = tensor(token_ids[model.context_size:], dtype=long)

		report = ConfidenceReport()
		num_batches = (total_examples + batch_size - 1) // batch_size

		if verbose:
			print("  Analyzing batches...")

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			batch_bits = all_bits[start:end]
			batch_targets = targets[start:end]

			# Forward pass -> raw scores
			scores = model.forward(batch_bits, backend=backend)

			# Softmax normalize to get proper probabilities
			probs = softmax(scores, dim=-1)

			# Compute entropy per example
			batch_entropies = ConfidenceAnalyzer.compute_entropy(probs)

			# Get max probability (confidence) per example
			batch_confidences = probs.max(dim=-1).values

			# Get target probabilities
			batch_indices = arange(end - start, device=scores.device)
			batch_target_probs = probs[batch_indices, batch_targets]

			# Check correctness
			predicted = probs.argmax(dim=-1)
			batch_correct = (predicted == batch_targets)

			# Accumulate results
			report.entropies.extend(batch_entropies.tolist())
			report.confidences.extend(batch_confidences.tolist())
			report.is_correct.extend(batch_correct.tolist())
			report.target_probs.extend(batch_target_probs.tolist())

			if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
				pct = (end / total_examples) * 100
				print(f"    {pct:5.1f}% ({end:,}/{total_examples:,})")

		report.total = total_examples

		if verbose:
			avg_entropy = sum(report.entropies) / len(report.entropies)
			avg_confidence = sum(report.confidences) / len(report.confidences)
			overall_acc = sum(report.is_correct) / len(report.is_correct)
			print(f"  Average entropy: {avg_entropy:.4f}")
			print(f"  Average confidence: {avg_confidence:.6f}")
			print(f"  Overall accuracy: {overall_acc:.2%}")

		return report

	@staticmethod
	def sweep_thresholds(
		report: ConfidenceReport,
		thresholds: Optional[list[float]] = None,
	) -> list[ThresholdResult]:
		"""Sweep entropy thresholds to find optimal RAM/transformer partition.

		For each threshold, partitions examples into:
		- Confident (entropy < threshold): RAM handles these
		- Uncertain (entropy >= threshold): Transformer handles these

		Args:
			report: ConfidenceReport from analyze_sequence.
			thresholds: List of entropy thresholds to try.
				Default covers a wide range from very selective to very permissive.

		Returns:
			List of ThresholdResult for each threshold, sorted by threshold.
		"""
		if thresholds is None:
			# Default sweep: from very selective (low entropy only) to permissive
			thresholds = [
				0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5,
				3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.5, 10.8,
			]

		results = []
		min_prob = 1e-10

		for threshold in sorted(thresholds):
			# Partition examples by entropy threshold
			confident_correct = 0
			confident_total = 0
			confident_log_probs = []

			uncertain_correct = 0
			uncertain_total = 0
			uncertain_log_probs = []

			for i in range(report.total):
				entropy = report.entropies[i]
				correct = report.is_correct[i]
				target_prob = max(report.target_probs[i], min_prob)
				lp = log(target_prob)

				if entropy < threshold:
					confident_total += 1
					confident_log_probs.append(lp)
					if correct:
						confident_correct += 1
				else:
					uncertain_total += 1
					uncertain_log_probs.append(lp)
					if correct:
						uncertain_correct += 1

			# Compute metrics for each partition
			coverage = confident_total / report.total if report.total > 0 else 0.0
			accuracy = confident_correct / confident_total if confident_total > 0 else 0.0
			avg_ce = -sum(confident_log_probs) / len(confident_log_probs) if confident_log_probs else float('inf')

			fallback_accuracy = uncertain_correct / uncertain_total if uncertain_total > 0 else 0.0
			fallback_ce = -sum(uncertain_log_probs) / len(uncertain_log_probs) if uncertain_log_probs else float('inf')

			results.append(ThresholdResult(
				threshold=threshold,
				coverage=coverage,
				accuracy=accuracy,
				avg_ce=avg_ce,
				fallback_accuracy=fallback_accuracy,
				fallback_ce=fallback_ce,
			))

		return results

	@staticmethod
	def find_best_threshold(
		results: list[ThresholdResult],
		min_coverage: float = 0.2,
		min_accuracy_ratio: float = 1.5,
	) -> Optional[ThresholdResult]:
		"""Find the best threshold that meets coverage and accuracy criteria.

		Args:
			results: ThresholdResult list from sweep_thresholds.
			min_coverage: Minimum fraction of examples RAM must handle.
			min_accuracy_ratio: Minimum ratio of confident accuracy to overall accuracy.

		Returns:
			Best ThresholdResult meeting criteria, or None if no threshold qualifies.
		"""
		# Compute overall accuracy
		best = None
		for r in results:
			if r.coverage >= min_coverage and r.coverage < 1.0:
				# Prefer higher coverage * accuracy product
				score = r.coverage * r.accuracy
				if best is None or score > best.coverage * best.accuracy:
					best = r
		return best

	@staticmethod
	def format_results(results: list[ThresholdResult]) -> str:
		"""Format threshold sweep results as a readable table.

		Args:
			results: ThresholdResult list from sweep_thresholds.

		Returns:
			Formatted string table.
		"""
		lines = [
			f"{'Threshold':>10} {'Coverage':>10} {'Acc(conf)':>10} {'CE(conf)':>10} "
			f"{'Acc(fall)':>10} {'CE(fall)':>10}",
			"-" * 65,
		]
		for r in results:
			lines.append(
				f"{r.threshold:10.2f} {r.coverage:10.1%} {r.accuracy:10.2%} "
				f"{r.avg_ce:10.4f} {r.fallback_accuracy:10.2%} {r.fallback_ce:10.4f}"
			)
		return "\n".join(lines)
