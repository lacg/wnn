"""
Confidence Analysis for RAM Language Models.

Measures prediction confidence and cache hit rate to determine
how useful RAM is as a "fast path" in a hybrid architecture.

Three complementary confidence metrics:

1. Sparse Entropy: Softmax only over non-zero clusters (those with at least
   one TRUE neuron). Avoids dilution from 50K zero-score clusters.
2. Score Margin: Gap between top-1 and top-2 raw scores. Bigger = more confident.
3. TRUE Count Ratio: Fraction of neurons that voted TRUE for the winner vs
   total non-empty neurons. Direct measure of evidence strength.

The original full-softmax entropy is also computed for comparison, but it
suffers from dilution: with 50K clusters where most score 0.0 (EMPTY=0.0),
softmax maps them all to exp(0)/Z, producing near-maximum entropy regardless
of actual RAM confidence.

Usage:
	from wnn.ram.strategies.confidence import ConfidenceAnalyzer

	report = analyzer.analyze_sequence(model, token_ids)
	results = analyzer.sweep_thresholds(report, metric='sparse_entropy')
"""

from dataclasses import dataclass, field
from math import log, exp
from typing import Optional

from torch import Tensor, tensor, long, arange, float32, zeros
from torch.nn.functional import softmax


@dataclass
class ConfidenceReport:
	"""Results from confidence analysis on a dataset.

	Contains per-example metrics from multiple confidence measures.

	Attributes:
		entropies: Full-softmax entropy H = -sum(p*log(p)) per example (nats).
		sparse_entropies: Entropy over non-zero clusters only (avoids dilution).
		score_margins: top1_score - top2_score per example (raw score gap).
		true_count_ratios: TRUE neurons for winner / total non-EMPTY neurons.
		confidences: max(p) per example from full softmax.
		is_correct: Whether argmax == target per example.
		target_probs: P(target) per example (after softmax normalization).
		num_nonzero_clusters: Count of clusters with score > 0 per example.
		total: Number of examples analyzed.
	"""
	entropies: list[float] = field(default_factory=list)
	sparse_entropies: list[float] = field(default_factory=list)
	score_margins: list[float] = field(default_factory=list)
	true_count_ratios: list[float] = field(default_factory=list)
	confidences: list[float] = field(default_factory=list)
	is_correct: list[bool] = field(default_factory=list)
	target_probs: list[float] = field(default_factory=list)
	num_nonzero_clusters: list[int] = field(default_factory=list)
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
	def compute_sparse_entropy(scores: Tensor) -> Tensor:
		"""Compute entropy over non-zero clusters only.

		Applies softmax only to clusters with score > 0, avoiding the
		dilution from thousands of zero-score (all-EMPTY) clusters.

		Args:
			scores: [batch, num_clusters] raw scores (before softmax).

		Returns:
			[batch] sparse entropy tensor (in nats).
		"""
		eps = 1e-10
		batch_size = scores.shape[0]
		result = zeros(batch_size, device=scores.device)

		for i in range(batch_size):
			nonzero_mask = scores[i] > 0
			nz_count = nonzero_mask.sum().item()
			if nz_count <= 1:
				result[i] = 0.0  # 0 or 1 non-zero = no uncertainty
				continue
			nz_scores = scores[i][nonzero_mask]
			nz_probs = softmax(nz_scores, dim=-1)
			result[i] = -(nz_probs * (nz_probs + eps).log()).sum()

		return result

	@staticmethod
	def compute_score_margin(scores: Tensor) -> Tensor:
		"""Compute gap between top-1 and top-2 raw scores.

		Larger margin = more confident prediction.

		Args:
			scores: [batch, num_clusters] raw scores.

		Returns:
			[batch] score margin tensor.
		"""
		if scores.shape[1] < 2:
			return zeros(scores.shape[0], device=scores.device)
		top2 = scores.topk(2, dim=-1).values  # [batch, 2]
		return top2[:, 0] - top2[:, 1]

	@staticmethod
	def compute_true_count_ratio(scores: Tensor, neurons_per_cluster: int) -> Tensor:
		"""Compute TRUE neuron ratio for the winning cluster.

		Score = count_TRUE / neurons_per_cluster (with empty_value=0.0).
		This directly measures evidence strength.

		Args:
			scores: [batch, num_clusters] raw scores.
			neurons_per_cluster: Neurons per cluster (for interpreting scores).

		Returns:
			[batch] ratio of TRUE neurons for the winner.
		"""
		top_scores = scores.max(dim=-1).values
		return top_scores

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

			# --- Metric 1: Full-softmax entropy (original) ---
			probs = softmax(scores, dim=-1)
			batch_entropies = ConfidenceAnalyzer.compute_entropy(probs)

			# --- Metric 2: Sparse entropy (non-zero clusters only) ---
			batch_sparse_entropies = ConfidenceAnalyzer.compute_sparse_entropy(scores)

			# --- Metric 3: Score margin (top1 - top2 raw scores) ---
			batch_margins = ConfidenceAnalyzer.compute_score_margin(scores)

			# --- Metric 4: TRUE count ratio (winner's score) ---
			batch_true_ratios = ConfidenceAnalyzer.compute_true_count_ratio(scores, 0)

			# Count non-zero clusters per example
			batch_nonzero = (scores > 0).sum(dim=-1)

			# Full-softmax confidence and target probs
			batch_confidences = probs.max(dim=-1).values
			batch_indices = arange(end - start, device=scores.device)
			batch_target_probs = probs[batch_indices, batch_targets]

			# Correctness (argmax of raw scores = same as argmax of softmax)
			predicted = scores.argmax(dim=-1)
			batch_correct = (predicted == batch_targets)

			# Accumulate all metrics
			report.entropies.extend(batch_entropies.tolist())
			report.sparse_entropies.extend(batch_sparse_entropies.tolist())
			report.score_margins.extend(batch_margins.tolist())
			report.true_count_ratios.extend(batch_true_ratios.tolist())
			report.confidences.extend(batch_confidences.tolist())
			report.is_correct.extend(batch_correct.tolist())
			report.target_probs.extend(batch_target_probs.tolist())
			report.num_nonzero_clusters.extend(batch_nonzero.tolist())

			if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
				pct = (end / total_examples) * 100
				print(f"    {pct:5.1f}% ({end:,}/{total_examples:,})")

		report.total = total_examples

		if verbose:
			avg_entropy = sum(report.entropies) / len(report.entropies)
			avg_sparse = sum(report.sparse_entropies) / len(report.sparse_entropies)
			avg_margin = sum(report.score_margins) / len(report.score_margins)
			avg_true_ratio = sum(report.true_count_ratios) / len(report.true_count_ratios)
			avg_nonzero = sum(report.num_nonzero_clusters) / len(report.num_nonzero_clusters)
			overall_acc = sum(report.is_correct) / len(report.is_correct)
			print(f"  Full-softmax entropy: {avg_entropy:.4f} (max={max(report.entropies):.4f})")
			print(f"  Sparse entropy:       {avg_sparse:.4f} (max={max(report.sparse_entropies):.4f})")
			print(f"  Score margin:         {avg_margin:.4f} (max={max(report.score_margins):.4f})")
			print(f"  TRUE count ratio:     {avg_true_ratio:.4f} (max={max(report.true_count_ratios):.4f})")
			print(f"  Avg non-zero clusters: {avg_nonzero:.0f}")
			print(f"  Overall accuracy:     {overall_acc:.2%}")

		return report

	@staticmethod
	def sweep_thresholds(
		report: ConfidenceReport,
		thresholds: Optional[list[float]] = None,
		metric: str = 'sparse_entropy',
	) -> list[ThresholdResult]:
		"""Sweep thresholds to find optimal RAM/transformer partition.

		For each threshold, partitions examples into:
		- Confident (metric_value < threshold for entropy metrics,
		  metric_value > threshold for margin/ratio metrics): RAM handles
		- Uncertain: Transformer handles

		Args:
			report: ConfidenceReport from analyze_sequence.
			thresholds: List of threshold values to try. If None, auto-generates
				appropriate range for the selected metric.
			metric: Which confidence metric to use:
				'full_entropy' - original full-softmax entropy (lower = more confident)
				'sparse_entropy' - entropy over non-zero clusters (lower = more confident)
				'score_margin' - top1-top2 score gap (higher = more confident)
				'true_count_ratio' - winner's raw score (higher = more confident)

		Returns:
			List of ThresholdResult for each threshold, sorted by threshold.
		"""
		# Select metric values and direction
		if metric == 'full_entropy':
			values = report.entropies
			lower_is_confident = True
		elif metric == 'sparse_entropy':
			values = report.sparse_entropies
			lower_is_confident = True
		elif metric == 'score_margin':
			values = report.score_margins
			lower_is_confident = False  # higher margin = more confident
		elif metric == 'true_count_ratio':
			values = report.true_count_ratios
			lower_is_confident = False  # higher ratio = more confident
		else:
			raise ValueError(f"Unknown metric: {metric}")

		# Auto-generate thresholds if not provided
		if thresholds is None:
			if not values:
				return []
			v_min = min(values)
			v_max = max(values)
			v_range = v_max - v_min
			if v_range < 1e-8:
				return []
			# 20 evenly spaced thresholds across the data range
			thresholds = [v_min + v_range * i / 19 for i in range(20)]

		results = []
		min_prob = 1e-10

		for threshold in sorted(thresholds):
			confident_correct = 0
			confident_total = 0
			confident_log_probs = []

			uncertain_correct = 0
			uncertain_total = 0
			uncertain_log_probs = []

			for i in range(report.total):
				value = values[i]
				correct = report.is_correct[i]
				target_prob = max(report.target_probs[i], min_prob)
				lp = log(target_prob)

				# Determine if this example is "confident"
				if lower_is_confident:
					is_confident = value < threshold
				else:
					is_confident = value > threshold

				if is_confident:
					confident_total += 1
					confident_log_probs.append(lp)
					if correct:
						confident_correct += 1
				else:
					uncertain_total += 1
					uncertain_log_probs.append(lp)
					if correct:
						uncertain_correct += 1

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
