#!/usr/bin/env python3
"""
WS0: Confidence Measurement & Cache Hit Rate Experiment.

Analyzes RAM prediction confidence on WikiText-2 to determine
whether RAM is useful as a "fast path" in a hybrid architecture.

Outputs:
- Threshold sweep table (coverage vs accuracy tradeoff)
- Best threshold recommendation
- JSON results for downstream use

Usage:
	python tests/test_confidence.py
	python tests/test_confidence.py --model-path checkpoints/ramlm.pt
	python tests/test_confidence.py --output experiments/confidence_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from wnn.ram.core import AccelerationMode
from wnn.ram.core.models import RAMLM
from wnn.ram.strategies.confidence import ConfidenceAnalyzer


def load_wikitext2_tokens(split: str = "validation") -> list[int]:
	"""Load WikiText-2 tokens using GPT-2 tokenizer."""
	from datasets import load_dataset

	ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
	text = "\n\n".join(ds["text"])

	from transformers import GPT2TokenizerFast
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokens = tokenizer.encode(text)
	print(f"Loaded {split}: {len(tokens):,} tokens")
	return tokens


def create_default_model(train_tokens: list[int]) -> RAMLM:
	"""Create and train a default RAMLM for testing."""
	from collections import Counter

	# Use best known config: asymmetric tiered
	counts = Counter(train_tokens)
	vocab_size = 50257
	seen_tokens = set(t for t, _ in counts.most_common())
	unseen_tokens = [t for t in range(vocab_size) if t not in seen_tokens]
	cluster_order = [t for t, _ in counts.most_common()] + unseen_tokens

	model = RAMLM(
		vocab_size=vocab_size,
		context_size=4,
		tiers=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
		cluster_order=cluster_order,
	)

	print(f"\n{model}\n")
	print("Training...")
	t0 = time.time()
	model.train_epoch_fast_auto(train_tokens, global_top_k=1000, batch_size=5000)
	print(f"Training took {time.time() - t0:.1f}s\n")

	return model


def main():
	parser = argparse.ArgumentParser(description="WS0: Confidence analysis for RAM LM")
	parser.add_argument("--model-path", type=str, help="Path to saved RAMLM model")
	parser.add_argument("--output", type=str, default="experiments/confidence_results.json",
						help="Output JSON path")
	parser.add_argument("--batch-size", type=int, default=5000)
	args = parser.parse_args()

	print("=" * 60)
	print("WS0: Confidence Measurement & Cache Hit Rate")
	print("=" * 60)

	# Load data
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")

	# Load or create model
	if args.model_path:
		print(f"\nLoading model from {args.model_path}...")
		model = RAMLM.load(args.model_path)
	else:
		model = create_default_model(train_tokens)

	# Run confidence analysis
	print("\n--- Confidence Analysis ---")
	t0 = time.time()
	report = ConfidenceAnalyzer.analyze_sequence(
		model, val_tokens,
		batch_size=args.batch_size,
		backend=AccelerationMode.AUTO,
	)
	analysis_time = time.time() - t0
	print(f"Analysis took {analysis_time:.1f}s")

	# Sweep all four metrics and compare
	metrics = ['full_entropy', 'sparse_entropy', 'score_margin', 'true_count_ratio']
	all_sweeps = {}
	all_bests = {}

	for metric in metrics:
		print(f"\n--- Threshold Sweep: {metric} ---")
		results = ConfidenceAnalyzer.sweep_thresholds(report, metric=metric)
		print(ConfidenceAnalyzer.format_results(results))
		all_sweeps[metric] = results

		best = ConfidenceAnalyzer.find_best_threshold(results, min_coverage=0.2)
		all_bests[metric] = best
		if best:
			print(f"\nBest threshold: {best.threshold:.4f}")
			print(f"  Coverage: {best.coverage:.1%} of examples handled by RAM")
			print(f"  Accuracy (confident): {best.accuracy:.2%}")
			print(f"  CE (confident): {best.avg_ce:.4f}")
			print(f"  Accuracy (fallback): {best.fallback_accuracy:.2%}")
			print(f"  CE (fallback): {best.fallback_ce:.4f}")
		else:
			print("\nNo threshold found meeting criteria (>= 20% coverage)")

	# Print comparison summary
	overall_acc = sum(report.is_correct) / len(report.is_correct)
	print("\n" + "=" * 80)
	print("METRIC COMPARISON SUMMARY")
	print("=" * 80)
	print(f"Overall accuracy: {overall_acc:.2%}")
	print(f"\n{'Metric':<20} {'Best Thresh':>12} {'Coverage':>10} {'Acc(conf)':>10} "
		  f"{'CE(conf)':>10} {'Acc Lift':>10}")
	print("-" * 72)
	for metric in metrics:
		best = all_bests[metric]
		if best:
			acc_lift = best.accuracy / overall_acc if overall_acc > 0 else 0
			print(f"{metric:<20} {best.threshold:>12.4f} {best.coverage:>10.1%} "
				  f"{best.accuracy:>10.2%} {best.avg_ce:>10.4f} {acc_lift:>10.2f}x")
		else:
			print(f"{metric:<20} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Distribution stats for each metric
	def dist_stats(values):
		if not values:
			return {}
		s = sorted(values)
		n = len(s)
		return {
			"min": s[0],
			"p25": s[n // 4],
			"median": s[n // 2],
			"p75": s[3 * n // 4],
			"max": s[-1],
			"mean": sum(s) / n,
		}

	output = {
		"experiment": "WS0_confidence_analysis_v2",
		"model_config": model.get_config() if not args.model_path else {"path": args.model_path},
		"dataset": "wikitext-2-raw-v1",
		"split": "validation",
		"total_examples": report.total,
		"analysis_time_s": round(analysis_time, 1),
		"summary": {
			"overall_accuracy": overall_acc,
			"avg_full_entropy": sum(report.entropies) / len(report.entropies),
			"avg_sparse_entropy": sum(report.sparse_entropies) / len(report.sparse_entropies),
			"avg_score_margin": sum(report.score_margins) / len(report.score_margins),
			"avg_true_count_ratio": sum(report.true_count_ratios) / len(report.true_count_ratios),
			"avg_num_nonzero_clusters": sum(report.num_nonzero_clusters) / len(report.num_nonzero_clusters),
		},
		"distributions": {
			"full_entropy": dist_stats(report.entropies),
			"sparse_entropy": dist_stats(report.sparse_entropies),
			"score_margin": dist_stats(report.score_margins),
			"true_count_ratio": dist_stats(report.true_count_ratios),
			"num_nonzero_clusters": dist_stats(report.num_nonzero_clusters),
		},
		"metric_sweeps": {},
		"metric_comparison": {},
	}

	for metric in metrics:
		results = all_sweeps[metric]
		best = all_bests[metric]
		output["metric_sweeps"][metric] = [
			{
				"threshold": r.threshold,
				"coverage": round(r.coverage, 4),
				"accuracy_confident": round(r.accuracy, 4),
				"ce_confident": round(r.avg_ce, 4),
				"accuracy_fallback": round(r.fallback_accuracy, 4),
				"ce_fallback": round(r.fallback_ce, 4),
			}
			for r in results
		]
		if best:
			acc_lift = best.accuracy / overall_acc if overall_acc > 0 else 0
			output["metric_comparison"][metric] = {
				"best_threshold": best.threshold,
				"coverage": round(best.coverage, 4),
				"accuracy": round(best.accuracy, 4),
				"ce": round(best.avg_ce, 4),
				"accuracy_lift": round(acc_lift, 2),
			}

	# Serialize: convert non-serializable types
	def make_serializable(obj):
		if isinstance(obj, (list, tuple)):
			if len(obj) > 1000:
				return f"<list of {len(obj)} items>"
			return [make_serializable(x) for x in obj]
		if isinstance(obj, dict):
			return {k: make_serializable(v) for k, v in obj.items()}
		return obj

	output = make_serializable(output)

	with open(output_path, "w") as f:
		json.dump(output, f, indent=2)
	print(f"\nResults saved to {output_path}")

	print("\n" + "=" * 60)
	print("WS0 Complete")
	print("=" * 60)


if __name__ == "__main__":
	main()
