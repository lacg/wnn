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

	# Sweep thresholds
	print("\n--- Threshold Sweep ---")
	results = ConfidenceAnalyzer.sweep_thresholds(report)
	print(ConfidenceAnalyzer.format_results(results))

	# Find best threshold
	best = ConfidenceAnalyzer.find_best_threshold(results, min_coverage=0.2)
	if best:
		print(f"\nBest threshold: {best.threshold:.2f}")
		print(f"  Coverage: {best.coverage:.1%} of examples handled by RAM")
		print(f"  Accuracy (confident): {best.accuracy:.2%}")
		print(f"  CE (confident): {best.avg_ce:.4f}")
		print(f"  Accuracy (fallback): {best.fallback_accuracy:.2%}")
		print(f"  CE (fallback): {best.fallback_ce:.4f}")
	else:
		print("\nNo threshold found meeting criteria (>= 20% coverage)")

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output = {
		"experiment": "WS0_confidence_analysis",
		"model_config": model.get_config() if not args.model_path else {"path": args.model_path},
		"dataset": "wikitext-2-raw-v1",
		"split": "validation",
		"total_examples": report.total,
		"analysis_time_s": round(analysis_time, 1),
		"summary": {
			"avg_entropy": sum(report.entropies) / len(report.entropies),
			"avg_confidence": sum(report.confidences) / len(report.confidences),
			"overall_accuracy": sum(report.is_correct) / len(report.is_correct),
		},
		"threshold_sweep": [
			{
				"threshold": r.threshold,
				"coverage": round(r.coverage, 4),
				"accuracy_confident": round(r.accuracy, 4),
				"ce_confident": round(r.avg_ce, 4),
				"accuracy_fallback": round(r.fallback_accuracy, 4),
				"ce_fallback": round(r.fallback_ce, 4),
			}
			for r in results
		],
		"best_threshold": {
			"threshold": best.threshold,
			"coverage": round(best.coverage, 4),
			"accuracy": round(best.accuracy, 4),
			"ce": round(best.avg_ce, 4),
		} if best else None,
	}

	# Serialize: convert non-serializable types
	def make_serializable(obj):
		if isinstance(obj, (list, tuple)):
			# Don't serialize large lists like cluster_order
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
