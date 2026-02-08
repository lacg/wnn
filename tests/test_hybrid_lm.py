#!/usr/bin/env python3
"""
WS2: Hybrid RAM+Transformer Experiment.

1. Load/train RAMLM + load distilgpt2 (or TinyTransformerLM)
2. Run WS0 analysis to find optimal threshold
3. Evaluate hybrid with different blending modes
4. Measure: total CE, compute savings, per-path breakdown

Usage:
	python tests/test_hybrid_lm.py
	python tests/test_hybrid_lm.py --transformer tiny --epochs 3
	python tests/test_hybrid_lm.py --transformer distilgpt2
	python tests/test_hybrid_lm.py --output experiments/hybrid_results.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

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


def create_ram_model(train_tokens: list[int]) -> RAMLM:
	"""Create and train default RAMLM."""
	counts = Counter(train_tokens)
	sorted_tokens = [tok for tok, _ in counts.most_common()]

	model = RAMLM(
		vocab_size=50257,
		context_size=4,
		tiers=[(100, 15, 20), (400, 10, 12), (None, 5, 8)],
		cluster_order=sorted_tokens,
	)
	print(f"\n{model}\n")
	print("Training RAM model...")
	t0 = time.time()
	model.train_epoch_fast_auto(train_tokens, global_top_k=1000, batch_size=5000)
	print(f"RAM training: {time.time() - t0:.1f}s\n")
	return model


def load_transformer(model_type: str, train_tokens: list[int], epochs: int = 3):
	"""Load or train transformer model."""
	if model_type == "distilgpt2":
		print("Loading distilgpt2...")
		from transformers import AutoModelForCausalLM
		model = AutoModelForCausalLM.from_pretrained("distilgpt2")
		model.eval()
		return model

	elif model_type == "tiny":
		from wnn.ram.core.models.tiny_transformer import TinyTransformerLM
		print("Creating TinyTransformerLM (5M params)...")
		model = TinyTransformerLM(
			vocab_size=50257, d_model=256, n_heads=4,
			n_layers=4, d_ff=1024, max_context=128,
		)
		print(f"  Parameters: {model.num_parameters():,}")
		print(f"  Training for {epochs} epochs...")
		for epoch in range(epochs):
			print(f"\n  Epoch {epoch + 1}/{epochs}:")
			model.train_epoch(train_tokens, batch_size=32, seq_len=64, lr=1e-4)
		return model

	else:
		raise ValueError(f"Unknown transformer type: {model_type}")


def main():
	parser = argparse.ArgumentParser(description="WS2: Hybrid experiment")
	parser.add_argument("--transformer", choices=["distilgpt2", "tiny"], default="tiny")
	parser.add_argument("--epochs", type=int, default=3, help="TinyTransformer training epochs")
	parser.add_argument("--output", type=str, default="experiments/hybrid_results.json")
	parser.add_argument("--batch-size", type=int, default=50)
	parser.add_argument("--model-path", type=str, help="Path to saved RAMLM")
	args = parser.parse_args()

	print("=" * 60)
	print("WS2: Hybrid RAM+Transformer Experiment")
	print("=" * 60)

	# Load data
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")

	# Create/load RAM model
	if args.model_path:
		print(f"Loading RAM model from {args.model_path}...")
		ram = RAMLM.load(args.model_path)
	else:
		ram = create_ram_model(train_tokens)

	# Evaluate RAM alone
	print("\n--- RAM-only evaluation ---")
	ram_stats = ram.evaluate_fast(val_tokens, batch_size=5000)

	# Find optimal threshold via WS0
	print("\n--- Confidence Analysis (WS0) ---")
	report = ConfidenceAnalyzer.analyze_sequence(ram, val_tokens, batch_size=5000)
	threshold_results = ConfidenceAnalyzer.sweep_thresholds(report)
	print(ConfidenceAnalyzer.format_results(threshold_results))

	best_threshold = ConfidenceAnalyzer.find_best_threshold(threshold_results)
	if best_threshold:
		optimal_threshold = best_threshold.threshold
		print(f"\nUsing optimal threshold: {optimal_threshold:.2f}")
	else:
		optimal_threshold = 2.0
		print(f"\nNo optimal threshold found, using default: {optimal_threshold:.2f}")

	# Load transformer
	print(f"\n--- Loading Transformer ({args.transformer}) ---")
	transformer = load_transformer(args.transformer, train_tokens, args.epochs)

	# Evaluate transformer alone
	if args.transformer == "tiny":
		print("\n--- Transformer-only evaluation ---")
		transformer_stats = transformer.evaluate(val_tokens, batch_size=32)
	else:
		print("\n--- Transformer-only evaluation (distilgpt2) ---")
		# Quick evaluation for HF model
		from wnn.ram.core.models.hybrid_lm import HybridRAMTransformerLM
		dummy_hybrid = HybridRAMTransformerLM(
			ram, transformer, entropy_threshold=0.0, blending_mode='select'
		)
		# threshold=0 means everything goes to transformer
		transformer_stats = dummy_hybrid.evaluate(
			val_tokens, batch_size=args.batch_size, verbose=True
		)
		transformer_stats = transformer_stats.get("overall", transformer_stats)

	# Evaluate hybrid with different modes and thresholds
	from wnn.ram.core.models.hybrid_lm import HybridRAMTransformerLM

	all_results = {
		"ram_only": {
			"cross_entropy": ram_stats["cross_entropy"],
			"perplexity": ram_stats["perplexity"],
			"accuracy": ram_stats["accuracy"],
		},
		"transformer_only": {
			"cross_entropy": transformer_stats.get("cross_entropy", float("inf")),
			"perplexity": transformer_stats.get("perplexity", float("inf")),
			"accuracy": transformer_stats.get("accuracy", 0.0),
		},
		"hybrid": {},
	}

	for mode in ["select", "interpolate"]:
		for threshold in [optimal_threshold, 2.0, 5.0, 8.0]:
			print(f"\n--- Hybrid: mode={mode}, threshold={threshold:.1f} ---")
			hybrid = HybridRAMTransformerLM(
				ram, transformer,
				entropy_threshold=threshold,
				blending_mode=mode,
			)

			result = hybrid.evaluate(
				val_tokens,
				batch_size=args.batch_size,
				verbose=True,
			)

			key = f"{mode}_t{threshold:.1f}"
			all_results["hybrid"][key] = {
				"overall_ce": result["overall"]["cross_entropy"],
				"overall_ppl": result["overall"]["perplexity"],
				"overall_acc": result["overall"]["accuracy"],
				"ram_fraction": result["ram_path"]["fraction"],
				"ram_ce": result["ram_path"]["cross_entropy"],
				"transformer_ce": result["transformer_path"]["cross_entropy"],
				"transformer_calls_avoided": result["compute_savings"]["fraction_avoided"],
			}

	# Summary
	print("\n" + "=" * 80)
	print("SUMMARY")
	print("=" * 80)
	print(f"{'Configuration':<30} {'CE':>10} {'PPL':>12} {'Acc':>10} {'RAM%':>10}")
	print("-" * 72)

	r = all_results["ram_only"]
	print(f"{'RAM only':<30} {r['cross_entropy']:>10.4f} {r['perplexity']:>12.2f} {r['accuracy']:>10.2%} {'100%':>10}")

	r = all_results["transformer_only"]
	print(f"{'Transformer only':<30} {r['cross_entropy']:>10.4f} {r['perplexity']:>12.2f} {r['accuracy']:>10.2%} {'0%':>10}")

	for key, r in all_results["hybrid"].items():
		print(f"{'Hybrid ' + key:<30} {r['overall_ce']:>10.4f} {r['overall_ppl']:>12.2f} "
			  f"{r['overall_acc']:>10.2%} {r['ram_fraction']:>10.1%}")

	# Save
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output = {
		"experiment": "WS2_hybrid_ram_transformer",
		"transformer_type": args.transformer,
		"optimal_threshold": optimal_threshold,
		"results": all_results,
	}

	with open(output_path, "w") as f:
		json.dump(output, f, indent=2, default=str)
	print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
	main()
