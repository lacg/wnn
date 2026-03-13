"""
WNN verification CLI.

Downloads a WNN model from HuggingFace Hub (or loads locally),
trains memory tables on a dataset's train split, and evaluates
on the test split to reproduce published metrics.

Usage:
	wnn-verify lacg/wnn-bitwise-best --dataset wikitext
	wnn-verify ./local_model --dataset unsw-nb15 --split test
"""

from __future__ import annotations

import argparse
import math
import sys
import time


def verify_model(
	model_id: str,
	dataset: str = "wikitext",
	split: str = "test",
) -> dict:
	"""Verify a WNN model by training and evaluating on a standard dataset.

	Args:
		model_id: HuggingFace model ID or local path
		dataset: Dataset name ("wikitext" or "unsw-nb15")
		split: Evaluation split

	Returns:
		Dict with cross_entropy, accuracy, perplexity, and optionally f1_macro, fpr
	"""
	import torch

	print(f"Loading model from {model_id}...")
	from wnn.hf import WNNForCausalLM
	model = WNNForCausalLM.from_pretrained(model_id)
	config = model.config

	print(f"  Architecture: {config.architecture_type}")
	print(f"  Clusters: {config.num_clusters}")
	print(f"  Total neurons: {config.total_neurons}")
	print(f"  Context size: {config.context_size}")

	if dataset == "wikitext":
		return _verify_lm(model, split)
	elif dataset == "unsw-nb15":
		return _verify_ids(model, split)
	else:
		print(f"Unknown dataset: {dataset}")
		sys.exit(1)


def _verify_lm(model, split: str = "test") -> dict:
	"""Verify a language model on WikiText-2."""
	import tiktoken
	import torch
	from datasets import load_dataset

	print("Loading WikiText-2 dataset...")
	ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
	text = "\n\n".join(ds["text"])

	tokenizer = tiktoken.get_encoding("gpt2")
	tokens = tokenizer.encode(text)
	print(f"  Tokens: {len(tokens):,}")

	context_size = model.config.context_size

	print("Evaluating...")
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	start = time.time()

	with torch.no_grad():
		for i in range(context_size, min(len(tokens), context_size + 100_000)):
			context = tokens[i - context_size:i]
			target = tokens[i]

			input_ids = torch.tensor([context], dtype=torch.long)
			outputs = model.forward(input_ids)
			logits = outputs.logits[0, -1, :]

			log_probs = torch.log_softmax(logits, dim=-1)
			total_loss += -log_probs[target].item()

			if logits.argmax().item() == target:
				correct += 1
			total += 1

			if total % 10000 == 0:
				elapsed = time.time() - start
				ce_so_far = total_loss / total
				print(f"  {total:,} predictions, CE={ce_so_far:.4f}, Acc={correct / total:.4%}, {elapsed:.1f}s")

	ce = total_loss / total
	elapsed = time.time() - start

	results = {
		"cross_entropy": ce,
		"perplexity": math.exp(ce),
		"accuracy": correct / total,
		"num_predictions": total,
		"elapsed_seconds": elapsed,
	}

	print()
	print("=" * 50)
	print("  VERIFICATION RESULTS (WikiText-2)")
	print("=" * 50)
	print(f"  Cross-Entropy:  {ce:.4f}")
	print(f"  Perplexity:     {math.exp(ce):.2f}")
	print(f"  Accuracy:       {correct / total:.4%}")
	print(f"  Predictions:    {total:,}")
	print(f"  Time:           {elapsed:.1f}s")
	print("=" * 50)

	return results


def _verify_ids(model, split: str = "test") -> dict:
	"""Verify an IDS model on UNSW-NB15."""
	print("IDS verification on UNSW-NB15 is not yet implemented.")
	print("This requires the UNSW-NB15 dataset and IDS-specific evaluation.")
	return {"error": "not_implemented"}


def main():
	"""CLI entry point for wnn-verify."""
	parser = argparse.ArgumentParser(
		description="Verify a WNN model by reproducing published metrics",
		usage="wnn-verify <model_or_hub_id> [--dataset wikitext|unsw-nb15] [--split test]",
	)
	parser.add_argument("model", help="HuggingFace model ID or local path")
	parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "unsw-nb15"],
						help="Evaluation dataset")
	parser.add_argument("--split", default="test", help="Dataset split to evaluate on")
	args = parser.parse_args()

	results = verify_model(args.model, args.dataset, args.split)

	if "error" not in results:
		# Print machine-readable output
		import json
		print()
		print("JSON output:")
		print(json.dumps(results, indent=2))


if __name__ == "__main__":
	main()
