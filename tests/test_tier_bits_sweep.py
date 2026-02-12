#!/usr/bin/env python3
"""
Tier Bits Sweep: Test different bit allocations across tiers.

Keeps neurons constant (15/10/5) and varies only bits per tier
to isolate the effect of bit allocation on performance.

Usage:
	python tests/test_tier_bits_sweep.py
	python tests/test_tier_bits_sweep.py --output experiments/tier_bits_sweep.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from wnn.ram.core.models import RAMLM


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


def build_cluster_order(train_tokens: list[int], vocab_size: int = 50257) -> list[int]:
	"""Build frequency-sorted cluster order."""
	counts = Counter(train_tokens)
	seen = set(t for t, _ in counts.most_common())
	unseen = [t for t in range(vocab_size) if t not in seen]
	return [t for t, _ in counts.most_common()] + unseen


def run_config(name: str, tiers: list, train_tokens: list[int], val_tokens: list[int],
			   cluster_order: list[int], vocab_size: int = 50257) -> dict:
	"""Train and evaluate a single tier configuration."""
	print(f"\n{'='*60}")
	print(f"Config: {name}")
	print(f"  Tiers: {tiers}")
	print(f"{'='*60}")

	model = RAMLM(
		vocab_size=vocab_size,
		context_size=4,
		tiers=tiers,
		cluster_order=cluster_order,
	)
	print(f"  Total neurons: {model.layer.total_neurons:,}")
	print(f"  Total memory cells: {model.layer.total_memory_cells:,}")

	# Train
	t0 = time.time()
	model.train_epoch_fast_auto(train_tokens, global_top_k=1000, batch_size=5000)
	train_time = time.time() - t0
	print(f"  Training: {train_time:.1f}s")

	# Evaluate
	t0 = time.time()
	stats = model.evaluate_fast(val_tokens, batch_size=5000)
	eval_time = time.time() - t0

	ce = stats["cross_entropy"]
	acc = stats["accuracy"]
	ppl = stats["perplexity"]
	print(f"  CE: {ce:.4f}  Accuracy: {acc:.2%}  PPL: {ppl:.0f}")
	print(f"  Evaluation: {eval_time:.1f}s")

	return {
		"name": name,
		"tiers": [(c, n, b) for c, n, b in tiers],
		"cross_entropy": round(ce, 4),
		"accuracy": round(acc, 4),
		"perplexity": round(ppl, 2),
		"total_neurons": model.layer.total_neurons,
		"total_memory_cells": model.layer.total_memory_cells,
		"train_time_s": round(train_time, 1),
	}


def main():
	parser = argparse.ArgumentParser(description="Tier bits sweep")
	parser.add_argument("--output", type=str, default="experiments/tier_bits_sweep.json")
	args = parser.parse_args()

	print("=" * 60)
	print("Tier Bits Sweep")
	print("Neurons fixed at 15/10/5, varying only bits per tier")
	print("=" * 60)

	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")
	cluster_order = build_cluster_order(train_tokens)

	# Configurations to test (neurons always 15/10/5)
	configs = [
		("22_12_8", [(100, 15, 22), (400, 10, 12), (None, 5, 8)]),
		("20_12_8", [(100, 15, 20), (400, 10, 12), (None, 5, 8)]),
		("16_12_8", [(100, 15, 16), (400, 10, 12), (None, 5, 8)]),
		("12_10_8", [(100, 15, 12), (400, 10, 10), (None, 5, 8)]),
		("8_8_8",   [(100, 15, 8),  (400, 10, 8),  (None, 5, 8)]),
	]

	results = []
	for name, tiers in configs:
		result = run_config(name, tiers, train_tokens, val_tokens, cluster_order)
		results.append(result)

	# Summary
	print(f"\n{'='*70}")
	print("SUMMARY")
	print(f"{'='*70}")
	print(f"{'Config':<15} {'T0 bits':>8} {'T1 bits':>8} {'T2 bits':>8} {'CE':>10} {'Acc':>10} {'Cells':>15}")
	print("-" * 70)
	for r in sorted(results, key=lambda x: x["cross_entropy"]):
		t = r["tiers"]
		print(f"{r['name']:<15} {t[0][2]:>8} {t[1][2]:>8} {t[2][2]:>8} {r['cross_entropy']:>10.4f} {r['accuracy']:>10.2%} {r['total_memory_cells']:>15,}")

	# Save
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump({"experiment": "tier_bits_sweep", "results": results}, f, indent=2)
	print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
	main()
