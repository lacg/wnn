#!/usr/bin/env python3
"""
WS1: Content-Dependent Routing Experiment.

Compares RoutedRAMClusterLayer vs baseline RAMLM with same total neurons
on WikiText-2 to measure whether routing improves performance.

Outputs:
- CE/PPL/accuracy comparison: routed vs baseline
- Per-route specialization analysis
- JSON results

Usage:
	python tests/test_routing.py
	python tests/test_routing.py --output experiments/routing_results.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from torch import tensor, long

from wnn.ram.core import AccelerationMode
from wnn.ram.core.models import RAMLM
from wnn.ram.core.routing import RoutedRAMClusterLayer
from wnn.ram.strategies.perplexity import PerplexityCalculator


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


def evaluate_routed_model(
	model: RAMLM,
	routed_layer: RoutedRAMClusterLayer,
	token_ids: list[int],
	batch_size: int = 5000,
) -> dict:
	"""Evaluate using the routed layer instead of model's own layer.

	Encodes sequences using the RAMLM's encoder, but forwards through
	the RoutedRAMClusterLayer instead.
	"""
	from torch.nn.functional import softmax
	from torch import arange

	calc = PerplexityCalculator(vocab_size=model.vocab_size)
	total_examples = len(token_ids) - model.context_size

	all_bits = model.encode_sequence(token_ids)
	targets = tensor(token_ids[model.context_size:], dtype=long)
	num_batches = (total_examples + batch_size - 1) // batch_size

	# Track per-route usage
	route_counts = Counter()

	for batch_idx in range(num_batches):
		start = batch_idx * batch_size
		end = min(start + batch_size, total_examples)

		batch_bits = all_bits[start:end]
		batch_targets = targets[start:end]

		# Forward through routed layer
		scores = routed_layer.forward(batch_bits)

		# Track routing decisions
		router_scores = routed_layer.router.forward(batch_bits)
		routes = router_scores.argmax(dim=-1).tolist()
		route_counts.update(routes)

		calc.add_from_scores_batch(scores, batch_targets, normalize=True)

	stats = calc.get_stats()
	stats["route_distribution"] = dict(route_counts.most_common())
	return stats


def main():
	parser = argparse.ArgumentParser(description="WS1: Routing experiment")
	parser.add_argument("--output", type=str, default="experiments/routing_results.json")
	parser.add_argument("--num-routes", type=int, default=8)
	parser.add_argument("--top-k", type=int, default=2)
	parser.add_argument("--neurons-per-route", type=int, default=3)
	parser.add_argument("--bits-per-neuron", type=int, default=10)
	parser.add_argument("--context-size", type=int, default=4)
	args = parser.parse_args()

	print("=" * 60)
	print("WS1: Content-Dependent Routing Experiment")
	print("=" * 60)

	# Load data
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")

	vocab_size = 50257
	context_size = args.context_size
	from wnn.ram.core.RAMClusterLayer import bits_needed
	total_input_bits = context_size * bits_needed(vocab_size)

	# --- Baseline: Standard RAMLM ---
	print("\n--- Baseline RAMLM ---")
	# Match total neurons: routed model has num_routes * neurons_per_route per cluster
	# So baseline should have num_routes * neurons_per_route / top_k neurons per cluster
	# to be comparable (since routed only uses top_k experts per input)
	baseline_neurons = args.num_routes * args.neurons_per_route
	print(f"Baseline: {baseline_neurons} neurons/cluster, {args.bits_per_neuron} bits/neuron")

	baseline = RAMLM(
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=baseline_neurons,
		bits_per_neuron=args.bits_per_neuron,
	)
	print(f"  Total neurons: {baseline.layer.total_neurons:,}")

	t0 = time.time()
	baseline.train_epoch_fast_auto(train_tokens, global_top_k=50, batch_size=5000)
	baseline_train_time = time.time() - t0
	print(f"  Training: {baseline_train_time:.1f}s")

	t0 = time.time()
	baseline_stats = baseline.evaluate_fast(val_tokens, batch_size=5000)
	baseline_eval_time = time.time() - t0
	print(f"  Evaluation: {baseline_eval_time:.1f}s")

	# --- Routed Model ---
	print(f"\n--- Routed Model ({args.num_routes} routes, top-{args.top_k}) ---")
	print(f"  {args.neurons_per_route} neurons/cluster/route, {args.bits_per_neuron} bits/neuron")

	routed_layer = RoutedRAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		num_routes=args.num_routes,
		neurons_per_cluster_per_route=args.neurons_per_route,
		bits_per_neuron=args.bits_per_neuron,
		top_k_routes=args.top_k,
	)
	print(f"  Total neurons: {routed_layer.total_neurons:,}")

	# Prepare training data (shared across routed model variants)
	all_bits = baseline.encode_sequence(train_tokens)
	targets = tensor(train_tokens[context_size:], dtype=long)

	# Prepare false clusters (50 negatives to keep memory manageable with 8 experts)
	counts = Counter(train_tokens)
	top_k_tokens = [t for t, _ in counts.most_common(50)]
	false_clusters_base = tensor(top_k_tokens, dtype=long)

	train_batch_size = 50000
	total_examples = all_bits.shape[0]
	num_batches = (total_examples + train_batch_size - 1) // train_batch_size

	def train_routed_batched(routed_layer, extra_training):
		"""Train routed model in batches to control peak memory."""
		for batch_idx in range(num_batches):
			start = batch_idx * train_batch_size
			end = min(start + train_batch_size, total_examples)
			batch_bits = all_bits[start:end]
			batch_targets = targets[start:end]
			batch_false = false_clusters_base.unsqueeze(0).expand(end - start, -1).contiguous()

			routed_layer.train_experts(
				batch_bits, batch_targets, batch_false,
				extra_training=extra_training,
			)
			if (batch_idx + 1) % 10 == 0:
				print(f"    Batch {batch_idx + 1}/{num_batches}")

	# --- Routed Model A: Specialized (no extra training) ---
	print("  Training specialized routed model (extra_training=False)...")
	t0 = time.time()
	train_routed_batched(routed_layer, extra_training=False)
	routed_train_time = time.time() - t0
	print(f"  Training: {routed_train_time:.1f}s")

	print("  Evaluating specialized routed model...")
	t0 = time.time()
	routed_stats = evaluate_routed_model(baseline, routed_layer, val_tokens)
	routed_eval_time = time.time() - t0
	print(f"  Evaluation: {routed_eval_time:.1f}s")

	# --- Routed Model B: Generalist (with extra training) ---
	print(f"\n--- Routed Model B: Generalist ({args.num_routes} routes, top-{args.top_k}, extra_training=True) ---")
	routed_gen = RoutedRAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		num_routes=args.num_routes,
		neurons_per_cluster_per_route=args.neurons_per_route,
		bits_per_neuron=args.bits_per_neuron,
		top_k_routes=args.top_k,
	)
	print("  Training generalist routed model (extra_training=True)...")
	t0 = time.time()
	train_routed_batched(routed_gen, extra_training=True)
	routed_gen_train_time = time.time() - t0
	print(f"  Training: {routed_gen_train_time:.1f}s")

	print("  Evaluating generalist routed model...")
	t0 = time.time()
	routed_gen_stats = evaluate_routed_model(baseline, routed_gen, val_tokens)
	routed_gen_eval_time = time.time() - t0
	print(f"  Evaluation: {routed_gen_eval_time:.1f}s")

	# --- Comparison ---
	print("\n" + "=" * 60)
	print("COMPARISON")
	print("=" * 60)
	print(f"{'Metric':<20} {'Baseline':>15} {'Specialized':>15} {'Generalist':>15}")
	print("-" * 65)

	b_ce = baseline_stats['cross_entropy']
	r_ce = routed_stats['cross_entropy']
	g_ce = routed_gen_stats['cross_entropy']
	print(f"{'Cross-Entropy':<20} {b_ce:>15.4f} {r_ce:>15.4f} {g_ce:>15.4f}")

	b_ppl = baseline_stats['perplexity']
	r_ppl = routed_stats['perplexity']
	g_ppl = routed_gen_stats['perplexity']
	print(f"{'Perplexity':<20} {b_ppl:>15.2f} {r_ppl:>15.2f} {g_ppl:>15.2f}")

	b_acc = baseline_stats['accuracy']
	r_acc = routed_stats['accuracy']
	g_acc = routed_gen_stats['accuracy']
	print(f"{'Accuracy':<20} {b_acc:>15.2%} {r_acc:>15.2%} {g_acc:>15.2%}")

	b_neurons = baseline.layer.total_neurons
	r_neurons = routed_layer.total_neurons
	print(f"{'Total Neurons':<20} {b_neurons:>15,} {r_neurons:>15,} {routed_gen.total_neurons:>15,}")

	# Per-route distribution (specialized)
	if "route_distribution" in routed_stats:
		print(f"\nRoute Distribution (Specialized):")
		rd = routed_stats["route_distribution"]
		total = sum(rd.values())
		for route, count in sorted(rd.items()):
			print(f"  Route {route}: {count:,} ({count/total:.1%})")

	# Per-route distribution (generalist)
	if "route_distribution" in routed_gen_stats:
		print(f"\nRoute Distribution (Generalist):")
		rd = routed_gen_stats["route_distribution"]
		total = sum(rd.values())
		for route, count in sorted(rd.items()):
			print(f"  Route {route}: {count:,} ({count/total:.1%})")

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output = {
		"experiment": "WS1_routing",
		"config": {
			"num_routes": args.num_routes,
			"top_k": args.top_k,
			"neurons_per_route": args.neurons_per_route,
			"bits_per_neuron": args.bits_per_neuron,
			"context_size": context_size,
		},
		"baseline": {
			"cross_entropy": round(b_ce, 4),
			"perplexity": round(b_ppl, 2),
			"accuracy": round(b_acc, 4),
			"total_neurons": b_neurons,
			"train_time_s": round(baseline_train_time, 1),
		},
		"routed_specialized": {
			"cross_entropy": round(r_ce, 4),
			"perplexity": round(r_ppl, 2),
			"accuracy": round(r_acc, 4),
			"total_neurons": r_neurons,
			"train_time_s": round(routed_train_time, 1),
			"route_distribution": routed_stats.get("route_distribution", {}),
			"extra_training": False,
		},
		"routed_generalist": {
			"cross_entropy": round(g_ce, 4),
			"perplexity": round(g_ppl, 2),
			"accuracy": round(g_acc, 4),
			"total_neurons": routed_gen.total_neurons,
			"train_time_s": round(routed_gen_train_time, 1),
			"route_distribution": routed_gen_stats.get("route_distribution", {}),
			"extra_training": True,
		},
		"delta_specialized": {
			"cross_entropy": round(r_ce - b_ce, 4),
			"perplexity": round(r_ppl - b_ppl, 2),
			"accuracy": round(r_acc - b_acc, 4),
		},
		"delta_generalist": {
			"cross_entropy": round(g_ce - b_ce, 4),
			"perplexity": round(g_ppl - b_ppl, 2),
			"accuracy": round(g_acc - b_acc, 4),
		},
	}

	with open(output_path, "w") as f:
		json.dump(output, f, indent=2)
	print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
	main()
