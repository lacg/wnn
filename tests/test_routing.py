#!/usr/bin/env python3
"""
WS1: Content-Dependent Routing Experiment (v2).

Compares three routing strategies for RoutedRAMClusterLayer vs baseline RAMLM:
  CONTEXT_HASH: Hash full context bits → route (balanced, context-agnostic)
  LAST_TOKEN: Last token identity → route (content-dependent, frequency-balanced)
  DISTRIBUTIONAL: K-means on last-token→target co-occurrence → route (semantic)

Each strategy is tested with extra_training=True (all experts see all data).
Routes are assigned based on OBSERVABLE input features, not targets.

Usage:
	python tests/test_routing.py
	python tests/test_routing.py --strategies LAST_TOKEN DISTRIBUTIONAL
	python tests/test_routing.py --output experiments/ws1_routing_v2.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "wnn"))

from torch import tensor, long

from wnn.ram.core.models import RAMLM
from wnn.ram.core.routing import RoutedRAMClusterLayer, RoutingStrategy
from wnn.ram.core.RAMClusterLayer import bits_needed
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
	"""Evaluate using the routed layer instead of model's own layer."""
	calc = PerplexityCalculator(vocab_size=model.vocab_size)
	total_examples = len(token_ids) - model.context_size

	all_bits = model.encode_sequence(token_ids)
	targets = tensor(token_ids[model.context_size:], dtype=long)
	num_batches = (total_examples + batch_size - 1) // batch_size

	route_counts = Counter()

	for batch_idx in range(num_batches):
		start = batch_idx * batch_size
		end = min(start + batch_size, total_examples)

		batch_bits = all_bits[start:end]
		batch_targets = targets[start:end]

		scores = routed_layer.forward(batch_bits)

		router_scores = routed_layer.router.forward(batch_bits)
		routes = router_scores.argmax(dim=-1).tolist()
		route_counts.update(routes)

		calc.add_from_scores_batch(scores, batch_targets, normalize=True)

		if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
			stats = calc.get_stats()
			pct = (batch_idx + 1) / num_batches * 100
			print(f"     {pct:5.1f}% - CE: {stats['cross_entropy']:.4f}, "
				  f"Acc: {stats['accuracy']:.2%}")

	stats = calc.get_stats()
	stats["route_distribution"] = dict(route_counts.most_common())
	return stats


def run_strategy(
	strategy: RoutingStrategy,
	all_bits,
	targets,
	false_clusters_base,
	baseline: RAMLM,
	val_tokens: list[int],
	args,
	deterministic: bool = False,
) -> dict:
	"""Train and evaluate a single routing strategy."""
	vocab_size = baseline.vocab_size
	context_size = baseline.context_size
	bpt = bits_needed(vocab_size)
	total_input_bits = context_size * bpt

	mode_str = "deterministic" if deterministic else "learned"
	print(f"\n{'=' * 60}")
	print(f"Strategy: {strategy.name} ({mode_str})")
	print(f"  {args.num_routes} routes, top-{args.top_k}, "
		  f"{args.neurons_per_route} neurons/route, {args.bits_per_neuron} bits")
	print(f"{'=' * 60}")

	routed = RoutedRAMClusterLayer(
		total_input_bits=total_input_bits,
		num_clusters=vocab_size,
		num_routes=args.num_routes,
		neurons_per_cluster_per_route=args.neurons_per_route,
		bits_per_neuron=args.bits_per_neuron,
		top_k_routes=args.top_k,
		bits_per_token=bpt,
		routing_strategy=strategy,
		use_deterministic_routing=deterministic,
	)
	print(f"  Total neurons: {routed.total_neurons:,}")

	# Train in batches
	train_batch_size = 50000
	total_examples = all_bits.shape[0]
	num_batches = (total_examples + train_batch_size - 1) // train_batch_size

	print(f"  Training (extra_training=True)...")
	t0 = time.time()
	for batch_idx in range(num_batches):
		start = batch_idx * train_batch_size
		end = min(start + train_batch_size, total_examples)

		stats = routed.train_experts(
			all_bits[start:end], targets[start:end], false_clusters_base,
			extra_training=True,
		)

		if batch_idx == 0:
			# Print route distribution from first batch
			rd = stats["router"]["route_distribution"]
			total = sum(rd.values())
			print(f"  Training route distribution (batch 0):")
			for r in sorted(rd.keys()):
				pct = rd[r] / total * 100
				print(f"    Route {r}: {rd[r]:,} ({pct:.1f}%)")

		if (batch_idx + 1) % 10 == 0:
			print(f"    Batch {batch_idx + 1}/{num_batches}")
	train_time = time.time() - t0
	print(f"  Training: {train_time:.1f}s")

	# Evaluate
	print(f"  Evaluating...")
	t0 = time.time()
	eval_stats = evaluate_routed_model(baseline, routed, val_tokens)
	eval_time = time.time() - t0
	print(f"  Evaluation: {eval_time:.1f}s")

	# Print eval route distribution
	rd = eval_stats["route_distribution"]
	total_eval = sum(rd.values())
	print(f"  Eval route distribution:")
	for r in sorted(rd.keys()):
		pct = rd[r] / total_eval * 100
		print(f"    Route {r}: {rd[r]:,} ({pct:.1f}%)")

	return {
		"strategy": strategy.name,
		"deterministic": deterministic,
		"cross_entropy": round(eval_stats["cross_entropy"], 4),
		"perplexity": round(eval_stats["perplexity"], 2),
		"accuracy": round(eval_stats["accuracy"], 4),
		"total_neurons": routed.total_neurons,
		"train_time_s": round(train_time, 1),
		"eval_time_s": round(eval_time, 1),
		"route_distribution": {str(k): v for k, v in rd.items()},
	}


def main():
	parser = argparse.ArgumentParser(description="WS1: Routing experiment v2")
	parser.add_argument("--output", type=str, default="experiments/ws1_routing_v2.json")
	parser.add_argument("--num-routes", type=int, default=8)
	parser.add_argument("--top-k", type=int, default=2)
	parser.add_argument("--neurons-per-route", type=int, default=3)
	parser.add_argument("--bits-per-neuron", type=int, default=10)
	parser.add_argument("--context-size", type=int, default=4)
	parser.add_argument(
		"--strategies", nargs="+",
		choices=["CONTEXT_HASH", "LAST_TOKEN", "DISTRIBUTIONAL"],
		default=["CONTEXT_HASH", "LAST_TOKEN", "DISTRIBUTIONAL"],
	)
	parser.add_argument(
		"--deterministic", action="store_true",
		help="Also test deterministic routing (skip learned router)",
	)
	args = parser.parse_args()

	strategy_map = {
		"CONTEXT_HASH": RoutingStrategy.CONTEXT_HASH,
		"LAST_TOKEN": RoutingStrategy.LAST_TOKEN,
		"DISTRIBUTIONAL": RoutingStrategy.DISTRIBUTIONAL,
	}
	strategies = [strategy_map[s] for s in args.strategies]

	print("=" * 60)
	print("WS1: Content-Dependent Routing Experiment (v2)")
	print(f"Strategies: {', '.join(s.name for s in strategies)}")
	print("=" * 60)

	# Load data
	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")

	vocab_size = 50257
	context_size = args.context_size
	bpt = bits_needed(vocab_size)
	total_input_bits = context_size * bpt

	# --- Baseline: Standard RAMLM ---
	print("\n--- Baseline RAMLM ---")
	baseline_neurons = args.num_routes * args.neurons_per_route
	print(f"  {baseline_neurons} neurons/cluster, {args.bits_per_neuron} bits/neuron")

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
	print(f"  CE: {baseline_stats['cross_entropy']:.4f}, "
		  f"PPL: {baseline_stats['perplexity']:.2f}, "
		  f"Acc: {baseline_stats['accuracy']:.2%}")
	print(f"  Evaluation: {baseline_eval_time:.1f}s")

	# Prepare shared training data
	all_bits = baseline.encode_sequence(train_tokens)
	targets = tensor(train_tokens[context_size:], dtype=long)
	counts = Counter(train_tokens)
	top_k_tokens = [t for t, _ in counts.most_common(50)]
	false_clusters_base = tensor(top_k_tokens, dtype=long)

	# --- Run each strategy ---
	strategy_results = {}
	for strategy in strategies:
		# Learned router
		result = run_strategy(
			strategy, all_bits, targets, false_clusters_base,
			baseline, val_tokens, args, deterministic=False,
		)
		strategy_results[strategy.name] = result

		# Deterministic routing (no learned router)
		if args.deterministic:
			det_result = run_strategy(
				strategy, all_bits, targets, false_clusters_base,
				baseline, val_tokens, args, deterministic=True,
			)
			strategy_results[f"{strategy.name}_DET"] = det_result

	# --- Comparison Table ---
	print("\n" + "=" * 80)
	print("COMPARISON")
	print("=" * 80)

	all_keys = list(strategy_results.keys())

	header = f"{'Metric':<20} {'Baseline':>12}"
	for key in all_keys:
		header += f" {key:>18}"
	print(header)
	print("-" * (34 + 19 * len(all_keys)))

	b_ce = baseline_stats['cross_entropy']
	b_ppl = baseline_stats['perplexity']
	b_acc = baseline_stats['accuracy']
	b_neurons = baseline.layer.total_neurons

	row = f"{'Cross-Entropy':<20} {b_ce:>12.4f}"
	for key in all_keys:
		r = strategy_results[key]
		delta = r['cross_entropy'] - b_ce
		row += f" {r['cross_entropy']:>12.4f}({delta:+.3f})"
	print(row)

	row = f"{'Perplexity':<20} {b_ppl:>12.2f}"
	for key in all_keys:
		r = strategy_results[key]
		row += f" {r['perplexity']:>18.2f}"
	print(row)

	row = f"{'Accuracy':<20} {b_acc:>12.2%}"
	for key in all_keys:
		r = strategy_results[key]
		row += f" {r['accuracy']:>18.2%}"
	print(row)

	row = f"{'Neurons':<20} {b_neurons:>12,}"
	for key in all_keys:
		r = strategy_results[key]
		row += f" {r['total_neurons']:>18,}"
	print(row)

	# Route balance analysis
	print("\nRoute Balance (eval):")
	for key in all_keys:
		rd = strategy_results[key]["route_distribution"]
		vals = list(rd.values())
		if vals:
			min_pct = min(vals) / sum(vals) * 100
			max_pct = max(vals) / sum(vals) * 100
			n_active = len(vals)
			print(f"  {key}: {n_active} active routes, "
				  f"range {min_pct:.1f}%-{max_pct:.1f}%")

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	output = {
		"experiment": "WS1_routing_v2",
		"config": {
			"num_routes": args.num_routes,
			"top_k": args.top_k,
			"neurons_per_route": args.neurons_per_route,
			"bits_per_neuron": args.bits_per_neuron,
			"context_size": context_size,
			"strategies": args.strategies,
		},
		"baseline": {
			"cross_entropy": round(b_ce, 4),
			"perplexity": round(b_ppl, 2),
			"accuracy": round(b_acc, 4),
			"total_neurons": b_neurons,
			"train_time_s": round(baseline_train_time, 1),
		},
		"strategies": strategy_results,
	}

	with open(output_path, "w") as f:
		json.dump(output, f, indent=2)
	print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
	main()
