#!/usr/bin/env python3
"""
Full Tier Bits Grid Sweep.

Tests all combinations of tier bits from 8 to 16 (step 2) for each of 3 tiers.
Each config is run 5 times with different random seeds and results are averaged.

Grid: [8, 10, 12, 14, 16] x [8, 10, 12, 14, 16] x [8, 10, 12, 14, 16] = 125 configs
Runs: 5 per config = 625 total runs
Neurons: fixed at 15/10/5 per tier

Results saved incrementally after each config completes (safe to interrupt).
Configs ordered from center (12,10,8) outward so best results come first.

Usage:
	python tests/test_tier_bits_grid.py
	python tests/test_tier_bits_grid.py --output experiments/tier_bits_grid.json
	python tests/test_tier_bits_grid.py --runs 3  # fewer runs for faster sweep
"""

import argparse
import json
import statistics
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


def force_sparse_mode(model):
	"""Force Rust sparse backend even for small-bit configs.

	Without this, configs where all tiers have <=10 bits use the
	Python dense path which is ~7x slower for evaluation.
	"""
	layer = model.layer
	if layer._use_sparse and layer._sparse_memory is not None:
		return

	import ram_accelerator
	rust_tier_configs = [
		(tc.cluster_end, tc.neurons_per_cluster, tc.bits_per_neuron)
		for tc in layer.tier_configs
	]
	layer._sparse_memory = ram_accelerator.TieredSparseMemory(
		rust_tier_configs, layer.num_clusters
	)
	layer._use_sparse = True


def run_single(
	tiers: list,
	train_tokens: list[int],
	val_tokens: list[int],
	cluster_order: list[int],
	rng_seed: int,
	vocab_size: int = 50257,
) -> dict:
	"""Train and evaluate a single config with a given seed."""
	model = RAMLM(
		vocab_size=vocab_size,
		context_size=4,
		tiers=tiers,
		cluster_order=cluster_order,
		rng=rng_seed,
	)
	force_sparse_mode(model)

	model.train_epoch_fast_auto(
		train_tokens, global_top_k=1000, batch_size=5000, verbose=False,
	)
	stats = model.evaluate_fast(val_tokens, batch_size=5000, verbose=False)

	return {
		"cross_entropy": stats["cross_entropy"],
		"accuracy": stats["accuracy"],
		"perplexity": stats["perplexity"],
	}


def load_existing_results(output_path: Path) -> dict:
	"""Load previously completed results for resume support."""
	if output_path.exists():
		with open(output_path) as f:
			data = json.load(f)
		completed = {r["name"]: r for r in data.get("results", [])}
		print(f"Resuming: {len(completed)} configs already completed")
		return completed
	return {}


def main():
	parser = argparse.ArgumentParser(description="Full tier bits grid sweep")
	parser.add_argument("--output", type=str, default="experiments/tier_bits_grid.json")
	parser.add_argument("--runs", type=int, default=5, help="Runs per config for averaging")
	args = parser.parse_args()

	bit_values = list(range(8, 17))  # 8, 9, 10, 11, 12, 13, 14, 15, 16
	total_configs = len(bit_values) ** 3

	print("=" * 70)
	print("Full Tier Bits Grid Sweep")
	print(f"Grid: {bit_values} ^ 3 = {total_configs} configs x {args.runs} runs = {total_configs * args.runs} total")
	print(f"Neurons: 15/10/5 (fixed)")
	print(f"Saving to: {args.output}")
	print("=" * 70)

	train_tokens = load_wikitext2_tokens("train")
	val_tokens = load_wikitext2_tokens("validation")
	cluster_order = build_cluster_order(train_tokens)

	# Generate all configs
	configs = [
		(t0, t1, t2)
		for t0 in bit_values
		for t1 in bit_values
		for t2 in bit_values
	]

	# Sort by Manhattan distance from known optimum (12, 10, 8)
	# so most promising configs run first
	center = (12, 10, 8)
	configs.sort(key=lambda c: abs(c[0] - center[0]) + abs(c[1] - center[1]) + abs(c[2] - center[2]))

	# Resume support: skip already-completed configs
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	completed = load_existing_results(output_path)
	all_results = list(completed.values())

	sweep_start = time.time()

	for i, (t0, t1, t2) in enumerate(configs):
		name = f"{t0}_{t1}_{t2}"

		# Skip if already completed
		if name in completed:
			continue

		tiers = [(100, 15, t0), (400, 10, t1), (None, 5, t2)]
		remaining = total_configs - len(all_results)
		elapsed = time.time() - sweep_start
		if len(all_results) > len(completed):
			done_this_session = len(all_results) - len(completed)
			eta_per_config = elapsed / done_this_session
			eta_remaining = eta_per_config * remaining
			eta_str = f" ETA: {eta_remaining/3600:.1f}h"
		else:
			eta_str = ""

		print(f"\n[{len(all_results)+1}/{total_configs}] Config {name} ({args.runs} runs){eta_str}")

		run_results = []
		for r in range(args.runs):
			t_start = time.time()
			try:
				result = run_single(tiers, train_tokens, val_tokens, cluster_order, rng_seed=r)
				run_time = time.time() - t_start
				run_results.append(result)
				print(f"  Run {r+1}/{args.runs}: CE={result['cross_entropy']:.4f} "
					  f"Acc={result['accuracy']:.2%} ({run_time:.0f}s)")
			except Exception as e:
				print(f"  Run {r+1}/{args.runs}: FAILED - {e}")
				continue

		if not run_results:
			print(f"  ALL RUNS FAILED, skipping")
			continue

		# Compute statistics
		avg_ce = statistics.mean(r["cross_entropy"] for r in run_results)
		avg_acc = statistics.mean(r["accuracy"] for r in run_results)
		avg_ppl = statistics.mean(r["perplexity"] for r in run_results)
		std_ce = statistics.stdev(r["cross_entropy"] for r in run_results) if len(run_results) > 1 else 0.0
		std_acc = statistics.stdev(r["accuracy"] for r in run_results) if len(run_results) > 1 else 0.0

		config_result = {
			"name": name,
			"t0_bits": t0,
			"t1_bits": t1,
			"t2_bits": t2,
			"avg_ce": round(avg_ce, 4),
			"avg_accuracy": round(avg_acc, 4),
			"avg_ppl": round(avg_ppl, 2),
			"std_ce": round(std_ce, 4),
			"std_accuracy": round(std_acc, 4),
			"num_runs": len(run_results),
			"runs": run_results,
		}
		all_results.append(config_result)

		print(f"  AVG: CE={avg_ce:.4f} +/-{std_ce:.4f}  Acc={avg_acc:.2%} +/-{std_acc:.2%}")

		# Save after each config (resume-safe)
		with open(output_path, "w") as f:
			json.dump({
				"experiment": "tier_bits_grid",
				"bit_values": bit_values,
				"num_runs_per_config": args.runs,
				"configs_completed": len(all_results),
				"configs_total": total_configs,
				"results": all_results,
			}, f, indent=2)

	# Final summary
	total_time = time.time() - sweep_start
	print(f"\n{'='*80}")
	print(f"SWEEP COMPLETE — {len(all_results)} configs in {total_time/3600:.1f}h")
	print(f"{'='*80}")
	print(f"{'Config':<12} {'T0':>4} {'T1':>4} {'T2':>4} {'Avg CE':>10} {'+-':>7} {'Avg Acc':>10} {'+-':>8}")
	print("-" * 65)
	for r in sorted(all_results, key=lambda x: x["avg_ce"])[:25]:
		print(f"{r['name']:<12} {r['t0_bits']:>4} {r['t1_bits']:>4} {r['t2_bits']:>4} "
			  f"{r['avg_ce']:>10.4f} {r['std_ce']:>7.4f} {r['avg_accuracy']:>10.2%} {r['std_accuracy']:>8.4f}")

	print(f"\nFull results: {output_path}")


if __name__ == "__main__":
	main()
