#!/usr/bin/env python3
"""
Bitwise Connectivity Optimization Pilot — GA/TS optimization for BitwiseRAMLM.

Three phases:
  Phase 1: Grid search over neurons × bits (random connections, QUAD_WEIGHTED)
  Phase 2: GA connectivity optimization on top-3 configs from Phase 1
  Phase 3: TS connectivity refinement starting from GA's best

Uses existing infrastructure:
  - BitwiseEvaluator (Rust+Metal batch evaluation)
  - ArchitectureGAStrategy / ArchitectureTSStrategy (GA/TS optimizers)
  - ClusterGenome (genome representation with per-neuron connections)

Usage:
	python run_bitwise_optimization.py --context 4 --rate 0.25
	python run_bitwise_optimization.py --ga-gens 100 --ts-iters 100 --patience 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext2_tokens(tokenizer_name="gpt2"):
	"""Load WikiText-2 train/test tokens using GPT-2 tokenizer."""
	try:
		from datasets import load_dataset
		from transformers import AutoTokenizer
	except ImportError:
		print("ERROR: datasets and transformers required. Install with:")
		print("  pip install datasets transformers")
		sys.exit(1)

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	train_text = "\n".join(dataset["train"]["text"])
	test_text = "\n".join(dataset["test"]["text"])

	train_tokens = tokenizer.encode(train_text)
	test_tokens = tokenizer.encode(test_text)

	vocab_size = tokenizer.vocab_size
	print(f"Loaded WikiText-2: {len(train_tokens):,} train, {len(test_tokens):,} test tokens, vocab={vocab_size}")

	return train_tokens, test_tokens, vocab_size


# ---------------------------------------------------------------------------
# Phase 1: Grid search
# ---------------------------------------------------------------------------

def phase1_grid_search(train_tokens, test_tokens, vocab_size, context_size, rate, top_k=3):
	"""Grid search over neurons × bits with random connections."""
	from wnn.ram.core.models import BitwiseRAMLM

	neurons_grid = [50, 100, 150, 200]
	bits_grid = [14, 16, 18, 20]

	results = []
	total = len(neurons_grid) * len(bits_grid)
	idx = 0

	print(f"\n{'='*70}")
	print(f"Phase 1: Grid Search ({total} configs)")
	print(f"  mode=QUAD_WEIGHTED, rate={rate}, context={context_size}")
	print(f"{'='*70}")

	for neurons in neurons_grid:
		for bits in bits_grid:
			idx += 1
			label = f"n={neurons}, b={bits}"
			print(f"\n[{idx}/{total}] {label}")

			model = BitwiseRAMLM(
				vocab_size=vocab_size,
				context_size=context_size,
				neurons_per_cluster=neurons,
				bits_per_neuron=bits,
				memory_mode=2,  # QUAD_WEIGHTED
				neuron_sample_rate=rate,
			)

			t0 = time.time()
			train_stats, eval_stats = model.train_and_eval_metal_split(
				train_tokens, test_tokens,
				verbose=False,
				per_bit=True,
			)
			elapsed = time.time() - t0

			ce = eval_stats["cross_entropy"]
			acc = eval_stats["accuracy"]
			ppl = eval_stats["perplexity"]
			per_bit = eval_stats.get("per_bit_accuracy", [])
			mean_bit_acc = eval_stats.get("mean_bit_accuracy", 0.0)

			print(f"  CE={ce:.4f}  PPL={ppl:.0f}  Acc={acc:.2%}  BitAcc={mean_bit_acc:.2%}  ({elapsed:.1f}s)")

			results.append({
				"neurons": neurons,
				"bits": bits,
				"cross_entropy": ce,
				"perplexity": ppl,
				"accuracy": acc,
				"mean_bit_accuracy": mean_bit_acc,
				"per_bit_accuracy": per_bit,
				"elapsed_s": round(elapsed, 1),
			})

	# Sort by CE (lower is better)
	results.sort(key=lambda r: r["cross_entropy"])

	print(f"\n{'─'*70}")
	print(f"Phase 1 Rankings (by CE):")
	for i, r in enumerate(results):
		marker = " ★" if i < top_k else ""
		print(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
			  f"CE={r['cross_entropy']:.4f}  Acc={r['accuracy']:.2%}{marker}")

	top_configs = [(r["neurons"], r["bits"]) for r in results[:top_k]]
	print(f"\nTop-{top_k} configs for Phase 2: {top_configs}")

	return results, top_configs


# ---------------------------------------------------------------------------
# Phase 2: GA connectivity optimization
# ---------------------------------------------------------------------------

def phase2_ga_optimization(
	train_tokens, test_tokens, vocab_size, context_size, rate,
	top_configs, ga_gens, population, patience,
):
	"""GA connectivity optimization for top configs from Phase 1."""
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
	from wnn.ram.strategies.connectivity.architecture_strategies import (
		ArchitectureConfig, ArchitectureGAStrategy,
	)
	from wnn.ram.strategies.connectivity.generic_strategies import GAConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	num_clusters = bits_needed(vocab_size)  # 16 for GPT-2
	total_input_bits = context_size * bits_needed(vocab_size)

	results = []

	for combo_idx, (neurons, bits) in enumerate(top_configs):
		print(f"\n{'='*70}")
		print(f"Phase 2 [{combo_idx+1}/{len(top_configs)}]: GA Connectivity — n={neurons}, b={bits}")
		print(f"  population={population}, generations={ga_gens}, patience={patience}")
		print(f"{'='*70}")

		evaluator = BitwiseEvaluator(
			train_tokens=train_tokens,
			eval_tokens=test_tokens,
			vocab_size=vocab_size,
			context_size=context_size,
			neurons_per_cluster=neurons,
			bits_per_neuron=bits,
			num_parts=3,
			memory_mode=2,  # QUAD_WEIGHTED
			neuron_sample_rate=rate,
		)

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
			default_bits=bits,
			default_neurons=neurons,
			total_input_bits=total_input_bits,
		)

		ga_config = GAConfig(
			population_size=population,
			generations=ga_gens,
			mutation_rate=0.1,
			crossover_rate=0.7,
			tournament_size=3,
			elitism_pct=0.1,
			patience=patience,
			check_interval=1,
		)

		# Create seed genome with random connections
		seed_genome = ClusterGenome(
			bits_per_cluster=[bits] * num_clusters,
			neurons_per_cluster=[neurons] * num_clusters,
		)
		seed_genome.initialize_connections(total_input_bits)

		def log(msg):
			print(f"  {msg}")

		strategy = ArchitectureGAStrategy(
			arch_config=arch_config,
			ga_config=ga_config,
			logger=log,
			batch_evaluator=evaluator,
		)

		t0 = time.time()
		result = strategy.optimize(
			evaluate_fn=lambda g: 999.0,  # unused when batch_evaluate_fn provided
			initial_genome=seed_genome,
			batch_evaluate_fn=lambda genomes: evaluator.evaluate_batch(genomes),
		)
		elapsed = time.time() - t0

		print(f"\n  GA Result: CE={result.final_fitness:.4f}, "
			  f"Acc={result.final_accuracy:.2%if result.final_accuracy else 'N/A'}")
		print(f"  Improvement: {result.improvement_percent:.1f}%")
		print(f"  Generations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
		print(f"  Stop reason: {result.stop_reason}")

		results.append({
			"neurons": neurons,
			"bits": bits,
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"initial_accuracy": result.initial_accuracy,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"history": result.history,
			"best_genome_connections": result.best_genome.connections,
		})

	return results


# ---------------------------------------------------------------------------
# Phase 3: TS connectivity refinement
# ---------------------------------------------------------------------------

def phase3_ts_refinement(
	train_tokens, test_tokens, vocab_size, context_size, rate,
	ga_results, ts_iters, neighbors, patience,
):
	"""TS connectivity refinement starting from GA's best genome."""
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
	from wnn.ram.strategies.connectivity.architecture_strategies import (
		ArchitectureConfig, ArchitectureTSStrategy,
	)
	from wnn.ram.strategies.connectivity.generic_strategies import TSConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	num_clusters = bits_needed(vocab_size)
	total_input_bits = context_size * bits_needed(vocab_size)

	results = []

	for combo_idx, ga_result in enumerate(ga_results):
		neurons = ga_result["neurons"]
		bits = ga_result["bits"]
		ga_ce = ga_result["final_ce"]

		print(f"\n{'='*70}")
		print(f"Phase 3 [{combo_idx+1}/{len(ga_results)}]: TS Refinement — n={neurons}, b={bits}")
		print(f"  GA baseline CE={ga_ce:.4f}")
		print(f"  neighbors={neighbors}, iterations={ts_iters}, patience={patience}")
		print(f"{'='*70}")

		evaluator = BitwiseEvaluator(
			train_tokens=train_tokens,
			eval_tokens=test_tokens,
			vocab_size=vocab_size,
			context_size=context_size,
			neurons_per_cluster=neurons,
			bits_per_neuron=bits,
			num_parts=3,
			memory_mode=2,  # QUAD_WEIGHTED
			neuron_sample_rate=rate,
		)

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
			default_bits=bits,
			default_neurons=neurons,
			total_input_bits=total_input_bits,
		)

		ts_config = TSConfig(
			iterations=ts_iters,
			neighbors_per_iter=neighbors,
			tabu_size=10,
			mutation_rate=0.1,
			patience=patience,
			check_interval=1,
		)

		# Reconstruct best genome from GA
		seed_genome = ClusterGenome(
			bits_per_cluster=[bits] * num_clusters,
			neurons_per_cluster=[neurons] * num_clusters,
			connections=ga_result["best_genome_connections"],
		)

		def log(msg):
			print(f"  {msg}")

		strategy = ArchitectureTSStrategy(
			arch_config=arch_config,
			ts_config=ts_config,
			logger=log,
			batch_evaluator=evaluator,
		)

		t0 = time.time()
		result = strategy.optimize(
			initial_genome=seed_genome,
			initial_fitness=ga_ce,
			evaluate_fn=lambda g: 999.0,
			batch_evaluate_fn=lambda genomes: evaluator.evaluate_batch(genomes),
		)
		elapsed = time.time() - t0

		print(f"\n  TS Result: CE={result.final_fitness:.4f}, "
			  f"Acc={result.final_accuracy:.2% if result.final_accuracy else 'N/A'}")
		print(f"  Improvement over GA: {((ga_ce - result.final_fitness) / ga_ce * 100):.1f}%")
		print(f"  Iterations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
		print(f"  Stop reason: {result.stop_reason}")

		results.append({
			"neurons": neurons,
			"bits": bits,
			"ga_ce": ga_ce,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_over_ga_pct": round((ga_ce - result.final_fitness) / ga_ce * 100, 2),
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"history": result.history,
			"best_genome_connections": result.best_genome.connections,
		})

	return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(phase1_results, ga_results, ts_results):
	"""Print comparison table: random vs GA vs TS connectivity."""
	print(f"\n{'='*70}")
	print(f"Summary: Random vs Optimized Connectivity")
	print(f"{'='*70}")
	print(f"{'Config':<16} {'Random CE':>10} {'GA CE':>10} {'TS CE':>10} {'Improvement':>12}")
	print(f"{'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

	for ga_r, ts_r in zip(ga_results, ts_results):
		n, b = ga_r["neurons"], ga_r["bits"]
		# Find random baseline from Phase 1
		random_ce = None
		for p1 in phase1_results:
			if p1["neurons"] == n and p1["bits"] == b:
				random_ce = p1["cross_entropy"]
				break
		random_ce = random_ce or ga_r["initial_ce"]
		ga_ce = ga_r["final_ce"]
		ts_ce = ts_r["final_ce"]
		improvement = (random_ce - ts_ce) / random_ce * 100

		print(f"n={n:3d}, b={b:2d}    {random_ce:10.4f} {ga_ce:10.4f} {ts_ce:10.4f} {improvement:+11.1f}%")

	# Best overall
	best_ts = min(ts_results, key=lambda r: r["final_ce"])
	print(f"\nBest overall: n={best_ts['neurons']}, b={best_ts['bits']}, "
		  f"CE={best_ts['final_ce']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(description="BitwiseRAMLM Connectivity Optimization Pilot")
	parser.add_argument("--context", type=int, default=4, help="Context window size (default: 4)")
	parser.add_argument("--rate", type=float, default=0.25, help="Neuron sample rate (default: 0.25)")
	parser.add_argument("--ga-gens", type=int, default=50, help="GA generations (default: 50)")
	parser.add_argument("--population", type=int, default=30, help="GA population size (default: 30)")
	parser.add_argument("--ts-iters", type=int, default=50, help="TS iterations (default: 50)")
	parser.add_argument("--neighbors", type=int, default=30, help="TS neighbors per iteration (default: 30)")
	parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (default: 2)")
	parser.add_argument("--top-k", type=int, default=3, help="Top configs from Phase 1 to optimize (default: 3)")
	parser.add_argument("--phase", type=str, default="all", help="Phase to run: 1, 2, 3, all (default: all)")
	parser.add_argument("--output", type=str, default="experiments/bitwise_optimization_results.json",
						help="Output JSON file")
	parser.add_argument("--phase1-input", type=str, default=None,
						help="Load Phase 1 results from JSON instead of running grid search")
	args = parser.parse_args()

	# Ensure output directory exists
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Load data
	train_tokens, test_tokens, vocab_size = load_wikitext2_tokens()

	all_results = {
		"config": {
			"context_size": args.context,
			"neuron_sample_rate": args.rate,
			"memory_mode": "QUAD_WEIGHTED",
			"ga_generations": args.ga_gens,
			"ga_population": args.population,
			"ts_iterations": args.ts_iters,
			"ts_neighbors": args.neighbors,
			"patience": args.patience,
			"vocab_size": vocab_size,
		},
		"phase1": None,
		"phase2_ga": None,
		"phase3_ts": None,
	}

	run_phase1 = args.phase in ("all", "1")
	run_phase2 = args.phase in ("all", "2")
	run_phase3 = args.phase in ("all", "3")

	# Phase 1
	if run_phase1:
		t0 = time.time()
		phase1_results, top_configs = phase1_grid_search(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context,
			rate=args.rate,
			top_k=args.top_k,
		)
		all_results["phase1"] = {
			"results": phase1_results,
			"top_configs": [{"neurons": n, "bits": b} for n, b in top_configs],
			"elapsed_s": round(time.time() - t0, 1),
		}
		# Save intermediate
		with open(args.output, "w") as f:
			json.dump(all_results, f, indent=2)
		print(f"\nPhase 1 saved to {args.output}")
	elif args.phase1_input:
		with open(args.phase1_input) as f:
			loaded = json.load(f)
		phase1_results = loaded["phase1"]["results"]
		top_configs = [(c["neurons"], c["bits"]) for c in loaded["phase1"]["top_configs"]]
		all_results["phase1"] = loaded["phase1"]
		print(f"Loaded Phase 1 from {args.phase1_input}: top={top_configs}")
	else:
		# Default top configs from known Phase 4 results
		top_configs = [(150, 20), (100, 20), (150, 18)]
		phase1_results = []
		print(f"Using default top configs: {top_configs}")

	# Phase 2
	if run_phase2:
		t0 = time.time()
		ga_results = phase2_ga_optimization(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context,
			rate=args.rate,
			top_configs=top_configs,
			ga_gens=args.ga_gens,
			population=args.population,
			patience=args.patience,
		)
		all_results["phase2_ga"] = {
			"results": [{k: v for k, v in r.items() if k != "best_genome_connections"} for r in ga_results],
			"elapsed_s": round(time.time() - t0, 1),
		}
		# Save intermediate (without huge connection lists in JSON)
		with open(args.output, "w") as f:
			json.dump(all_results, f, indent=2)
		print(f"\nPhase 2 saved to {args.output}")
	else:
		ga_results = None

	# Phase 3
	if run_phase3 and ga_results is not None:
		t0 = time.time()
		ts_results = phase3_ts_refinement(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context,
			rate=args.rate,
			ga_results=ga_results,
			ts_iters=args.ts_iters,
			neighbors=args.neighbors,
			patience=args.patience,
		)
		all_results["phase3_ts"] = {
			"results": [{k: v for k, v in r.items() if k != "best_genome_connections"} for r in ts_results],
			"elapsed_s": round(time.time() - t0, 1),
		}
		# Save best genome separately (connections can be huge)
		best_ts = min(ts_results, key=lambda r: r["final_ce"])
		genome_path = output_path.with_suffix(".best_genome.json")
		with open(genome_path, "w") as f:
			json.dump({
				"neurons": best_ts["neurons"],
				"bits": best_ts["bits"],
				"final_ce": best_ts["final_ce"],
				"final_accuracy": best_ts["final_accuracy"],
				"connections": best_ts["best_genome_connections"],
			}, f)
		print(f"\nBest genome saved to {genome_path}")

		# Print summary
		print_summary(phase1_results, ga_results, ts_results)

	# Save final results
	with open(args.output, "w") as f:
		json.dump(all_results, f, indent=2)
	print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
