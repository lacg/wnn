#!/usr/bin/env python3
"""
Bitwise Architecture + Connectivity Optimization for BitwiseRAMLM.

7-phase pipeline optimizing one dimension at a time using ArchitectureGA/TSStrategy
with BitwiseEvaluator (16 clusters, Rust+Metal).

Phases:
  Phase 1: Grid search over neurons × bits (quick landscape scan → top-K uniform configs)
  Phase 2: GA neurons per cluster (bits fixed from Phase 1)
  Phase 3: TS neurons refinement (from Phase 2 best)
  Phase 4: GA bits per cluster (neurons fixed from Phase 3)
  Phase 5: TS bits refinement (from Phase 4 best)
  Phase 6: GA connections (bits+neurons fixed from Phase 5)
  Phase 7: TS connections refinement (from Phase 6 best)

Each phase seeds from the previous phase's best genome. The ArchitectureConfig
flags control what gets mutated. BitwiseEvaluator handles heterogeneous
per-cluster configs via Rust+Metal batch evaluation.

Usage:
	python run_bitwise_optimization.py --context 4 --rate 0.25
	python run_bitwise_optimization.py --ga-gens 50 --population 30 --ts-iters 50
	python run_bitwise_optimization.py --phase 3 --input experiments/bitwise_optimization_results.json
"""

import argparse
import json
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
# Phase 1: Grid search (quick landscape scan)
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
			print(f"\n[{idx}/{total}] n={neurons}, b={bits}")

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
			mean_bit_acc = eval_stats.get("mean_bit_accuracy", 0.0)

			print(f"  CE={ce:.4f}  PPL={ppl:.0f}  Acc={acc:.2%}  BitAcc={mean_bit_acc:.2%}  ({elapsed:.1f}s)")

			results.append({
				"neurons": neurons,
				"bits": bits,
				"cross_entropy": ce,
				"perplexity": ppl,
				"accuracy": acc,
				"mean_bit_accuracy": mean_bit_acc,
				"per_bit_accuracy": eval_stats.get("per_bit_accuracy", []),
				"elapsed_s": round(elapsed, 1),
			})

	results.sort(key=lambda r: r["cross_entropy"])

	print(f"\n{'─'*70}")
	print(f"Phase 1 Rankings (by CE):")
	for i, r in enumerate(results):
		marker = " ★" if i < top_k else ""
		print(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
			  f"CE={r['cross_entropy']:.4f}  Acc={r['accuracy']:.2%}{marker}")

	best = results[0]
	print(f"\nBest config: n={best['neurons']}, b={best['bits']}, CE={best['cross_entropy']:.4f}")

	return results


# ---------------------------------------------------------------------------
# Helper: Create evaluator and configs
# ---------------------------------------------------------------------------

def create_evaluator(train_tokens, test_tokens, vocab_size, context_size, rate,
					 default_neurons, default_bits):
	"""Create a BitwiseEvaluator with default config."""
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
	return BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=default_neurons,
		bits_per_neuron=default_bits,
		num_parts=3,
		memory_mode=2,  # QUAD_WEIGHTED
		neuron_sample_rate=rate,
	)


def create_seed_genome(num_clusters, bits, neurons, total_input_bits):
	"""Create a uniform seed genome with random connections."""
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
	return ClusterGenome.create_uniform(
		num_clusters=num_clusters,
		bits=bits,
		neurons=neurons,
		total_input_bits=total_input_bits,
	)


def run_ga_phase(
	evaluator, arch_config, ga_gens, population, patience,
	seed_genome=None, phase_name="GA", logger=print,
):
	"""Run a GA optimization phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import GAConfig

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

	strategy = ArchitectureGAStrategy(
		arch_config=arch_config,
		ga_config=ga_config,
		logger=logger,
		batch_evaluator=evaluator,
	)

	t0 = time.time()
	result = strategy.optimize(
		evaluate_fn=lambda g: 999.0,
		initial_genome=seed_genome,
		batch_evaluate_fn=lambda genomes: evaluator.evaluate_batch(genomes),
	)
	elapsed = time.time() - t0

	logger(f"\n  {phase_name} Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		logger(f"  Accuracy: {result.final_accuracy:.2%}")
	logger(f"  Improvement: {result.improvement_percent:.1f}%")
	logger(f"  Generations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	logger(f"  Stop reason: {result.stop_reason}")

	return result, elapsed


def run_ts_phase(
	evaluator, arch_config, ts_iters, neighbors, patience,
	seed_genome, seed_fitness, phase_name="TS", logger=print,
):
	"""Run a TS refinement phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureTSStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import TSConfig

	ts_config = TSConfig(
		iterations=ts_iters,
		neighbors_per_iter=neighbors,
		tabu_size=10,
		mutation_rate=0.1,
		patience=patience,
		check_interval=1,
	)

	strategy = ArchitectureTSStrategy(
		arch_config=arch_config,
		ts_config=ts_config,
		logger=logger,
		batch_evaluator=evaluator,
	)

	t0 = time.time()
	result = strategy.optimize(
		initial_genome=seed_genome,
		initial_fitness=seed_fitness,
		evaluate_fn=lambda g: 999.0,
		batch_evaluate_fn=lambda genomes: evaluator.evaluate_batch(genomes),
	)
	elapsed = time.time() - t0

	improvement_over_seed = (seed_fitness - result.final_fitness) / seed_fitness * 100 if seed_fitness > 0 else 0
	logger(f"\n  {phase_name} Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		logger(f"  Accuracy: {result.final_accuracy:.2%}")
	logger(f"  Improvement over seed: {improvement_over_seed:.1f}%")
	logger(f"  Iterations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	logger(f"  Stop reason: {result.stop_reason}")

	return result, elapsed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(phase_results):
	"""Print end-to-end optimization summary."""
	print(f"\n{'='*70}")
	print(f"Optimization Summary")
	print(f"{'='*70}")

	for name, data in phase_results.items():
		if data is None:
			continue
		ce = data.get("final_ce") or data.get("best_ce", "?")
		acc = data.get("final_accuracy")
		acc_str = f"  Acc={acc:.2%}" if acc is not None else ""
		print(f"  {name}: CE={ce:.4f}{acc_str}")

	# Overall improvement
	phase_names = list(phase_results.keys())
	first = next((phase_results[n] for n in phase_names if phase_results[n] is not None), None)
	last = next((phase_results[n] for n in reversed(phase_names) if phase_results[n] is not None), None)
	if first and last:
		baseline = first.get("best_ce") or first.get("final_ce", 0)
		final = last.get("final_ce", 0)
		if baseline > 0 and final > 0:
			improvement = (baseline - final) / baseline * 100
			print(f"\n  Overall: CE {baseline:.4f} → {final:.4f}  ({improvement:+.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="BitwiseRAMLM 7-Phase Architecture + Connectivity Optimization"
	)
	# General
	parser.add_argument("--context", type=int, default=4, help="Context window size (default: 4)")
	parser.add_argument("--rate", type=float, default=0.25, help="Neuron sample rate (default: 0.25)")
	parser.add_argument("--phase", type=str, default="all",
						help="Phase to run: 1-7 or 'all' (default: all)")
	parser.add_argument("--output", type=str, default="experiments/bitwise_optimization_results.json",
						help="Output JSON file")
	parser.add_argument("--input", type=str, default=None,
						help="Load previous results from JSON (for resuming)")
	# Phase 1
	parser.add_argument("--top-k", type=int, default=3,
						help="Top configs from Phase 1 grid search (default: 3)")
	# GA/TS parameters (shared across phases 2-7)
	parser.add_argument("--ga-gens", type=int, default=50,
						help="GA generations per phase (default: 50)")
	parser.add_argument("--population", type=int, default=30,
						help="GA population size (default: 30)")
	parser.add_argument("--ts-iters", type=int, default=50,
						help="TS iterations per phase (default: 50)")
	parser.add_argument("--neighbors", type=int, default=30,
						help="TS neighbors per iteration (default: 30)")
	parser.add_argument("--patience", type=int, default=2,
						help="Early stop patience (default: 2)")
	args = parser.parse_args()

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Import dependencies
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	train_tokens, test_tokens, vocab_size = load_wikitext2_tokens()
	num_clusters = bits_needed(vocab_size)  # 16 for GPT-2
	total_input_bits = args.context * bits_needed(vocab_size)

	all_results = {
		"config": {
			"context_size": args.context,
			"neuron_sample_rate": args.rate,
			"memory_mode": "QUAD_WEIGHTED",
			"vocab_size": vocab_size,
			"num_clusters": num_clusters,
			"total_input_bits": total_input_bits,
			"ga_gens": args.ga_gens,
			"population": args.population,
			"ts_iters": args.ts_iters,
			"neighbors": args.neighbors,
			"patience": args.patience,
		},
		"phase1_grid": None,
		"phase2_ga_neurons": None,
		"phase3_ts_neurons": None,
		"phase4_ga_bits": None,
		"phase5_ts_bits": None,
		"phase6_ga_connections": None,
		"phase7_ts_connections": None,
	}

	# Load previous results if available
	input_path = args.input or (str(output_path) if output_path.exists() else None)
	if input_path:
		try:
			with open(input_path) as f:
				prev = json.load(f)
			for key in all_results:
				if key != "config" and key in prev and prev[key] is not None:
					all_results[key] = prev[key]
			print(f"Loaded previous results from {input_path}")
		except Exception as e:
			print(f"Warning: Could not load {input_path}: {e}")

	def save():
		with open(args.output, "w") as f:
			json.dump(all_results, f, indent=2)

	def should_run(phase_num):
		return args.phase in ("all", str(phase_num))

	def log(msg):
		print(f"  {msg}")

	# Track best genome flowing through phases
	best_genome = None
	best_ce = None
	best_bits = 20   # defaults, overwritten by Phase 1
	best_neurons = 150

	# ── Phase 1: Grid search ─────────────────────────────────────────────
	if should_run(1):
		t0 = time.time()
		phase1_results = phase1_grid_search(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate, top_k=args.top_k,
		)
		best_p1 = phase1_results[0]  # sorted by CE
		best_bits = best_p1["bits"]
		best_neurons = best_p1["neurons"]
		best_ce = best_p1["cross_entropy"]

		# Create seed genome from Phase 1 best
		best_genome = create_seed_genome(num_clusters, best_bits, best_neurons, total_input_bits)

		all_results["phase1_grid"] = {
			"results": phase1_results,
			"best_bits": best_bits,
			"best_neurons": best_neurons,
			"best_ce": best_ce,
			"elapsed_s": round(time.time() - t0, 1),
		}
		save()
		print(f"\nPhase 1 saved to {args.output}")
	elif all_results.get("phase1_grid"):
		p1 = all_results["phase1_grid"]
		best_bits = p1["best_bits"]
		best_neurons = p1["best_neurons"]
		best_ce = p1["best_ce"]
		best_genome = create_seed_genome(num_clusters, best_bits, best_neurons, total_input_bits)
		print(f"Loaded Phase 1: bits={best_bits}, neurons={best_neurons}, CE={best_ce:.4f}")
	else:
		best_genome = create_seed_genome(num_clusters, best_bits, best_neurons, total_input_bits)
		print(f"Using default config: bits={best_bits}, neurons={best_neurons}")

	# Create evaluator (reused across phases 2-7)
	evaluator = create_evaluator(
		train_tokens, test_tokens, vocab_size,
		context_size=args.context, rate=args.rate,
		default_neurons=best_neurons, default_bits=best_bits,
	)

	# If no best_ce from Phase 1, evaluate the seed genome
	if best_ce is None and best_genome is not None:
		print("Evaluating seed genome...")
		results = evaluator.evaluate_batch([best_genome])
		best_ce = results[0][0]
		print(f"  Seed CE={best_ce:.4f}")

	# ── Phase 2: GA neurons (bits fixed) ─────────────────────────────────
	if should_run(2):
		print(f"\n{'='*70}")
		print(f"Phase 2: GA Neurons per Cluster (bits fixed at {best_bits})")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_neurons=True,
			optimize_bits=False,
			optimize_connections=False,
			default_bits=best_bits,
			default_neurons=best_neurons,
			min_neurons=10,
			max_neurons=300,
			min_bits=best_bits,
			max_bits=best_bits,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ga_phase(
			evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name="Phase 2: GA Neurons", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase2_ga_neurons"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
		}
		save()
	elif all_results.get("phase2_ga_neurons"):
		p2 = all_results["phase2_ga_neurons"]
		best_ce = p2["final_ce"]
		# Reconstruct genome from stats (uniform bits, varied neurons)
		stats = p2["genome_stats"]
		print(f"Loaded Phase 2: CE={best_ce:.4f}")

	# ── Phase 3: TS neurons refinement ───────────────────────────────────
	if should_run(3) and best_genome is not None:
		print(f"\n{'='*70}")
		print(f"Phase 3: TS Neurons Refinement")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_neurons=True,
			optimize_bits=False,
			optimize_connections=False,
			default_bits=best_bits,
			min_neurons=10,
			max_neurons=300,
			min_bits=best_bits,
			max_bits=best_bits,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ts_phase(
			evaluator, arch_config,
			ts_iters=args.ts_iters, neighbors=args.neighbors,
			patience=args.patience,
			seed_genome=best_genome, seed_fitness=best_ce,
			phase_name="Phase 3: TS Neurons", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase3_ts_neurons"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100 if result.initial_fitness > 0 else 0,
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
		}
		save()

	# ── Phase 4: GA bits (neurons fixed) ─────────────────────────────────
	if should_run(4) and best_genome is not None:
		print(f"\n{'='*70}")
		print(f"Phase 4: GA Bits per Cluster (neurons fixed)")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=True,
			optimize_neurons=False,
			optimize_connections=False,
			min_bits=10,
			max_bits=24,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ga_phase(
			evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name="Phase 4: GA Bits", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase4_ga_bits"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
		}
		save()

	# ── Phase 5: TS bits refinement ──────────────────────────────────────
	if should_run(5) and best_genome is not None:
		print(f"\n{'='*70}")
		print(f"Phase 5: TS Bits Refinement")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=True,
			optimize_neurons=False,
			optimize_connections=False,
			min_bits=10,
			max_bits=24,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ts_phase(
			evaluator, arch_config,
			ts_iters=args.ts_iters, neighbors=args.neighbors,
			patience=args.patience,
			seed_genome=best_genome, seed_fitness=best_ce,
			phase_name="Phase 5: TS Bits", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase5_ts_bits"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100 if result.initial_fitness > 0 else 0,
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
		}
		save()

	# ── Phase 6: GA connections (architecture fixed) ─────────────────────
	if should_run(6) and best_genome is not None:
		print(f"\n{'='*70}")
		print(f"Phase 6: GA Connections (architecture fixed)")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ga_phase(
			evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name="Phase 6: GA Connections", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase6_ga_connections"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
		}
		save()

	# ── Phase 7: TS connections refinement ───────────────────────────────
	if should_run(7) and best_genome is not None:
		print(f"\n{'='*70}")
		print(f"Phase 7: TS Connections Refinement")
		print(f"{'='*70}")

		arch_config = ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
			total_input_bits=total_input_bits,
		)

		result, elapsed = run_ts_phase(
			evaluator, arch_config,
			ts_iters=args.ts_iters, neighbors=args.neighbors,
			patience=args.patience,
			seed_genome=best_genome, seed_fitness=best_ce,
			phase_name="Phase 7: TS Connections", logger=log,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results["phase7_ts_connections"] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100 if result.initial_fitness > 0 else 0,
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
		}

		# Save best genome separately
		if best_genome is not None:
			genome_path = output_path.with_suffix(".best_genome.json")
			best_genome.save(str(genome_path), fitness=best_ce, accuracy=result.final_accuracy)
			print(f"\nBest genome saved to {genome_path}")

	# ── Summary ──────────────────────────────────────────────────────────
	phase_results = {
		"Phase 1 (grid)": all_results.get("phase1_grid"),
		"Phase 2 (GA neurons)": all_results.get("phase2_ga_neurons"),
		"Phase 3 (TS neurons)": all_results.get("phase3_ts_neurons"),
		"Phase 4 (GA bits)": all_results.get("phase4_ga_bits"),
		"Phase 5 (TS bits)": all_results.get("phase5_ts_bits"),
		"Phase 6 (GA connections)": all_results.get("phase6_ga_connections"),
		"Phase 7 (TS connections)": all_results.get("phase7_ts_connections"),
	}
	print_summary(phase_results)

	save()
	print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
