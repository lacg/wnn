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
import math
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext2_tokens(tokenizer_name="gpt2"):
	"""Load all 3 WikiText-2 splits using GPT-2 tokenizer.

	Returns: (train_tokens, test_tokens, validation_tokens, vocab_size)
	- train: dataset["train"] → ~2.4M tokens (used for training during GA/TS)
	- test: dataset["test"] → ~288K tokens (used for fitness scoring during GA/TS)
	- validation: dataset["validation"] → ~251K tokens (held out for full eval only)
	"""
	try:
		from datasets import load_dataset
		from transformers import AutoTokenizer
	except ImportError:
		print("ERROR: datasets and transformers required. Install with:")
		print("  pip install datasets transformers")
		sys.exit(1)

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	tokenizer.model_max_length = int(1e12)  # We only use BPE encoding, not the model
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	train_text = "\n".join(dataset["train"]["text"])
	train_tokens = tokenizer.encode(train_text)

	test_text = "\n".join(dataset["test"]["text"])
	test_tokens = tokenizer.encode(test_text)

	validation_text = "\n".join(dataset["validation"]["text"])
	validation_tokens = tokenizer.encode(validation_text)

	vocab_size = tokenizer.vocab_size
	print(f"Loaded WikiText-2: {len(train_tokens):,} train, "
		  f"{len(test_tokens):,} test, {len(validation_tokens):,} validation, vocab={vocab_size}")

	return train_tokens, test_tokens, validation_tokens, vocab_size


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

def create_evaluators(train_tokens, test_tokens, validation_tokens, vocab_size,
					  context_size, rate, default_neurons, default_bits,
					  num_train_parts=36, num_eval_parts=6):
	"""Create two BitwiseEvaluators: one for optimization, one for full validation.

	opt_evaluator: trains on train_tokens (rotated ~67K), scores on test_tokens (rotated ~50K)
	full_evaluator: trains on full train_tokens, evaluates on full validation_tokens
	"""
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

	opt_evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=default_neurons,
		bits_per_neuron=default_bits,
		num_parts=num_train_parts,
		num_eval_parts=num_eval_parts,
		memory_mode=2,  # QUAD_WEIGHTED
		neuron_sample_rate=rate,
	)

	full_evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=validation_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=default_neurons,
		bits_per_neuron=default_bits,
		num_parts=1,
		num_eval_parts=1,
		memory_mode=2,  # QUAD_WEIGHTED
		neuron_sample_rate=rate,
	)

	return opt_evaluator, full_evaluator


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
	seed_genome=None, phase_name="GA", logger=print, check_interval=10,
	initial_population=None,
):
	"""Run a GA optimization phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import GAConfig
	from wnn.ram.fitness import FitnessCalculatorType

	ga_config = GAConfig(
		population_size=population,
		generations=ga_gens,
		mutation_rate=0.1,
		crossover_rate=0.7,
		tournament_size=3,
		elitism_pct=0.1,
		patience=patience,
		check_interval=check_interval,
		# Start at 3% accuracy, +0.01%/gen continuously
		min_accuracy=0.03,
		threshold_delta=ga_gens * 0.0001,
		threshold_reference=ga_gens,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=1.0,
		fitness_weight_acc=1.0,
	)

	strategy = ArchitectureGAStrategy(
		arch_config=arch_config,
		ga_config=ga_config,
		logger=logger,
		batch_evaluator=evaluator,
	)

	t0 = time.time()
	optimize_kwargs = dict(
		evaluate_fn=lambda g: 999.0,
		initial_genome=seed_genome,
		batch_evaluate_fn=lambda genomes, **kw: evaluator.evaluate_batch(genomes, **kw),
	)
	if initial_population is not None:
		optimize_kwargs["initial_population"] = initial_population
	result = strategy.optimize(**optimize_kwargs)
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
	seed_genome, seed_fitness, phase_name="TS", logger=print, check_interval=10,
):
	"""Run a TS refinement phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureTSStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import TSConfig
	from wnn.ram.fitness import FitnessCalculatorType

	ts_config = TSConfig(
		iterations=ts_iters,
		neighbors_per_iter=neighbors,
		tabu_size=10,
		mutation_rate=0.1,
		patience=patience,
		check_interval=check_interval,
		# Start at 3% accuracy, +0.01%/iter continuously
		min_accuracy=0.03,
		threshold_delta=ts_iters * 0.0001,
		threshold_reference=ts_iters,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=1.0,
		fitness_weight_acc=1.0,
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
		batch_evaluate_fn=lambda genomes, **kw: evaluator.evaluate_batch(genomes, **kw),
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
# Top-K full-eval summary (mirrors wnn-exp PhaseComparisonTable)
# ---------------------------------------------------------------------------

def harmonic_weighted(ce_rank, acc_rank, w_ce=1.0, w_acc=1.0):
	"""Weighted harmonic mean of ranks (lower = better)."""
	if ce_rank == 0 or acc_rank == 0:
		return float("inf")
	return (w_ce + w_acc) / (w_ce / ce_rank + w_acc / acc_rank)


def evaluate_phase_top_k(full_evaluator, result, phase_name):
	"""Full-eval 1-3 representative genomes from a phase on validation data.

	Pre-filters the population by fitness rank (using optimization metrics from result),
	then full-evals only best CE, best Acc (if different), best Fitness (if different).
	This avoids expensive full evaluation on the entire population.

	Args:
		full_evaluator: BitwiseEvaluator configured with validation data
		result: OptimizationResult with final_population and metrics
		phase_name: Display name for the phase

	Returns dict with keys: best_ce, best_acc, best_fitness.
	"""
	pop = result.final_population if result.final_population else []
	if not pop:
		pop = [result.best_genome]

	# Use optimization-time metrics (from the result) to rank population
	# result.population_metrics is list of (genome, ce, acc) if available
	pop_metrics = getattr(result, "population_metrics", None)
	if pop_metrics and len(pop_metrics) == len(pop):
		# Pre-rank by optimization metrics to select candidates
		by_ce = sorted(range(len(pop)), key=lambda i: pop_metrics[i][0])
		by_acc = sorted(range(len(pop)), key=lambda i: -pop_metrics[i][1])

		# Compute fitness ranks
		n = len(pop)
		ce_ranks = {i: rank + 1 for rank, i in enumerate(by_ce)}
		acc_ranks = {i: rank + 1 for rank, i in enumerate(by_acc)}
		by_fitness = sorted(
			range(n),
			key=lambda i: harmonic_weighted(ce_ranks[i], acc_ranks[i]),
		)

		best_ce_idx = by_ce[0]
		best_acc_idx = by_acc[0]
		best_fit_idx = by_fitness[0]
	else:
		# No optimization metrics — just use best_genome for all three
		best_ce_idx = 0
		best_acc_idx = 0
		best_fit_idx = 0

	# Collect unique genomes to full-eval (1-3 max)
	candidates = {}  # idx → genome
	candidates[best_ce_idx] = pop[best_ce_idx]
	if best_acc_idx != best_ce_idx:
		candidates[best_acc_idx] = pop[best_acc_idx]
	if best_fit_idx not in candidates:
		candidates[best_fit_idx] = pop[best_fit_idx]

	# Full evaluation on validation data (1-3 genomes)
	genome_list = list(candidates.values())
	idx_list = list(candidates.keys())
	print(f"  Full-eval {len(genome_list)} genome(s) on validation...")
	full_results = full_evaluator.evaluate_batch_full(genome_list)

	# Map results back
	eval_map = {}
	for i, idx in enumerate(idx_list):
		eval_map[idx] = full_results[i]

	def make_entry(idx):
		ce, acc = eval_map[idx]
		return {"ce": ce, "acc": acc, "ppl": math.exp(ce)}

	return {
		"phase_name": phase_name,
		"n_full_evaluated": len(genome_list),
		"n_population": len(pop),
		"best_ce": make_entry(best_ce_idx),
		"best_acc": make_entry(best_acc_idx),
		"best_fitness": make_entry(best_fit_idx),
	}


def print_phase_comparison(phase_metrics_list):
	"""Print cumulative phase comparison table (all on full validation)."""
	W = 82
	print(f"\n{'='*W}")
	print(f"  Phase Comparison (Full Validation)")
	print(f"{'='*W}")
	print(f"  {'Phase':<28} {'Metric':<14} {'CE':>10} {'PPL':>12} {'Accuracy':>10}")
	print(f"  {'-'*(W-2)}")

	baseline_ce = phase_metrics_list[0]["best_ce"]["ce"] if phase_metrics_list else None

	for i, pm in enumerate(phase_metrics_list):
		name = pm["phase_name"]

		# Row 1: best CE
		m = pm["best_ce"]
		print(f"  {name:<28} {'best CE':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		# Row 2: best Acc (only if different from best CE)
		m = pm["best_acc"]
		if m != pm["best_ce"]:
			print(f"  {'':<28} {'best Acc':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		# Row 3: best Fitness (only if different from both)
		m = pm["best_fitness"]
		if m != pm["best_ce"] and m != pm["best_acc"]:
			print(f"  {'':<28} {'best Fitness':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		if i < len(phase_metrics_list) - 1:
			print(f"  {'-'*(W-2)}")

	print(f"  {'='*(W-2)}")

	# Overall improvement
	if len(phase_metrics_list) >= 2 and baseline_ce:
		final = phase_metrics_list[-1]
		best_ce_impr = (1 - final["best_ce"]["ce"] / baseline_ce) * 100
		print(f"  Improvement vs baseline: best CE {best_ce_impr:+.2f}%")


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
	parser.add_argument("--population", type=int, default=50,
						help="GA population size (default: 50)")
	parser.add_argument("--ts-iters", type=int, default=50,
						help="TS iterations per phase (default: 50)")
	parser.add_argument("--neighbors", type=int, default=30,
						help="TS neighbors per iteration (default: 30)")
	parser.add_argument("--patience", type=int, default=2,
						help="Early stop patience (default: 2)")
	parser.add_argument("--check-interval", type=int, default=10,
						help="Patience check every N generations/iterations (default: 10)")
	# Data rotation
	parser.add_argument("--train-parts", type=int, default=36,
						help="Number of train subsets for rotation (default: 36)")
	parser.add_argument("--eval-parts", type=int, default=6,
						help="Number of test subsets for rotation (default: 6)")
	args = parser.parse_args()

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Import dependencies
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	# Load all 3 WikiText-2 splits
	train_tokens, test_tokens, validation_tokens, vocab_size = load_wikitext2_tokens()
	num_clusters = bits_needed(vocab_size)  # 16 for GPT-2
	total_input_bits = args.context * bits_needed(vocab_size)

	all_results = {
		"config": {
			"context_size": args.context,
			"neuron_sample_rate": args.rate,
			"memory_mode": "QUAD_WEIGHTED",
			"fitness": "HARMONIC_RANK",
			"vocab_size": vocab_size,
			"num_clusters": num_clusters,
			"total_input_bits": total_input_bits,
			"train_tokens": len(train_tokens),
			"test_tokens": len(test_tokens),
			"validation_tokens": len(validation_tokens),
			"train_parts": args.train_parts,
			"eval_parts": args.eval_parts,
			"ga_gens": args.ga_gens,
			"population": args.population,
			"ts_iters": args.ts_iters,
			"neighbors": args.neighbors,
			"patience": args.patience,
			"check_interval": args.check_interval,
		},
		"phase1_grid": None,
		"init_population": None,
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
	initial_population = None  # 50 genomes seeded from Phase 1

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

		# ── Create 50-genome initial population ──
		# 3 genomes per grid config (different random connections) = 48
		# 2 extra from best 2 configs = 50 total
		print(f"\nSeeding population from {len(phase1_results)} grid configs...")
		pop_genomes = []
		for config in phase1_results:
			for _ in range(3):
				g = create_seed_genome(
					num_clusters, config["bits"], config["neurons"], total_input_bits
				)
				pop_genomes.append(g)

		# Add 2 extra from top-2 configs
		for config in phase1_results[:2]:
			g = create_seed_genome(
				num_clusters, config["bits"], config["neurons"], total_input_bits
			)
			pop_genomes.append(g)

		initial_population = pop_genomes[:args.population]
		best_genome = initial_population[0]  # Best config, first random seed

		print(f"  Created {len(initial_population)} genomes for initial population")

		all_results["phase1_grid"] = {
			"results": phase1_results,
			"best_bits": best_bits,
			"best_neurons": best_neurons,
			"best_ce": best_ce,
			"init_population_size": len(initial_population),
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

	# Create two evaluators:
	# opt_evaluator: rotated train+test for GA/TS fitness
	# full_evaluator: full train + full validation for phase comparison
	opt_evaluator, full_evaluator = create_evaluators(
		train_tokens, test_tokens, validation_tokens, vocab_size,
		context_size=args.context, rate=args.rate,
		default_neurons=best_neurons, default_bits=best_bits,
		num_train_parts=args.train_parts, num_eval_parts=args.eval_parts,
	)

	# Full-eval initial population baseline on validation
	phase_metrics_list = []
	print(f"\nEvaluating baseline on full validation...")
	baseline_ce, baseline_acc = full_evaluator.evaluate_single_full(best_genome)
	print(f"  Baseline CE={baseline_ce:.4f}, Acc={baseline_acc:.2%}, PPL={math.exp(baseline_ce):.0f}")
	baseline_entry = {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)}
	phase_metrics_list.append({
		"phase_name": "Init Population",
		"n_full_evaluated": 1,
		"n_population": len(initial_population) if initial_population else 1,
		"best_ce": baseline_entry,
		"best_acc": baseline_entry,
		"best_fitness": baseline_entry,
	})

	ci = args.check_interval

	# Helper to run a phase and collect metrics
	def run_and_eval_ga(phase_num, phase_name, result_key, arch_config, init_pop=None):
		nonlocal best_genome, best_ce
		if not should_run(phase_num) or best_genome is None:
			return
		print(f"\n{'='*70}")
		print(f"Phase {phase_num}: {phase_name}")
		print(f"{'='*70}")

		result, elapsed = run_ga_phase(
			opt_evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name=f"Phase {phase_num}: {phase_name}", logger=log,
			check_interval=ci, initial_population=init_pop,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results[result_key] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
		}

		# Full-eval 1-3 genomes on validation
		metrics = evaluate_phase_top_k(full_evaluator, result, f"P{phase_num} {phase_name}")
		phase_metrics_list.append(metrics)
		all_results[result_key]["full_eval"] = metrics
		print_phase_comparison(phase_metrics_list)
		save()

	def run_and_eval_ts(phase_num, phase_name, result_key, arch_config):
		nonlocal best_genome, best_ce
		if not should_run(phase_num) or best_genome is None:
			return
		print(f"\n{'='*70}")
		print(f"Phase {phase_num}: {phase_name}")
		print(f"{'='*70}")

		result, elapsed = run_ts_phase(
			opt_evaluator, arch_config,
			ts_iters=args.ts_iters, neighbors=args.neighbors,
			patience=args.patience,
			seed_genome=best_genome, seed_fitness=best_ce,
			phase_name=f"Phase {phase_num}: {phase_name}", logger=log,
			check_interval=ci,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness

		all_results[result_key] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"improvement_pct": (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100 if result.initial_fitness > 0 else 0,
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats() if hasattr(best_genome, "stats") else {},
		}

		# Full-eval 1-3 genomes on validation
		metrics = evaluate_phase_top_k(full_evaluator, result, f"P{phase_num} {phase_name}")
		phase_metrics_list.append(metrics)
		all_results[result_key]["full_eval"] = metrics
		print_phase_comparison(phase_metrics_list)
		save()

	# ── Phase 2: GA neurons (bits fixed) ─────────────────────────────────
	run_and_eval_ga(2, f"GA Neurons (bits={best_bits})", "phase2_ga_neurons", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_neurons=True, optimize_bits=False, optimize_connections=False,
		default_bits=best_bits, default_neurons=best_neurons,
		min_neurons=10, max_neurons=300,
		min_bits=best_bits, max_bits=best_bits,
		total_input_bits=total_input_bits,
	), init_pop=initial_population)

	# ── Phase 3: TS neurons refinement ───────────────────────────────────
	run_and_eval_ts(3, "TS Neurons", "phase3_ts_neurons", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_neurons=True, optimize_bits=False, optimize_connections=False,
		default_bits=best_bits,
		min_neurons=10, max_neurons=300,
		min_bits=best_bits, max_bits=best_bits,
		total_input_bits=total_input_bits,
	))

	# ── Phase 4: GA bits (neurons fixed) ─────────────────────────────────
	run_and_eval_ga(4, "GA Bits", "phase4_ga_bits", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=True, optimize_neurons=False, optimize_connections=False,
		min_bits=10, max_bits=24,
		total_input_bits=total_input_bits,
	))

	# ── Phase 5: TS bits refinement ──────────────────────────────────────
	run_and_eval_ts(5, "TS Bits", "phase5_ts_bits", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=True, optimize_neurons=False, optimize_connections=False,
		min_bits=10, max_bits=24,
		total_input_bits=total_input_bits,
	))

	# ── Phase 6: GA connections (architecture fixed) ─────────────────────
	run_and_eval_ga(6, "GA Connections", "phase6_ga_connections", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=False, optimize_neurons=False, optimize_connections=True,
		total_input_bits=total_input_bits,
	))

	# ── Phase 7: TS connections refinement ───────────────────────────────
	run_and_eval_ts(7, "TS Connections", "phase7_ts_connections", ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=False, optimize_neurons=False, optimize_connections=True,
		total_input_bits=total_input_bits,
	))

	# Save best genome separately
	if best_genome is not None:
		genome_path = output_path.with_suffix(".best_genome.json")
		best_genome.save(str(genome_path), fitness=best_ce)
		print(f"\nBest genome saved to {genome_path}")

	# ── Final Summary ────────────────────────────────────────────────────
	print(f"\n{'='*82}")
	print(f"  FINAL RESULTS (All metrics on FULL validation data)")
	print(f"{'='*82}")
	print_phase_comparison(phase_metrics_list)

	all_results["phase_metrics"] = phase_metrics_list
	save()
	print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
