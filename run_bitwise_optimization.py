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

def load_wikitext2_tokens(tokenizer_name="gpt2", train_tokens_limit=200_000, eval_tokens_limit=50_000):
	"""Load WikiText-2 tokens using GPT-2 tokenizer, with configurable limits.

	Default limits match run_coarse_fine_search.py for comparable speed:
	- train: 200K tokens (split into 3 subsets of ~67K for rotation)
	- eval: 50K tokens
	"""
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
	all_train_tokens = tokenizer.encode(train_text)

	# Use eval split (not test) for optimization; test is held out for final eval
	eval_text = "\n".join(dataset["validation"]["text"])
	all_eval_tokens = tokenizer.encode(eval_text)

	# Apply limits
	train_tokens = all_train_tokens[:train_tokens_limit]
	eval_tokens = all_eval_tokens[:eval_tokens_limit]

	vocab_size = tokenizer.vocab_size
	print(f"Loaded WikiText-2: {len(train_tokens):,}/{len(all_train_tokens):,} train, "
		  f"{len(eval_tokens):,}/{len(all_eval_tokens):,} eval tokens, vocab={vocab_size}")

	return train_tokens, eval_tokens, vocab_size


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
	seed_genome=None, phase_name="GA", logger=print, check_interval=10,
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
		check_interval=check_interval,
		# Threshold: 0.01%/gen → 1% every 100 gens
		threshold_delta=0.01,
		threshold_reference=100,
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
		batch_evaluate_fn=lambda genomes, **kw: evaluator.evaluate_batch(genomes, **kw),
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
	seed_genome, seed_fitness, phase_name="TS", logger=print, check_interval=10,
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
		check_interval=check_interval,
		# Threshold: 0.01%/iter → 1% every 100 iters
		threshold_delta=0.01,
		threshold_reference=100,
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


def evaluate_phase_top_k(evaluator, result, phase_name, k=10):
	"""Evaluate a phase's population on full validation, return top-K metrics.

	Returns dict with keys: best_ce, best_acc, best_fitness, top_k_mean.
	Each has (ce, acc, ppl).
	"""
	pop = result.final_population if result.final_population else []
	if not pop:
		# Single genome fallback
		pop = [result.best_genome]

	# Evaluate all on full train+eval
	full_evals = []
	for genome in pop:
		ce, acc = evaluator.evaluate_single_full(genome)
		full_evals.append((genome, ce, acc))

	# Sort by CE (ascending), Acc (descending)
	by_ce = sorted(full_evals, key=lambda x: x[1])
	by_acc = sorted(full_evals, key=lambda x: -x[2])

	# Harmonic weighted fitness ranking
	n = len(full_evals)
	ce_ranks = {id(g): i + 1 for i, (g, _, _) in enumerate(by_ce)}
	acc_ranks = {id(g): i + 1 for i, (g, _, _) in enumerate(by_acc)}
	by_fitness = sorted(
		full_evals,
		key=lambda x: harmonic_weighted(ce_ranks[id(x[0])], acc_ranks[id(x[0])]),
	)

	# Top-K mean (by CE)
	top_k = by_ce[:k]
	top_k_ce = sum(e[1] for e in top_k) / len(top_k)
	top_k_acc = sum(e[2] for e in top_k) / len(top_k)

	return {
		"phase_name": phase_name,
		"n_evaluated": len(full_evals),
		"k": min(k, len(top_k)),
		"top_k_mean": {"ce": top_k_ce, "acc": top_k_acc, "ppl": math.exp(top_k_ce)},
		"best_ce": {"ce": by_ce[0][1], "acc": by_ce[0][2], "ppl": math.exp(by_ce[0][1])},
		"best_acc": {"ce": by_acc[0][1], "acc": by_acc[0][2], "ppl": math.exp(by_acc[0][1])},
		"best_fitness": {"ce": by_fitness[0][1], "acc": by_fitness[0][2], "ppl": math.exp(by_fitness[0][1])},
	}


def print_phase_comparison(phase_metrics_list):
	"""Print cumulative phase comparison table (all on full validation)."""
	W = 92
	print(f"\n{'='*W}")
	print(f"  Phase Comparison (Full Validation)")
	print(f"{'='*W}")
	print(f"  {'Phase':<28} {'Metric':<14} {'CE':>10} {'PPL':>12} {'Accuracy':>10}")
	print(f"  {'-'*(W-2)}")

	baseline_ce = phase_metrics_list[0]["best_ce"]["ce"] if phase_metrics_list else None

	for i, pm in enumerate(phase_metrics_list):
		name = pm["phase_name"]
		k = pm["k"]

		# Row 1: top-K mean
		m = pm["top_k_mean"]
		print(f"  {name:<28} {'top-'+str(k)+' mean':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		# Row 2: best CE
		m = pm["best_ce"]
		print(f"  {'':<28} {'best CE':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		# Row 3: best Acc
		m = pm["best_acc"]
		print(f"  {'':<28} {'best Acc':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		# Row 4: best Fitness (harmonic weighted)
		m = pm["best_fitness"]
		print(f"  {'':<28} {'best Fitness':<14} {m['ce']:>10.4f} {m['ppl']:>12.0f} {m['acc']:>9.2%}")

		if i < len(phase_metrics_list) - 1:
			print(f"  {'-'*(W-2)}")

	print(f"  {'='*(W-2)}")

	# Overall improvement
	if len(phase_metrics_list) >= 2 and baseline_ce:
		final = phase_metrics_list[-1]
		top_k_impr = (1 - final["top_k_mean"]["ce"] / baseline_ce) * 100
		best_ce_impr = (1 - final["best_ce"]["ce"] / baseline_ce) * 100
		print(f"  Improvement vs baseline: top-{final['k']} CE {top_k_impr:+.2f}%, best CE {best_ce_impr:+.2f}%")


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
	parser.add_argument("--train-tokens", type=int, default=200_000,
						help="Total training tokens from WikiText-2 (default: 200000)")
	parser.add_argument("--eval-tokens", type=int, default=50_000,
						help="Eval tokens from WikiText-2 validation split (default: 50000)")
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
	parser.add_argument("--check-interval", type=int, default=10,
						help="Patience check every N generations/iterations (default: 10)")
	args = parser.parse_args()

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Import dependencies
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	train_tokens, test_tokens, vocab_size = load_wikitext2_tokens(
		train_tokens_limit=args.train_tokens,
		eval_tokens_limit=args.eval_tokens,
	)
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
			"train_tokens": len(train_tokens),
			"eval_tokens": len(test_tokens),
			"ga_gens": args.ga_gens,
			"population": args.population,
			"ts_iters": args.ts_iters,
			"neighbors": args.neighbors,
			"patience": args.patience,
			"check_interval": args.check_interval,
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

	# Evaluate baseline on full validation
	if best_ce is None and best_genome is not None:
		print("Evaluating seed genome...")
		results = evaluator.evaluate_batch([best_genome])
		best_ce = results[0][0]
		print(f"  Seed CE={best_ce:.4f}")

	# Baseline full-eval
	phase_metrics_list = []
	print(f"\nEvaluating baseline on full validation...")
	baseline_ce, baseline_acc = evaluator.evaluate_single_full(best_genome)
	print(f"  Baseline CE={baseline_ce:.4f}, Acc={baseline_acc:.2%}, PPL={math.exp(baseline_ce):.0f}")
	phase_metrics_list.append({
		"phase_name": "Baseline",
		"n_evaluated": 1,
		"k": 1,
		"top_k_mean": {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)},
		"best_ce": {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)},
		"best_acc": {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)},
		"best_fitness": {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)},
	})

	ci = args.check_interval

	# Helper to run a phase and collect top-K metrics
	def run_and_eval_ga(phase_num, phase_name, result_key, arch_config):
		nonlocal best_genome, best_ce
		if not should_run(phase_num) or best_genome is None:
			return
		print(f"\n{'='*70}")
		print(f"Phase {phase_num}: {phase_name}")
		print(f"{'='*70}")

		result, elapsed = run_ga_phase(
			evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name=f"Phase {phase_num}: {phase_name}", logger=log,
			check_interval=ci,
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

		# Full-eval top-K
		print(f"\n  Evaluating Phase {phase_num} population on full validation...")
		metrics = evaluate_phase_top_k(evaluator, result, f"P{phase_num} {phase_name}", k=args.top_k)
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
			evaluator, arch_config,
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

		# Full-eval top-K
		print(f"\n  Evaluating Phase {phase_num} population on full validation...")
		metrics = evaluate_phase_top_k(evaluator, result, f"P{phase_num} {phase_name}", k=args.top_k)
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
	))

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
	print(f"\n{'='*92}")
	print(f"  FINAL RESULTS (All metrics on FULL validation data)")
	print(f"{'='*92}")
	print_phase_comparison(phase_metrics_list)

	all_results["phase_metrics"] = phase_metrics_list
	save()
	print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
