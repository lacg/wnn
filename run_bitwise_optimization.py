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
from dataclasses import asdict
from pathlib import Path

from wnn.logger import Logger


# ---------------------------------------------------------------------------
# Correlation analysis (Pearson + Spearman, pure Python — no scipy needed)
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> float:
	"""Pearson correlation coefficient between x and y."""
	n = len(x)
	if n < 3:
		return 0.0
	mx = sum(x) / n
	my = sum(y) / n
	cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
	sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
	sy = sum((yi - my) ** 2 for yi in y) ** 0.5
	if sx == 0 or sy == 0:
		return 0.0
	return cov / (sx * sy)


def _rank(values: list[float]) -> list[float]:
	"""Assign ranks (1-based, average ties) to values."""
	n = len(values)
	indexed = sorted(range(n), key=lambda i: values[i])
	ranks = [0.0] * n
	i = 0
	while i < n:
		j = i
		while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
			j += 1
		avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
		for k in range(i, j + 1):
			ranks[indexed[k]] = avg_rank
		i = j + 1
	return ranks


def _spearman(x: list[float], y: list[float]) -> float:
	"""Spearman rank correlation coefficient between x and y."""
	if len(x) < 3:
		return 0.0
	return _pearson(_rank(x), _rank(y))


def compute_phase_correlations(
	generation_log: list[tuple[int, float, float, float, float, float, float]],
	logger=print,
) -> dict:
	"""Compute Pearson and Spearman correlations between CE, Acc, and BitAcc.

	Uses per-generation data from BitwiseEvaluator.generation_log.
	Each entry: (gen, best_ce, best_acc, best_bit_acc, mean_ce, mean_acc, mean_bit_acc)

	Returns dict with correlation values for JSON serialization.
	"""
	if len(generation_log) < 5:
		logger(f"  Correlations: insufficient data ({len(generation_log)} generations, need >= 5)")
		return {"insufficient_data": True, "n_generations": len(generation_log)}

	# Extract series
	best_ce = [e[1] for e in generation_log]
	best_acc = [e[2] for e in generation_log]
	best_bit_acc = [e[3] for e in generation_log]
	mean_ce = [e[4] for e in generation_log]
	mean_acc = [e[5] for e in generation_log]
	mean_bit_acc = [e[6] for e in generation_log]

	# Compute correlations (best genome metrics)
	corr = {
		"n_generations": len(generation_log),
		"best": {
			"ce_vs_bit_acc": {
				"pearson": round(_pearson(best_ce, best_bit_acc), 4),
				"spearman": round(_spearman(best_ce, best_bit_acc), 4),
			},
			"acc_vs_bit_acc": {
				"pearson": round(_pearson(best_acc, best_bit_acc), 4),
				"spearman": round(_spearman(best_acc, best_bit_acc), 4),
			},
			"ce_vs_acc": {
				"pearson": round(_pearson(best_ce, best_acc), 4),
				"spearman": round(_spearman(best_ce, best_acc), 4),
			},
		},
		"mean": {
			"ce_vs_bit_acc": {
				"pearson": round(_pearson(mean_ce, mean_bit_acc), 4),
				"spearman": round(_spearman(mean_ce, mean_bit_acc), 4),
			},
			"acc_vs_bit_acc": {
				"pearson": round(_pearson(mean_acc, mean_bit_acc), 4),
				"spearman": round(_spearman(mean_acc, mean_bit_acc), 4),
			},
			"ce_vs_acc": {
				"pearson": round(_pearson(mean_ce, mean_acc), 4),
				"spearman": round(_spearman(mean_ce, mean_acc), 4),
			},
		},
	}

	# Log summary
	logger(f"\n  Correlation Analysis ({len(generation_log)} generations):")
	logger(f"  ┌─────────────────────┬──────────┬──────────┐")
	logger(f"  │ Metric Pair (best)  │ Pearson  │ Spearman │")
	logger(f"  ├─────────────────────┼──────────┼──────────┤")
	for pair_name, label in [("ce_vs_bit_acc", "CE ↔ BitAcc"), ("acc_vs_bit_acc", "Acc ↔ BitAcc"), ("ce_vs_acc", "CE ↔ Acc")]:
		p = corr["best"][pair_name]["pearson"]
		s = corr["best"][pair_name]["spearman"]
		logger(f"  │ {label:<19s} │ {p:+7.4f}  │ {s:+7.4f}  │")
	logger(f"  └─────────────────────┴──────────┴──────────┘")

	# Interpret
	ce_ba_p = corr["best"]["ce_vs_bit_acc"]["pearson"]
	if ce_ba_p < -0.5:
		logger(f"  → CE and BitAcc are negatively correlated (r={ce_ba_p:+.3f}): BitAcc improves as CE improves")
		logger(f"    BitAcc is a good proxy for CE — consider using it in fitness ranking")
	elif abs(ce_ba_p) < 0.3:
		logger(f"  → CE and BitAcc are weakly correlated (r={ce_ba_p:+.3f}): CE improvements come from elsewhere")
		logger(f"    BitAcc alone may not capture what drives CE — reconstruction matters more")
	else:
		logger(f"  → CE and BitAcc correlation: r={ce_ba_p:+.3f}")

	return corr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext2_tokens(tokenizer_name="gpt2", logger=print):
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
		logger("ERROR: datasets and transformers required. Install with:")
		logger("  pip install datasets transformers")
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
	logger(f"Loaded WikiText-2: {len(train_tokens):,} train, "
		   f"{len(test_tokens):,} test, {len(validation_tokens):,} validation, vocab={vocab_size}")

	return train_tokens, test_tokens, validation_tokens, vocab_size


# ---------------------------------------------------------------------------
# Phase 1: Grid search (quick landscape scan)
# ---------------------------------------------------------------------------

def phase1_grid_search(train_tokens, test_tokens, vocab_size, context_size, rate, top_k=3, logger=print,
					   fitness_type=None):
	"""Grid search over neurons × bits with random connections."""
	from wnn.ram.core.models import BitwiseRAMLM
	from wnn.ram.fitness import FitnessCalculatorType, FitnessCalculatorFactory

	if fitness_type is None:
		fitness_type = FitnessCalculatorType.HARMONIC_RANK
	calculator = FitnessCalculatorFactory.create(fitness_type)

	neurons_grid = [50, 100, 150, 200]
	bits_grid = [14, 16, 18, 20]

	results = []
	total = len(neurons_grid) * len(bits_grid)
	idx = 0

	logger(f"\n{'='*70}")
	logger(f"Phase 1: Grid Search ({total} configs)")
	logger(f"  mode=QUAD_WEIGHTED, rate={rate}, context={context_size}")
	logger(f"{'='*70}")

	for neurons in neurons_grid:
		for bits in bits_grid:
			idx += 1
			logger(f"[{idx}/{total}] n={neurons}, b={bits}")

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

			logger(f"  CE={ce:.4f}  PPL={ppl:.0f}  Acc={acc:.2%}  BitAcc={mean_bit_acc:.2%}  ({elapsed:.1f}s)")

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

	# Rank by fitness calculator (same as GA/TS phases)
	population = [(r, r["cross_entropy"], r["accuracy"]) for r in results]
	fitness_scores = calculator.fitness(population)
	for r, score in zip(results, fitness_scores):
		r["fitness"] = score
	results.sort(key=lambda r: r["fitness"])

	logger(f"\n{'─'*70}")
	logger(f"Phase 1 Rankings (by {calculator.name}):")
	for i, r in enumerate(results):
		marker = " ★" if i < top_k else ""
		logger(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
			   f"CE={r['cross_entropy']:.4f}  Acc={r['accuracy']:.2%}  "
			   f"Fit={r['fitness']:.4f}{marker}")

	best = results[0]
	logger(f"\nBest config: n={best['neurons']}, b={best['bits']}, "
		   f"CE={best['cross_entropy']:.4f}, Fit={best['fitness']:.4f}")

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
	initial_population=None, checkpoint_config=None,
	fitness_type=None,
):
	"""Run a GA optimization phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import GAConfig
	from wnn.ram.fitness import FitnessCalculatorType

	if fitness_type is None:
		fitness_type = FitnessCalculatorType.HARMONIC_RANK

	ga_config = GAConfig(
		population_size=population,
		generations=ga_gens,
		mutation_rate=0.1,
		crossover_rate=0.7,
		tournament_size=3,
		elitism_pct=0.1,
		patience=patience,
		check_interval=check_interval,
		# Start at 3% accuracy, +0.001%/gen continuously
		min_accuracy=0.03,
		threshold_delta=ga_gens * 0.00001,
		threshold_reference=ga_gens,
		fitness_calculator_type=fitness_type,
		fitness_weight_ce=1.0,
		fitness_weight_acc=1.0,
	)

	strategy = ArchitectureGAStrategy(
		arch_config=arch_config,
		ga_config=ga_config,
		logger=logger,
		batch_evaluator=evaluator,
		checkpoint_config=checkpoint_config,
		phase_name=phase_name,
	)

	t0 = time.time()
	optimize_kwargs = dict(
		evaluate_fn=lambda g: 999.0,
		initial_genome=seed_genome,
		batch_evaluate_fn=lambda genomes, **kw: evaluator.evaluate_batch(genomes, logger=logger, **kw),
	)
	if initial_population is not None:
		optimize_kwargs["initial_population"] = initial_population
	result = strategy.optimize(**optimize_kwargs)
	elapsed = time.time() - t0

	logger(f"  {phase_name} Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		logger(f"  Accuracy: {result.final_accuracy:.2%}")
	logger(f"  Improvement: {result.improvement_percent:.1f}%")
	logger(f"  Generations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	logger(f"  Stop reason: {result.stop_reason}")

	return result, elapsed


def run_ts_phase(
	evaluator, arch_config, ts_iters, neighbors, patience,
	seed_genome, seed_fitness, phase_name="TS", logger=print, check_interval=10,
	initial_neighbors=None, diversity_sources_pct=0.0,
	fitness_type=None,
):
	"""Run a TS refinement phase."""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureTSStrategy
	from wnn.ram.strategies.connectivity.generic_strategies import TSConfig
	from wnn.ram.fitness import FitnessCalculatorType

	if fitness_type is None:
		fitness_type = FitnessCalculatorType.HARMONIC_RANK

	ts_config = TSConfig(
		iterations=ts_iters,
		neighbors_per_iter=neighbors,
		tabu_size=10,
		mutation_rate=0.1,
		patience=patience,
		check_interval=check_interval,
		# Start at 3% accuracy, +0.001%/iter continuously
		min_accuracy=0.03,
		threshold_delta=ts_iters * 0.00001,
		threshold_reference=ts_iters,
		fitness_calculator_type=fitness_type,
		fitness_weight_ce=1.0,
		fitness_weight_acc=1.0,
		diversity_sources_pct=diversity_sources_pct,
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
		batch_evaluate_fn=lambda genomes, **kw: evaluator.evaluate_batch(genomes, logger=logger, **kw),
		initial_neighbors=initial_neighbors,
	)
	elapsed = time.time() - t0

	improvement_over_seed = (seed_fitness - result.final_fitness) / seed_fitness * 100 if seed_fitness > 0 else 0
	logger(f"  {phase_name} Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		logger(f"  Accuracy: {result.final_accuracy:.2%}")
	logger(f"  Improvement over seed: {improvement_over_seed:.1f}%")
	logger(f"  Iterations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	logger(f"  Stop reason: {result.stop_reason}")

	return result, elapsed


# ---------------------------------------------------------------------------
# Top-K full-eval summary (reuses PhaseComparisonTable from reporting.py)
# ---------------------------------------------------------------------------

from wnn.ram.core.reporting import PhaseMetrics, PhaseComparisonTable


def harmonic_weighted(ce_rank, acc_rank, w_ce=1.0, w_acc=1.0):
	"""Weighted harmonic mean of ranks (lower = better)."""
	if ce_rank == 0 or acc_rank == 0:
		return float("inf")
	return (w_ce + w_acc) / (w_ce / ce_rank + w_acc / acc_rank)


def evaluate_phase_top_k(full_evaluator, result, phase_name, logger=print):
	"""Full-eval 1-3 representative genomes from a phase on validation data.

	Pre-filters the population by fitness rank (using optimization metrics from result),
	then full-evals only best CE, best Acc (if different), best Fitness (if different).

	Returns PhaseMetrics for use with PhaseComparisonTable.
	"""
	pop = result.final_population if result.final_population else []
	if not pop:
		pop = [result.best_genome]

	# Use optimization-time metrics (from the result) to rank population
	pop_metrics = getattr(result, "population_metrics", None)
	if pop_metrics and len(pop_metrics) == len(pop):
		by_ce = sorted(range(len(pop)), key=lambda i: pop_metrics[i][0])
		by_acc = sorted(range(len(pop)), key=lambda i: -pop_metrics[i][1])

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
		best_ce_idx = 0
		best_acc_idx = 0
		best_fit_idx = 0

	# Collect unique genomes to full-eval (1-3 max)
	candidates = {}
	candidates[best_ce_idx] = pop[best_ce_idx]
	if best_acc_idx != best_ce_idx:
		candidates[best_acc_idx] = pop[best_acc_idx]
	if best_fit_idx not in candidates:
		candidates[best_fit_idx] = pop[best_fit_idx]

	genome_list = list(candidates.values())
	idx_list = list(candidates.keys())
	logger(f"  Full-eval {len(genome_list)} genome(s) on validation...")
	full_results = full_evaluator.evaluate_batch_full(genome_list)

	eval_map = {}
	for i, idx in enumerate(idx_list):
		eval_map[idx] = full_results[i]

	def get_metrics(idx):
		r = eval_map[idx]
		return r[0], r[1], r[2] if len(r) > 2 else 0.0

	fit_ce, fit_acc, fit_bit_acc = get_metrics(best_fit_idx)
	ce_ce, ce_acc, ce_bit_acc = get_metrics(best_ce_idx)
	acc_ce, acc_acc, acc_bit_acc = get_metrics(best_acc_idx)

	# Log BitAcc for correlation tracking
	logger(f"  Full-eval BitAcc: fitness={fit_bit_acc:.2%}, best_ce={ce_bit_acc:.2%}, best_acc={acc_bit_acc:.2%}")

	return PhaseMetrics(
		phase_name=phase_name,
		top_k_ce=fit_ce,
		top_k_acc=fit_acc,
		best_ce_ce=ce_ce,
		best_ce_acc=ce_acc,
		best_acc_ce=acc_ce,
		best_acc_acc=acc_acc,
		first_metric_label="best fitness",
	)


def print_phase_comparison(phase_metrics_list):
	"""Print cumulative phase comparison table using PhaseComparisonTable."""
	table = PhaseComparisonTable("Phase Comparison (Full Validation)")
	for pm in phase_metrics_list:
		table.add_phase(pm)
	table.print()


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
	parser.add_argument("--neighbors", type=int, default=50,
						help="TS neighbors per iteration (default: 50)")
	parser.add_argument("--ts-diversity", type=float, default=0.0,
						help="TS diversity: fraction of top genomes as neighbor sources (0=top-1, 0.2=top 20%%)")
	parser.add_argument("--patience", type=int, default=2,
						help="Early stop patience (default: 2)")
	parser.add_argument("--check-interval", type=int, default=10,
						help="Patience check every N generations/iterations (default: 10)")
	# Data rotation
	parser.add_argument("--train-parts", type=int, default=4,
						help="Number of train subsets for rotation (default: 4)")
	parser.add_argument("--eval-parts", type=int, default=1,
						help="Number of test subsets for rotation (default: 1)")
	# Checkpointing
	parser.add_argument("--checkpoint-dir", type=str, default=None,
						help="Directory for intra-phase checkpoints (enables resume on crash)")
	# Fitness
	parser.add_argument("--fitness", type=str, default="HARMONIC_RANK",
						choices=["CE", "HARMONIC_RANK", "NORMALIZED", "NORMALIZED_HARMONIC"],
						help="Fitness calculator type (default: HARMONIC_RANK)")
	# Gating
	parser.add_argument("--enable-gating", action="store_true",
						help="Enable gating phase after optimization")
	parser.add_argument("--gating-neurons", type=int, default=8,
						help="Neurons per gate (default: 8)")
	parser.add_argument("--gating-bits", type=int, default=12,
						help="Bits per gating neuron (default: 12)")
	parser.add_argument("--gating-threshold", type=float, default=0.5,
						help="Gating threshold (default: 0.5)")
	parser.add_argument("--gating-mode", type=int, default=0,
						choices=[0, 1, 2],
						help="0=TOKEN_LEVEL, 1=BIT_LEVEL, 2=DUAL_STAGE (default: 0)")
	args = parser.parse_args()

	# Resolve fitness calculator type
	from wnn.ram.fitness import FitnessCalculatorType
	fitness_type = FitnessCalculatorType[args.fitness]

	# Checkpoint config for intra-phase resume
	checkpoint_config = None
	if args.checkpoint_dir:
		from wnn.ram.strategies.connectivity.architecture_strategies import CheckpointConfig
		cp_dir = Path(args.checkpoint_dir)
		cp_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_config = CheckpointConfig(enabled=True, interval=5, checkpoint_dir=cp_dir)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Create logger with timestamps
	logger = Logger(f"bitwise_opt_ctx{args.context}")
	logger(f"Output: {args.output}")

	# Import dependencies
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	# Load all 3 WikiText-2 splits
	train_tokens, test_tokens, validation_tokens, vocab_size = load_wikitext2_tokens(logger=logger)
	num_clusters = bits_needed(vocab_size)  # 16 for GPT-2
	total_input_bits = args.context * bits_needed(vocab_size)

	all_results = {
		"config": {
			"context_size": args.context,
			"neuron_sample_rate": args.rate,
			"memory_mode": "QUAD_WEIGHTED",
			"fitness": args.fitness,
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
			logger(f"Loaded previous results from {input_path}")
		except Exception as e:
			logger(f"Warning: Could not load {input_path}: {e}")

	def save():
		with open(args.output, "w") as f:
			json.dump(all_results, f, indent=2)

	phase_result_keys = {
		1: "phase1_grid",
		2: "phase2_ga_neurons", 3: "phase3_ts_neurons",
		4: "phase4_ga_bits", 5: "phase5_ts_bits",
		6: "phase6_ga_connections", 7: "phase7_ts_connections",
	}

	def should_run(phase_num):
		if args.phase not in ("all", str(phase_num)):
			return False
		key = phase_result_keys.get(phase_num)
		if not key:
			return True
		data = all_results.get(key)
		if not data:
			return True
		# Phase 1 uses a different format (no best_genome_data)
		if phase_num == 1 and data:
			logger(f"  Phase 1 already completed, skipping")
			return False
		if "best_genome_data" in data:
			logger(f"  Phase {phase_num} already completed, skipping")
			return False
		return True

	def log(msg):
		logger(f"  {msg}")

	# Track best genome flowing through phases
	best_genome = None
	best_ce = None
	best_bits = 20   # defaults, overwritten by Phase 1
	best_neurons = 150
	initial_population = None  # 50 genomes seeded from Phase 1

	# Restore genome + population from last completed phase (inter-phase resume)
	phase_order = [
		("phase7_ts_connections", 7), ("phase6_ga_connections", 6),
		("phase5_ts_bits", 5), ("phase4_ga_bits", 4),
		("phase3_ts_neurons", 3), ("phase2_ga_neurons", 2),
	]
	for key, pnum in phase_order:
		data = all_results.get(key)
		if data and "best_genome_data" in data:
			best_genome = ClusterGenome.deserialize(data["best_genome_data"])
			best_ce = data["final_ce"]
			if data.get("final_population"):
				initial_population = [ClusterGenome.deserialize(g) for g in data["final_population"]]
			logger(f"Resumed from {key}: CE={best_ce:.4f}, population={len(initial_population) if initial_population else 0}")
			break

	# ── Phase 1: Grid search ─────────────────────────────────────────────
	if should_run(1):
		t0 = time.time()
		phase1_results = phase1_grid_search(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate, top_k=args.top_k,
			logger=logger, fitness_type=fitness_type,
		)
		best_p1 = phase1_results[0]  # sorted by fitness (HARMONIC_RANK)
		best_bits = best_p1["bits"]
		best_neurons = best_p1["neurons"]
		best_ce = best_p1["cross_entropy"]

		# ── Create initial population with tiered seeding ──
		# Top 1-5: 5 variants each (best configs get most diversity)
		# 6-10: 3 variants each
		# 11-16: fill remaining slots
		logger(f"\nSeeding population from {len(phase1_results)} grid configs...")
		pop_genomes = []
		target = args.population

		# Tier 1: top 5 configs × 5 variants = 25
		for config in phase1_results[:5]:
			for _ in range(5):
				g = create_seed_genome(
					num_clusters, config["bits"], config["neurons"], total_input_bits
				)
				pop_genomes.append(g)

		# Tier 2: configs 6-10 × 3 variants = 15
		for config in phase1_results[5:10]:
			for _ in range(3):
				g = create_seed_genome(
					num_clusters, config["bits"], config["neurons"], total_input_bits
				)
				pop_genomes.append(g)

		# Tier 3: configs 11-16, fill remaining slots
		remaining = phase1_results[10:]
		for config in remaining:
			if len(pop_genomes) >= target:
				break
			g = create_seed_genome(
				num_clusters, config["bits"], config["neurons"], total_input_bits
			)
			pop_genomes.append(g)

		# Top up any remaining slots from top configs (round-robin)
		i = 0
		while len(pop_genomes) < target:
			config = phase1_results[i % len(phase1_results)]
			g = create_seed_genome(
				num_clusters, config["bits"], config["neurons"], total_input_bits
			)
			pop_genomes.append(g)
			i += 1

		initial_population = pop_genomes[:target]
		best_genome = initial_population[0]  # Best config, first random seed

		logger(f"  Created {len(initial_population)} genomes for initial population")

		all_results["phase1_grid"] = {
			"results": phase1_results,
			"best_bits": best_bits,
			"best_neurons": best_neurons,
			"best_ce": best_ce,
			"init_population_size": len(initial_population),
			"elapsed_s": round(time.time() - t0, 1),
			"initial_population": [g.serialize() for g in initial_population],
		}
		save()
		logger(f"\nPhase 1 saved to {args.output}")
	elif all_results.get("phase1_grid"):
		p1 = all_results["phase1_grid"]
		best_bits = p1["best_bits"]
		best_neurons = p1["best_neurons"]
		if best_genome is None:
			best_ce = p1["best_ce"]
		# Restore population: prefer serialized, fallback to recreation
		if initial_population is None:
			if p1.get("initial_population"):
				# Load serialized population from checkpoint
				loaded = [ClusterGenome.deserialize(g) for g in p1["initial_population"]]
				if len(loaded) >= args.population:
					initial_population = loaded[:args.population]
				else:
					# Fill gap with mutations of existing genomes
					initial_population = list(loaded)
					while len(initial_population) < args.population:
						src = loaded[len(initial_population) % len(loaded)]
						initial_population.append(create_seed_genome(
							num_clusters, src.clusters[0].bits, src.clusters[0].neurons, total_input_bits
						))
				logger(f"Loaded {len(loaded)} genomes from checkpoint (using {len(initial_population)})")
			elif "results" in p1:
				# Recreate from grid configs (old checkpoint without serialized pop)
				pop_genomes = []
				for config in p1["results"]:
					for _ in range(3):
						pop_genomes.append(create_seed_genome(
							num_clusters, config["bits"], config["neurons"], total_input_bits
						))
				for config in p1["results"][:2]:
					pop_genomes.append(create_seed_genome(
						num_clusters, config["bits"], config["neurons"], total_input_bits
					))
				initial_population = pop_genomes[:args.population]
				logger(f"Recreated {len(initial_population)} genomes from grid configs")
		if best_genome is None:
			best_genome = initial_population[0] if initial_population else create_seed_genome(
				num_clusters, best_bits, best_neurons, total_input_bits
			)
		logger(f"Loaded Phase 1: bits={best_bits}, neurons={best_neurons}, CE={best_ce:.4f}")
	else:
		if best_genome is None:
			best_genome = create_seed_genome(num_clusters, best_bits, best_neurons, total_input_bits)
		logger(f"Using default config: bits={best_bits}, neurons={best_neurons}")

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
	logger(f"\nEvaluating baseline on full validation...")
	baseline_result = full_evaluator.evaluate_batch_full([best_genome])[0]
	baseline_ce, baseline_acc = baseline_result[0], baseline_result[1]
	baseline_bit_acc = baseline_result[2] if len(baseline_result) > 2 else 0.0
	logger(f"  Baseline CE={baseline_ce:.4f}, Acc={baseline_acc:.2%}, PPL={math.exp(baseline_ce):.0f}, BitAcc={baseline_bit_acc:.2%}")
	phase_metrics_list.append(PhaseMetrics(
		phase_name="Init Population",
		top_k_ce=baseline_ce,
		top_k_acc=baseline_acc,
		best_ce_ce=baseline_ce,
		best_ce_acc=baseline_acc,
		best_acc_ce=baseline_ce,
		best_acc_acc=baseline_acc,
		first_metric_label="best fitness",
	))

	ci = args.check_interval

	# Helper to run a phase and collect metrics
	def run_and_eval_ga(phase_num, phase_name, result_key, arch_config, init_pop=None):
		nonlocal best_genome, best_ce, initial_population
		if not should_run(phase_num) or best_genome is None:
			return
		logger(f"\n{'='*70}")
		logger(f"Phase {phase_num}: {phase_name}")
		logger(f"{'='*70}")

		result, elapsed = run_ga_phase(
			opt_evaluator, arch_config,
			ga_gens=args.ga_gens, population=args.population,
			patience=args.patience, seed_genome=best_genome,
			phase_name=f"Phase {phase_num}: {phase_name}", logger=log,
			check_interval=ci, initial_population=init_pop or initial_population,
			checkpoint_config=checkpoint_config,
			fitness_type=fitness_type,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness
		initial_population = result.final_population

		# Track BitAcc from cached evaluation on best genome
		best_bit_acc = getattr(best_genome, '_cached_bit_acc', None)

		all_results[result_key] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"final_bit_accuracy": best_bit_acc,
			"improvement_pct": result.improvement_percent,
			"generations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats(),
			"best_genome_data": best_genome.serialize(),
			"final_population": [g.serialize() for g in result.final_population] if result.final_population else None,
		}

		if best_bit_acc is not None:
			logger(f"  Best genome BitAcc={best_bit_acc:.2%}")

		# Correlation analysis from per-generation metrics
		correlations = compute_phase_correlations(opt_evaluator.generation_log, logger=log)
		all_results[result_key]["correlations"] = correlations
		opt_evaluator.clear_generation_log()

		# Full-eval 1-3 genomes on validation
		metrics = evaluate_phase_top_k(full_evaluator, result, f"P{phase_num} {phase_name}", logger=log)
		phase_metrics_list.append(metrics)
		all_results[result_key]["full_eval"] = asdict(metrics)
		print_phase_comparison(phase_metrics_list)
		save()

	def run_and_eval_ts(phase_num, phase_name, result_key, arch_config):
		nonlocal best_genome, best_ce, initial_population
		if not should_run(phase_num) or best_genome is None:
			return
		logger(f"\n{'='*70}")
		logger(f"Phase {phase_num}: {phase_name}")
		logger(f"{'='*70}")

		result, elapsed = run_ts_phase(
			opt_evaluator, arch_config,
			ts_iters=args.ts_iters, neighbors=args.neighbors,
			patience=args.patience,
			seed_genome=best_genome, seed_fitness=best_ce,
			phase_name=f"Phase {phase_num}: {phase_name}", logger=log,
			check_interval=ci,
			initial_neighbors=initial_population,
			diversity_sources_pct=args.ts_diversity,
			fitness_type=fitness_type,
		)

		best_genome = result.best_genome
		best_ce = result.final_fitness
		initial_population = result.final_population

		# Track BitAcc from cached evaluation on best genome
		best_bit_acc = getattr(best_genome, '_cached_bit_acc', None)

		all_results[result_key] = {
			"initial_ce": result.initial_fitness,
			"final_ce": result.final_fitness,
			"final_accuracy": result.final_accuracy,
			"final_bit_accuracy": best_bit_acc,
			"improvement_pct": (result.initial_fitness - result.final_fitness) / result.initial_fitness * 100 if result.initial_fitness > 0 else 0,
			"iterations_run": result.iterations_run,
			"stop_reason": str(result.stop_reason),
			"elapsed_s": round(elapsed, 1),
			"genome_stats": best_genome.stats() if hasattr(best_genome, "stats") else {},
			"best_genome_data": best_genome.serialize(),
			"final_population": [g.serialize() for g in result.final_population] if result.final_population else None,
		}

		if best_bit_acc is not None:
			logger(f"  Best genome BitAcc={best_bit_acc:.2%}")

		# Correlation analysis from per-generation metrics
		correlations = compute_phase_correlations(opt_evaluator.generation_log, logger=log)
		all_results[result_key]["correlations"] = correlations
		opt_evaluator.clear_generation_log()

		# Full-eval 1-3 genomes on validation
		metrics = evaluate_phase_top_k(full_evaluator, result, f"P{phase_num} {phase_name}", logger=log)
		phase_metrics_list.append(metrics)
		all_results[result_key]["full_eval"] = asdict(metrics)
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

	# ── Phase 8 (optional): Gating ──────────────────────────────────────
	if args.enable_gating and best_genome is not None:
		from wnn.ram.core.gating_trainer import GatingTrainer, GatingConfig, GatingMode

		logger(f"\n{'='*70}")
		logger(f"Phase 8: Gating ({GatingMode(args.gating_mode).name})")
		logger(f"{'='*70}")

		gating_config = GatingConfig(
			enabled=True,
			neurons_per_gate=args.gating_neurons,
			bits_per_neuron=args.gating_bits,
			threshold=args.gating_threshold,
			mode=GatingMode(args.gating_mode),
		)
		trainer = GatingTrainer(gating_config, logger=log)
		gating_result = trainer.train(
			total_input_bits=total_input_bits,
			train_tokens=train_tokens,
			vocab_size=vocab_size,
			context_size=args.context,
			# No cluster_order → bitwise encoding
		)

		# Use full_evaluator's eval tokens for gated evaluation
		gated_eval = opt_evaluator.evaluate_with_gating(
			genome=best_genome,
			train_tokens=train_tokens,
			gating_result=gating_result,
			logger=log,
		)
		all_results["gating"] = gated_eval
		save()

	# Save best genome separately
	if best_genome is not None:
		genome_path = output_path.with_suffix(".best_genome.json")
		best_genome.save(str(genome_path), fitness=best_ce)
		logger(f"\nBest genome saved to {genome_path}")

	# ── Final Summary ────────────────────────────────────────────────────
	logger(f"\n{'='*82}")
	logger(f"  FINAL RESULTS (All metrics on FULL validation data)")
	logger(f"{'='*82}")
	print_phase_comparison(phase_metrics_list)

	all_results["phase_metrics"] = [asdict(pm) for pm in phase_metrics_list]
	save()
	logger(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
