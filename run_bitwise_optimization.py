#!/usr/bin/env python3
"""
Bitwise Architecture + Connectivity Optimization Pilot for BitwiseRAMLM.

Four phases:
  Phase 1: Grid search over neurons × bits (quick landscape scan)
  Phase 2: GA architecture optimization — refine (bits, neurons) around Phase 1 best
  Phase 3: GA connectivity optimization — optimize which input bits each neuron observes
  Phase 4: TS connectivity refinement — local search from Phase 3's best

BitwiseRAMLM has 16 uniform clusters (one per output bit for 50K vocab).
All clusters share the same (bits, neurons) config, so Phase 2 uses a custom
2D GA over two integers rather than ArchitectureGAStrategy (which mutates
per-cluster independently and would break the uniformity constraint).

Phases 3-4 use the existing ArchitectureGA/TSStrategy + BitwiseEvaluator
for Rust+Metal batch evaluation of connection genomes.

Usage:
	python run_bitwise_optimization.py --context 4 --rate 0.25
	python run_bitwise_optimization.py --arch-gens 30 --conn-ga-gens 50 --conn-ts-iters 50
	python run_bitwise_optimization.py --phase 3 --phase2-input experiments/bitwise_optimization_results.json
"""

import argparse
import json
import random
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

	top_configs = [(r["neurons"], r["bits"]) for r in results[:top_k]]
	print(f"\nTop-{top_k} for Phase 2: {top_configs}")

	return results, top_configs


# ---------------------------------------------------------------------------
# Phase 2: GA architecture optimization (bits + neurons)
# ---------------------------------------------------------------------------

def _evaluate_arch_genome(bits, neurons, train_tokens, test_tokens, vocab_size, context_size, rate):
	"""Evaluate a single (bits, neurons) config with random connections."""
	from wnn.ram.core.models import BitwiseRAMLM

	model = BitwiseRAMLM(
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=neurons,
		bits_per_neuron=bits,
		memory_mode=2,
		neuron_sample_rate=rate,
	)
	_, eval_stats = model.train_and_eval_metal_split(
		train_tokens, test_tokens,
		verbose=False,
		per_bit=False,
	)
	return eval_stats["cross_entropy"], eval_stats["accuracy"]


def phase2_architecture_ga(
	train_tokens, test_tokens, vocab_size, context_size, rate,
	top_configs, arch_gens, arch_pop, patience,
	min_bits=10, max_bits=24, min_neurons=20, max_neurons=300,
):
	"""
	GA optimization over the 2D (bits, neurons) space.

	BitwiseRAMLM has 16 uniform clusters so bits and neurons are scalars,
	not per-cluster vectors. A simple custom GA is cleaner than forcing
	ArchitectureGAStrategy to keep all clusters identical.
	"""
	print(f"\n{'='*70}")
	print(f"Phase 2: GA Architecture Optimization (bits + neurons)")
	print(f"  population={arch_pop}, generations={arch_gens}, patience={patience}")
	print(f"  bits=[{min_bits},{max_bits}], neurons=[{min_neurons},{max_neurons}]")
	print(f"{'='*70}")

	# Initialize population around Phase 1 top configs
	population = []
	# Seed with top configs
	for n, b in top_configs:
		population.append((b, n))
	# Fill rest with random mutations of top configs + random
	while len(population) < arch_pop:
		if random.random() < 0.7 and top_configs:
			# Mutate a top config
			base_n, base_b = random.choice(top_configs)
			b = max(min_bits, min(max_bits, base_b + random.randint(-3, 3)))
			n = max(min_neurons, min(max_neurons, base_n + random.choice([-30, -20, -10, 0, 10, 20, 30])))
		else:
			# Random
			b = random.randint(min_bits, max_bits)
			n = random.randint(min_neurons, max_neurons)
		population.append((b, n))

	# Evaluate initial population
	def eval_pop(pop):
		results = []
		for b, n in pop:
			ce, acc = _evaluate_arch_genome(b, n, train_tokens, test_tokens, vocab_size, context_size, rate)
			results.append((ce, acc))
		return results

	fitness = eval_pop(population)
	best_ce = min(f[0] for f in fitness)
	best_idx = [f[0] for f in fitness].index(best_ce)
	best_genome = population[best_idx]
	best_acc = fitness[best_idx][1]

	print(f"  Initial best: b={best_genome[0]}, n={best_genome[1]}, CE={best_ce:.4f}, Acc={best_acc:.2%}")

	history = [(0, best_ce)]
	no_improve = 0
	elitism_count = max(2, arch_pop // 5)

	for gen in range(1, arch_gens + 1):
		t0 = time.time()

		# Sort by CE (lower = better)
		ranked = sorted(zip(population, fitness), key=lambda x: x[1][0])

		# Elitism: keep top genomes
		new_pop = [g for g, _ in ranked[:elitism_count]]

		# Generate offspring
		while len(new_pop) < arch_pop:
			# Tournament selection (size 3)
			candidates = random.sample(list(range(len(ranked))), min(3, len(ranked)))
			parent1_idx = min(candidates)  # lower index = better fitness
			candidates = random.sample(list(range(len(ranked))), min(3, len(ranked)))
			parent2_idx = min(candidates)

			p1_b, p1_n = ranked[parent1_idx][0]
			p2_b, p2_n = ranked[parent2_idx][0]

			# Crossover
			if random.random() < 0.7:
				child_b = random.choice([p1_b, p2_b])
				child_n = random.choice([p1_n, p2_n])
			else:
				child_b, child_n = p1_b, p1_n

			# Mutation
			if random.random() < 0.3:
				child_b += random.randint(-2, 2)
				child_b = max(min_bits, min(max_bits, child_b))
			if random.random() < 0.3:
				child_n += random.choice([-20, -10, -5, 0, 5, 10, 20])
				child_n = max(min_neurons, min(max_neurons, child_n))

			new_pop.append((child_b, child_n))

		population = new_pop
		fitness = eval_pop(population)
		elapsed = time.time() - t0

		gen_best_ce = min(f[0] for f in fitness)
		gen_best_idx = [f[0] for f in fitness].index(gen_best_ce)
		gen_best = population[gen_best_idx]
		gen_best_acc = fitness[gen_best_idx][1]

		if gen_best_ce < best_ce:
			best_ce = gen_best_ce
			best_genome = gen_best
			best_acc = gen_best_acc
			no_improve = 0
		else:
			no_improve += 1

		history.append((gen, best_ce))
		print(f"  [Gen {gen:02d}/{arch_gens}] best: b={best_genome[0]}, n={best_genome[1]}, "
			  f"CE={best_ce:.4f}, Acc={best_acc:.2%} "
			  f"(gen_best: b={gen_best[0]}, n={gen_best[1]}, CE={gen_best_ce:.4f}) "
			  f"[{elapsed:.1f}s]")

		if no_improve >= patience:
			print(f"  Early stop: {patience} generations without improvement")
			break

	best_bits, best_neurons = best_genome
	print(f"\n  Phase 2 Result: bits={best_bits}, neurons={best_neurons}, "
		  f"CE={best_ce:.4f}, Acc={best_acc:.2%}")

	return {
		"best_bits": best_bits,
		"best_neurons": best_neurons,
		"best_ce": best_ce,
		"best_accuracy": best_acc,
		"generations_run": gen if 'gen' in dir() else 0,
		"history": history,
		"final_population": [(b, n, ce, acc) for (b, n), (ce, acc) in zip(population, fitness)],
	}


# ---------------------------------------------------------------------------
# Phase 3: GA connectivity optimization
# ---------------------------------------------------------------------------

def phase3_ga_connections(
	train_tokens, test_tokens, vocab_size, context_size, rate,
	best_bits, best_neurons, conn_ga_gens, population, patience,
):
	"""GA connectivity optimization with fixed (bits, neurons) from Phase 2."""
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
	from wnn.ram.strategies.connectivity.architecture_strategies import (
		ArchitectureConfig, ArchitectureGAStrategy,
	)
	from wnn.ram.strategies.connectivity.generic_strategies import GAConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	num_clusters = bits_needed(vocab_size)  # 16 for GPT-2
	total_input_bits = context_size * bits_needed(vocab_size)

	print(f"\n{'='*70}")
	print(f"Phase 3: GA Connectivity Optimization")
	print(f"  Fixed architecture: bits={best_bits}, neurons={best_neurons}")
	print(f"  population={population}, generations={conn_ga_gens}, patience={patience}")
	print(f"{'='*70}")

	evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=best_neurons,
		bits_per_neuron=best_bits,
		num_parts=3,
		memory_mode=2,
		neuron_sample_rate=rate,
	)

	arch_config = ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=False,
		optimize_neurons=False,
		optimize_connections=True,
		default_bits=best_bits,
		default_neurons=best_neurons,
		total_input_bits=total_input_bits,
	)

	ga_config = GAConfig(
		population_size=population,
		generations=conn_ga_gens,
		mutation_rate=0.1,
		crossover_rate=0.7,
		tournament_size=3,
		elitism_pct=0.1,
		patience=patience,
		check_interval=1,
	)

	seed_genome = ClusterGenome(
		bits_per_cluster=[best_bits] * num_clusters,
		neurons_per_cluster=[best_neurons] * num_clusters,
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
		evaluate_fn=lambda g: 999.0,
		initial_genome=seed_genome,
		batch_evaluate_fn=lambda genomes: evaluator.evaluate_batch(genomes),
	)
	elapsed = time.time() - t0

	print(f"\n  GA Connections Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		print(f"  Accuracy: {result.final_accuracy:.2%}")
	print(f"  Improvement: {result.improvement_percent:.1f}%")
	print(f"  Generations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	print(f"  Stop reason: {result.stop_reason}")

	return {
		"bits": best_bits,
		"neurons": best_neurons,
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
	}


# ---------------------------------------------------------------------------
# Phase 4: TS connectivity refinement
# ---------------------------------------------------------------------------

def phase4_ts_connections(
	train_tokens, test_tokens, vocab_size, context_size, rate,
	ga_result, ts_iters, neighbors, patience,
):
	"""TS connectivity refinement starting from Phase 3 GA's best."""
	from wnn.ram.core.RAMClusterLayer import bits_needed
	from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator
	from wnn.ram.strategies.connectivity.architecture_strategies import (
		ArchitectureConfig, ArchitectureTSStrategy,
	)
	from wnn.ram.strategies.connectivity.generic_strategies import TSConfig
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	num_clusters = bits_needed(vocab_size)
	total_input_bits = context_size * bits_needed(vocab_size)

	best_bits = ga_result["bits"]
	best_neurons = ga_result["neurons"]
	ga_ce = ga_result["final_ce"]

	print(f"\n{'='*70}")
	print(f"Phase 4: TS Connectivity Refinement")
	print(f"  Fixed architecture: bits={best_bits}, neurons={best_neurons}")
	print(f"  GA baseline CE={ga_ce:.4f}")
	print(f"  neighbors={neighbors}, iterations={ts_iters}, patience={patience}")
	print(f"{'='*70}")

	evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=context_size,
		neurons_per_cluster=best_neurons,
		bits_per_neuron=best_bits,
		num_parts=3,
		memory_mode=2,
		neuron_sample_rate=rate,
	)

	arch_config = ArchitectureConfig(
		num_clusters=num_clusters,
		optimize_bits=False,
		optimize_neurons=False,
		optimize_connections=True,
		default_bits=best_bits,
		default_neurons=best_neurons,
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

	seed_genome = ClusterGenome(
		bits_per_cluster=[best_bits] * num_clusters,
		neurons_per_cluster=[best_neurons] * num_clusters,
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

	print(f"\n  TS Connections Result: CE={result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		print(f"  Accuracy: {result.final_accuracy:.2%}")
	improvement_over_ga = (ga_ce - result.final_fitness) / ga_ce * 100
	print(f"  Improvement over GA: {improvement_over_ga:.1f}%")
	print(f"  Iterations: {result.iterations_run}, Elapsed: {elapsed:.0f}s")
	print(f"  Stop reason: {result.stop_reason}")

	return {
		"bits": best_bits,
		"neurons": best_neurons,
		"ga_ce": ga_ce,
		"final_ce": result.final_fitness,
		"final_accuracy": result.final_accuracy,
		"improvement_over_ga_pct": round(improvement_over_ga, 2),
		"iterations_run": result.iterations_run,
		"stop_reason": str(result.stop_reason),
		"elapsed_s": round(elapsed, 1),
		"history": result.history,
		"best_genome_connections": result.best_genome.connections,
	}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(phase1_results, arch_result, ga_conn_result, ts_conn_result):
	"""Print end-to-end optimization summary."""
	print(f"\n{'='*70}")
	print(f"Optimization Summary")
	print(f"{'='*70}")

	# Phase 1 baseline: best grid search CE
	if phase1_results:
		grid_best = min(phase1_results, key=lambda r: r["cross_entropy"])
		print(f"Phase 1 (grid):   n={grid_best['neurons']:3d}, b={grid_best['bits']:2d}, "
			  f"CE={grid_best['cross_entropy']:.4f}  (random connections)")

	# Phase 2: architecture optimization
	if arch_result:
		print(f"Phase 2 (arch GA): n={arch_result['best_neurons']:3d}, b={arch_result['best_bits']:2d}, "
			  f"CE={arch_result['best_ce']:.4f}  (random connections)")

	# Phase 3: GA connections
	if ga_conn_result:
		print(f"Phase 3 (conn GA): n={ga_conn_result['neurons']:3d}, b={ga_conn_result['bits']:2d}, "
			  f"CE={ga_conn_result['final_ce']:.4f}  (optimized connections)")

	# Phase 4: TS connections
	if ts_conn_result:
		print(f"Phase 4 (conn TS): n={ts_conn_result['neurons']:3d}, b={ts_conn_result['bits']:2d}, "
			  f"CE={ts_conn_result['final_ce']:.4f}  (refined connections)")

	# Overall improvement
	if phase1_results and ts_conn_result:
		baseline = grid_best["cross_entropy"]
		final = ts_conn_result["final_ce"]
		improvement = (baseline - final) / baseline * 100
		print(f"\nOverall: CE {baseline:.4f} → {final:.4f}  ({improvement:+.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(description="BitwiseRAMLM Architecture + Connectivity Optimization")
	# General
	parser.add_argument("--context", type=int, default=4, help="Context window size (default: 4)")
	parser.add_argument("--rate", type=float, default=0.25, help="Neuron sample rate (default: 0.25)")
	parser.add_argument("--phase", type=str, default="all",
						help="Phase to run: 1, 2, 3, 4, all (default: all)")
	parser.add_argument("--output", type=str, default="experiments/bitwise_optimization_results.json",
						help="Output JSON file")
	# Phase 1
	parser.add_argument("--top-k", type=int, default=3,
						help="Top configs from Phase 1 grid search (default: 3)")
	# Phase 2: architecture GA
	parser.add_argument("--arch-gens", type=int, default=20,
						help="Architecture GA generations (default: 20)")
	parser.add_argument("--arch-pop", type=int, default=20,
						help="Architecture GA population (default: 20)")
	parser.add_argument("--arch-patience", type=int, default=5,
						help="Architecture GA patience (default: 5)")
	# Phase 3: connectivity GA
	parser.add_argument("--conn-ga-gens", type=int, default=50,
						help="Connectivity GA generations (default: 50)")
	parser.add_argument("--conn-population", type=int, default=30,
						help="Connectivity GA population (default: 30)")
	parser.add_argument("--conn-patience", type=int, default=2,
						help="Connectivity GA/TS patience (default: 2)")
	# Phase 4: connectivity TS
	parser.add_argument("--conn-ts-iters", type=int, default=50,
						help="Connectivity TS iterations (default: 50)")
	parser.add_argument("--conn-neighbors", type=int, default=30,
						help="Connectivity TS neighbors per iteration (default: 30)")
	# Resume from previous run
	parser.add_argument("--phase2-input", type=str, default=None,
						help="Load Phase 2 result from previous JSON output")
	args = parser.parse_args()

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	train_tokens, test_tokens, vocab_size = load_wikitext2_tokens()

	all_results = {
		"config": {
			"context_size": args.context,
			"neuron_sample_rate": args.rate,
			"memory_mode": "QUAD_WEIGHTED",
			"vocab_size": vocab_size,
			"arch_gens": args.arch_gens,
			"arch_pop": args.arch_pop,
			"conn_ga_gens": args.conn_ga_gens,
			"conn_population": args.conn_population,
			"conn_ts_iters": args.conn_ts_iters,
			"conn_neighbors": args.conn_neighbors,
		},
		"phase1_grid": None,
		"phase2_arch_ga": None,
		"phase3_conn_ga": None,
		"phase4_conn_ts": None,
	}

	run_phase = lambda p: args.phase in ("all", str(p))
	phase1_results = []
	arch_result = None
	ga_conn_result = None
	ts_conn_result = None

	def save():
		with open(args.output, "w") as f:
			json.dump(all_results, f, indent=2)

	# ── Phase 1: Grid search ─────────────────────────────────────────────
	if run_phase(1):
		t0 = time.time()
		phase1_results, top_configs = phase1_grid_search(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate, top_k=args.top_k,
		)
		all_results["phase1_grid"] = {
			"results": phase1_results,
			"top_configs": [{"neurons": n, "bits": b} for n, b in top_configs],
			"elapsed_s": round(time.time() - t0, 1),
		}
		save()
		print(f"\nPhase 1 saved to {args.output}")
	else:
		# Try loading from previous output
		if output_path.exists():
			with open(output_path) as f:
				prev = json.load(f)
			if prev.get("phase1_grid"):
				phase1_results = prev["phase1_grid"]["results"]
				top_configs = [(c["neurons"], c["bits"]) for c in prev["phase1_grid"]["top_configs"]]
				all_results["phase1_grid"] = prev["phase1_grid"]
				print(f"Loaded Phase 1 from {args.output}: top={top_configs}")
			else:
				top_configs = [(150, 20), (100, 20), (150, 18)]
				print(f"Using default top configs: {top_configs}")
		else:
			top_configs = [(150, 20), (100, 20), (150, 18)]
			print(f"Using default top configs: {top_configs}")

	# ── Phase 2: Architecture GA (bits + neurons) ────────────────────────
	if run_phase(2):
		t0 = time.time()
		arch_result = phase2_architecture_ga(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate,
			top_configs=top_configs,
			arch_gens=args.arch_gens, arch_pop=args.arch_pop,
			patience=args.arch_patience,
		)
		all_results["phase2_arch_ga"] = arch_result
		all_results["phase2_arch_ga"]["elapsed_s"] = round(time.time() - t0, 1)
		save()
		print(f"\nPhase 2 saved to {args.output}")
	else:
		# Try loading from previous output or --phase2-input
		source = args.phase2_input or (str(output_path) if output_path.exists() else None)
		if source:
			with open(source) as f:
				prev = json.load(f)
			if prev.get("phase2_arch_ga"):
				arch_result = prev["phase2_arch_ga"]
				all_results["phase2_arch_ga"] = arch_result
				print(f"Loaded Phase 2: bits={arch_result['best_bits']}, neurons={arch_result['best_neurons']}, "
					  f"CE={arch_result['best_ce']:.4f}")

	# ── Phase 3: Connectivity GA ─────────────────────────────────────────
	if run_phase(3):
		if arch_result is None:
			# Fall back to Phase 1 best
			if phase1_results:
				best_p1 = min(phase1_results, key=lambda r: r["cross_entropy"])
				best_bits, best_neurons = best_p1["bits"], best_p1["neurons"]
			else:
				best_bits, best_neurons = 20, 150
			print(f"No Phase 2 result, using: bits={best_bits}, neurons={best_neurons}")
		else:
			best_bits = arch_result["best_bits"]
			best_neurons = arch_result["best_neurons"]

		t0 = time.time()
		ga_conn_result = phase3_ga_connections(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate,
			best_bits=best_bits, best_neurons=best_neurons,
			conn_ga_gens=args.conn_ga_gens,
			population=args.conn_population,
			patience=args.conn_patience,
		)
		# Don't serialize huge connection lists in the main JSON
		all_results["phase3_conn_ga"] = {
			k: v for k, v in ga_conn_result.items() if k != "best_genome_connections"
		}
		save()
		print(f"\nPhase 3 saved to {args.output}")

	# ── Phase 4: Connectivity TS ─────────────────────────────────────────
	if run_phase(4) and ga_conn_result is not None:
		t0 = time.time()
		ts_conn_result = phase4_ts_connections(
			train_tokens, test_tokens, vocab_size,
			context_size=args.context, rate=args.rate,
			ga_result=ga_conn_result,
			ts_iters=args.conn_ts_iters,
			neighbors=args.conn_neighbors,
			patience=args.conn_patience,
		)
		all_results["phase4_conn_ts"] = {
			k: v for k, v in ts_conn_result.items() if k != "best_genome_connections"
		}

		# Save best genome separately
		genome_path = output_path.with_suffix(".best_genome.json")
		with open(genome_path, "w") as f:
			json.dump({
				"neurons": ts_conn_result["neurons"],
				"bits": ts_conn_result["bits"],
				"final_ce": ts_conn_result["final_ce"],
				"final_accuracy": ts_conn_result["final_accuracy"],
				"connections": ts_conn_result["best_genome_connections"],
			}, f)
		print(f"\nBest genome saved to {genome_path}")

	# ── Summary ──────────────────────────────────────────────────────────
	print_summary(phase1_results, arch_result, ga_conn_result, ts_conn_result)

	save()
	print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
	main()
