"""IDS training runner — WNN classifier for UNSW-NB15.

Usage:
	python -m wnn.ids.train --classification binary
	python -m wnn.ids.train --classification multi

Ten optimization phases (GA → Adapt → TS per dimension):
	0.  Grid search    — evaluate neuron × bit combos to find best seed
	1a. GA neurons     — explore neuron counts per class
	1b. Neurogenesis   — stats-guided neuron add/remove (planned, needs Rust hooks)
	1c. TS neurons     — refine neuron counts
	2a. GA bits        — explore address bits per neuron
	2b. Synaptogenesis — stats-guided bit grow/prune (planned, needs Rust hooks)
	2c. TS bits        — refine address bits
	3a. GA connections — explore input wiring
	3b. Axonogenesis   — stats-guided connection rewiring (planned, needs Rust hooks)
	3c. TS connections — refine input wiring

Currently runs 0 → 1a → 1c → 2a → 2c → 3a → 3c (7 phases).
Adaptation phases (1b, 2b, 3b) require Rust-side training stats in IDSEvaluator.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from wnn.ids import load_unsw_nb15
from wnn.ids.metrics import compute_ids_metrics, format_ids_report
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome, PhaseType
from wnn.ram.strategies.connectivity.architecture_strategies import (
	ArchitectureConfig,
	ArchitectureGAStrategy,
	ArchitectureTSStrategy,
	GridSearchConfig,
	GridSearchStrategy,
)
from wnn.ram.strategies.connectivity.generic_strategies import GAConfig, TSConfig
from wnn.ram.fitness import FitnessCalculatorType


@dataclass
class IDSTrainConfig:
	classification: str = "binary"
	n_bits: int = 8
	# Initial genome defaults (overridden by grid search or asymmetric allocation)
	neurons: int = 10
	address_bits: int = 12
	# Optimization
	ga_gens: int = 250
	ts_iters: int = 250
	patience: int = 5
	population: int = 150
	neighbors: int = 150
	num_parts: int = 3
	seed: int = 42
	# Grid search
	neurons_grid: list[int] = field(default_factory=lambda: [3, 5, 8, 10, 15, 20])
	bits_grid: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 14])
	grid_top_k: int = 10
	# Fitness: accuracy-dominant for IDS (CE is just a training signal)
	fitness_weight_ce: float = 0.3
	fitness_weight_acc: float = 1.0
	output: Optional[str] = None


def create_asymmetric_genome(
	y_train: np.ndarray,
	num_classes: int,
	total_features: int,
	seed: int = 42,
) -> ClusterGenome:
	"""Create a genome with per-class bits/neurons matched to training density.

	Address space (bits) is chosen so each neuron sees >= 1 visit per slot
	on average. More neurons for classes with more data (more patterns to learn).

	Density rule: visits_per_slot = examples / (2^bits)
	We want visits_per_slot >= min_density (default 2.0).
	So: bits <= log2(examples / min_density)
	"""
	min_density = 2.0  # Minimum visits per address slot for meaningful learning
	min_bits = 4
	max_bits = 14
	min_neurons = 3
	max_neurons = 20

	# Count examples per class
	class_counts = []
	for c in range(num_classes):
		class_counts.append(int((y_train == c).sum()))

	bits_per_class = []
	neurons_per_class = []

	for c in range(num_classes):
		count = class_counts[c]

		# Bits: largest address space where density >= min_density
		if count > 0:
			ideal_bits = int(math.log2(count / min_density))
			bits = max(min_bits, min(max_bits, ideal_bits))
		else:
			bits = min_bits

		# Neurons: proportional to sqrt(count), scaled to [min_neurons, max_neurons]
		# sqrt because doubling data doesn't need doubling neurons
		if count > 0:
			raw = math.sqrt(count)
		else:
			raw = 0
		bits_per_class.append(bits)
		neurons_per_class.append(raw)

	# Normalize neurons to [min_neurons, max_neurons]
	raw_values = neurons_per_class
	if max(raw_values) > 0:
		scale = max_neurons / max(raw_values)
		neurons_per_class = [
			max(min_neurons, min(max_neurons, int(round(r * scale))))
			for r in raw_values
		]
	else:
		neurons_per_class = [min_neurons] * num_classes

	# Build per-neuron bits list
	all_bits = []
	for c in range(num_classes):
		all_bits.extend([bits_per_class[c]] * neurons_per_class[c])

	genome = ClusterGenome(
		bits_per_neuron=all_bits,
		neurons_per_cluster=neurons_per_class,
		connections=None,
	)

	# Initialize random connections
	import random
	rng = random.Random(seed)
	conns = []
	for b in all_bits:
		conns.extend(rng.sample(range(total_features), min(b, total_features)))
	genome.connections = conns

	# Log the allocation
	print("Asymmetric genome allocation:")
	for c in range(num_classes):
		print(f"  Class {c:2d} ({class_counts[c]:>7,} examples): "
			  f"{bits_per_class[c]:2d} bits, {neurons_per_class[c]:2d} neurons "
			  f"(density: {class_counts[c] / 2**bits_per_class[c]:.1f}/slot)")

	return genome


def run_ids_experiment(cfg: IDSTrainConfig) -> dict:
	"""Run full IDS training experiment."""
	print(f"=== WNN IDS Classifier ({cfg.classification}) ===")
	print(f"Config: pop={cfg.population}, GA={cfg.ga_gens}gen, TS={cfg.ts_iters}iter, "
		  f"patience={cfg.patience}, fitness_weights=(CE={cfg.fitness_weight_ce}, Acc={cfg.fitness_weight_acc})")

	# Load dataset
	t0 = time.time()
	dataset = load_unsw_nb15(n_bits=cfg.n_bits)
	total_features = dataset.X_train.shape[1]
	print(f"Dataset: {dataset.X_train.shape[0]:,} train, {dataset.X_test.shape[0]:,} test, "
		  f"{total_features} features ({time.time()-t0:.1f}s)")

	if cfg.classification == "binary":
		num_classes = 2
		class_names = ["Normal", "Attack"]
		y_train = dataset.y_train_binary
	else:
		num_classes = len(dataset.category_names)
		class_names = dataset.category_names
		y_train = dataset.y_train_multi

	# Create evaluator
	t0 = time.time()
	evaluator = IDSEvaluator(
		dataset=dataset,
		classification=cfg.classification,
		num_parts=cfg.num_parts,
		seed=cfg.seed,
	)
	print(f"Evaluator: {repr(evaluator)} ({time.time()-t0:.1f}s)")

	# ── Phase 0: Grid search ─────────────────────────────────────────
	grid_config = GridSearchConfig(
		num_clusters=num_classes,
		neurons_grid=cfg.neurons_grid,
		bits_grid=cfg.bits_grid,
		top_k=cfg.grid_top_k,
		population_size=cfg.population,
		total_input_bits=total_features,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=cfg.fitness_weight_ce,
		fitness_weight_acc=cfg.fitness_weight_acc,
	)
	grid_strategy = GridSearchStrategy(
		config=grid_config,
		batch_evaluator=evaluator,
		seed=cfg.seed,
		logger=print,
	)

	t0 = time.time()
	grid_result = grid_strategy.optimize()
	grid_time = time.time() - t0
	print(f"Grid search: CE={grid_result.final_fitness:.4f}, "
		  f"Acc={grid_result.final_accuracy*100:.2f}% "
		  f"({grid_time:.1f}s)")

	best_genome = grid_result.best_genome
	best_ce = grid_result.final_fitness
	best_acc = grid_result.final_accuracy or 0.0
	seed_population = grid_result.final_population

	# For multi-class: override with asymmetric genome if it's better
	if cfg.classification == "multi":
		asym_genome = create_asymmetric_genome(
			y_train, num_classes, total_features, seed=cfg.seed,
		)
		asym_result = evaluator.evaluate_batch_full([asym_genome])[0]
		print(f"Asymmetric genome: CE={asym_result.ce:.4f}, Acc={asym_result.accuracy*100:.2f}%")
		if asym_result.ce < best_ce:
			best_genome = asym_genome
			best_ce = asym_result.ce
			best_acc = asym_result.accuracy
			print("  → Using asymmetric genome (better than grid search)")
		else:
			print("  → Keeping grid search genome (better than asymmetric)")

	print(f"Total: {best_genome.total_neurons} neurons, {best_genome.total_memory_cells():,} memory cells")

	# ── Phases 1-3: GA + TS (neurons → bits → connections) ───────────

	# Phase 1: Optimize neurons (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"neurons", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=True, optimize_connections=False,
		num_classes=num_classes, total_features=total_features,
		seed_population=seed_population,
	)

	# Phase 2: Optimize bits (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"bits", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=True, optimize_neurons=False, optimize_connections=False,
		num_classes=num_classes, total_features=total_features,
	)

	# Phase 3: Optimize connections (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"connections", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=False, optimize_connections=True,
		num_classes=num_classes, total_features=total_features,
	)

	# ── Final evaluation ──────────────────────────────────────────────
	print("\n=== Final Evaluation ===")
	final_result = evaluator.evaluate_batch_full([best_genome])[0]
	print(f"Final: CE={final_result.ce:.4f}, Acc={final_result.accuracy*100:.2f}%")
	print(f"Genome: neurons={best_genome.neurons_per_cluster}, "
		  f"bits={best_genome.bits_per_neuron[:10]}{'...' if len(best_genome.bits_per_neuron) > 10 else ''}")

	metrics = {
		"classification": cfg.classification,
		"num_classes": num_classes,
		"total_features": total_features,
		"ce": final_result.ce,
		"accuracy": final_result.accuracy,
		"genome": {
			"bits_per_neuron": best_genome.bits_per_neuron,
			"neurons_per_cluster": best_genome.neurons_per_cluster,
			"total_neurons": best_genome.total_neurons,
			"total_memory_cells": best_genome.total_memory_cells(),
		},
		"config": {
			"n_bits": cfg.n_bits,
			"neurons": cfg.neurons,
			"address_bits": cfg.address_bits,
			"ga_gens": cfg.ga_gens,
			"ts_iters": cfg.ts_iters,
			"patience": cfg.patience,
			"population": cfg.population,
			"seed": cfg.seed,
			"fitness_weight_ce": cfg.fitness_weight_ce,
			"fitness_weight_acc": cfg.fitness_weight_acc,
		},
	}

	if cfg.output:
		output_path = Path(cfg.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(metrics, indent=2))
		print(f"\nResults saved to {cfg.output}")

	return metrics


def _run_phase(
	phase_name: str,
	evaluator: IDSEvaluator,
	genome: ClusterGenome,
	best_ce: float,
	best_acc: float,
	cfg: IDSTrainConfig,
	optimize_bits: bool,
	optimize_neurons: bool,
	optimize_connections: bool,
	num_classes: int,
	total_features: int,
	seed_population: list[ClusterGenome] | None = None,
) -> tuple[ClusterGenome, float, float]:
	"""Run GA + TS for a single optimization phase."""
	print(f"\n--- Phase: {phase_name} ---")

	arch_config = ArchitectureConfig(
		num_clusters=num_classes,
		min_bits=4,
		max_bits=24,
		min_neurons=3,
		max_neurons=30,
		optimize_bits=optimize_bits,
		optimize_neurons=optimize_neurons,
		optimize_connections=optimize_connections,
		default_bits=cfg.address_bits,
		default_neurons=cfg.neurons,
		total_input_bits=total_features,
	)

	# GA phase — accuracy-dominant fitness for IDS
	ga_config = GAConfig(
		population_size=cfg.population,
		generations=cfg.ga_gens,
		patience=cfg.patience,
		mutation_rate=0.1,
		crossover_rate=0.7,
		tournament_size=3,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=cfg.fitness_weight_ce,
		fitness_weight_acc=cfg.fitness_weight_acc,
	)

	ga_strategy = ArchitectureGAStrategy(
		arch_config=arch_config,
		ga_config=ga_config,
		seed=cfg.seed,
		logger=print,
		cached_evaluator=evaluator,
	)

	# Seed GA with grid search population if available, otherwise just the genome
	initial_pop = seed_population if seed_population else [genome]

	t0 = time.time()
	ga_result = ga_strategy.optimize(initial_population=initial_pop)
	ga_time = time.time() - t0
	print(f"  GA {phase_name}: CE {best_ce:.4f} -> {ga_result.final_fitness:.4f}, "
		  f"Acc {(ga_result.final_accuracy or 0)*100:.2f}% "
		  f"({ga_result.iterations_run} gens, {ga_time:.1f}s)")

	best_genome = ga_result.best_genome
	best_ce = ga_result.final_fitness
	best_acc = ga_result.final_accuracy or best_acc

	# TS phase — same accuracy-dominant fitness
	ts_config = TSConfig(
		iterations=cfg.ts_iters,
		neighbors_per_iter=cfg.neighbors,
		total_neighbors_size=cfg.neighbors,
		patience=cfg.patience,
		mutation_rate=0.1,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=cfg.fitness_weight_ce,
		fitness_weight_acc=cfg.fitness_weight_acc,
	)

	ts_strategy = ArchitectureTSStrategy(
		arch_config=arch_config,
		ts_config=ts_config,
		seed=cfg.seed + 1000,
		logger=print,
		cached_evaluator=evaluator,
	)

	t0 = time.time()
	ts_result = ts_strategy.optimize(
		initial_genome=best_genome,
		initial_fitness=best_ce,
		initial_neighbors=ga_result.final_population,
	)
	ts_time = time.time() - t0
	print(f"  TS {phase_name}: CE {best_ce:.4f} -> {ts_result.final_fitness:.4f}, "
		  f"Acc {(ts_result.final_accuracy or 0)*100:.2f}% "
		  f"({ts_result.iterations_run} iters, {ts_time:.1f}s)")

	if ts_result.final_fitness < best_ce:
		best_genome = ts_result.best_genome
		best_ce = ts_result.final_fitness
		best_acc = ts_result.final_accuracy or best_acc

	return best_genome, best_ce, best_acc


def main():
	parser = argparse.ArgumentParser(description="WNN IDS Classifier Training")
	parser.add_argument("--classification", choices=["binary", "multi"], default="binary")
	parser.add_argument("--n-bits", type=int, default=8, help="Thermometer encoding bits")
	parser.add_argument("--neurons", type=int, default=10, help="Initial neurons per class (binary)")
	parser.add_argument("--bits", type=int, default=12, help="Initial address bits per neuron (binary)")
	parser.add_argument("--ga-gens", type=int, default=250)
	parser.add_argument("--ts-iters", type=int, default=250)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--population", type=int, default=150)
	parser.add_argument("--neighbors", type=int, default=150)
	parser.add_argument("--num-parts", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--fitness-ce-weight", type=float, default=0.3,
		help="CE weight in fitness ranking (lower = more accuracy-focused)")
	parser.add_argument("--fitness-acc-weight", type=float, default=1.0,
		help="Accuracy weight in fitness ranking")
	parser.add_argument("--output", type=str, default=None, help="JSON output path")
	args = parser.parse_args()

	cfg = IDSTrainConfig(
		classification=args.classification,
		n_bits=args.n_bits,
		neurons=args.neurons,
		address_bits=args.bits,
		ga_gens=args.ga_gens,
		ts_iters=args.ts_iters,
		patience=args.patience,
		population=args.population,
		neighbors=args.neighbors,
		num_parts=args.num_parts,
		seed=args.seed,
		fitness_weight_ce=args.fitness_ce_weight,
		fitness_weight_acc=args.fitness_acc_weight,
		output=args.output,
	)

	run_ids_experiment(cfg)


if __name__ == "__main__":
	main()
