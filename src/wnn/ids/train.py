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

from wnn.ids import load_unsw_nb15, create_attack_only_dataset, split_train_validation
from wnn.ids.bitwise import create_bitwise_codebook, create_per_bit_dataset
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
	fitness_weight_f1: float = 0.0  # F1-macro weight (0.0=off, >0=included in ranking)
	# Tunable knobs for FPR vs recall tradeoff (used in final reporting)
	beta: float = 1.0  # F-beta: <1 favors precision/low FPR, >1 favors recall
	class_weights: Optional[list[float]] = None  # Per-class importance (None=uniform)
	split: str = "standard"  # "standard" or "random"
	val_fraction: float = 0.25  # Validation holdout fraction from training data (0=disabled)
	bitwise_quick: bool = False  # Quick mode: grid search + connections only
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


class _OverfitControl:
	"""Simple control object returned by overfitting callback."""
	def __init__(self, early_stop: bool = False):
		self.early_stop = early_stop


def create_overfitting_callback(
	evaluator: IDSEvaluator,
	baseline_acc: float | None = None,
	patience: int = 3,
	logger=print,
):
	"""Create an overfitting detection callback for GA/TS optimization.

	Evaluates the current best genome on the FULL training + validation set
	(via evaluate_batch_full) and compares validation accuracy against the
	subset-rotation training accuracy. If validation accuracy drops below
	the baseline for `patience` consecutive checks, signals early stop.

	Args:
		evaluator: IDSEvaluator (with validation data in its "test" slot)
		baseline_acc: Initial validation accuracy to compare against (auto-set on first call)
		patience: How many consecutive degradations before stopping
		logger: Print function for logging

	Returns:
		Callable compatible with overfitting_callback(genome, fitness) -> control
	"""
	state = {"best_val_acc": baseline_acc, "degradation_count": 0, "call_count": 0}

	def callback(genome, train_fitness):
		state["call_count"] += 1
		# Evaluate on full train → validation
		result = evaluator.evaluate_batch_full([genome])[0]
		val_acc = result.accuracy
		val_ce = result.ce

		# Auto-set baseline on first call
		if state["best_val_acc"] is None:
			state["best_val_acc"] = val_acc

		# Compare: is validation improving?
		if val_acc >= state["best_val_acc"]:
			state["best_val_acc"] = val_acc
			state["degradation_count"] = 0
			logger(f"  [overfit check #{state['call_count']}] val_acc={val_acc*100:.2f}% "
				   f"(best={state['best_val_acc']*100:.2f}%) ✓")
		else:
			state["degradation_count"] += 1
			gap = state["best_val_acc"] - val_acc
			logger(f"  [overfit check #{state['call_count']}] val_acc={val_acc*100:.2f}% "
				   f"(best={state['best_val_acc']*100:.2f}%, gap={gap*100:.2f}%, "
				   f"degradation {state['degradation_count']}/{patience})")

		if state["degradation_count"] >= patience:
			logger(f"  [overfit check] EARLY STOP: validation degraded {patience} consecutive times")
			return _OverfitControl(early_stop=True)

		return _OverfitControl(early_stop=False)

	return callback


def run_ids_experiment(cfg: IDSTrainConfig) -> dict:
	"""Run full IDS training experiment."""
	print(f"=== WNN IDS Classifier ({cfg.classification}, split={cfg.split}) ===")
	f1_str = f", F1={cfg.fitness_weight_f1}" if cfg.fitness_weight_f1 > 0 else ""
	print(f"Config: pop={cfg.population}, GA={cfg.ga_gens}gen, TS={cfg.ts_iters}iter, "
		  f"patience={cfg.patience}, fitness_weights=(CE={cfg.fitness_weight_ce}, Acc={cfg.fitness_weight_acc}{f1_str})")

	# Load dataset
	t0 = time.time()
	full_dataset = load_unsw_nb15(n_bits=cfg.n_bits, split=cfg.split)
	total_features = full_dataset.X_train.shape[1]
	print(f"Dataset: {full_dataset.X_train.shape[0]:,} train, {full_dataset.X_test.shape[0]:,} test, "
		  f"{total_features} features ({time.time()-t0:.1f}s)")

	if cfg.classification == "binary":
		num_classes = 2
		class_names = ["Normal", "Attack"]
		y_train = full_dataset.y_train_binary
	else:
		num_classes = len(full_dataset.category_names)
		class_names = full_dataset.category_names
		y_train = full_dataset.y_train_multi

	# Validation holdout: split training data so optimizer never sees 82K test set
	overfitting_cb = None
	if cfg.val_fraction > 0:
		train_val_dataset, _ = split_train_validation(
			full_dataset, val_fraction=cfg.val_fraction, seed=cfg.seed,
		)
		opt_dataset = train_val_dataset  # optimization: 131K train (3-part rotation), 44K val as eval
		y_train = opt_dataset.y_train_binary if cfg.classification == "binary" else opt_dataset.y_train_multi
	else:
		opt_dataset = full_dataset  # no validation split — optimizer uses full 175K train + 82K test

	# Create optimizer evaluator (per-generation fitness: trains on subset of opt_dataset, evals on opt_dataset.X_test)
	t0 = time.time()
	evaluator = IDSEvaluator(
		dataset=opt_dataset,
		classification=cfg.classification,
		num_parts=cfg.num_parts,
		seed=cfg.seed,
	)
	print(f"Optimizer evaluator: {repr(evaluator)} ({time.time()-t0:.1f}s)")

	# Create test evaluator (175K train → 82K test) for overfitting checks + end-of-phase reporting
	t0 = time.time()
	test_evaluator = IDSEvaluator(
		dataset=full_dataset,
		classification=cfg.classification,
		num_parts=1,  # no subset rotation — always full 175K train → 82K test
		seed=cfg.seed,
	)
	print(f"Test evaluator: {repr(test_evaluator)} ({time.time()-t0:.1f}s)")

	# Overfitting callback: every 10 gens, evaluate best genome on 82K test set
	if cfg.val_fraction > 0:
		overfitting_cb = create_overfitting_callback(
			evaluator=test_evaluator,  # uses 175K train → 82K test
			patience=3,
			logger=print,
		)

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
		fitness_weight_f1=cfg.fitness_weight_f1,
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
		overfitting_callback=overfitting_cb,
		test_evaluator=test_evaluator,
	)

	# Phase 2: Optimize bits (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"bits", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=True, optimize_neurons=False, optimize_connections=False,
		num_classes=num_classes, total_features=total_features,
		overfitting_callback=overfitting_cb,
		test_evaluator=test_evaluator,
	)

	# Phase 3: Optimize connections (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"connections", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=False, optimize_connections=True,
		num_classes=num_classes, total_features=total_features,
		overfitting_callback=overfitting_cb,
		test_evaluator=test_evaluator,
	)

	# ── Final evaluation on 82K test set (175K train → 82K test) ────
	print("\n=== Final Evaluation (Test Set: 175K train → 82K test) ===")
	final_result = test_evaluator.evaluate_batch_full([best_genome])[0]
	print(f"Final: CE={final_result.ce:.4f}, Acc={final_result.accuracy*100:.2f}%"
		  + (f", F1={final_result.f1_macro:.4f}" if final_result.f1_macro is not None else ""))
	print(f"Genome: neurons={best_genome.neurons_per_cluster}, "
		  f"bits={best_genome.bits_per_neuron[:10]}{'...' if len(best_genome.bits_per_neuron) > 10 else ''}")

	# Per-example predictions for detailed IDS metrics (on 82K test set)
	predictions = test_evaluator.predict(best_genome)
	if cfg.classification == "binary":
		y_test = full_dataset.y_test_binary
	else:
		y_test = full_dataset.y_test_multi
	ids_metrics = compute_ids_metrics(
		y_test, predictions, num_classes,
		beta=cfg.beta, class_weights=cfg.class_weights,
	)
	print()
	print(format_ids_report(ids_metrics, class_names=class_names))

	metrics = {
		"classification": cfg.classification,
		"split": cfg.split,
		"num_classes": num_classes,
		"total_features": total_features,
		"ce": final_result.ce,
		"accuracy": final_result.accuracy,
		"f1_macro": ids_metrics["f1_macro"],
		"ids_metrics": ids_metrics,
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
			"fitness_weight_f1": cfg.fitness_weight_f1,
			"beta": cfg.beta,
			"val_fraction": cfg.val_fraction,
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
	overfitting_callback=None,
	test_evaluator: IDSEvaluator | None = None,
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
		fitness_weight_f1=cfg.fitness_weight_f1,
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
	ga_kwargs = {}
	if overfitting_callback is not None:
		ga_kwargs["overfitting_callback"] = overfitting_callback
	ga_result = ga_strategy.optimize(initial_population=initial_pop, **ga_kwargs)
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
		fitness_weight_f1=cfg.fitness_weight_f1,
	)

	ts_strategy = ArchitectureTSStrategy(
		arch_config=arch_config,
		ts_config=ts_config,
		seed=cfg.seed + 1000,
		logger=print,
		cached_evaluator=evaluator,
	)

	t0 = time.time()
	ts_kwargs = {}
	if overfitting_callback is not None:
		ts_kwargs["overfitting_callback"] = overfitting_callback
	ts_result = ts_strategy.optimize(
		initial_genome=best_genome,
		initial_fitness=best_ce,
		initial_neighbors=ga_result.final_population,
		**ts_kwargs,
	)
	ts_time = time.time() - t0
	print(f"  TS {phase_name}: CE {best_ce:.4f} -> {ts_result.final_fitness:.4f}, "
		  f"Acc {(ts_result.final_accuracy or 0)*100:.2f}% "
		  f"({ts_result.iterations_run} iters, {ts_time:.1f}s)")

	if ts_result.final_fitness < best_ce:
		best_genome = ts_result.best_genome
		best_ce = ts_result.final_fitness
		best_acc = ts_result.final_accuracy or best_acc

	# End-of-phase test evaluation (175K train → 82K test)
	if test_evaluator is not None:
		test_result = test_evaluator.evaluate_batch_full([best_genome])[0]
		print(f"  Phase {phase_name} test: CE={test_result.ce:.4f}, "
			  f"Acc={test_result.accuracy*100:.2f}%"
			  + (f", F1={test_result.f1_macro:.4f}" if test_result.f1_macro is not None else ""))

	return best_genome, best_ce, best_acc


def run_hierarchical_experiment(cfg: IDSTrainConfig) -> dict:
	"""Run two-stage hierarchical IDS experiment.

	Stage 0: Binary gate (normal vs attack) on full dataset.
	Stage 1: 9-class attack classifier on attack-only examples.
	Both stages use the same input features (thermometer-encoded flow).
	"""
	print(f"=== Hierarchical IDS Classifier (split={cfg.split}) ===")
	print(f"  Stage 0: Binary gate (normal vs attack)")
	print(f"  Stage 1: 9-class attack type classifier")
	print()

	# ── Load full dataset ─────────────────────────────────────────────
	t0 = time.time()
	dataset = load_unsw_nb15(n_bits=cfg.n_bits, split=cfg.split)
	total_features = dataset.X_train.shape[1]
	load_time = time.time() - t0

	# ── Stage 0: Binary classification ────────────────────────────────
	print("=" * 60)
	print("STAGE 0: Binary Gate (Normal vs Attack)")
	print("=" * 60)

	s0_cfg = IDSTrainConfig(
		classification="binary",
		n_bits=cfg.n_bits,
		neurons=cfg.neurons,
		address_bits=cfg.address_bits,
		ga_gens=cfg.ga_gens,
		ts_iters=cfg.ts_iters,
		patience=cfg.patience,
		population=cfg.population,
		neighbors=cfg.neighbors,
		num_parts=cfg.num_parts,
		seed=cfg.seed,
		fitness_weight_ce=cfg.fitness_weight_ce,
		fitness_weight_acc=cfg.fitness_weight_acc,
		fitness_weight_f1=cfg.fitness_weight_f1,
		split=cfg.split,
		val_fraction=cfg.val_fraction,
	)

	# Reuse loaded dataset — run_ids_experiment will reload, but that's fine
	# for correctness. For the hierarchical runner, Stage 0 is binary.
	s0_metrics = run_ids_experiment(s0_cfg)

	# ── Stage 1: Attack type classification ───────────────────────────
	print()
	print("=" * 60)
	print("STAGE 1: Attack Type Classification (9 classes)")
	print("=" * 60)

	full_attack_dataset = create_attack_only_dataset(dataset)
	attack_features = full_attack_dataset.X_train.shape[1]
	num_attack_classes = len(full_attack_dataset.category_names)

	print(f"Attack dataset: {full_attack_dataset.X_train.shape[0]:,} train, "
		  f"{full_attack_dataset.X_test.shape[0]:,} test, {num_attack_classes} classes")

	# Validation split for Stage 1
	s1_overfit_cb = None
	if cfg.val_fraction > 0:
		s1_train_val, _ = split_train_validation(
			full_attack_dataset, val_fraction=cfg.val_fraction, seed=cfg.seed + 2000,
		)
		s1_opt_dataset = s1_train_val
	else:
		s1_opt_dataset = full_attack_dataset

	# Create Stage 1 optimizer evaluator
	s1_evaluator = IDSEvaluator(
		dataset=s1_opt_dataset,
		classification="multi",
		num_parts=cfg.num_parts,
		seed=cfg.seed + 2000,
	)
	print(f"Stage 1 optimizer evaluator: {repr(s1_evaluator)}")

	# Stage 1 test evaluator (full attack train → attack test)
	s1_test_evaluator = IDSEvaluator(
		dataset=full_attack_dataset,
		classification="multi",
		num_parts=1,
		seed=cfg.seed + 2000,
	)

	if cfg.val_fraction > 0:
		s1_overfit_cb = create_overfitting_callback(
			evaluator=s1_test_evaluator,
			patience=3,
			logger=print,
		)

	# Grid search for Stage 1
	grid_config = GridSearchConfig(
		num_clusters=num_attack_classes,
		neurons_grid=cfg.neurons_grid,
		bits_grid=cfg.bits_grid,
		top_k=cfg.grid_top_k,
		population_size=cfg.population,
		total_input_bits=attack_features,
		fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
		fitness_weight_ce=cfg.fitness_weight_ce,
		fitness_weight_acc=cfg.fitness_weight_acc,
		fitness_weight_f1=cfg.fitness_weight_f1,
	)
	grid_strategy = GridSearchStrategy(
		config=grid_config,
		batch_evaluator=s1_evaluator,
		seed=cfg.seed + 2000,
		logger=print,
	)

	t0 = time.time()
	grid_result = grid_strategy.optimize()
	print(f"Grid search: CE={grid_result.final_fitness:.4f}, "
		  f"Acc={grid_result.final_accuracy*100:.2f}% ({time.time()-t0:.1f}s)")

	best_genome = grid_result.best_genome
	best_ce = grid_result.final_fitness
	best_acc = grid_result.final_accuracy or 0.0
	seed_population = grid_result.final_population

	# Asymmetric genome for 9-class with varying attack counts
	asym_genome = create_asymmetric_genome(
		s1_opt_dataset.y_train_multi, num_attack_classes, attack_features,
		seed=cfg.seed + 2000,
	)
	asym_result = s1_evaluator.evaluate_batch_full([asym_genome])[0]
	print(f"Asymmetric genome: CE={asym_result.ce:.4f}, Acc={asym_result.accuracy*100:.2f}%")
	if asym_result.ce < best_ce:
		best_genome = asym_genome
		best_ce = asym_result.ce
		best_acc = asym_result.accuracy
		print("  → Using asymmetric genome")
	else:
		print("  → Keeping grid search genome")

	# Phase 1: Neurons
	best_genome, best_ce, best_acc = _run_phase(
		"neurons", s1_evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=True, optimize_connections=False,
		num_classes=num_attack_classes, total_features=attack_features,
		seed_population=seed_population,
		overfitting_callback=s1_overfit_cb,
		test_evaluator=s1_test_evaluator,
	)

	# Phase 2: Bits
	best_genome, best_ce, best_acc = _run_phase(
		"bits", s1_evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=True, optimize_neurons=False, optimize_connections=False,
		num_classes=num_attack_classes, total_features=attack_features,
		overfitting_callback=s1_overfit_cb,
		test_evaluator=s1_test_evaluator,
	)

	# Phase 3: Connections
	best_genome, best_ce, best_acc = _run_phase(
		"connections", s1_evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=False, optimize_connections=True,
		num_classes=num_attack_classes, total_features=attack_features,
		overfitting_callback=s1_overfit_cb,
		test_evaluator=s1_test_evaluator,
	)

	# Final Stage 1 evaluation (full attack train → attack test)
	print("\n=== Stage 1 Final Evaluation ===")
	s1_final = s1_test_evaluator.evaluate_batch_full([best_genome])[0]
	print(f"Stage 1: CE={s1_final.ce:.4f}, Acc={s1_final.accuracy*100:.2f}%")
	print(f"Genome: neurons={best_genome.neurons_per_cluster}")

	# ── Combined report ───────────────────────────────────────────────
	print()
	print("=" * 60)
	print("COMBINED RESULTS")
	print("=" * 60)
	print(f"Stage 0 (Binary gate):       {s0_metrics['accuracy']*100:.2f}%")
	print(f"Stage 1 (Attack classifier): {s1_final.accuracy*100:.2f}%")
	print(f"  (Stage 1 tested on {full_attack_dataset.X_test.shape[0]:,} attack examples only)")
	print()
	print("At inference:")
	print("  1. All traffic → Stage 0 (< 1 KB, 6 neurons)")
	print("  2. If attack → Stage 1 classifies type")
	print("  3. If normal → pass through")

	metrics = {
		"mode": "hierarchical",
		"split": cfg.split,
		"stage0": s0_metrics,
		"stage1": {
			"classification": "attack_type",
			"num_classes": num_attack_classes,
			"class_names": full_attack_dataset.category_names,
			"ce": s1_final.ce,
			"accuracy": s1_final.accuracy,
			"genome": {
				"bits_per_neuron": best_genome.bits_per_neuron,
				"neurons_per_cluster": best_genome.neurons_per_cluster,
				"total_neurons": best_genome.total_neurons,
			},
			"train_examples": int(full_attack_dataset.X_train.shape[0]),
			"test_examples": int(full_attack_dataset.X_test.shape[0]),
		},
	}

	if cfg.output:
		output_path = Path(cfg.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(metrics, indent=2))
		print(f"\nResults saved to {cfg.output}")

	return metrics


def run_bitwise_experiment(cfg: IDSTrainConfig) -> dict:
	"""Run bitwise ECOC attack classification with error-correcting codes.

	Trains 7 independent binary classifiers, one per output bit of a [7,4,3]
	Hamming codeword. At inference, the 7 predicted bits are decoded to the
	nearest valid codeword (Hamming distance). Single-bit error correction
	is free from the code's minimum distance of 3.

	Pipeline per bit:
		Grid search → GA neurons → TS neurons → GA bits → TS bits
		→ GA connections → TS connections

	With --bitwise-quick: Grid search → GA connections → TS connections only.
	"""
	print(f"=== Bitwise ECOC Attack Classifier (split={cfg.split}) ===")
	print(f"Config: pop={cfg.population}, GA={cfg.ga_gens}gen, TS={cfg.ts_iters}iter, "
		  f"patience={cfg.patience}")
	print()

	# ── Load attack-only dataset ──────────────────────────────────────
	t0 = time.time()
	full_dataset = load_unsw_nb15(n_bits=cfg.n_bits, split=cfg.split)
	attack_dataset = create_attack_only_dataset(full_dataset)
	total_features = attack_dataset.X_train.shape[1]
	num_attack_classes = len(attack_dataset.category_names)
	load_time = time.time() - t0
	print(f"Loaded in {load_time:.1f}s: {total_features} features, {num_attack_classes} classes")

	# ── Create Hamming codebook ───────────────────────────────────────
	codebook = create_bitwise_codebook(attack_dataset.category_names, num_bits=7)
	print()

	# ── Train one binary classifier per output bit ────────────────────
	bit_genomes = []
	bit_accuracies = []
	total_train_time = 0.0

	for bit_idx in range(codebook.num_bits):
		print(f"\n{'='*60}")
		print(f"BIT {bit_idx}/{codebook.num_bits}: Training binary classifier")
		print(f"{'='*60}")

		# Create per-bit binary dataset
		bit_dataset = create_per_bit_dataset(attack_dataset, codebook, bit_idx)

		# Show bit class balance
		n_zeros = int((bit_dataset.y_train_binary == 0).sum())
		n_ones = int((bit_dataset.y_train_binary == 1).sum())
		print(f"  Bit {bit_idx} balance: {n_zeros:,} zeros ({n_zeros*100/(n_zeros+n_ones):.1f}%), "
			  f"{n_ones:,} ones ({n_ones*100/(n_zeros+n_ones):.1f}%)")

		# Create evaluator for this bit (binary: 2 classes)
		evaluator = IDSEvaluator(
			dataset=bit_dataset,
			classification="binary",
			num_parts=cfg.num_parts,
			seed=cfg.seed + bit_idx * 100,
		)

		# Grid search
		grid_config = GridSearchConfig(
			num_clusters=2,
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
			seed=cfg.seed + bit_idx * 100,
			logger=print,
		)

		t_bit = time.time()
		grid_result = grid_strategy.optimize()
		print(f"  Grid: CE={grid_result.final_fitness:.4f}, "
			  f"Acc={grid_result.final_accuracy*100:.2f}%")

		best_genome = grid_result.best_genome
		best_ce = grid_result.final_fitness
		best_acc = grid_result.final_accuracy or 0.0
		seed_pop = grid_result.final_population

		if cfg.bitwise_quick:
			# Quick mode: only connections phase
			best_genome, best_ce, best_acc = _run_phase(
				"connections", evaluator, best_genome, best_ce, best_acc, cfg,
				optimize_bits=False, optimize_neurons=False, optimize_connections=True,
				num_classes=2, total_features=total_features,
				seed_population=seed_pop,
			)
		else:
			# Full pipeline: neurons → bits → connections
			best_genome, best_ce, best_acc = _run_phase(
				"neurons", evaluator, best_genome, best_ce, best_acc, cfg,
				optimize_bits=False, optimize_neurons=True, optimize_connections=False,
				num_classes=2, total_features=total_features,
				seed_population=seed_pop,
			)
			best_genome, best_ce, best_acc = _run_phase(
				"bits", evaluator, best_genome, best_ce, best_acc, cfg,
				optimize_bits=True, optimize_neurons=False, optimize_connections=False,
				num_classes=2, total_features=total_features,
			)
			best_genome, best_ce, best_acc = _run_phase(
				"connections", evaluator, best_genome, best_ce, best_acc, cfg,
				optimize_bits=False, optimize_neurons=False, optimize_connections=True,
				num_classes=2, total_features=total_features,
			)

		# Final eval for this bit
		final = evaluator.evaluate_batch_full([best_genome])[0]
		bit_time = time.time() - t_bit
		total_train_time += bit_time

		print(f"  Bit {bit_idx} final: Acc={final.accuracy*100:.2f}%, "
			  f"CE={final.ce:.4f} ({bit_time:.1f}s)")
		print(f"  Genome: neurons={best_genome.neurons_per_cluster}, "
			  f"bits={best_genome.bits_per_neuron}")

		bit_genomes.append(best_genome)
		bit_accuracies.append(final.accuracy)

	# ── Combined evaluation via codeword decoding ─────────────────────
	print(f"\n{'='*60}")
	print("COMBINED EVALUATION: Codeword Decoding")
	print(f"{'='*60}")
	print(f"Per-bit accuracies: {[f'{a*100:.1f}%' for a in bit_accuracies]}")
	print(f"Mean per-bit accuracy: {np.mean(bit_accuracies)*100:.2f}%")
	print(f"Total training time: {total_train_time:.1f}s")

	# Predict each bit on the test set using Rust GPU accelerator
	# Each bit's classifier is re-trained on full data and predicts on eval set
	predicted_bits = np.zeros((len(attack_dataset.X_test), codebook.num_bits), dtype=bool)

	print("  Running per-example prediction (Rust GPU)...")
	for bit_idx in range(codebook.num_bits):
		bit_dataset = create_per_bit_dataset(attack_dataset, codebook, bit_idx)

		# Create evaluator for this bit to get full-data predictions
		pred_evaluator = IDSEvaluator(
			dataset=bit_dataset,
			classification="binary",
			num_parts=1,
			seed=cfg.seed + bit_idx * 100,
		)
		# predict() trains on full training data and returns per-example eval predictions
		predictions = np.array(pred_evaluator.predict(bit_genomes[bit_idx]))
		# Binary classifier: class 1 means bit is set
		predicted_bits[:, bit_idx] = predictions == 1
		bit_acc = np.mean(predictions == bit_dataset.y_test_binary)
		print(f"  Bit {bit_idx}: test acc={bit_acc*100:.2f}%")

	# Decode predicted codewords to class indices
	predicted_classes = codebook.decode_batch(predicted_bits)
	true_classes = attack_dataset.y_test_multi

	# Compute metrics
	from sklearn.metrics import accuracy_score, classification_report
	combined_acc = accuracy_score(true_classes, predicted_classes)
	print(f"\n  Combined accuracy (codeword decoding): {combined_acc*100:.2f}%")
	print(f"  Error correction: can correct {(codebook.min_distance - 1) // 2} bit errors")
	print()
	print(classification_report(
		true_classes, predicted_classes,
		target_names=attack_dataset.category_names,
		zero_division=0,
	))

	metrics = {
		"mode": "bitwise",
		"split": cfg.split,
		"num_classes": num_attack_classes,
		"class_names": attack_dataset.category_names,
		"codebook": {
			"num_bits": codebook.num_bits,
			"min_distance": codebook.min_distance,
			"codewords": codebook.codewords.astype(int).tolist(),
		},
		"per_bit_accuracy": bit_accuracies,
		"mean_bit_accuracy": float(np.mean(bit_accuracies)),
		"combined_accuracy": float(combined_acc),
		"total_train_time": total_train_time,
		"bit_genomes": [
			{
				"bits_per_neuron": g.bits_per_neuron,
				"neurons_per_cluster": g.neurons_per_cluster,
				"total_neurons": g.total_neurons,
			}
			for g in bit_genomes
		],
		"config": {
			"n_bits": cfg.n_bits,
			"ga_gens": cfg.ga_gens,
			"ts_iters": cfg.ts_iters,
			"patience": cfg.patience,
			"population": cfg.population,
			"bitwise_quick": cfg.bitwise_quick,
		},
	}

	if cfg.output:
		output_path = Path(cfg.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(metrics, indent=2, default=str))
		print(f"\nResults saved to {cfg.output}")

	return metrics


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
	parser.add_argument("--fitness-f1-weight", type=float, default=0.0,
		help="F1-macro weight in fitness ranking (0=disabled, >0=included)")
	parser.add_argument("--beta", type=float, default=1.0,
		help="F-beta score: <1 favors precision/low FPR, >1 favors recall")
	parser.add_argument("--class-weights", type=str, default=None,
		help="Per-class importance weights, comma-separated (e.g. '1.0,2.0')")
	parser.add_argument("--val-fraction", type=float, default=0.25,
		help="Validation holdout fraction from training data (0=disabled, default 0.25)")
	parser.add_argument("--split", choices=["standard", "random"], default="standard",
		help="Evaluation protocol: 'standard' (temporal) or 'random' (i.i.d.)")
	parser.add_argument("--hierarchical", action="store_true",
		help="Two-stage: binary gate → attack type classifier")
	parser.add_argument("--bitwise", action="store_true",
		help="Bitwise ECOC: 7 binary classifiers with Hamming error correction")
	parser.add_argument("--bitwise-quick", action="store_true",
		help="Bitwise quick mode: grid search + connections only (skip neurons/bits phases)")
	parser.add_argument("--output", type=str, default=None, help="JSON output path")
	args = parser.parse_args()

	# Parse class weights if provided
	class_weights = None
	if args.class_weights:
		class_weights = [float(w) for w in args.class_weights.split(",")]

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
		fitness_weight_f1=args.fitness_f1_weight,
		beta=args.beta,
		class_weights=class_weights,
		split=args.split,
		val_fraction=args.val_fraction,
		bitwise_quick=args.bitwise_quick,
		output=args.output,
	)

	if args.bitwise or args.bitwise_quick:
		if args.bitwise_quick:
			cfg.bitwise_quick = True
		run_bitwise_experiment(cfg)
	elif args.hierarchical:
		run_hierarchical_experiment(cfg)
	else:
		run_ids_experiment(cfg)


if __name__ == "__main__":
	main()
