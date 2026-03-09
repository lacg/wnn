"""IDS training runner — WNN classifier for UNSW-NB15.

Usage:
	python -m wnn.ids.train --classification binary --ga-gens 50 --ts-iters 100
	python -m wnn.ids.train --classification multi --neurons 10 --bits 12
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wnn.ids import load_unsw_nb15
from wnn.ids.metrics import compute_ids_metrics, format_ids_report
from wnn.ram.architecture.ids_evaluator import IDSEvaluator
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome, PhaseType
from wnn.ram.strategies.connectivity.architecture_strategies import (
	ArchitectureConfig,
	ArchitectureGAStrategy,
	ArchitectureTSStrategy,
)
from wnn.ram.strategies.connectivity.generic_strategies import GAConfig, TSConfig


@dataclass
class IDSTrainConfig:
	classification: str = "binary"
	n_bits: int = 8
	neurons: int = 5
	address_bits: int = 12
	ga_gens: int = 50
	ts_iters: int = 100
	patience: int = 5
	population: int = 30
	neighbors: int = 30
	num_parts: int = 3
	seed: int = 42
	output: Optional[str] = None


def run_ids_experiment(cfg: IDSTrainConfig) -> dict:
	"""Run full IDS training experiment.

	Returns dict with final metrics.
	"""
	print(f"=== WNN IDS Classifier ({cfg.classification}) ===")
	print(f"Config: neurons={cfg.neurons}, bits={cfg.address_bits}, "
		  f"GA={cfg.ga_gens}gen, TS={cfg.ts_iters}iter, pop={cfg.population}")

	# Load dataset
	t0 = time.time()
	dataset = load_unsw_nb15(n_bits=cfg.n_bits)
	total_features = dataset.X_train.shape[1]
	print(f"Dataset loaded: {dataset.X_train.shape[0]} train, {dataset.X_test.shape[0]} test, "
		  f"{total_features} features ({time.time()-t0:.1f}s)")

	if cfg.classification == "binary":
		num_classes = 2
		class_names = ["Normal", "Attack"]
	else:
		num_classes = len(dataset.category_names)
		class_names = dataset.category_names

	# Create evaluator
	t0 = time.time()
	evaluator = IDSEvaluator(
		dataset=dataset,
		classification=cfg.classification,
		num_parts=cfg.num_parts,
		seed=cfg.seed,
	)
	print(f"Evaluator created ({time.time()-t0:.1f}s)")

	# Create initial genome
	genome = ClusterGenome.create_uniform(
		num_clusters=num_classes,
		bits=cfg.address_bits,
		neurons=cfg.neurons,
		total_input_bits=total_features,
		rng=cfg.seed,
	)
	print(f"Initial genome: {num_classes} clusters, {genome.total_neurons} neurons, "
		  f"{cfg.address_bits} bits/neuron")

	# Evaluate initial genome
	result = evaluator.evaluate_batch_full([genome])[0]
	print(f"Initial: CE={result.ce:.4f}, Acc={result.accuracy*100:.2f}%")

	best_genome = genome
	best_ce = result.ce
	best_acc = result.accuracy

	# Phase 1: Optimize neurons (GA + TS)
	best_genome, best_ce, best_acc = _run_phase(
		"neurons", evaluator, best_genome, best_ce, best_acc, cfg,
		optimize_bits=False, optimize_neurons=True, optimize_connections=False,
		num_classes=num_classes, total_features=total_features,
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

	# Final evaluation on full data
	print("\n=== Final Evaluation ===")
	final_result = evaluator.evaluate_batch_full([best_genome])[0]
	print(f"Final: CE={final_result.ce:.4f}, Acc={final_result.accuracy*100:.2f}%")

	# Compute IDS-specific metrics
	# For now, report the CE/accuracy from the evaluator
	# Full per-class metrics would require forward-pass prediction (future work)
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
		},
	}

	if cfg.output:
		output_path = Path(cfg.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(metrics, indent=2))
		print(f"Results saved to {cfg.output}")

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

	# GA phase
	ga_config = GAConfig(
		population_size=cfg.population,
		generations=cfg.ga_gens,
		patience=cfg.patience,
		mutation_rate=0.1,
		crossover_rate=0.7,
		tournament_size=3,
	)

	ga_strategy = ArchitectureGAStrategy(
		arch_config=arch_config,
		ga_config=ga_config,
		seed=cfg.seed,
		logger=print,
		cached_evaluator=evaluator,
	)

	t0 = time.time()
	ga_result = ga_strategy.optimize(
		initial_population=[genome],
	)
	ga_time = time.time() - t0
	print(f"  GA {phase_name}: CE {best_ce:.4f} -> {ga_result.final_fitness:.4f}, "
		  f"Acc {ga_result.final_accuracy*100:.2f}% ({ga_result.iterations_run} gens, {ga_time:.1f}s)")

	best_genome = ga_result.best_genome
	best_ce = ga_result.final_fitness
	best_acc = ga_result.final_accuracy or best_acc

	# TS phase (refinement)
	ts_config = TSConfig(
		iterations=cfg.ts_iters,
		neighbors_per_iter=cfg.neighbors,
		total_neighbors_size=cfg.neighbors,
		patience=cfg.patience,
		mutation_rate=0.1,
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
		  f"Acc {ts_result.final_accuracy*100:.2f}% ({ts_result.iterations_run} iters, {ts_time:.1f}s)")

	if ts_result.final_fitness < best_ce:
		best_genome = ts_result.best_genome
		best_ce = ts_result.final_fitness
		best_acc = ts_result.final_accuracy or best_acc

	return best_genome, best_ce, best_acc


def main():
	parser = argparse.ArgumentParser(description="WNN IDS Classifier Training")
	parser.add_argument("--classification", choices=["binary", "multi"], default="binary")
	parser.add_argument("--n-bits", type=int, default=8, help="Thermometer encoding bits")
	parser.add_argument("--neurons", type=int, default=5, help="Initial neurons per class")
	parser.add_argument("--bits", type=int, default=12, help="Initial address bits per neuron")
	parser.add_argument("--ga-gens", type=int, default=50)
	parser.add_argument("--ts-iters", type=int, default=100)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--population", type=int, default=30)
	parser.add_argument("--neighbors", type=int, default=30)
	parser.add_argument("--num-parts", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
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
		output=args.output,
	)

	run_ids_experiment(cfg)


if __name__ == "__main__":
	main()
