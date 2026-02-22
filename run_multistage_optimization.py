#!/usr/bin/env python3
"""
Multi-Stage Architecture Optimization for RAM LM.

Uses MultiStageEvaluator with Rust+Metal backend. Each stage predicts
a portion of the vocabulary factorization:
  P(token) = P(group|ctx) × P(token|group,ctx)

10-phase pipeline (5 per stage):
  Stage 0: Grid → GA Neurons → TS Neurons → GA Connections → TS Connections
  Stage 1: Grid → GA Neurons → TS Neurons → GA Connections → TS Connections

Usage:
	python run_multistage_optimization.py
	python run_multistage_optimization.py --num-stages 2 --stage-k auto
	python run_multistage_optimization.py --stage-k 256,256 --stage-cluster-type bitwise,bitwise
	python run_multistage_optimization.py --ga-gens 50 --ts-iters 50 --patience 2
"""

import argparse
import sys
import time

from wnn.logger import Logger


def load_wikitext2_tokens(logger=print):
	"""Load all 3 WikiText-2 splits using GPT-2 tokenizer."""
	try:
		from datasets import load_dataset
		from transformers import AutoTokenizer
	except ImportError:
		logger("ERROR: datasets and transformers required.")
		sys.exit(1)

	tokenizer = AutoTokenizer.from_pretrained("gpt2")
	tokenizer.model_max_length = int(1e12)
	dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

	train_tokens = tokenizer.encode("\n".join(dataset["train"]["text"]))
	test_tokens = tokenizer.encode("\n".join(dataset["test"]["text"]))
	validation_tokens = tokenizer.encode("\n".join(dataset["validation"]["text"]))
	vocab_size = tokenizer.vocab_size

	logger(f"Loaded WikiText-2: {len(train_tokens):,} train, "
		   f"{len(test_tokens):,} test, {len(validation_tokens):,} validation, vocab={vocab_size}")

	return train_tokens, test_tokens, validation_tokens, vocab_size


def main():
	parser = argparse.ArgumentParser(
		description="Multi-Stage RAM LM Optimization (2-stage factorized prediction)"
	)
	# Multi-stage config
	parser.add_argument("--num-stages", type=int, default=2,
						help="Number of prediction stages (default: 2)")
	parser.add_argument("--stage-k", type=str, default="auto",
						help="K per stage: 'auto' or comma-separated (e.g. '256,256')")
	parser.add_argument("--stage-cluster-type", type=str, default="bitwise,bitwise",
						help="Comma-separated cluster types per stage (default: bitwise,bitwise)")
	parser.add_argument("--stage-mode", type=str, default="input_concat",
						help="Stage connection mode (default: input_concat)")

	# General
	parser.add_argument("--context", type=int, default=4, help="Context window size")
	parser.add_argument("--memory-mode", type=str, default="QUAD_WEIGHTED",
						choices=["TERNARY", "QUAD_BINARY", "QUAD_WEIGHTED"],
						help="Memory mode (default: QUAD_WEIGHTED)")
	parser.add_argument("--neuron-sample-rate", type=float, default=0.25,
						help="Neuron sample rate (default: 0.25)")

	# Search params
	parser.add_argument("--ga-gens", type=int, default=250,
						help="GA generations per phase (default: 250)")
	parser.add_argument("--ts-iters", type=int, default=250,
						help="TS iterations per phase (default: 250)")
	parser.add_argument("--population", type=int, default=50,
						help="GA population / TS neighbor cache size (default: 50)")
	parser.add_argument("--neighbors", type=int, default=50,
						help="TS neighbors per iteration (default: 50)")
	parser.add_argument("--patience", type=int, default=10,
						help="Early stop patience (default: 10)")

	# Bounds
	parser.add_argument("--min-bits", type=int, default=4,
						help="Min bits per neuron (default: 4)")
	parser.add_argument("--max-bits", type=int, default=24,
						help="Max bits per neuron (default: 24)")
	parser.add_argument("--min-neurons", type=int, default=5,
						help="Min neurons per cluster (default: 5)")
	parser.add_argument("--max-neurons", type=int, default=300,
						help="Max neurons per cluster (default: 300)")

	# Fitness
	parser.add_argument("--fitness", type=str, default="HARMONIC_RANK",
						choices=["CE", "HARMONIC_RANK", "NORMALIZED", "NORMALIZED_HARMONIC"],
						help="Fitness calculator type (default: HARMONIC_RANK)")

	# Data rotation
	parser.add_argument("--train-parts", type=int, default=4,
						help="Number of train subsets for rotation (default: 4)")
	parser.add_argument("--eval-parts", type=int, default=1,
						help="Number of eval subsets for rotation (default: 1)")

	args = parser.parse_args()

	# Parse stage config
	num_stages = args.num_stages
	stage_cluster_type = [s.strip() for s in args.stage_cluster_type.split(",")]
	assert len(stage_cluster_type) == num_stages, \
		f"Expected {num_stages} cluster types, got {len(stage_cluster_type)}"

	# Parse stage K
	if args.stage_k == "auto":
		from wnn.ram.architecture.multistage_evaluator import compute_default_k
		stage_k = compute_default_k(num_stages, stage_cluster_type)
	else:
		stage_k = [int(k.strip()) for k in args.stage_k.split(",")]
		assert len(stage_k) == num_stages, \
			f"Expected {num_stages} K values, got {len(stage_k)}"

	# Parse stage mode
	from wnn.ram.experiments.experiment import StageMode
	mode_map = {"input_concat": StageMode.INPUT_CONCAT}
	stage_mode_val = mode_map.get(args.stage_mode)
	if stage_mode_val is None:
		print(f"Unknown stage mode: {args.stage_mode}")
		sys.exit(1)
	stage_mode = [stage_mode_val]

	# Parse memory mode
	memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2}
	memory_mode_int = memory_mode_map[args.memory_mode]

	# Parse fitness
	from wnn.ram.fitness import FitnessCalculatorType
	fitness_type = FitnessCalculatorType[args.fitness]

	# Create logger
	logger = Logger(f"multistage_{num_stages}s_k{'x'.join(str(k) for k in stage_k)}")
	logger(f"Multi-Stage Optimization")
	logger(f"  Stages: {num_stages}, K: {stage_k}, Types: {stage_cluster_type}")
	logger(f"  Mode: {args.stage_mode}, Context: {args.context}")
	logger(f"  Bounds: neurons=[{args.min_neurons},{args.max_neurons}], bits=[{args.min_bits},{args.max_bits}]")
	logger(f"  GA: {args.ga_gens} gens, TS: {args.ts_iters} iters, Patience: {args.patience}")

	# Load data
	train_tokens, test_tokens, validation_tokens, vocab_size = load_wikitext2_tokens(logger=logger)

	# Create evaluator
	from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator

	logger("Creating MultiStageEvaluator...")
	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=test_tokens,
		vocab_size=vocab_size,
		context_size=args.context,
		num_stages=num_stages,
		stage_k=stage_k,
		stage_cluster_type=stage_cluster_type,
		stage_mode=stage_mode,
		target_stage=0,  # Flow.run() switches this at stage boundaries
		num_parts=args.train_parts,
		num_eval_parts=args.eval_parts,
		memory_mode=memory_mode_int,
		neuron_sample_rate=args.neuron_sample_rate,
	)

	# Create flow config
	from wnn.ram.experiments.flow import FlowConfig, Flow

	flow_config = FlowConfig.multi_stage_flow(
		name=f"MultiStage-{num_stages}s-K{'x'.join(str(k) for k in stage_k)}",
		num_stages=num_stages,
		stage_k=stage_k,
		stage_cluster_type=stage_cluster_type,
		stage_mode=stage_mode,
		ga_generations=args.ga_gens,
		ts_iterations=args.ts_iters,
		population_size=args.population,
		neighbors_per_iter=args.neighbors,
		patience=args.patience,
		context_size=args.context,
		memory_mode=args.memory_mode,
		neuron_sample_rate=args.neuron_sample_rate,
		min_bits=args.min_bits,
		max_bits=args.max_bits,
		min_neurons=args.min_neurons,
		max_neurons=args.max_neurons,
		fitness_calculator_type=fitness_type,
	)

	# Create and run flow
	flow = Flow(
		config=flow_config,
		evaluator=evaluator,
		logger=logger,
	)

	logger(f"\nStarting flow with {len(flow_config.experiments)} experiments...")
	t0 = time.time()
	result = flow.run()
	elapsed = time.time() - t0

	# Summary
	logger(f"\n{'='*70}")
	logger(f"COMPLETED in {elapsed:.0f}s")
	logger(f"{'='*70}")
	logger(f"  Final CE (last stage): {result.final_fitness:.4f}")
	if result.final_accuracy is not None:
		logger(f"  Final Accuracy: {result.final_accuracy:.2%}")

	if result.combined_ce is not None:
		logger(f"\n  Combined CE: {result.combined_ce:.4f}")
	if result.combined_accuracy is not None:
		logger(f"  Combined Accuracy: {result.combined_accuracy:.2%}")
	if result.per_stage_ce:
		for i, ce in enumerate(result.per_stage_ce):
			logger(f"  Stage {i} CE: {ce:.4f}")

	logger(f"\n  Experiment results:")
	for exp_result in result.experiment_results:
		logger(f"    {exp_result.experiment_name}: CE={exp_result.final_fitness:.4f}")


if __name__ == "__main__":
	main()
