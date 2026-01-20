#!/usr/bin/env python3
"""
Coarse-to-Fine Phased Architecture Search

Implements a multi-pass approach where:
- Pass 1 (coarse): Low patience for quick exploration (patience=2)
- Pass 2 (fine): Higher patience for refinement (patience=4)
- Pass 3 (final): Return to low patience to escape local minima (patience=2)

Each pass runs through all phases (neurons -> bits -> connections).
Later passes can be seeded from previous pass results.

Usage:
  # First pass (coarse exploration, patience=2)
  python run_coarse_fine_search.py --pass 1 --base-patience 2

  # Second pass (refinement, patience=4)
  python run_coarse_fine_search.py --pass 2 --patience 4 --seed-from results_pass1.json

  # Third pass (patience=2, reset to explore new areas)
  python run_coarse_fine_search.py --pass 3 --patience 2 --seed-from results_pass2.json

  # Or use --base-patience for automatic calculation (doubles each pass)
  python run_coarse_fine_search.py --pass 2 --base-patience 2 --seed-from results_pass1.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wnn.logger import Logger
from wnn.ram.experiments import PhasedSearchConfig, PhasedSearchRunner
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


# Global logger instance
logger: Optional[Logger] = None


def log(msg: str):
	"""Log wrapper that uses the global logger."""
	if logger:
		logger(msg)
	else:
		print(msg)


def load_seed_data(seed_file: str) -> tuple[Optional[ClusterGenome], Optional[list[ClusterGenome]]]:
	"""
	Load seed genome and population from a previous run's JSON output.

	Returns:
		Tuple of (seed_genome, seed_population). Population may be None if not available.
	"""
	# Try ClusterGenome.load() first (direct genome file - no population)
	try:
		genome, _ = ClusterGenome.load(seed_file)
		return genome, None
	except (KeyError, TypeError):
		pass

	# Fall back to phased search results format
	try:
		with open(seed_file, 'r') as f:
			data = json.load(f)

		genome = None
		population = None

		# Load genome from final.genome
		if 'final' in data and 'genome' in data['final']:
			genome = ClusterGenome.deserialize(data['final']['genome'])

		# Load population from final.final_population (new format)
		if 'final' in data and 'final_population' in data['final'] and data['final']['final_population']:
			population = [ClusterGenome.deserialize(g) for g in data['final']['final_population']]

		# Fall back to genome_stats (old format - limited, no population)
		if genome is None and 'final' in data and 'genome_stats' in data['final']:
			stats = data['final']['genome_stats']
			genome = ClusterGenome(
				bits_per_cluster=stats.get('bits_per_cluster', []),
				neurons_per_cluster=stats.get('neurons_per_cluster', []),
				connections=stats.get('connections'),
			)

		return genome, population
	except Exception as e:
		print(f"Warning: Could not load seed data from {seed_file}: {e}")

	return None, None


def main():
	global logger

	parser = argparse.ArgumentParser(description="Coarse-to-Fine Phased Architecture Search")

	# Config file (YAML)
	parser.add_argument("--config", type=str, default=None,
						help="YAML config file (CLI args override config file values)")

	# Pass configuration
	parser.add_argument("--pass", dest="pass_num", type=int, default=1,
						help="Pass number (1=coarse, 2+=fine). Patience doubles each pass by default.")
	parser.add_argument("--base-patience", type=int, default=2,
						help="Base patience for pass 1 (default: 2)")
	parser.add_argument("--patience", type=int, default=None,
						help="Override patience directly (ignores base-patience calculation)")
	parser.add_argument("--seed-from", type=str, default=None,
						help="JSON file from previous pass to seed from")

	# Data configuration
	parser.add_argument("--train-tokens", type=int, default=200000, help="Total training tokens")
	parser.add_argument("--eval-tokens", type=int, default=50000, help="Eval tokens")
	parser.add_argument("--context", type=int, default=4, help="Context size")

	# Optimization configuration
	parser.add_argument("--ga-gens", type=int, default=100, help="GA generations per phase")
	parser.add_argument("--ts-iters", type=int, default=200, help="TS iterations per phase")
	parser.add_argument("--population", type=int, default=50, help="GA population size")
	parser.add_argument("--neighbors", type=int, default=50, help="TS neighbors per iteration")

	# Architecture defaults
	parser.add_argument("--default-bits", type=int, default=8, help="Default bits for Phase 1")
	parser.add_argument("--default-neurons", type=int, default=5, help="Default neurons for Phase 2")

	# CE pressure
	parser.add_argument("--ce-percentile", type=float, default=None,
						help="CE percentile filter (e.g., 0.75 = keep top 75%% by CE). None=disabled")

	# Checkpointing
	parser.add_argument("--checkpoint-dir", type=str, default=None,
						help="Directory to save/load phase checkpoints (enables resume)")
	parser.add_argument("--resume-from", type=str, default=None,
						help="Resume from phase (1a, 1b, 2a, 2b, 3a, 3b). Requires --checkpoint-dir")

	# Other settings
	parser.add_argument("--token-parts", type=int, default=3, help="Number of token subsets (3=thirds)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for rotation (None=time-based)")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file (auto-generated if not specified)")

	args = parser.parse_args()

	# Load config from YAML if specified (CLI args override)
	if args.config:
		config_obj = PhasedSearchConfig.load_yaml(args.config)
		# Map config fields to args (only if arg was not explicitly set)
		config_to_arg = {
			'context_size': 'context',
			'token_parts': 'token_parts',
			'ga_generations': 'ga_gens',
			'ts_iterations': 'ts_iters',
			'population_size': 'population',
			'neighbors_per_iter': 'neighbors',
			'patience': 'patience',
			'default_bits': 'default_bits',
			'default_neurons': 'default_neurons',
			'ce_percentile': 'ce_percentile',
			'rotation_seed': 'seed',
		}
		for cfg_field, arg_name in config_to_arg.items():
			cfg_value = getattr(config_obj, cfg_field)
			# Only apply if not None and arg wasn't explicitly provided
			if cfg_value is not None and getattr(args, arg_name) is None:
				setattr(args, arg_name, cfg_value)

	# Validate checkpoint args
	if args.resume_from and not args.checkpoint_dir:
		parser.error("--resume-from requires --checkpoint-dir")

	# Calculate patience: use override if provided, otherwise base_patience * 2^(pass-1)
	if args.patience is not None:
		patience = args.patience
	else:
		patience = args.base_patience * (2 ** (args.pass_num - 1))

	# Auto-generate output filename if not specified
	if args.output is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		args.output = f"results_pass{args.pass_num}_{timestamp}.json"

	# Determine seed: time-based if not specified
	rotation_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)

	# Setup logger
	logger = Logger(name=f"coarse_fine_pass{args.pass_num}")

	logger.header(f"Coarse-to-Fine Search - Pass {args.pass_num}")
	log(f"  Log file: {logger.log_file}")
	if args.patience is not None:
		log(f"  Pass: {args.pass_num} (patience: {patience} [override])")
	else:
		log(f"  Pass: {args.pass_num} (patience: {args.base_patience} * 2^{args.pass_num - 1} = {patience})")
	if args.seed_from:
		log(f"  Seeding from: {args.seed_from}")
	log(f"  Total train tokens: {args.train_tokens:,}")
	log(f"  Per-iteration rotation: 1/{args.token_parts} (~{args.train_tokens // args.token_parts:,} tokens per iteration)")
	log(f"  Rotation seed: {rotation_seed}" + (" (time-based)" if args.seed is None else " (explicit)"))
	log(f"  Eval tokens: {args.eval_tokens:,}")
	log(f"  Context: {args.context}")
	log(f"  GA generations: {args.ga_gens}, population: {args.population}")
	log(f"  TS iterations: {args.ts_iters}, neighbors: {args.neighbors}")
	log(f"  Patience: {patience}")
	if args.ce_percentile is not None:
		log(f"  CE percentile filter: {args.ce_percentile:.0%} (keep top {args.ce_percentile:.0%} by CE)")
	else:
		log(f"  CE percentile filter: disabled")
	log(f"  Default bits (Phase 1): {args.default_bits}")
	log(f"  Default neurons (Phase 2): {args.default_neurons}")
	log(f"  Output: {args.output}")
	if args.checkpoint_dir:
		log(f"  Checkpoint dir: {args.checkpoint_dir}")
		if args.resume_from:
			log(f"  Resume from: phase {args.resume_from}")
	log("")

	# Load seed genome and population if specified
	seed_genome = None
	seed_population = None
	if args.seed_from:
		seed_genome, seed_population = load_seed_data(args.seed_from)
		if seed_genome:
			log(f"Loaded seed genome: {seed_genome}")
		if seed_population:
			log(f"Loaded seed population: {len(seed_population)} genomes")
		if not seed_genome and not seed_population:
			log("No seed data loaded, starting fresh")
		log("")

	# Load data
	log("Loading WikiText-2 dataset...")
	try:
		from datasets import load_dataset
		import tiktoken

		dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
		text = " ".join([t for t in dataset["text"] if t.strip()])

		enc = tiktoken.get_encoding("gpt2")
		all_tokens = enc.encode(text)

		train_tokens = all_tokens[:args.train_tokens]
		eval_tokens = all_tokens[args.train_tokens:args.train_tokens + args.eval_tokens]
		vocab_size = enc.n_vocab

		log(f"  Train: {len(train_tokens):,} tokens")
		log(f"  Eval: {len(eval_tokens):,} tokens")
		log(f"  Vocab: {vocab_size:,}")

	except ImportError as e:
		log(f"ERROR: Missing dependencies: {e}")
		log("Please install: pip install datasets tiktoken")
		return 1

	# Create configuration with calculated patience
	config = PhasedSearchConfig(
		context_size=args.context,
		token_parts=args.token_parts,
		ga_generations=args.ga_gens,
		ts_iterations=args.ts_iters,
		population_size=args.population,
		neighbors_per_iter=args.neighbors,
		patience=patience,  # Calculated based on pass number
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		ce_percentile=args.ce_percentile,
		rotation_seed=rotation_seed,
		log_path=logger.log_file,
	)

	# Create runner and setup
	runner = PhasedSearchRunner(config=config, logger=log, checkpoint_dir=args.checkpoint_dir)
	log("")
	runner.setup(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
	)

	# Run all phases (optionally seeded from previous pass, or resume from checkpoint)
	results = runner.run_all_phases(
		seed_genome=seed_genome,
		seed_population=seed_population,
		resume_from=args.resume_from,
	)

	# Add metadata
	results["pass"] = args.pass_num
	results["patience"] = patience
	results["base_patience"] = args.base_patience
	results["args"] = vars(args)
	results["timestamp"] = datetime.now().isoformat()
	results["per_iteration_rotation"] = {
		"num_parts": args.token_parts,
		"tokens_per_part": len(train_tokens) // args.token_parts,
	}

	# Save results
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with open(output_path, "w") as f:
		json.dump(results, f, indent=2, default=str)
	log("")
	log(f"Results saved to: {output_path}")

	# Print next pass command suggestion
	if args.pass_num < 3:
		# Default pattern: 2, 4, 2 (pass 3 resets to base)
		if args.pass_num == 1:
			next_patience = args.base_patience * 2  # Pass 2: double
		else:
			next_patience = args.base_patience  # Pass 3: reset to base
		log("")
		log(f"To run pass {args.pass_num + 1} (suggested patience={next_patience}):")
		log(f"  python run_coarse_fine_search.py --pass {args.pass_num + 1} --patience {next_patience} --seed-from {args.output}")
		log(f"  (or use --base-patience to calculate automatically)")

	return 0


if __name__ == "__main__":
	sys.exit(main())
