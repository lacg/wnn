#!/usr/bin/env python3
"""
Phased Architecture Search with Per-Iteration Token Rotation

Demonstrates the three-phase optimization approach:
1. Phase 1: Optimize neurons only (bits fixed at default_bits)
2. Phase 2: Optimize bits only (neurons from Phase 1)
3. Phase 3: Optimize connections only (architecture from Phase 2)

Each phase uses GA then TS refinement.

Per-iteration rotation: Each generation/iteration uses a different 1/3 of
training tokens (cycling randomly through thirds). This acts as a regularizer
that forces genomes to generalize across all data subsets.
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


# Global logger instance
logger: Optional[Logger] = None


def log(msg: str):
	"""Log wrapper that uses the global logger."""
	if logger:
		logger(msg)
	else:
		print(msg)


def main():
	global logger

	parser = argparse.ArgumentParser(description="Phased Architecture Search")
	parser.add_argument("--train-tokens", type=int, default=200000, help="Total training tokens")
	parser.add_argument("--eval-tokens", type=int, default=50000, help="Eval tokens")
	parser.add_argument("--context", type=int, default=4, help="Context size")
	parser.add_argument("--ga-gens", type=int, default=50, help="GA generations per phase")
	parser.add_argument("--ts-iters", type=int, default=100, help="TS iterations per phase")
	parser.add_argument("--population", type=int, default=30, help="GA population size")
	parser.add_argument("--neighbors", type=int, default=20, help="TS neighbors per iteration")
	parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
	parser.add_argument("--default-bits", type=int, default=8, help="Default bits for Phase 1")
	parser.add_argument("--default-neurons", type=int, default=5, help="Default neurons for Phase 2")
	parser.add_argument("--token-parts", type=int, default=3, help="Number of token subsets (3=thirds)")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for rotation (None=time-based)")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file")
	# New tiered config options
	parser.add_argument(
		"--tier-config", type=str, default=None,
		help="Tiered config: 'clusters,neurons,bits;...' e.g., '100,15,20;400,10,12;rest,5,8'"
	)
	parser.add_argument(
		"--phase-order", type=str, default="neurons_first",
		choices=["neurons_first", "bits_first"],
		help="Phase order: neurons→bits→connections or bits→neurons→connections"
	)
	parser.add_argument(
		"--tier0-only", action="store_true",
		help="Only optimize tier0 clusters (requires --tier-config)"
	)
	args = parser.parse_args()

	# Determine seed: time-based if not specified
	rotation_seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)

	# Parse tier config if provided
	tier_config = None
	if args.tier_config:
		tier_config = []
		for tier_str in args.tier_config.split(";"):
			parts = tier_str.strip().split(",")
			if len(parts) != 3:
				print(f"ERROR: Invalid tier config format: {tier_str}")
				print("Expected: 'clusters,neurons,bits' e.g., '100,15,20'")
				return 1
			num_clusters_str, neurons_str, bits_str = parts
			num_clusters = None if num_clusters_str.lower() == "rest" else int(num_clusters_str)
			neurons = int(neurons_str)
			bits = int(bits_str)
			tier_config.append((num_clusters, neurons, bits))

	# Setup logger
	logger = Logger(name="phased_search")

	logger.header("Phased Architecture Search with Per-Iteration Rotation")
	log(f"  Log file: {logger.log_file}")
	log(f"  Total train tokens: {args.train_tokens:,}")
	log(f"  Per-iteration rotation: 1/{args.token_parts} (~{args.train_tokens // args.token_parts:,} tokens per iteration)")
	log(f"  Rotation seed: {rotation_seed}" + (" (time-based)" if args.seed is None else " (explicit)"))
	log(f"  Eval tokens: {args.eval_tokens:,}")
	log(f"  Context: {args.context}")
	log(f"  GA generations: {args.ga_gens}, population: {args.population}")
	log(f"  TS iterations: {args.ts_iters}, neighbors: {args.neighbors}")
	log(f"  Patience: {args.patience}")
	log(f"  Default bits: {args.default_bits}")
	log(f"  Default neurons: {args.default_neurons}")
	log(f"  Phase order: {args.phase_order}")
	if tier_config:
		log(f"  Tier config: {tier_config}")
		log(f"  Tier0-only: {args.tier0_only}")
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

	# Create configuration
	config = PhasedSearchConfig(
		context_size=args.context,
		token_parts=args.token_parts,
		ga_generations=args.ga_gens,
		ts_iterations=args.ts_iters,
		population_size=args.population,
		neighbors_per_iter=args.neighbors,
		patience=args.patience,
		default_bits=args.default_bits,
		default_neurons=args.default_neurons,
		tier_config=tier_config,
		phase_order=args.phase_order,
		optimize_tier0_only=args.tier0_only,
		rotation_seed=rotation_seed,
		log_path=logger.log_file,
	)

	# Create runner and setup
	runner = PhasedSearchRunner(config=config, logger=log)
	log("")
	runner.setup(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=vocab_size,
	)

	# Run all phases
	results = runner.run_all_phases()

	# Add metadata
	results["args"] = vars(args)
	results["timestamp"] = datetime.now().isoformat()
	results["per_iteration_rotation"] = {
		"num_parts": args.token_parts,
		"tokens_per_part": len(train_tokens) // args.token_parts,
	}

	# Save results
	if args.output:
		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with open(output_path, "w") as f:
			json.dump(results, f, indent=2, default=str)
		log("")
		log(f"Results saved to: {output_path}")

	return 0


if __name__ == "__main__":
	sys.exit(main())
