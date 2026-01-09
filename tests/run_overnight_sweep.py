#!/usr/bin/env python3
"""
Overnight Sweep - High-capacity experiments using SPARSE backend.

These experiments push beyond the 10-bit dense limit using the new
SPARSE memory backend (Rust FxHashMap) for 12-20 bits per neuron.

Usage:
    # Quick overnight (~4-6 hours, 4 experiments)
    python tests/run_overnight_sweep.py --full-data --set quick

    # Standard overnight (~8-10 hours, 6 experiments)
    python tests/run_overnight_sweep.py --full-data --set standard

    # Extended weekend run (~16-20 hours, 10 experiments)
    python tests/run_overnight_sweep.py --full-data --set extended

    # Single experiment test
    python tests/run_overnight_sweep.py --experiments tier0_16bit
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
	"""Configuration for a single experiment."""
	name: str
	tiered: str
	context: int = 4
	mode: str = "full"
	full_data: bool = True
	description: str = ""
	priority: int = 1  # 1=quick, 2=standard, 3=extended


@dataclass
class TierResult:
	"""Results for a single tier."""
	tier: int
	name: str
	clusters: int
	neurons: int
	bits: int
	data_pct: float
	ppl: float
	accuracy: float


@dataclass
class ExperimentResult:
	"""Results from a single experiment."""
	name: str
	config: str
	context: int
	overall_ppl: float
	overall_accuracy: float
	tier_results: list[TierResult]
	train_time: float
	eval_time: float
	timestamp: str


def parse_tier_results(output: str) -> tuple[dict, list[TierResult]]:
	"""Parse the benchmark output to extract per-tier results."""
	import re
	lines = output.split('\n')

	overall = {'ppl': 0.0, 'accuracy': 0.0}
	tier_results = []

	for line in lines:
		ppl_match = re.search(r'Test PPL:\s*([\d.]+)', line)
		if ppl_match:
			overall['ppl'] = float(ppl_match.group(1))

		acc_match = re.search(r'Test Acc:\s*([\d.]+)%', line)
		if acc_match:
			overall['accuracy'] = float(acc_match.group(1)) / 100

	tier_pattern = re.compile(
		r'Tier\s+(\d+)\s+'
		r'([\d,]+)\s+'
		r'(\d+)\s+'
		r'([\d.]+)%\s+'
		r'([\d.]+)\s+'
		r'([\d.]+)%'
	)

	in_tier_section = False
	for line in lines:
		if 'Per-Tier Test Results' in line:
			in_tier_section = True
			continue
		if in_tier_section and 'TOTAL' in line:
			in_tier_section = False
			continue

		if in_tier_section:
			match = tier_pattern.search(line)
			if match:
				tier_idx = int(match.group(1))
				clusters = int(match.group(2).replace(',', ''))
				neurons = int(match.group(3))
				data_pct = float(match.group(4))
				ppl = float(match.group(5))
				accuracy = float(match.group(6)) / 100

				tier_results.append(TierResult(
					tier=tier_idx,
					name=f"tier_{tier_idx}",
					clusters=clusters,
					neurons=neurons,
					bits=8,
					data_pct=data_pct,
					ppl=ppl,
					accuracy=accuracy,
				))

	return overall, tier_results


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
	"""Run a single experiment and return results."""
	print(f"\n{'='*70}")
	print(f"Running: {config.name}")
	print(f"Config: {config.tiered}")
	print(f"Context: {config.context}")
	print(f"Description: {config.description}")
	print(f"{'='*70}\n")

	cmd = [
		sys.executable, "tests/ramlm_full_benchmark.py",
		"--mode", config.mode,
		"--tiered", config.tiered,
		"--context", str(config.context),
		"--per-tier",
	]

	if config.full_data:
		cmd.append("--full-data")

	start_time = time.perf_counter()

	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=7200,  # 2 hour timeout per experiment
			cwd=Path(__file__).parent.parent,
		)
		elapsed = time.perf_counter() - start_time

		if result.returncode != 0:
			print(f"ERROR: Experiment failed with return code {result.returncode}")
			print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
			return None

		output = result.stdout + result.stderr
		overall, tier_results = parse_tier_results(output)

		print(f"\nCompleted in {elapsed/60:.1f} minutes")
		print(f"Overall PPL: {overall['ppl']:.1f}, Accuracy: {overall['accuracy']:.2%}")

		return ExperimentResult(
			name=config.name,
			config=config.tiered,
			context=config.context,
			overall_ppl=overall['ppl'],
			overall_accuracy=overall['accuracy'],
			tier_results=tier_results,
			train_time=elapsed,
			eval_time=0.0,
			timestamp=datetime.now().isoformat(),
		)

	except subprocess.TimeoutExpired:
		print(f"ERROR: Experiment timed out after 2 hours")
		return None
	except Exception as e:
		print(f"ERROR: {e}")
		return None


def define_experiments() -> list[ExperimentConfig]:
	"""Define overnight experiments with priority levels."""
	experiments = []

	# ========================================
	# Priority 1: Quick overnight (~4-6 hours)
	# ========================================

	experiments.append(ExperimentConfig(
		name="tier0_16bit",
		tiered="100,15,16;400,10,10;rest,5,8",
		context=4,
		description="Tier0: 15 neurons, 16 bits (SPARSE) - best capacity boost",
		priority=1,
	))

	experiments.append(ExperimentConfig(
		name="balanced_14bit",
		tiered="100,12,14;400,8,12;rest,5,10",
		context=4,
		description="Balanced SPARSE: 14→12→10 bits gradient",
		priority=1,
	))

	experiments.append(ExperimentConfig(
		name="neurons_20_tier0",
		tiered="100,20,12;400,12,10;rest,7,8",
		context=4,
		description="High neurons: 20→12→7 with moderate bits",
		priority=1,
	))

	experiments.append(ExperimentConfig(
		name="context_6_sparse",
		tiered="100,12,14;400,8,10;rest,5,8",
		context=6,
		description="Context 6 with SPARSE tier0/tier1",
		priority=1,
	))

	# ========================================
	# Priority 2: Standard overnight (~8-10 hours)
	# ========================================

	experiments.append(ExperimentConfig(
		name="tier0_18bit",
		tiered="100,15,18;400,10,12;rest,5,8",
		context=4,
		description="Tier0: 18 bits (262K addresses per neuron)",
		priority=2,
	))

	experiments.append(ExperimentConfig(
		name="neurons_25_gradient",
		tiered="100,25,14;400,15,10;rest,7,8",
		context=4,
		description="Max neurons: 25→15→7 gradient",
		priority=2,
	))

	# ========================================
	# Priority 3: Extended/weekend (~16-20 hours)
	# ========================================

	experiments.append(ExperimentConfig(
		name="tier0_20bit",
		tiered="100,15,20;400,10,12;rest,5,8",
		context=4,
		description="Tier0: 20 bits (1M addresses per neuron)",
		priority=3,
	))

	experiments.append(ExperimentConfig(
		name="all_sparse_16bit",
		tiered="100,15,16;400,12,16;rest,8,16",
		context=4,
		description="All tiers 16-bit SPARSE",
		priority=3,
	))

	experiments.append(ExperimentConfig(
		name="context_8_high_cap",
		tiered="100,15,12;400,10,10;rest,7,8",
		context=8,
		description="Context 8 with high capacity",
		priority=3,
	))

	experiments.append(ExperimentConfig(
		name="extreme_tier0",
		tiered="100,30,16;400,12,12;rest,5,8",
		context=5,
		description="Extreme: 30 neurons, 16 bits tier0",
		priority=3,
	))

	return experiments


def print_summary_table(results: list[ExperimentResult]):
	"""Print a summary table of all results."""
	print("\n" + "="*110)
	print("OVERNIGHT SWEEP RESULTS")
	print("="*110)

	print(f"\n{'Experiment':<25} {'Config':<35} {'PPL':>10} {'Acc':>8} | {'T0 PPL':>9} {'T0 Acc':>7}")
	print("-"*110)

	for r in sorted(results, key=lambda x: x.overall_ppl):
		t0 = next((t for t in r.tier_results if t.tier == 0), None)
		t0_ppl = f"{t0.ppl:.0f}" if t0 else "N/A"
		t0_acc = f"{t0.accuracy:.1%}" if t0 else "N/A"

		print(f"{r.name:<25} {r.config:<35} {r.overall_ppl:>10.1f} {r.overall_accuracy:>8.2%} | {t0_ppl:>9} {t0_acc:>7}")

	print("-"*110)

	# Best result
	best = min(results, key=lambda x: x.overall_ppl)
	print(f"\nBest PPL: {best.name} with {best.overall_ppl:.1f}")


def estimate_runtime(experiments: list[ExperimentConfig]) -> float:
	"""Estimate total runtime in hours (rough estimate)."""
	# Base estimate: ~45 min per full-mode experiment
	# Adjust for bits (more bits = more training time for sparse)
	total_minutes = 0
	for exp in experiments:
		base = 45  # minutes
		# Parse max bits from config
		parts = exp.tiered.split(';')
		max_bits = max(int(p.split(',')[2]) for p in parts)
		# Bits adjustment: +5 min per 2 bits above 10
		bits_adj = max(0, (max_bits - 10) // 2) * 5
		# Context adjustment: +10 min per context above 4
		ctx_adj = max(0, exp.context - 4) * 10
		total_minutes += base + bits_adj + ctx_adj
	return total_minutes / 60


def main():
	parser = argparse.ArgumentParser(description="Overnight High-Capacity Sweep")
	parser.add_argument("--full-data", action="store_true",
		help="Use full dataset (required for meaningful results)")
	parser.add_argument("--output", type=str, default="overnight_sweep_results.json",
		help="Output JSON file for results")
	parser.add_argument("--set", type=str, choices=["quick", "standard", "extended"],
		default="standard", help="Experiment set to run")
	parser.add_argument("--experiments", type=str, default=None,
		help="Comma-separated list of specific experiments to run")
	args = parser.parse_args()

	# Define experiments
	all_experiments = define_experiments()

	# Filter by set or specific experiments
	if args.experiments:
		names = [n.strip() for n in args.experiments.split(',')]
		experiments = [e for e in all_experiments if e.name in names]
		if not experiments:
			print(f"ERROR: No experiments matched: {names}")
			print(f"Available: {[e.name for e in all_experiments]}")
			return 1
	else:
		max_priority = {"quick": 1, "standard": 2, "extended": 3}[args.set]
		experiments = [e for e in all_experiments if e.priority <= max_priority]

	# Update mode based on full-data flag
	mode = "full" if args.full_data else "fast"
	for exp in experiments:
		exp.mode = mode
		exp.full_data = args.full_data

	# Estimate runtime
	est_hours = estimate_runtime(experiments)

	print(f"\n{'#'*70}")
	print(f"# Overnight High-Capacity Sweep")
	print(f"# Set: {args.set} ({len(experiments)} experiments)")
	print(f"# Mode: {mode}")
	print(f"# Estimated runtime: {est_hours:.1f} hours")
	print(f"# Output: {args.output}")
	print(f"{'#'*70}")

	print(f"\nExperiments to run:")
	for i, exp in enumerate(experiments, 1):
		print(f"  {i}. {exp.name}: {exp.description}")

	print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"Expected completion: ~{est_hours:.1f} hours\n")

	results = []
	for i, config in enumerate(experiments):
		print(f"\n[{i+1}/{len(experiments)}] {config.name}")
		result = run_experiment(config)
		if result:
			results.append(result)
			# Save intermediate results
			output_path = Path("experiments") / args.output
			output_path.parent.mkdir(exist_ok=True)
			with open(output_path, 'w') as f:
				json.dump([asdict(r) for r in results], f, indent=2, default=str)
			print(f"Saved intermediate results to {output_path}")

	# Print summary
	if results:
		print_summary_table(results)

		output_path = Path("experiments") / args.output
		with open(output_path, 'w') as f:
			json.dump([asdict(r) for r in results], f, indent=2, default=str)
		print(f"\nFinal results saved to: {output_path}")
	else:
		print("\nNo successful experiments!")
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
