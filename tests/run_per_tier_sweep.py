#!/usr/bin/env python3
"""
Per-Tier Metrics Sweep - Systematic experiments to understand tier behavior.

This script runs experiments varying bits, neurons, and context to understand
how each configuration affects per-tier PPL and accuracy.

Usage:
    python tests/run_per_tier_sweep.py [--full-data] [--output results.json]
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
	tiered: str  # Format: "100,11,8;400,7,8;rest,5,8"
	context: int = 4
	mode: str = "fast"
	full_data: bool = False
	description: str = ""


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
	"""Parse the benchmark output to extract per-tier results.

	Handles timestamped log output like:
		21:50:53 |   Test PPL: 43339.71
		21:50:53 | Tier 0          100       11    47.5%      38173.3    16.21%
	"""
	import re
	lines = output.split('\n')

	overall = {'ppl': 0.0, 'accuracy': 0.0}
	tier_results = []

	# Find overall results - use regex to handle timestamp prefix
	# Match pattern like "Test PPL: 43339.71" anywhere in line
	for line in lines:
		ppl_match = re.search(r'Test PPL:\s*([\d.]+)', line)
		if ppl_match:
			overall['ppl'] = float(ppl_match.group(1))

		acc_match = re.search(r'Test Acc:\s*([\d.]+)%', line)
		if acc_match:
			overall['accuracy'] = float(acc_match.group(1)) / 100

	# Find per-tier results
	# Match lines like: "Tier 0          100       11    47.5%      38173.3    16.21%"
	# Pattern: Tier <idx> <clusters> <neurons> <data%> <ppl> <acc%>
	tier_pattern = re.compile(
		r'Tier\s+(\d+)\s+'           # Tier index
		r'([\d,]+)\s+'               # Clusters (may have commas)
		r'(\d+)\s+'                  # Neurons
		r'([\d.]+)%\s+'              # Data percentage
		r'([\d.]+)\s+'               # PPL
		r'([\d.]+)%'                 # Accuracy
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
					bits=8,  # Will be updated from config
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
			timeout=1800,  # 30 min timeout
			cwd=Path(__file__).parent.parent,
		)
		elapsed = time.perf_counter() - start_time

		if result.returncode != 0:
			print(f"ERROR: Experiment failed with return code {result.returncode}")
			print(result.stderr)
			return None

		output = result.stdout + result.stderr
		overall, tier_results = parse_tier_results(output)

		return ExperimentResult(
			name=config.name,
			config=config.tiered,
			context=config.context,
			overall_ppl=overall['ppl'],
			overall_accuracy=overall['accuracy'],
			tier_results=tier_results,
			train_time=elapsed,
			eval_time=0.0,  # Included in train_time
			timestamp=datetime.now().isoformat(),
		)

	except subprocess.TimeoutExpired:
		print(f"ERROR: Experiment timed out after 30 minutes")
		return None
	except Exception as e:
		print(f"ERROR: {e}")
		return None


def define_experiments(full_data: bool = False) -> list[ExperimentConfig]:
	"""Define the experiments to run."""
	mode = "full" if full_data else "fast"

	experiments = []

	# ========================================
	# Group 1: Baseline configurations
	# ========================================

	experiments.append(ExperimentConfig(
		name="baseline_8bit",
		tiered="100,11,8;400,7,8;rest,5,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="Baseline: 8-bit all tiers",
	))

	# ========================================
	# Group 2: Bits sweep (uniform across tiers)
	# ========================================

	experiments.append(ExperimentConfig(
		name="bits_10_uniform",
		tiered="100,11,10;400,7,10;rest,5,10",
		context=4,
		mode=mode,
		full_data=full_data,
		description="10-bit uniform",
	))

	experiments.append(ExperimentConfig(
		name="bits_12_uniform",
		tiered="100,11,12;400,7,12;rest,5,12",
		context=4,
		mode=mode,
		full_data=full_data,
		description="12-bit uniform",
	))

	# ========================================
	# Group 3: Bits sweep (asymmetric)
	# ========================================

	experiments.append(ExperimentConfig(
		name="bits_12_tier0_8_rest",
		tiered="100,11,12;400,7,8;rest,5,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="12-bit tier0, 8-bit rest (accuracy for common, smooth for rare)",
	))

	experiments.append(ExperimentConfig(
		name="bits_8_tier0_12_rest",
		tiered="100,11,8;400,7,12;rest,5,12",
		context=4,
		mode=mode,
		full_data=full_data,
		description="8-bit tier0, 12-bit rest (opposite strategy)",
	))

	# ========================================
	# Group 4: Neuron sweep (uniform)
	# ========================================

	experiments.append(ExperimentConfig(
		name="neurons_11_uniform",
		tiered="100,11,8;400,11,8;rest,11,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="11 neurons uniform (max capacity)",
	))

	experiments.append(ExperimentConfig(
		name="neurons_7_uniform",
		tiered="100,7,8;400,7,8;rest,7,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="7 neurons uniform",
	))

	experiments.append(ExperimentConfig(
		name="neurons_5_uniform",
		tiered="100,5,8;400,5,8;rest,5,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="5 neurons uniform",
	))

	experiments.append(ExperimentConfig(
		name="neurons_3_uniform",
		tiered="100,3,8;400,3,8;rest,3,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="3 neurons uniform",
	))

	# ========================================
	# Group 5: Neuron sweep (asymmetric - tier 0 focus)
	# ========================================

	experiments.append(ExperimentConfig(
		name="neurons_11_tier0_3_rest",
		tiered="100,11,8;400,3,8;rest,3,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="Max tier0, minimal rest",
	))

	experiments.append(ExperimentConfig(
		name="neurons_11_tier0_1_rest",
		tiered="100,11,8;400,1,8;rest,1,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="Max tier0, single neuron rest",
	))

	# ========================================
	# Group 6: Context sweep
	# ========================================

	experiments.append(ExperimentConfig(
		name="context_5",
		tiered="100,11,8;400,7,8;rest,5,8",
		context=5,
		mode=mode,
		full_data=full_data,
		description="Context 5 tokens",
	))

	experiments.append(ExperimentConfig(
		name="context_6",
		tiered="100,11,8;400,7,8;rest,5,8",
		context=6,
		mode=mode,
		full_data=full_data,
		description="Context 6 tokens",
	))

	# ========================================
	# Group 7: Combined optimizations
	# ========================================

	experiments.append(ExperimentConfig(
		name="optimal_candidate_1",
		tiered="100,11,12;400,7,10;rest,5,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="Gradient bits: 12→10→8",
	))

	experiments.append(ExperimentConfig(
		name="optimal_candidate_2",
		tiered="100,15,10;400,7,8;rest,3,8",
		context=4,
		mode=mode,
		full_data=full_data,
		description="Focus tier0: 15n, 10b",
	))

	return experiments


def print_summary_table(results: list[ExperimentResult]):
	"""Print a summary table of all results."""
	print("\n" + "="*100)
	print("SUMMARY: Per-Tier Sweep Results")
	print("="*100)

	# Overall results header
	print(f"\n{'Experiment':<30} {'PPL':>12} {'Accuracy':>10} | {'T0 PPL':>10} {'T0 Acc':>8} | {'T1 PPL':>10} {'T1 Acc':>8} | {'T2 PPL':>10} {'T2 Acc':>8}")
	print("-"*100)

	for r in results:
		t0 = next((t for t in r.tier_results if t.tier == 0), None)
		t1 = next((t for t in r.tier_results if t.tier == 1), None)
		t2 = next((t for t in r.tier_results if t.tier == 2), None)

		t0_ppl = f"{t0.ppl:.0f}" if t0 else "N/A"
		t0_acc = f"{t0.accuracy:.2%}" if t0 else "N/A"
		t1_ppl = f"{t1.ppl:.0f}" if t1 else "N/A"
		t1_acc = f"{t1.accuracy:.2%}" if t1 else "N/A"
		t2_ppl = f"{t2.ppl:.0f}" if t2 else "N/A"
		t2_acc = f"{t2.accuracy:.2%}" if t2 else "N/A"

		print(f"{r.name:<30} {r.overall_ppl:>12.1f} {r.overall_accuracy:>10.2%} | "
			  f"{t0_ppl:>10} {t0_acc:>8} | {t1_ppl:>10} {t1_acc:>8} | {t2_ppl:>10} {t2_acc:>8}")

	print("-"*100)


def main():
	parser = argparse.ArgumentParser(description="Per-Tier Metrics Sweep")
	parser.add_argument("--full-data", action="store_true",
		help="Use full dataset (slower but more accurate)")
	parser.add_argument("--output", type=str, default="per_tier_sweep_results.json",
		help="Output JSON file for results")
	parser.add_argument("--experiments", type=str, default=None,
		help="Comma-separated list of experiment names to run (default: all)")
	args = parser.parse_args()

	# Define experiments
	all_experiments = define_experiments(full_data=args.full_data)

	# Filter if specific experiments requested
	if args.experiments:
		names = [n.strip() for n in args.experiments.split(',')]
		experiments = [e for e in all_experiments if e.name in names]
		if not experiments:
			print(f"ERROR: No experiments matched: {names}")
			print(f"Available: {[e.name for e in all_experiments]}")
			return 1
	else:
		experiments = all_experiments

	print(f"\n{'#'*70}")
	print(f"# Per-Tier Metrics Sweep")
	print(f"# {len(experiments)} experiments to run")
	print(f"# Mode: {'full' if args.full_data else 'fast'}")
	print(f"# Output: {args.output}")
	print(f"{'#'*70}\n")

	results = []
	for i, config in enumerate(experiments):
		print(f"\n[{i+1}/{len(experiments)}] {config.name}: {config.description}")
		result = run_experiment(config)
		if result:
			results.append(result)
			# Save intermediate results
			with open(args.output, 'w') as f:
				json.dump([asdict(r) for r in results], f, indent=2, default=str)

	# Print summary
	if results:
		print_summary_table(results)

		# Save final results
		output_path = Path(args.output)
		with open(output_path, 'w') as f:
			json.dump([asdict(r) for r in results], f, indent=2, default=str)
		print(f"\nResults saved to: {output_path}")
	else:
		print("\nNo successful experiments!")
		return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
