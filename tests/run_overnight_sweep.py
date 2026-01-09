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

    # With GA+TS optimization (captures improvement metrics)
    python tests/run_overnight_sweep.py --full-data --set quick --optimize

    # With specific optimization strategy
    python tests/run_overnight_sweep.py --full-data --set quick --optimize --strategy GA
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
	# Optimization settings
	optimize: bool = False
	strategy: str = "GA,TS"  # GA, TS, or GA,TS


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
class OptimizationResult:
	"""Results from optimization phase."""
	strategy: str
	initial_ce: float
	final_ce: float
	improvement_pct: float
	iterations: int


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
	# Optimization results (None if not optimized)
	initial_val_ppl: Optional[float] = None  # Validation PPL before optimization
	final_val_ppl: Optional[float] = None    # Validation PPL after optimization (same as overall_ppl)
	val_ppl_improvement_pct: Optional[float] = None  # Actual PPL improvement on validation
	optimization_results: Optional[list[OptimizationResult]] = None
	total_ce_improvement_pct: Optional[float] = None  # CE improvement from GA+TS


def parse_tier_results(output: str) -> tuple[dict, list[TierResult], dict]:
	"""Parse the benchmark output to extract per-tier results and optimization data."""
	import re
	lines = output.split('\n')

	overall = {'ppl': 0.0, 'accuracy': 0.0}
	tier_results = []
	opt_data = {
		'initial_val_ppl': None,
		'final_val_ppl': None,
		'optimizations': [],
		'total_ce_improvement': None,
		'val_ppl_improvement': None,
	}

	# Track which section we're in for PPL parsing
	in_initial_eval = False
	in_final_eval = False

	for i, line in enumerate(lines):
		# Track sections
		if 'Initial Evaluation (Validation Set)' in line:
			in_initial_eval = True
			in_final_eval = False
		elif 'Final Evaluation (Validation Set)' in line:
			in_initial_eval = False
			in_final_eval = True
		elif 'Final Evaluation (Test Set)' in line:
			in_initial_eval = False
			in_final_eval = False

		# Parse PPL based on section
		ppl_match = re.search(r'Perplexity:\s*([\d.]+)', line)
		if ppl_match:
			ppl_val = float(ppl_match.group(1))
			if in_initial_eval and opt_data['initial_val_ppl'] is None:
				opt_data['initial_val_ppl'] = ppl_val
			elif in_final_eval and opt_data['final_val_ppl'] is None:
				opt_data['final_val_ppl'] = ppl_val

		# Test PPL for overall results
		test_ppl_match = re.search(r'Test PPL:\s*([\d.]+)', line)
		if test_ppl_match:
			overall['ppl'] = float(test_ppl_match.group(1))

		acc_match = re.search(r'Test Acc:\s*([\d.]+)%', line)
		if acc_match:
			overall['accuracy'] = float(acc_match.group(1)) / 100

	# Parse optimization results (GA/TS)
	# Format: "[Joint tiered] GA Results:" or "[Joint tiered] TS Results:"
	# "    Initial CE: 10.7152"
	# "    Final CE: 10.7094"
	# "    Improvement: 0.05%"
	# "    Iterations: 5"
	current_strategy = None
	current_opt = {}

	for i, line in enumerate(lines):
		# Detect strategy result section
		strat_match = re.search(r'\[(.*?)\]\s*(GA|TS)\s*Results:', line)
		if strat_match:
			if current_strategy and current_opt:
				opt_data['optimizations'].append(OptimizationResult(
					strategy=current_strategy,
					initial_ce=current_opt.get('initial_ce', 0),
					final_ce=current_opt.get('final_ce', 0),
					improvement_pct=current_opt.get('improvement', 0),
					iterations=current_opt.get('iterations', 0),
				))
			current_strategy = strat_match.group(2)
			current_opt = {}

		if current_strategy:
			init_ce_match = re.search(r'Initial CE:\s*([\d.]+)', line)
			if init_ce_match:
				current_opt['initial_ce'] = float(init_ce_match.group(1))

			final_ce_match = re.search(r'Final CE:\s*([\d.]+)', line)
			if final_ce_match:
				current_opt['final_ce'] = float(final_ce_match.group(1))

			imp_match = re.search(r'Improvement:\s*([\d.]+)%', line)
			if imp_match:
				current_opt['improvement'] = float(imp_match.group(1))

			iter_match = re.search(r'Iterations:\s*(\d+)', line)
			if iter_match:
				current_opt['iterations'] = int(iter_match.group(1))

	# Add last strategy if present
	if current_strategy and current_opt:
		opt_data['optimizations'].append(OptimizationResult(
			strategy=current_strategy,
			initial_ce=current_opt.get('initial_ce', 0),
			final_ce=current_opt.get('final_ce', 0),
			improvement_pct=current_opt.get('improvement', 0),
			iterations=current_opt.get('iterations', 0),
		))

	# Calculate total CE improvement from GA+TS
	if opt_data['optimizations']:
		total_imp = sum(o.improvement_pct for o in opt_data['optimizations'])
		opt_data['total_ce_improvement'] = total_imp

	# Calculate validation PPL improvement (lower is better, so improvement = (initial - final) / initial)
	if opt_data['initial_val_ppl'] and opt_data['final_val_ppl']:
		init_ppl = opt_data['initial_val_ppl']
		final_ppl = opt_data['final_val_ppl']
		if init_ppl > 0:
			opt_data['val_ppl_improvement'] = ((init_ppl - final_ppl) / init_ppl) * 100

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

	return overall, tier_results, opt_data


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
	"""Run a single experiment and return results."""
	print(f"\n{'='*70}")
	print(f"Running: {config.name}")
	print(f"Config: {config.tiered}")
	print(f"Context: {config.context}")
	print(f"Optimize: {config.optimize} ({config.strategy})" if config.optimize else "Optimize: False")
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

	if config.optimize:
		cmd.append("--optimize")
		cmd.extend(["--strategy", config.strategy])

	start_time = time.perf_counter()

	# Longer timeout for optimized experiments
	timeout = 14400 if config.optimize else 7200  # 4 hours vs 2 hours

	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=timeout,
			cwd=Path(__file__).parent.parent,
		)
		elapsed = time.perf_counter() - start_time

		if result.returncode != 0:
			print(f"ERROR: Experiment failed with return code {result.returncode}")
			print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
			return None

		output = result.stdout + result.stderr
		overall, tier_results, opt_data = parse_tier_results(output)

		print(f"\nCompleted in {elapsed/60:.1f} minutes")
		print(f"Overall PPL: {overall['ppl']:.1f}, Accuracy: {overall['accuracy']:.2%}")
		if opt_data['optimizations']:
			for opt in opt_data['optimizations']:
				print(f"  {opt.strategy}: {opt.improvement_pct:.2f}% CE improvement")
			if opt_data['total_ce_improvement']:
				print(f"  Total CE: {opt_data['total_ce_improvement']:.2f}% improvement")
		if opt_data['val_ppl_improvement']:
			print(f"  Val PPL: {opt_data['initial_val_ppl']:.0f} → {opt_data['final_val_ppl']:.0f} ({opt_data['val_ppl_improvement']:.2f}% improvement)")

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
			initial_val_ppl=opt_data['initial_val_ppl'],
			final_val_ppl=opt_data['final_val_ppl'],
			val_ppl_improvement_pct=opt_data['val_ppl_improvement'],
			optimization_results=opt_data['optimizations'] if opt_data['optimizations'] else None,
			total_ce_improvement_pct=opt_data['total_ce_improvement'],
		)

	except subprocess.TimeoutExpired:
		timeout_hrs = timeout / 3600
		print(f"ERROR: Experiment timed out after {timeout_hrs:.0f} hours")
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
	# Check if any results have optimization data
	has_opt = any(r.optimization_results for r in results)

	print("\n" + "="*200)
	print("OVERNIGHT SWEEP RESULTS")
	print("="*200)

	if has_opt:
		# CE % = cross-entropy improvement (GA+TS optimizer fitness)
		# PPL % = actual validation PPL improvement
		print(f"\n{'Experiment':<20} {'Config':<26} {'PPL':>7} {'Acc':>5} | {'GA%':>5} {'TS%':>5} {'Val%':>5} | {'T0 PPL':>7} {'T0 Ac':>5} {'T1 PPL':>7} {'T1 Ac':>5} {'T2 PPL':>7} {'T2 Ac':>5}")
		print("-"*200)
	else:
		print(f"\n{'Experiment':<20} {'Config':<30} {'PPL':>8} {'Acc':>6} | {'T0 PPL':>7} {'T0 Ac':>5} {'T1 PPL':>7} {'T1 Ac':>5} {'T2 PPL':>7} {'T2 Ac':>5}")
		print("-"*150)

	for r in sorted(results, key=lambda x: x.overall_ppl):
		t0 = next((t for t in r.tier_results if t.tier == 0), None)
		t1 = next((t for t in r.tier_results if t.tier == 1), None)
		t2 = next((t for t in r.tier_results if t.tier == 2), None)

		t0_ppl = f"{t0.ppl:.0f}" if t0 else "-"
		t0_acc = f"{t0.accuracy:.1%}" if t0 else "-"
		t1_ppl = f"{t1.ppl:.0f}" if t1 else "-"
		t1_acc = f"{t1.accuracy:.1%}" if t1 else "-"
		t2_ppl = f"{t2.ppl:.0f}" if t2 else "-"
		t2_acc = f"{t2.accuracy:.1%}" if t2 else "-"

		if has_opt:
			ga_imp = ts_imp = val_imp = "-"
			if r.optimization_results:
				for opt in r.optimization_results:
					if opt.strategy == "GA":
						ga_imp = f"{opt.improvement_pct:.1f}"
					elif opt.strategy == "TS":
						ts_imp = f"{opt.improvement_pct:.1f}"
			if r.val_ppl_improvement_pct is not None:
				val_imp = f"{r.val_ppl_improvement_pct:.1f}"

			print(f"{r.name:<20} {r.config:<26} {r.overall_ppl:>7.0f} {r.overall_accuracy:>5.1%} | {ga_imp:>5} {ts_imp:>5} {val_imp:>5} | {t0_ppl:>7} {t0_acc:>5} {t1_ppl:>7} {t1_acc:>5} {t2_ppl:>7} {t2_acc:>5}")
		else:
			print(f"{r.name:<20} {r.config:<30} {r.overall_ppl:>8.0f} {r.overall_accuracy:>6.1%} | {t0_ppl:>7} {t0_acc:>5} {t1_ppl:>7} {t1_acc:>5} {t2_ppl:>7} {t2_acc:>5}")

	if has_opt:
		print("-"*200)
	else:
		print("-"*150)

	# Best result
	best = min(results, key=lambda x: x.overall_ppl)
	print(f"\nBest PPL: {best.name} with {best.overall_ppl:.1f}")

	# Best optimization improvement if available
	if has_opt:
		# Best CE improvement
		opt_results = [r for r in results if r.total_ce_improvement_pct]
		if opt_results:
			best_ce = max(opt_results, key=lambda x: x.total_ce_improvement_pct or 0)
			print(f"Best CE improvement: {best_ce.name} with {best_ce.total_ce_improvement_pct:.2f}%")

		# Best Val PPL improvement
		ppl_results = [r for r in results if r.val_ppl_improvement_pct]
		if ppl_results:
			best_ppl = max(ppl_results, key=lambda x: x.val_ppl_improvement_pct or 0)
			print(f"Best Val PPL improvement: {best_ppl.name} with {best_ppl.val_ppl_improvement_pct:.2f}%")


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
		# Optimization adjustment: +60-90 min for GA+TS
		opt_adj = 75 if exp.optimize else 0
		total_minutes += base + bits_adj + ctx_adj + opt_adj
	return total_minutes / 60


def load_completed_experiments(output_path: Path) -> set[str]:
	"""Load names of already-completed experiments from results file."""
	if not output_path.exists():
		return set()
	try:
		with open(output_path) as f:
			data = json.load(f)
		return {r['name'] for r in data}
	except (json.JSONDecodeError, KeyError):
		return set()


def load_existing_results(output_path: Path) -> list[dict]:
	"""Load existing results for merging."""
	if not output_path.exists():
		return []
	try:
		with open(output_path) as f:
			return json.load(f)
	except json.JSONDecodeError:
		return []


def deserialize_result(r: dict) -> ExperimentResult:
	"""Deserialize a result dict to ExperimentResult, handling optional fields."""
	tier_results = [TierResult(**t) for t in r.get('tier_results', [])]

	# Handle optimization results if present
	opt_results = None
	if r.get('optimization_results'):
		opt_results = [OptimizationResult(**o) for o in r['optimization_results']]

	return ExperimentResult(
		name=r['name'],
		config=r['config'],
		context=r['context'],
		overall_ppl=r['overall_ppl'],
		overall_accuracy=r['overall_accuracy'],
		tier_results=tier_results,
		train_time=r.get('train_time', 0.0),
		eval_time=r.get('eval_time', 0.0),
		timestamp=r.get('timestamp', ''),
		initial_val_ppl=r.get('initial_val_ppl') or r.get('initial_ppl'),  # backwards compat
		final_val_ppl=r.get('final_val_ppl'),
		val_ppl_improvement_pct=r.get('val_ppl_improvement_pct'),
		optimization_results=opt_results,
		total_ce_improvement_pct=r.get('total_ce_improvement_pct') or r.get('total_improvement_pct'),  # backwards compat
	)


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
	parser.add_argument("--skip-completed", action="store_true", default=True,
		help="Skip experiments already in output file (default: True)")
	parser.add_argument("--force-rerun", action="store_true",
		help="Re-run all experiments even if already completed")
	parser.add_argument("--optimize", action="store_true",
		help="Run GA+TS connectivity optimization after training")
	parser.add_argument("--strategy", type=str, default="GA,TS",
		help="Optimization strategy: GA, TS, or GA,TS (default: GA,TS)")
	args = parser.parse_args()

	# Define experiments
	all_experiments = define_experiments()

	# Output path
	output_path = Path("experiments") / args.output

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

	# Skip completed experiments (unless --force-rerun)
	completed = set()
	if args.skip_completed and not args.force_rerun:
		completed = load_completed_experiments(output_path)
		if completed:
			original_count = len(experiments)
			experiments = [e for e in experiments if e.name not in completed]
			skipped = original_count - len(experiments)
			if skipped > 0:
				print(f"\nSkipping {skipped} already-completed experiments: {sorted(completed)}")

	if not experiments:
		print("\nAll experiments already completed! Use --force-rerun to re-run them.")
		# Show existing results
		existing = load_existing_results(output_path)
		if existing:
			print_summary_table([deserialize_result(r) for r in existing])
		return 0

	# Update mode based on full-data flag and optimization settings
	mode = "full" if args.full_data else "fast"
	for exp in experiments:
		exp.mode = mode
		exp.full_data = args.full_data
		exp.optimize = args.optimize
		exp.strategy = args.strategy

	# Estimate runtime
	est_hours = estimate_runtime(experiments)

	print(f"\n{'#'*70}")
	print(f"# Overnight High-Capacity Sweep")
	print(f"# Set: {args.set} ({len(experiments)} new + {len(completed)} completed)")
	print(f"# Mode: {mode}")
	if args.optimize:
		print(f"# Optimization: {args.strategy}")
	print(f"# Estimated runtime: {est_hours:.1f} hours")
	print(f"# Output: {args.output}")
	print(f"{'#'*70}")

	print(f"\nExperiments to run:")
	for i, exp in enumerate(experiments, 1):
		print(f"  {i}. {exp.name}: {exp.description}")

	print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"Expected completion: ~{est_hours:.1f} hours\n")

	# Load existing results for merging
	existing_results = load_existing_results(output_path)
	results = []

	for i, config in enumerate(experiments):
		print(f"\n[{i+1}/{len(experiments)}] {config.name}")
		result = run_experiment(config)
		if result:
			results.append(result)
			# Merge with existing and save
			merged = existing_results + [asdict(r) for r in results]
			output_path.parent.mkdir(exist_ok=True)
			with open(output_path, 'w') as f:
				json.dump(merged, f, indent=2, default=str)
			print(f"Saved to {output_path} ({len(existing_results)} previous + {len(results)} new)")

	# Print summary (new experiments only)
	if results:
		print("\n--- New Experiments ---")
		print_summary_table(results)

	# Print combined summary if there were previous results
	if existing_results:
		all_results = [deserialize_result(r) for r in existing_results] + results
		print("\n--- All Experiments (Previous + New) ---")
		print_summary_table(all_results)

	if results:
		print(f"\nResults saved to: {output_path}")
		print(f"  Previous: {len(existing_results)}, New: {len(results)}, Total: {len(existing_results) + len(results)}")
	else:
		print("\nNo new experiments completed!")
		if not existing_results:
			return 1

	return 0


if __name__ == "__main__":
	sys.exit(main())
