#!/usr/bin/env python3
"""
Bitwise Context Sweep: Full 7-phase optimization for top-K configs across context sizes.

Reads the top-K (neurons, bits) winners from a previous bitwise optimization run,
then runs the full 7-phase pipeline for each (config, context_size) pair.

Context=4 is skipped (already done in the baseline run) and its results are pulled
into the final comparison table automatically.

Usage:
	python run_bitwise_context_sweep.py \
		--input experiments/bitwise_overnight_v3.json \
		--top-k 3 \
		--ga-gens 250 --ts-iters 250 --patience 5 \
		--output experiments/bitwise_context_sweep.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Import shared helpers from the single-context optimization script
from run_bitwise_optimization import (
	load_wikitext2_tokens,
	create_evaluators,
	create_seed_genome,
	run_ga_phase,
	run_ts_phase,
	evaluate_phase_top_k,
	print_phase_comparison,
)


# ---------------------------------------------------------------------------
# Single-context full pipeline (phases 2-7)
# ---------------------------------------------------------------------------

def run_full_pipeline(
	neurons, bits, context_size, train_tokens, test_tokens, validation_tokens,
	vocab_size, num_clusters, args,
):
	"""Run full 7-phase optimization for one (neurons, bits, context) config.

	Returns a dict with per-phase results and phase_metrics for the comparison table.
	"""
	from wnn.ram.strategies.connectivity.architecture_strategies import ArchitectureConfig

	total_input_bits = context_size * num_clusters
	ci = args.check_interval

	print(f"\n{'#'*80}")
	print(f"  Config: n={neurons}, b={bits}, context={context_size}")
	print(f"  Total input bits: {total_input_bits}")
	print(f"{'#'*80}")

	# Create evaluators for this context size
	opt_evaluator, full_evaluator = create_evaluators(
		train_tokens, test_tokens, validation_tokens, vocab_size,
		context_size=context_size, rate=args.rate,
		default_neurons=neurons, default_bits=bits,
		num_train_parts=args.train_parts, num_eval_parts=args.eval_parts,
	)

	# Create initial population (all same architecture, different random connections)
	initial_population = [
		create_seed_genome(num_clusters, bits, neurons, total_input_bits)
		for _ in range(args.population)
	]
	best_genome = initial_population[0]
	best_ce = None

	# Baseline on full validation
	phase_metrics_list = []
	baseline_ce, baseline_acc = full_evaluator.evaluate_single_full(best_genome)
	print(f"  Baseline CE={baseline_ce:.4f}, Acc={baseline_acc:.2%}, PPL={math.exp(baseline_ce):.0f}")
	baseline_entry = {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)}
	phase_metrics_list.append({
		"phase_name": "Init Population",
		"n_full_evaluated": 1,
		"n_population": len(initial_population),
		"best_ce": baseline_entry,
		"best_acc": baseline_entry,
		"best_fitness": baseline_entry,
	})

	results = {
		"neurons": neurons,
		"bits": bits,
		"context_size": context_size,
		"total_input_bits": total_input_bits,
		"baseline": {"ce": baseline_ce, "acc": baseline_acc, "ppl": math.exp(baseline_ce)},
	}

	def log(msg):
		print(f"  {msg}")

	# ── Phase 2: GA Neurons (bits fixed) ─────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 2: GA Neurons (bits={bits})")
	print(f"{'='*70}")

	result, elapsed = run_ga_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_neurons=True, optimize_bits=False, optimize_connections=False,
			default_bits=bits, default_neurons=neurons,
			min_neurons=10, max_neurons=300,
			min_bits=bits, max_bits=bits,
			total_input_bits=total_input_bits,
		),
		ga_gens=args.ga_gens, population=args.population, patience=args.patience,
		seed_genome=best_genome, phase_name="Phase 2: GA Neurons",
		logger=log, check_interval=ci, initial_population=initial_population,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, f"P2 GA Neurons (bits={bits})")
	phase_metrics_list.append(metrics)
	results["phase2_ga_neurons"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# ── Phase 3: TS Neurons ──────────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 3: TS Neurons")
	print(f"{'='*70}")

	result, elapsed = run_ts_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_neurons=True, optimize_bits=False, optimize_connections=False,
			default_bits=bits,
			min_neurons=10, max_neurons=300,
			min_bits=bits, max_bits=bits,
			total_input_bits=total_input_bits,
		),
		ts_iters=args.ts_iters, neighbors=args.neighbors, patience=args.patience,
		seed_genome=best_genome, seed_fitness=best_ce,
		phase_name="Phase 3: TS Neurons", logger=log, check_interval=ci,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, "P3 TS Neurons")
	phase_metrics_list.append(metrics)
	results["phase3_ts_neurons"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# ── Phase 4: GA Bits ─────────────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 4: GA Bits")
	print(f"{'='*70}")

	result, elapsed = run_ga_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=True, optimize_neurons=False, optimize_connections=False,
			min_bits=10, max_bits=24,
			total_input_bits=total_input_bits,
		),
		ga_gens=args.ga_gens, population=args.population, patience=args.patience,
		seed_genome=best_genome, phase_name="Phase 4: GA Bits",
		logger=log, check_interval=ci,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, "P4 GA Bits")
	phase_metrics_list.append(metrics)
	results["phase4_ga_bits"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# ── Phase 5: TS Bits ─────────────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 5: TS Bits")
	print(f"{'='*70}")

	result, elapsed = run_ts_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=True, optimize_neurons=False, optimize_connections=False,
			min_bits=10, max_bits=24,
			total_input_bits=total_input_bits,
		),
		ts_iters=args.ts_iters, neighbors=args.neighbors, patience=args.patience,
		seed_genome=best_genome, seed_fitness=best_ce,
		phase_name="Phase 5: TS Bits", logger=log, check_interval=ci,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, "P5 TS Bits")
	phase_metrics_list.append(metrics)
	results["phase5_ts_bits"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# ── Phase 6: GA Connections ──────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 6: GA Connections")
	print(f"{'='*70}")

	result, elapsed = run_ga_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False, optimize_neurons=False, optimize_connections=True,
			total_input_bits=total_input_bits,
		),
		ga_gens=args.ga_gens, population=args.population, patience=args.patience,
		seed_genome=best_genome, phase_name="Phase 6: GA Connections",
		logger=log, check_interval=ci,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, "P6 GA Connections")
	phase_metrics_list.append(metrics)
	results["phase6_ga_connections"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# ── Phase 7: TS Connections ──────────────────────────────────────────
	print(f"\n{'='*70}")
	print(f"Phase 7: TS Connections")
	print(f"{'='*70}")

	result, elapsed = run_ts_phase(
		opt_evaluator,
		ArchitectureConfig(
			num_clusters=num_clusters,
			optimize_bits=False, optimize_neurons=False, optimize_connections=True,
			total_input_bits=total_input_bits,
		),
		ts_iters=args.ts_iters, neighbors=args.neighbors, patience=args.patience,
		seed_genome=best_genome, seed_fitness=best_ce,
		phase_name="Phase 7: TS Connections", logger=log, check_interval=ci,
	)
	best_genome = result.best_genome
	best_ce = result.final_fitness
	metrics = evaluate_phase_top_k(full_evaluator, result, "P7 TS Connections")
	phase_metrics_list.append(metrics)
	results["phase7_ts_connections"] = {
		"final_ce": result.final_fitness, "final_accuracy": result.final_accuracy,
		"iterations_run": result.iterations_run, "elapsed_s": round(elapsed, 1),
		"full_eval": metrics,
	}

	# Phase comparison for this run
	print_phase_comparison(phase_metrics_list)

	results["phase_metrics"] = phase_metrics_list

	# Final result = last phase's best CE on full validation
	final_pm = phase_metrics_list[-1]
	results["final"] = final_pm["best_ce"]

	return results, best_genome


# ---------------------------------------------------------------------------
# Final comparison table across all context sizes
# ---------------------------------------------------------------------------

def print_context_comparison(all_runs, baseline_input):
	"""Print a comparison table across all context sizes for each config."""

	# Group runs by config
	configs = {}
	for run in all_runs:
		key = (run["neurons"], run["bits"])
		if key not in configs:
			configs[key] = {}
		configs[key][run["context_size"]] = run["final"]

	all_contexts = sorted({r["context_size"] for r in all_runs})

	# Header
	W = 14 + 30 * len(all_contexts)
	print(f"\n{'='*W}")
	print(f"  Context Size Comparison (Final CE / PPL / Accuracy on Full Validation)")
	print(f"{'='*W}")

	# Column headers
	header = f"  {'Config':<14}"
	for ctx in all_contexts:
		src = " [v3]" if ctx == 4 else ""
		header += f" | {'ctx=' + str(ctx) + src:^27}"
	print(header)
	print(f"  {'-'*(W-2)}")

	for (neurons, bits), ctx_results in sorted(configs.items()):
		label = f"n={neurons},b={bits}"

		# CE row
		row_ce = f"  {label:<14}"
		for ctx in all_contexts:
			if ctx in ctx_results:
				ce = ctx_results[ctx]["ce"]
				row_ce += f" | {'CE=' + f'{ce:.4f}':^27}"
			else:
				row_ce += f" | {'—':^27}"
		print(row_ce)

		# PPL row
		row_ppl = f"  {'':<14}"
		for ctx in all_contexts:
			if ctx in ctx_results:
				ppl = ctx_results[ctx]["ppl"]
				row_ppl += f" | {'PPL=' + f'{ppl:.0f}':^27}"
			else:
				row_ppl += f" | {'':^27}"
		print(row_ppl)

		# Accuracy row
		row_acc = f"  {'':<14}"
		for ctx in all_contexts:
			if ctx in ctx_results:
				acc = ctx_results[ctx]["acc"]
				row_acc += f" | {'Acc=' + f'{acc:.2%}':^27}"
			else:
				row_acc += f" | {'':^27}"
		print(row_acc)

		print(f"  {'-'*(W-2)}")

	print(f"{'='*W}")

	# Best config per context
	print(f"\n  Best config per context size:")
	for ctx in all_contexts:
		best_key = None
		best_ce = float("inf")
		for (neurons, bits), ctx_results in configs.items():
			if ctx in ctx_results and ctx_results[ctx]["ce"] < best_ce:
				best_ce = ctx_results[ctx]["ce"]
				best_key = (neurons, bits)
		if best_key:
			n, b = best_key
			r = configs[best_key][ctx]
			src = " [v3]" if ctx == 4 else ""
			print(f"    ctx={ctx:>2}{src}: n={n}, b={b} → "
				  f"CE={r['ce']:.4f}, PPL={r['ppl']:.0f}, Acc={r['acc']:.2%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(
		description="Bitwise Context Sweep: full 7-phase optimization across context sizes"
	)
	parser.add_argument("--input", type=str, required=True,
						help="Input JSON from previous bitwise optimization (e.g., bitwise_overnight_v3.json)")
	parser.add_argument("--output", type=str, default="experiments/bitwise_context_sweep.json",
						help="Output JSON file (default: experiments/bitwise_context_sweep.json)")
	parser.add_argument("--top-k", type=int, default=3,
						help="Number of top configs to sweep (default: 3)")
	parser.add_argument("--contexts", type=str, default="2,3,5,6,7,8,16",
						help="Comma-separated context sizes to run (default: 2,3,5,6,7,8,16)")
	parser.add_argument("--baseline-context", type=int, default=4,
						help="Context size already done in baseline run (default: 4)")
	# Shared params
	parser.add_argument("--rate", type=float, default=0.25)
	parser.add_argument("--ga-gens", type=int, default=250)
	parser.add_argument("--population", type=int, default=50)
	parser.add_argument("--ts-iters", type=int, default=250)
	parser.add_argument("--neighbors", type=int, default=30)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--check-interval", type=int, default=10)
	parser.add_argument("--train-parts", type=int, default=36)
	parser.add_argument("--eval-parts", type=int, default=6)
	args = parser.parse_args()

	contexts_to_run = [int(c) for c in args.contexts.split(",")]
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# ── Load baseline results and extract top-K configs ──────────────────
	with open(args.input) as f:
		baseline = json.load(f)

	grid_results = baseline["phase1_grid"]["results"]  # already sorted by CE
	top_configs = grid_results[:args.top_k]

	print(f"Top {args.top_k} configs from {args.input}:")
	for i, cfg in enumerate(top_configs):
		print(f"  {i+1}. n={cfg['neurons']}, b={cfg['bits']}, CE={cfg['cross_entropy']:.4f}")

	print(f"\nContext sizes to run: {contexts_to_run}")
	print(f"Baseline context (from input): {args.baseline_context}")
	print(f"Total runs: {len(top_configs)} configs × {len(contexts_to_run)} contexts "
		  f"= {len(top_configs) * len(contexts_to_run)}")

	# ── Load dataset once ────────────────────────────────────────────────
	from wnn.ram.core.RAMClusterLayer import bits_needed
	train_tokens, test_tokens, validation_tokens, vocab_size = load_wikitext2_tokens()
	num_clusters = bits_needed(vocab_size)

	# ── Load or initialize sweep results ─────────────────────────────────
	if output_path.exists():
		with open(output_path) as f:
			sweep_results = json.load(f)
		print(f"Resuming from {output_path} ({len(sweep_results.get('runs', []))} runs completed)")
	else:
		sweep_results = {
			"config": {
				"top_configs": top_configs,
				"contexts_to_run": contexts_to_run,
				"baseline_context": args.baseline_context,
				"baseline_input": args.input,
				"ga_gens": args.ga_gens,
				"ts_iters": args.ts_iters,
				"population": args.population,
				"neighbors": args.neighbors,
				"patience": args.patience,
				"check_interval": args.check_interval,
			},
			"runs": [],
		}

	# Build set of completed runs for resume support
	completed = {
		(r["neurons"], r["bits"], r["context_size"])
		for r in sweep_results["runs"]
	}

	def save():
		with open(output_path, "w") as f:
			json.dump(sweep_results, f, indent=2)

	# ── Include baseline context=4 results from input ────────────────────
	# Extract the final phase metrics from the baseline run for each top config
	baseline_ctx = args.baseline_context
	if baseline.get("phase_metrics"):
		# The baseline ran with a single config (the grid winner)
		# Pull its final results for context=4
		final_pm = baseline["phase_metrics"][-1]
		best_cfg = top_configs[0]

		baseline_run = {
			"neurons": best_cfg["neurons"],
			"bits": best_cfg["bits"],
			"context_size": baseline_ctx,
			"total_input_bits": baseline_ctx * num_clusters,
			"final": final_pm["best_ce"],
			"source": "baseline",
		}

		# Add to runs if not already there
		key = (best_cfg["neurons"], best_cfg["bits"], baseline_ctx)
		if key not in completed:
			sweep_results["runs"].append(baseline_run)
			completed.add(key)
			save()
			print(f"\nIncluded baseline ctx={baseline_ctx} result: "
				  f"n={best_cfg['neurons']}, b={best_cfg['bits']}, "
				  f"CE={final_pm['best_ce']['ce']:.4f}")

	# ── Run the sweep ────────────────────────────────────────────────────
	total = len(top_configs) * len(contexts_to_run)
	run_idx = 0
	t_sweep_start = time.time()

	for cfg in top_configs:
		neurons = cfg["neurons"]
		bits = cfg["bits"]

		for context_size in contexts_to_run:
			run_idx += 1
			key = (neurons, bits, context_size)

			if key in completed:
				print(f"\n[{run_idx}/{total}] Skipping n={neurons}, b={bits}, ctx={context_size} (already done)")
				continue

			print(f"\n{'#'*80}")
			print(f"  [{run_idx}/{total}] n={neurons}, b={bits}, ctx={context_size}")
			print(f"{'#'*80}")

			t0 = time.time()
			try:
				run_result, best_genome = run_full_pipeline(
					neurons=neurons, bits=bits, context_size=context_size,
					train_tokens=train_tokens, test_tokens=test_tokens,
					validation_tokens=validation_tokens,
					vocab_size=vocab_size, num_clusters=num_clusters,
					args=args,
				)
				run_result["elapsed_total_s"] = round(time.time() - t0, 1)
				sweep_results["runs"].append(run_result)
				completed.add(key)
				save()

				print(f"\n  Run complete: CE={run_result['final']['ce']:.4f}, "
					  f"Acc={run_result['final']['acc']:.2%} "
					  f"({run_result['elapsed_total_s']:.0f}s)")

			except Exception as e:
				print(f"\n  ERROR in run n={neurons}, b={bits}, ctx={context_size}: {e}")
				import traceback
				traceback.print_exc()
				# Save partial results and continue
				save()
				continue

	# ── Final comparison table ───────────────────────────────────────────
	sweep_elapsed = time.time() - t_sweep_start
	print(f"\n{'#'*80}")
	print(f"  SWEEP COMPLETE — {len(sweep_results['runs'])} runs in {sweep_elapsed:.0f}s")
	print(f"{'#'*80}")

	print_context_comparison(sweep_results["runs"], args.input)

	save()
	print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
	main()
