"""
GridSearchConfig + GridSearchStrategy — one-shot evaluation of neuron × bit configurations

Split out of architecture_strategies.py (D3, 11/06/2026); that module
re-exports everything, so existing imports keep working.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.strategies.connectivity.framework import OptimizerResult, StopReason
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

@dataclass
class GridSearchConfig:
	"""Configuration for grid search over neuron × bit combinations."""
	num_clusters: int
	neurons_grid: list[int] = field(default_factory=lambda: [50, 100, 150, 200])
	bits_grid: list[int] = field(default_factory=lambda: [14, 16, 18, 20])
	top_k: int = 15
	population_size: int = 50  # Total genomes in output population
	total_input_bits: Optional[int] = None
	fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
	fitness_weight_ce: float = 1.0
	fitness_weight_acc: float = 1.0
	fitness_weight_f1: float = 0.0
	fitness_weight_fpr: float = 0.0
	grid_source: str = "random"  # "random" or "leaderboard"


class GridSearchStrategy:
	"""
	One-shot evaluation of neuron × bit configurations.

	Unlike GA/TS which iteratively optimize, grid search evaluates
	each (neurons, bits) combination once and ranks by fitness.
	Top-K configurations get proportionally more representation
	in the output population, which seeds the next phase.

	Each grid point is recorded as an iteration for dashboard tracking.

	Evaluation uses the BitwiseEvaluator (Rust + Metal) for fast
	train+eval. All grid configs are batched together so the evaluator
	can leverage parallel CPU training + GPU forward passes.
	"""

	def __init__(
		self,
		config: GridSearchConfig,
		batch_evaluator: Optional[Any] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		self._config = config
		self._batch_evaluator = batch_evaluator
		self._seed = seed
		self._rng = random.Random(seed)
		self._log = logger or (lambda x: None)
		self._shutdown_check = shutdown_check
		self._tracker = None
		self._tracker_experiment_id = None

	@property
	def name(self) -> str:
		return "GridSearch"

	def set_tracker(self, tracker: "ExperimentTracker", experiment_id: int, _unused: Optional[int] = None) -> None:
		"""Set the experiment tracker for iteration recording."""
		self._tracker = tracker
		self._tracker_experiment_id = experiment_id

	def _create_genome(self, neurons_per_cluster: int, bits_per_neuron: int) -> 'ClusterGenome':
		"""Create a genome with uniform neurons/bits and random connections."""
		import numpy as np
		from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

		cfg = self._config
		neurons = [neurons_per_cluster] * cfg.num_clusters
		total_neurons = sum(neurons)
		bits = [bits_per_neuron] * total_neurons

		connections = None
		if cfg.total_input_bits is not None:
			from wnn.ram.strategies.connectivity.adaptive_cluster import generate_connections
			connections = generate_connections(bits, cfg.total_input_bits, self._rng.randint(0, 2**63))

		return ClusterGenome(bits_per_neuron=bits, neurons_per_cluster=neurons, connections=connections)

	def optimize(
		self,
		evaluate_fn: Optional[Callable] = None,
		initial_genome: Optional['ClusterGenome'] = None,
		initial_population: Optional[list['ClusterGenome']] = None,
		**kwargs,
	) -> OptimizerResult['ClusterGenome']:
		"""
		Run grid search: evaluate each (neurons, bits) config and rank by fitness.

		All grid genomes are created first, then evaluated in a single batch
		through the Rust+Metal evaluator for maximum throughput.
		"""
		import time
		from wnn.ram.fitness import FitnessCalculatorFactory

		cfg = self._config
		from wnn.ram.metrics import FitnessWeights
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weights=FitnessWeights(ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
								   f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr),
		)

		# Phase 1: Build config list from grid or leaderboard
		is_leaderboard = cfg.grid_source == "leaderboard"
		if is_leaderboard and initial_population:
			config_list = []  # [(neurons, bits, genome)]
			for genome in initial_population:
				n = genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0
				b = genome.bits_per_neuron[0] if genome.bits_per_neuron else 0
				config_list.append((n, b, genome))
			total_configs = len(config_list)
			self._log(f"\n{'='*70}")
			self._log(f"Grid Search — Leaderboard ({total_configs} genomes)")
			self._log(f"  source: leaderboard top-{total_configs}")
			self._log(f"  clusters: {cfg.num_clusters}")
			self._log(f"{'='*70}")
		else:
			config_list = []
			for neurons in cfg.neurons_grid:
				for bits in cfg.bits_grid:
					genome = self._create_genome(neurons, bits)
					config_list.append((neurons, bits, genome))
			total_configs = len(config_list)
			self._log(f"\n{'='*70}")
			self._log(f"Grid Search ({total_configs} configs)")
			self._log(f"  neurons: {cfg.neurons_grid}")
			self._log(f"  bits:    {cfg.bits_grid}")
			self._log(f"  clusters: {cfg.num_clusters}")
			self._log(f"{'='*70}")

		# Phase 2: Evaluate one config at a time for real-time dashboard progress
		# Fix train subset to a single index so all configs are compared fairly
		# (auto-advancing would evaluate each config on a different subset)
		grid_train_idx = kwargs.get('train_subset_idx', None)
		if grid_train_idx is None and self._batch_evaluator is not None and hasattr(self._batch_evaluator, 'random_train_idx'):
			grid_train_idx = self._batch_evaluator.random_train_idx(self._rng)
		if grid_train_idx is not None:
			self._log(f"  Using fixed train subset {grid_train_idx} for all {total_configs} configs")

		t0 = time.time()
		results = []
		best_ce_so_far = float('inf')
		best_acc_so_far = 0.0
		best_f1_so_far: Optional[float] = None
		best_fpr_so_far: Optional[float] = None

		# Concurrent grid_search: configs are independent — each is a single-
		# genome `evaluate_batch` call. The Rust accelerator releases the GIL
		# inside `evaluate_genomes_parallel_hybrid` via `py.allow_threads`,
		# so Python threads run truly in parallel. GPU dispatches serialize
		# at the Metal queue, but CPU train/eval + buffer alloc + export
		# overlap across threads.
		#
		# WNN_GRID_SEARCH_PARALLEL=N caps the concurrent worker count.
		# Default 4: empirical sweet spot on M4 Max (16 cores + 40-core GPU).
		# Set to 1 to disable concurrency.
		import os as _os
		from concurrent.futures import ThreadPoolExecutor as _TPE
		grid_max_workers = max(1, int(_os.environ.get('WNN_GRID_SEARCH_PARALLEL', '4')))

		def _eval_one_config(idx_neurons_bits_genome):
			idx, neurons, bits, genome = idx_neurons_bits_genome
			if self._shutdown_check and self._shutdown_check():
				return idx, neurons, bits, genome, None
			t_config = time.time()
			if self._batch_evaluator is not None:
				grid_evals = self._batch_evaluator.evaluate_batch(
					[genome], train_subset_idx=grid_train_idx,
				)
				grid_ev = grid_evals[0]
				ce, acc, bit_acc = grid_ev.ce, grid_ev.acc, grid_ev.bit_accuracy or 0.0
				f1_macro = grid_ev.f1
				fpr = grid_ev.fpr
			elif evaluate_fn is not None:
				ce = evaluate_fn(genome)
				acc, bit_acc = 0.0, 0.0
				f1_macro, fpr = None, None
			else:
				raise ValueError("GridSearchStrategy requires a batch_evaluator or evaluate_fn")
			config_elapsed = time.time() - t_config
			return idx, neurons, bits, genome, {
				"neurons": neurons, "bits": bits,
				"ce": ce, "accuracy": acc,
				"bit_accuracy": bit_acc, "f1_macro": f1_macro, "fpr": fpr,
				"elapsed_s": config_elapsed, "genome": genome,
			}

		indexed_configs = [(idx, n, b, g) for idx, (n, b, g) in enumerate(config_list)]

		# Stream results as they complete so the dashboard sees per-config
		# updates in real time. Use as_completed instead of pool.map to
		# avoid buffering all 48 results until the batch finishes.
		def _iter_results(configs_iter):
			"""Yield (idx, neurons, bits, genome, out) tuples in completion order."""
			if grid_max_workers > 1 and len(configs_iter) > 1:
				self._log(f"  Concurrent eval: {len(configs_iter)} configs × parallelism={grid_max_workers} (streaming)")
				from concurrent.futures import as_completed as _as_completed
				with _TPE(max_workers=grid_max_workers) as _pool:
					_futures = [_pool.submit(_eval_one_config, c) for c in configs_iter]
					for _fut in _as_completed(_futures):
						yield _fut.result()
			else:
				for c in configs_iter:
					yield _eval_one_config(c)

		# Process each result as it completes — log + update best_*_so_far +
		# write DB row immediately so the dashboard ticks in real time.
		# iteration_num in the DB tracks COMPLETION order; the per-config grid
		# position is still in the row data (neurons, bits).
		_completion_num = 0
		for _idx, neurons, bits, genome, out in _iter_results(indexed_configs):
			if out is None:
				self._log(f"  Shutdown requested at config {_idx}")
				break
			_completion_num += 1
			ce = out["ce"]; acc = out["accuracy"]
			f1_macro = out["f1_macro"]; fpr = out["fpr"]
			config_elapsed = out["elapsed_s"]
			self._log(f"  [{_completion_num}/{total_configs}] n={neurons:3d}, b={bits:2d}: "
					  f"CE={ce:.4f}  Acc={acc:.2%}  ({config_elapsed:.1f}s)")
			best_ce_so_far = min(best_ce_so_far, ce)
			best_acc_so_far = max(best_acc_so_far, acc)
			results.append(out)

			# Record each config as a separate iteration for real-time dashboard tracking.
			# iteration_num tracks COMPLETION order so the dashboard sees a
			# monotonically growing list. Grid position (neurons, bits) is in
			# the row body if reconstruction is needed.
			if self._tracker and self._tracker_experiment_id:
				try:
					avg_ce = sum(r["ce"] for r in results) / len(results)
					avg_acc = sum(r["accuracy"] for r in results) / len(results)
					cur_f1 = results[-1].get("f1_macro")
					cur_fpr = results[-1].get("fpr")
					if cur_f1 is not None and (best_f1_so_far is None or cur_f1 > best_f1_so_far):
						best_f1_so_far = cur_f1
					if cur_fpr is not None and (best_fpr_so_far is None or cur_fpr < best_fpr_so_far):
						best_fpr_so_far = cur_fpr
					iter_id = self._tracker.record_iteration(
						experiment_id=self._tracker_experiment_id,
						iteration_num=_completion_num,
						best_ce=best_ce_so_far,
						best_accuracy=best_acc_so_far,
						avg_ce=avg_ce,
						avg_accuracy=avg_acc,
						elapsed_secs=config_elapsed,
						candidates_total=total_configs,
						best_f1=best_f1_so_far,
						best_fpr=best_fpr_so_far,
					)
					if HAS_GENOME_TRACKING and iter_id:
						genome_config = self._genome_to_config(genome)
						if genome_config:
							genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, genome_config
							)
							eval_id = self._tracker.record_genome_evaluation(
								iteration_id=iter_id,
								genome_id=genome_id,
								position=0,
								role=GenomeRole.INIT,
								ce=ce,
								accuracy=acc,
								fitness_score=None,
								f1_macro=f1_macro,
								fpr=fpr,
							)
							results[-1]["eval_id"] = eval_id
					self._tracker.update_experiment_progress(
						self._tracker_experiment_id,
						current_iteration=_completion_num,
						best_ce=best_ce_so_far,
						best_accuracy=best_acc_so_far,
					)
				except Exception as e:
					self._log(f"  Warning: tracker error: {e}")

		batch_elapsed = time.time() - t0
		self._log(f"  Total evaluation: {batch_elapsed:.1f}s, "
				  f"{batch_elapsed/len(results):.1f}s/config avg")

		if not results:
			raise ValueError("Grid search produced no results")

		# Refuse to mark the experiment "completed" with a partial grid. If
		# shutdown_check fired mid-grid (worker SIGINT/SIGTERM, flow cancelled
		# from dashboard, etc.), only some of the planned configs evaluated.
		# Continuing to ranking + offspring with that partial result would
		# (a) seed GA Neurons with only the smallest-architecture elite and
		# (b) write a bogus best_ce checkpoint that auto-resume then skips
		# past. Re-raise here so the experiment is marked failed/cancelled and
		# re-runs cleanly on next pickup. Observed on flow 2721 on 18/05/2026:
		# watcher SIGINT during grid_search left a 5n × 16b checkpoint.
		if len(results) < total_configs:
			if self._shutdown_check and self._shutdown_check():
				raise RuntimeError(
					f"Grid search aborted by shutdown after {len(results)}/{total_configs} "
					"configs evaluated — refusing to mark experiment completed with partial data"
				)

		# Phase 4: Rank by fitness
		from wnn.ram.metrics import Metrics as _M
		rank_metrics = [_M(ce=r["ce"], acc=r["accuracy"], f1=r.get("f1_macro"), fpr=r.get("fpr")) for r in results]
		fitness_scores = calculator.fitness(rank_metrics)
		for r, score in zip(results, fitness_scores):
			r["fitness"] = score
		results.sort(key=lambda r: r["fitness"])

		# Update fitness scores in DB for per-config genome evaluations
		if self._tracker and self._tracker_experiment_id:
			fitness_updates = [
				(r["eval_id"], r["fitness"])
				for r in results if "eval_id" in r
			]
			if fitness_updates:
				try:
					self._tracker.update_genome_evaluation_fitness_batch(fitness_updates)
				except Exception as e:
					self._log(f"  Warning: failed to update fitness scores: {e}")

		self._log(f"\n{'─'*70}")
		self._log(f"Grid Search Rankings (by {calculator.name}):")
		for i, r in enumerate(results):
			marker = " ★" if i < cfg.top_k else ""
			self._log(f"  {i+1:2d}. n={r['neurons']:3d}, b={r['bits']:2d}: "
					  f"CE={r['ce']:.4f}  Acc={r['accuracy']:.2%}  "
					  f"Fit={r['fitness']:.4f}{marker}")

		# Phase 5: Build output population with balanced representation.
		# Generate population_size * 1.1 genomes (10% extra for fitness trimming),
		# distributed evenly across top_k configs with extra going to top-ranked.
		best_result = results[0]
		best_genome = best_result["genome"]

		top_k = min(cfg.top_k, len(results))
		target_total = max(top_k, int(cfg.population_size * 1.1))
		base_per_config = target_total // top_k
		remainder = target_total - base_per_config * top_k
		genomes_per_config = []
		for i in range(top_k):
			extra = 1 if i < remainder else 0
			genomes_per_config.append(max(1, base_per_config + extra))

		# Reuse original evaluated genomes + create new variations for the rest.
		# Each config's first genome is the already-evaluated original;
		# additional genomes get fresh random connections and need evaluation.
		output_population = []
		population_metrics = []
		new_genomes = []       # genomes needing evaluation
		new_genome_indices = []  # their positions in output_population

		for i in range(top_k):
			r = results[i]
			n_genomes = genomes_per_config[i]

			# First genome: reuse the original (already evaluated)
			from wnn.ram.metrics import Metrics as _Metrics
			output_population.append(r["genome"])
			population_metrics.append(_Metrics(ce=r["ce"], acc=r["accuracy"], f1=r.get("f1_macro"), fpr=r.get("fpr")))

			# Remaining: fresh random connections (need evaluation)
			for _ in range(n_genomes - 1):
				genome = self._create_genome(r["neurons"], r["bits"])
				new_genome_indices.append(len(output_population))
				output_population.append(genome)
				population_metrics.append(_Metrics(ce=0.0, acc=0.0))  # placeholder
				new_genomes.append(genome)

			self._log(f"  #{i+1:2d} n={r['neurons']:3d}, b={r['bits']:2d} (CE={r['ce']:.4f}) → {n_genomes} genomes (1 original + {n_genomes - 1} new)")

		self._log(f"\nPopulation: {len(output_population)} genomes ({top_k} original + {len(new_genomes)} new)")

		# Evaluate the NEW genomes — per-genome dispatch, concurrent + streaming.
		# Earlier attempt (15/05/2026) grouped genomes by shape and called
		# batched_train_offspring per group. For Phase 5's small group sizes
		# (~2-3 genomes/group for 40 new), GPU kernel-launch overhead dominated
		# AND 4 concurrent threads serialized on the Metal queue → ~8× slower
		# per genome (4.4s → 32-41s). Per-genome dispatch lets B11 affinity
		# route these small workloads to CPU and parallelize cleanly across
		# the thread pool. Streaming via as_completed preserves dashboard ticks.
		num_seed_recorded = 0
		if new_genomes and self._batch_evaluator is not None:
			self._log(f"  Evaluating {len(new_genomes)} new genomes (per-genome, concurrent)...")
			t1 = time.time()
			expand_elapsed = 0.0

			def _eval_one_new_genome(idx_genome):
				idx_in_new, genome = idx_genome
				t_g = time.time()
				_evals = self._batch_evaluator.evaluate_batch(
					[genome], train_subset_idx=grid_train_idx,
				)
				return idx_in_new, genome, _evals[0], time.time() - t_g

			indexed_new = list(enumerate(new_genomes))

			def _iter_new_genomes(items):
				if grid_max_workers > 1 and len(items) > 1:
					from concurrent.futures import as_completed as _as_completed
					with _TPE(max_workers=grid_max_workers) as _pool:
						_futures = [_pool.submit(_eval_one_new_genome, ig) for ig in items]
						for _fut in _as_completed(_futures):
							yield _fut.result()
				else:
					for ig in items:
						yield _eval_one_new_genome(ig)

			# Process each genome's result as it completes — log + tracker writes
			# immediately so the dashboard ticks in real time.
			for idx_in_new, genome, ev, per_genome_elapsed in _iter_new_genomes(indexed_new):
				expand_elapsed += per_genome_elapsed
				ce, acc = ev.ce, ev.acc
				pop_idx = new_genome_indices[idx_in_new]
				population_metrics[pop_idx] = _Metrics(ce=ce, acc=acc, f1=ev.f1, fpr=ev.fpr)
				g = output_population[pop_idx]
				neurons = g.neurons_per_cluster[0] if g.neurons_per_cluster else 0
				bits = g.bits_per_neuron[0] if g.bits_per_neuron else 0
				self._log(f"  [{idx_in_new+1}/{len(new_genomes)}] n={neurons:3d}, b={bits:2d}: "
				          f"CE={ce:.4f}  Acc={acc:.2%}  ({per_genome_elapsed:.1f}s)")
				if ce < best_result["ce"]:
					best_result = {"ce": ce, "accuracy": acc, "genome": output_population[pop_idx]}
					best_genome = output_population[pop_idx]
				if self._tracker and self._tracker_experiment_id:
					try:
						best_ce_so_far = min(best_ce_so_far, ce)
						best_acc_so_far = max(best_acc_so_far, acc)
						seed_iter_num = len(results) + idx_in_new + 1
						ev_f1 = ev.f1
						ev_fpr = ev.fpr
						if ev_f1 is not None and (best_f1_so_far is None or ev_f1 > best_f1_so_far):
							best_f1_so_far = ev_f1
						if ev_fpr is not None and (best_fpr_so_far is None or ev_fpr < best_fpr_so_far):
							best_fpr_so_far = ev_fpr
						iter_id = self._tracker.record_iteration(
							experiment_id=self._tracker_experiment_id,
							iteration_num=seed_iter_num,
							best_ce=best_ce_so_far,
							best_accuracy=best_acc_so_far,
							avg_ce=ce,
							avg_accuracy=acc,
							elapsed_secs=per_genome_elapsed,
							candidates_total=len(new_genomes),
							best_f1=best_f1_so_far,
							best_fpr=best_fpr_so_far,
						)
						if HAS_GENOME_TRACKING and iter_id:
							genome_config = self._genome_to_config(genome)
							if genome_config:
								genome_id = self._tracker.get_or_create_genome(
									self._tracker_experiment_id, genome_config
								)
								self._tracker.record_genome_evaluation(
									iteration_id=iter_id,
									genome_id=genome_id,
									position=0,
									role=GenomeRole.INIT,
									ce=ce,
									accuracy=acc,
									fitness_score=None,
									f1_macro=ev_f1,
									fpr=ev_fpr,
								)
						self._tracker.update_experiment_progress(
							self._tracker_experiment_id,
							current_iteration=seed_iter_num,
							best_ce=best_ce_so_far,
							best_accuracy=best_acc_so_far,
						)
						num_seed_recorded += 1
					except Exception as e:
						self._log(f"  Warning: seed tracker error: {e}")
			batch_elapsed += expand_elapsed
			self._log(f"  New genome eval total: {expand_elapsed:.1f}s "
				  f"({expand_elapsed/len(new_genomes):.1f}s/genome avg, "
				  f"per-genome × parallelism={grid_max_workers})")

		# Phase 6: Rank full population by fitness and sort
		pop_fitness_scores = calculator.fitness(population_metrics)

		# Sort population by fitness score (lower = better)
		sorted_indices = sorted(range(len(output_population)), key=lambda i: pop_fitness_scores[i])
		output_population = [output_population[i] for i in sorted_indices]
		population_metrics = [population_metrics[i] for i in sorted_indices]
		pop_fitness_scores = [pop_fitness_scores[i] for i in sorted_indices]

		# Trim to population_size (balanced seeding may overshoot)
		if len(output_population) > cfg.population_size:
			dropped = len(output_population) - cfg.population_size
			self._log(f"  Trimming population: {len(output_population)} → {cfg.population_size} (dropped {dropped} weakest)")
			output_population = output_population[:cfg.population_size]
			population_metrics = population_metrics[:cfg.population_size]
			pop_fitness_scores = pop_fitness_scores[:cfg.population_size]

		# Three independent bests from potentially different genomes (recompute after trim)
		pop_bests = calculator.bests(output_population, population_metrics)
		best_genome = pop_bests.best_fitness.genome

		# Record final iteration with ALL population genomes sorted by fitness
		final_iter_num = len(results) + num_seed_recorded + 1  # After per-config + seed iterations
		if self._tracker and self._tracker_experiment_id:
			try:
				avg_ce = sum(m.ce for m in population_metrics) / len(population_metrics)
				avg_acc = sum(m.acc for m in population_metrics) / len(population_metrics)
				pm_f1s = [m.f1 for m in population_metrics if m.f1 is not None]
				pm_fprs = [m.fpr for m in population_metrics if m.fpr is not None]
				if pm_f1s:
					final_f1 = max(pm_f1s)
					if best_f1_so_far is None or final_f1 > best_f1_so_far:
						best_f1_so_far = final_f1
				if pm_fprs:
					final_fpr = min(pm_fprs)
					if best_fpr_so_far is None or final_fpr < best_fpr_so_far:
						best_fpr_so_far = final_fpr
				iter_id = self._tracker.record_iteration(
					experiment_id=self._tracker_experiment_id,
					iteration_num=final_iter_num,
					best_ce=pop_bests.best_ce.metrics.ce,
					best_accuracy=pop_bests.best_acc.metrics.acc,
					avg_ce=avg_ce,
					avg_accuracy=avg_acc,
					elapsed_secs=batch_elapsed,
					candidates_total=len(output_population),
					best_f1=best_f1_so_far,
					best_fpr=best_fpr_so_far,
				)
				if HAS_GENOME_TRACKING:
					for pos, (genome, m, fit) in enumerate(zip(output_population, population_metrics, pop_fitness_scores)):
						ce, acc = m.ce, m.acc
						genome_config = self._genome_to_config(genome)
						if genome_config:
							genome_id = self._tracker.get_or_create_genome(
								self._tracker_experiment_id, genome_config
							)
							self._tracker.record_genome_evaluation(
								iteration_id=iter_id,
								genome_id=genome_id,
								position=pos,
								role=GenomeRole.INIT,
								ce=ce,
								accuracy=acc,
								fitness_score=fit,
								f1_macro=m.f1,
								fpr=m.fpr,
							)
				self._tracker.update_experiment_progress(
					self._tracker_experiment_id,
					current_iteration=1,
					best_ce=pop_bests.best_ce.metrics.ce,
					best_accuracy=pop_bests.best_acc.metrics.acc,
				)
			except Exception as e:
				self._log(f"  Warning: tracker error: {e}")

		# Build OptimizerResult
		worst_ce = results[-1]["ce"]
		improvement = ((worst_ce - pop_bests.best_ce.metrics.ce) / worst_ce * 100) if worst_ce > 0 else 0.0

		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=worst_ce,
			final_fitness=pop_bests.best_ce.metrics.ce,
			improvement_percent=improvement,
			iterations_run=1,
			method_name="GridSearch",
			history=[(1, pop_bests.best_ce.metrics.ce)],
			early_stopped=False,
			stop_reason=StopReason.MAX_ITERATIONS,
			final_population=output_population,
			population_metrics=population_metrics,
			initial_accuracy=results[-1]["accuracy"],
			final_accuracy=pop_bests.best_acc.metrics.acc,
			final_threshold=None,
		)

	def _genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert genome to GenomeConfig for tracker."""
		if not HAS_GENOME_TRACKING:
			return None
		cfg = self._config
		tiers = [TierConfig(
			tier=0,
			clusters=cfg.num_clusters,
			neurons=genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0,
			bits=genome.bits_per_neuron[0] if genome.bits_per_neuron else 0,
		)]
		return GenomeConfig(tiers=tiers)


# =============================================================================
# Stats-Guided Adaptation Strategies (Neurogenesis, Synaptogenesis, Axonogenesis)
# =============================================================================
