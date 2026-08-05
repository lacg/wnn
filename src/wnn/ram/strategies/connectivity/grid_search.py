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
from wnn.ram.metrics import IDSMetrics, Metrics
from wnn.ram.strategies.connectivity.framework import OptimizerResult, StopReason
from wnn.ram.strategies.connectivity.generic_grid_search import GenericGridSearch
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
		Run grid search on the shared `GenericGridSearch` core: evaluate each
		(neurons, bits) config once, rank by fitness, expand the top-K into a seed
		population, and trim. The invariant top-K / expand / trim algebra lives in
		the core (shared with the controller); this method builds the config list,
		wires an `_IDSGridSearchCore` adapter (eval concurrency + dashboard tracker
		writes + shutdown handling live in its `_evaluate` hook), and converts the
		neutral `GridSearchOutcome` back into the IDS `OptimizerResult`.
		"""
		import time
		from wnn.ram.fitness import FitnessCalculatorFactory
		from wnn.ram.metrics import FitnessWeights

		cfg = self._config
		calculator = FitnessCalculatorFactory.create(
			cfg.fitness_calculator_type,
			weights=FitnessWeights(ce=cfg.fitness_weight_ce, acc=cfg.fitness_weight_acc,
								   f1=cfg.fitness_weight_f1, fpr=cfg.fitness_weight_fpr),
		)

		# Phase 1: Build the config list (grid or leaderboard). For grid configs the
		# genome is created lazily by the core's _make_genome (same RNG order: the
		# core builds ALL config genomes before drawing the train subset, exactly as
		# the old inline build did). Leaderboard configs carry a pre-built genome.
		config_list = self._build_config_list(cfg, initial_population)
		total_configs = len(config_list)

		t0 = time.time()
		core = _IDSGridSearchCore(self, calculator, config_list, evaluate_fn,
		                          kwargs.get('train_subset_idx', None))
		outcome = core.run()
		batch_elapsed = time.time() - t0

		return self._build_result(core, outcome, calculator, batch_elapsed)

	def _build_config_list(self, cfg: GridSearchConfig,
	                       initial_population: Optional[list]) -> list:
		"""[(neurons, bits, genome-or-None)] from the grid or a leaderboard pop."""
		if cfg.grid_source == "leaderboard" and initial_population:
			config_list = [
				(g.neurons_per_cluster[0] if g.neurons_per_cluster else 0,
				 g.bits_per_neuron[0] if g.bits_per_neuron else 0, g)
				for g in initial_population
			]
			self._log(f"\n{'='*70}")
			self._log(f"Grid Search — Leaderboard ({len(config_list)} genomes)")
			self._log(f"  source: leaderboard top-{len(config_list)}")
			self._log(f"  clusters: {cfg.num_clusters}")
			self._log(f"{'='*70}")
			return config_list
		config_list = [(neurons, bits, None)
		               for neurons in cfg.neurons_grid for bits in cfg.bits_grid]
		self._log(f"\n{'='*70}")
		self._log(f"Grid Search ({len(config_list)} configs)")
		self._log(f"  neurons: {cfg.neurons_grid}")
		self._log(f"  bits:    {cfg.bits_grid}")
		self._log(f"  clusters: {cfg.num_clusters}")
		self._log(f"{'='*70}")
		return config_list

	def _build_result(self, core: '_IDSGridSearchCore', outcome, calculator,
	                  batch_elapsed: float) -> OptimizerResult['ClusterGenome']:
		"""Convert the neutral GridSearchOutcome + the adapter's stashed state into
		the IDS OptimizerResult, and record the final full-population iteration.
		Mirrors the old Phase-6 tail exactly (pop_bests on the trimmed population,
		worst-config CE as the initial fitness)."""
		output_population = outcome.seed_population
		population_metrics = outcome.seed_metrics
		pop_fitness_scores = core.final_scores          # pre-trim scores, sorted+trimmed

		pop_bests = calculator.bests(output_population, population_metrics)
		best_genome = pop_bests.best_fitness.genome

		# Final iteration: ALL surviving genomes, sorted by fitness (dashboard).
		final_iter_num = len(core.results) + core.num_seed_recorded + 1
		core.record_final_iteration(outcome, pop_bests, pop_fitness_scores,
		                            final_iter_num, batch_elapsed)

		worst = core.ranked[-1][2]  # worst-fitness config's metric
		worst_ce = worst.ce
		best_ce = pop_bests.best_ce.metrics.ce
		improvement = ((worst_ce - best_ce) / worst_ce * 100) if worst_ce > 0 else 0.0
		return OptimizerResult(
			initial_genome=best_genome,
			best_genome=best_genome,
			initial_fitness=worst_ce,
			final_fitness=best_ce,
			improvement_percent=improvement,
			iterations_run=1,
			method_name="GridSearch",
			history=[(1, best_ce)],
			early_stopped=False,
			stop_reason=StopReason.MAX_ITERATIONS,
			final_population=output_population,
			population_metrics=population_metrics,
			initial_accuracy=worst.acc,
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


@dataclass
class _IDSPoint:
	"""One grid cell for the IDS adapter. `genome` is pre-built only for the
	leaderboard source; for the grid source it stays None and the core builds it
	via `_make_genome` (preserving the original's RNG draw order)."""
	neurons: int
	bits: int
	genome: Any = None


class _IDSGridSearchCore(GenericGridSearch):
	"""IDS adapter onto the shared GenericGridSearch core (tightly coupled to
	GridSearchStrategy — kept in this file). The invariant top-K / expand / trim
	algebra lives in the base; this adapter owns the IDS-specific pieces: genome
	construction, the concurrent+streaming batch evaluation, the dashboard tracker
	writes, the shutdown/partial-abort guard, and the fixed train-subset draw.

	Parity note: `_evaluate` returns metrics ALIGNED to the input genome order (the
	base zips genome↔metric then stably sorts), while the tracker records still fire
	in completion order. Under WNN_GRID_SEARCH_PARALLEL=1 (sequential) completion
	order == enumeration order, so the ranking is byte-identical to the old inline
	pipeline; the concurrency only reorders the (order-independent) tracker writes.
	"""

	def __init__(self, strategy: 'GridSearchStrategy', calculator, config_list: list,
	             evaluate_fn, explicit_train_idx):
		import time as _time
		cfg = strategy._config
		super().__init__(top_k=cfg.top_k, population_size=cfg.population_size,
		                 log=strategy._log)
		self._s = strategy
		self._calc = calculator
		self._config_list = config_list
		self._evaluate_fn = evaluate_fn
		self._explicit_train_idx = explicit_train_idx
		self._grid_train_idx = None
		self._train_idx_drawn = False
		self._t0 = _time.time()
		# Running dashboard/report state (spans the config + expand phases).
		self.results: list = []            # per-config records (completion order)
		self.num_seed_recorded = 0
		self.ranked: list = []
		self.final_scores: list = []       # pre-trim fitness scores, sorted+trimmed
		self._completion_num = 0
		self._best_ce = float('inf')
		self._best_acc = 0.0
		self._best_f1: Optional[float] = None
		self._best_fpr: Optional[float] = None
		self._grid_max_workers = self._resolve_parallelism()

	def _resolve_parallelism(self) -> int:
		"""WNN_GRID_SEARCH_PARALLEL override, else 4 — dropped to 1 on 10M+-row
		datasets (each concurrent eval carries a multi-GB working set; 4× on 46M
		rows SIGKILLed the runner — flows 4299/4300). Same policy as the old inline
		grid; sequential when the env is 1 (the parity-deterministic path)."""
		import os
		env = os.environ.get('WNN_GRID_SEARCH_PARALLEL')
		if env is not None:
			return max(1, int(env))
		rows = getattr(self._s._batch_evaluator, 'train_rows_hint', None)
		if rows is not None and rows > 10_000_000:
			self._log(f"  [grid-parallel] train rows {rows:,} > 10,000,000 — sequential "
			          f"grid eval (memory guard; override via WNN_GRID_SEARCH_PARALLEL)")
			return 1
		return 4

	# ---- GenericGridSearch hooks -----------------------------------------
	def _enumerate_points(self) -> list:
		return [_IDSPoint(n, b, g) for (n, b, g) in self._config_list]

	def _make_genome(self, point: _IDSPoint):
		# Leaderboard config → the pre-built genome (no RNG draw); grid config →
		# a fresh genome (draws self._s._rng, in enumeration order — parity).
		if point.genome is not None:
			return point.genome
		return self._s._create_genome(point.neurons, point.bits)

	def _make_variant(self, point: _IDSPoint):
		return self._s._create_genome(point.neurons, point.bits)

	def _fitness(self, metrics: list) -> list:
		return self._calc.fitness(metrics)

	def _sort_and_trim(self, pts, gens, mets):
		# Capture the pre-trim fitness scores (sorted + trimmed) the final tracker
		# iteration records per surviving genome — the base computes but drops them.
		out = super()._sort_and_trim(pts, gens, mets)
		self.final_scores = out[3]
		return out

	def _evaluate(self, genomes: list, is_expansion: bool) -> list:
		"""Concurrent + streaming batch eval. Returns metrics aligned to `genomes`;
		fires dashboard tracker writes in completion order as each finishes."""
		# Expansion variants are only evaluated when a batch_evaluator is present —
		# the evaluate_fn-only fallback leaves them at the neutral placeholder (the
		# old inline pipeline gated the whole expand-eval block on batch_evaluator).
		if is_expansion and self._s._batch_evaluator is None:
			return [IDSMetrics(ce=0.0, acc=0.0) for _ in genomes]
		if not is_expansion:
			self._ensure_train_idx()
		metrics: list = [None] * len(genomes)
		for i, genome, metric, elapsed in self._iter(list(enumerate(genomes)), is_expansion):
			if metric is None:  # shutdown sentinel (config phase only)
				done = sum(1 for m in metrics if m is not None)
				raise RuntimeError(
					f"Grid search aborted by shutdown after {done}/{len(genomes)} "
					"configs evaluated — refusing to mark experiment completed with partial data")
			metrics[i] = metric
			if is_expansion:
				self._record_seed(i, genome, metric, len(genomes))
			else:
				self._record_config(genome, metric, elapsed)
		if not is_expansion:
			import time as _time
			self._log(f"  Total evaluation: {_time.time() - self._t0:.1f}s over {len(genomes)} configs")
		return metrics

	def _on_ranked(self, ranked: list) -> None:
		self.ranked = ranked
		# Push fitness scores back onto the per-config records (by genome identity)
		# for the dashboard fitness-DB update.
		by_gid = {id(r["genome"]): r for r in self.results}
		for _point, genome, _metric, score in ranked:
			rec = by_gid.get(id(genome))
			if rec is not None:
				rec["fitness"] = score
		if self._s._tracker and self._s._tracker_experiment_id:
			updates = [(r["eval_id"], r["fitness"]) for r in self.results
			           if "eval_id" in r and "fitness" in r]
			if updates:
				try:
					self._s._tracker.update_genome_evaluation_fitness_batch(updates)
				except Exception as e:
					self._log(f"  Warning: failed to update fitness scores: {e}")
		self._log(f"\n{'─'*70}")
		self._log(f"Grid Search Rankings (by {self._calc.name}):")
		top_k = self._s._config.top_k
		for i, (point, _genome, metric, score) in enumerate(ranked):
			marker = " ★" if i < top_k else ""
			self._log(f"  {i+1:2d}. n={point.neurons:3d}, b={point.bits:2d}: "
			          f"CE={metric.ce:.4f}  Acc={metric.acc:.2%}  Fit={score:.4f}{marker}")

	# ---- eval + tracker plumbing -----------------------------------------
	def _ensure_train_idx(self) -> None:
		if self._train_idx_drawn:
			return
		self._train_idx_drawn = True
		idx = self._explicit_train_idx
		be = self._s._batch_evaluator
		if idx is None and be is not None and hasattr(be, 'random_train_idx'):
			idx = be.random_train_idx(self._s._rng)
		self._grid_train_idx = idx
		if idx is not None:
			self._log(f"  Using fixed train subset {idx} for all {len(self._config_list)} configs")

	def _eval_one(self, i: int, genome, is_expansion: bool):
		"""(idx, genome, Metrics|None, elapsed). None metric = shutdown sentinel."""
		import time as _time
		if not is_expansion and self._s._shutdown_check and self._s._shutdown_check():
			return i, genome, None, 0.0
		t = _time.time()
		be = self._s._batch_evaluator
		if be is not None:
			ev = be.evaluate_batch([genome], train_subset_idx=self._grid_train_idx)[0]
			metric = IDSMetrics(ce=ev.ce, acc=ev.acc, f1=ev.f1, fpr=ev.fpr)
		elif self._evaluate_fn is not None:
			metric = IDSMetrics(ce=self._evaluate_fn(genome), acc=0.0, f1=None, fpr=None)
		else:
			raise ValueError("GridSearchStrategy requires a batch_evaluator or evaluate_fn")
		return i, genome, metric, _time.time() - t

	def _iter(self, indexed_genomes: list, is_expansion: bool):
		"""Yield eval results, concurrently (streaming) or sequentially."""
		if self._grid_max_workers > 1 and len(indexed_genomes) > 1:
			from concurrent.futures import ThreadPoolExecutor, as_completed
			with ThreadPoolExecutor(max_workers=self._grid_max_workers) as pool:
				futs = [pool.submit(self._eval_one, i, g, is_expansion)
				        for i, g in indexed_genomes]
				for fut in as_completed(futs):
					yield fut.result()
		else:
			for i, g in indexed_genomes:
				yield self._eval_one(i, g, is_expansion)

	def _record_config(self, genome, metric: Metrics, elapsed: float) -> None:
		"""Per-config: log, update running bests, append to results, tracker row."""
		self._completion_num += 1
		n = genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0
		b = genome.bits_per_neuron[0] if genome.bits_per_neuron else 0
		ce, acc = metric.ce, metric.acc
		self._log(f"  [{self._completion_num}/{len(self._config_list)}] n={n:3d}, b={b:2d}: "
		          f"CE={ce:.4f}  Acc={acc:.2%}  ({elapsed:.1f}s)")
		self._best_ce = min(self._best_ce, ce)
		self._best_acc = max(self._best_acc, acc)
		rec = {"neurons": n, "bits": b, "ce": ce, "accuracy": acc,
		       "f1_macro": metric.f1, "fpr": metric.fpr, "genome": genome}
		self.results.append(rec)
		if not (self._s._tracker and self._s._tracker_experiment_id):
			return
		try:
			avg_ce = sum(r["ce"] for r in self.results) / len(self.results)
			avg_acc = sum(r["accuracy"] for r in self.results) / len(self.results)
			self._bump_f1_fpr(metric.f1, metric.fpr)
			iter_id = self._s._tracker.record_iteration(
				experiment_id=self._s._tracker_experiment_id,
				iteration_num=self._completion_num,
				best_ce=self._best_ce, best_accuracy=self._best_acc,
				avg_ce=avg_ce, avg_accuracy=avg_acc, elapsed_secs=elapsed,
				candidates_total=len(self._config_list),
				best_f1=self._best_f1, best_fpr=self._best_fpr)
			if HAS_GENOME_TRACKING and iter_id:
				gc = self._s._genome_to_config(genome)
				if gc:
					gid = self._s._tracker.get_or_create_genome(self._s._tracker_experiment_id, gc)
					rec["eval_id"] = self._s._tracker.record_genome_evaluation(
						iteration_id=iter_id, genome_id=gid, position=0, role=GenomeRole.INIT,
						ce=ce, accuracy=acc, fitness_score=None, f1_macro=metric.f1, fpr=metric.fpr)
			self._s._tracker.update_experiment_progress(
				self._s._tracker_experiment_id, current_iteration=self._completion_num,
				best_ce=self._best_ce, best_accuracy=self._best_acc)
		except Exception as e:
			self._log(f"  Warning: tracker error: {e}")

	def _record_seed(self, idx_in_new: int, genome, metric: Metrics, num_new: int) -> None:
		"""Per-expansion-genome tracker row (iteration_num continues after configs)."""
		ce, acc = metric.ce, metric.acc
		n = genome.neurons_per_cluster[0] if genome.neurons_per_cluster else 0
		b = genome.bits_per_neuron[0] if genome.bits_per_neuron else 0
		self._log(f"  [{idx_in_new+1}/{num_new}] n={n:3d}, b={b:2d}: CE={ce:.4f}  Acc={acc:.2%}")
		if not (self._s._tracker and self._s._tracker_experiment_id):
			return
		try:
			self._best_ce = min(self._best_ce, ce)
			self._best_acc = max(self._best_acc, acc)
			self._bump_f1_fpr(metric.f1, metric.fpr)
			seed_iter_num = len(self.results) + idx_in_new + 1
			iter_id = self._s._tracker.record_iteration(
				experiment_id=self._s._tracker_experiment_id,
				iteration_num=seed_iter_num,
				best_ce=self._best_ce, best_accuracy=self._best_acc,
				avg_ce=ce, avg_accuracy=acc, elapsed_secs=0.0,
				candidates_total=num_new, best_f1=self._best_f1, best_fpr=self._best_fpr)
			if HAS_GENOME_TRACKING and iter_id:
				gc = self._s._genome_to_config(genome)
				if gc:
					gid = self._s._tracker.get_or_create_genome(self._s._tracker_experiment_id, gc)
					self._s._tracker.record_genome_evaluation(
						iteration_id=iter_id, genome_id=gid, position=0, role=GenomeRole.INIT,
						ce=ce, accuracy=acc, fitness_score=None, f1_macro=metric.f1, fpr=metric.fpr)
			self._s._tracker.update_experiment_progress(
				self._s._tracker_experiment_id, current_iteration=seed_iter_num,
				best_ce=self._best_ce, best_accuracy=self._best_acc)
			self.num_seed_recorded += 1
		except Exception as e:
			self._log(f"  Warning: seed tracker error: {e}")

	def _bump_f1_fpr(self, f1, fpr) -> None:
		if f1 is not None and (self._best_f1 is None or f1 > self._best_f1):
			self._best_f1 = f1
		if fpr is not None and (self._best_fpr is None or fpr < self._best_fpr):
			self._best_fpr = fpr

	def record_final_iteration(self, outcome, pop_bests, pop_fitness_scores,
	                           final_iter_num: int, batch_elapsed: float) -> None:
		"""Final dashboard iteration: the whole surviving population, sorted by
		fitness. No-op without a tracker."""
		if not (self._s._tracker and self._s._tracker_experiment_id):
			return
		pop = outcome.seed_population
		mets = outcome.seed_metrics
		try:
			avg_ce = sum(m.ce for m in mets) / len(mets)
			avg_acc = sum(m.acc for m in mets) / len(mets)
			f1s = [m.f1 for m in mets if m.f1 is not None]
			fprs = [m.fpr for m in mets if m.fpr is not None]
			if f1s:
				self._bump_f1_fpr(max(f1s), None)
			if fprs:
				self._bump_f1_fpr(None, min(fprs))
			iter_id = self._s._tracker.record_iteration(
				experiment_id=self._s._tracker_experiment_id, iteration_num=final_iter_num,
				best_ce=pop_bests.best_ce.metrics.ce, best_accuracy=pop_bests.best_acc.metrics.acc,
				avg_ce=avg_ce, avg_accuracy=avg_acc, elapsed_secs=batch_elapsed,
				candidates_total=len(pop), best_f1=self._best_f1, best_fpr=self._best_fpr)
			if HAS_GENOME_TRACKING:
				for pos, (genome, m, fit) in enumerate(zip(pop, mets, pop_fitness_scores)):
					gc = self._s._genome_to_config(genome)
					if gc:
						gid = self._s._tracker.get_or_create_genome(self._s._tracker_experiment_id, gc)
						self._s._tracker.record_genome_evaluation(
							iteration_id=iter_id, genome_id=gid, position=pos, role=GenomeRole.INIT,
							ce=m.ce, accuracy=m.acc, fitness_score=fit, f1_macro=m.f1, fpr=m.fpr)
			self._s._tracker.update_experiment_progress(
				self._s._tracker_experiment_id, current_iteration=1,
				best_ce=pop_bests.best_ce.metrics.ce, best_accuracy=pop_bests.best_acc.metrics.acc)
		except Exception as e:
			self._log(f"  Warning: tracker error: {e}")


# =============================================================================
# Stats-Guided Adaptation Strategies (Neurogenesis, Synaptogenesis, Axonogenesis)
# =============================================================================
