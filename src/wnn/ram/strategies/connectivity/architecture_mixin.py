"""
ArchitectureStrategyMixin — shared ClusterGenome strategy machinery (phase typing, batch eval, progress, percentile filters)

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
from wnn.ram.strategies.filters import PercentileFilter, FilterMode
from wnn.ram.strategies.connectivity.framework import OptimizerResult, StopReason
from wnn.ram.strategies.connectivity.adaptive_cluster import PhaseType
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole
from wnn.ram.strategies.connectivity.live_progress import LiveProgressObserver
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

class ArchitectureStrategyMixin:
	"""
	Mixin providing common functionality for GA and TS architecture strategies.

	Reduces code duplication by extracting:
	- Metal cleanup logic
	- Shutdown checking
	- genome_to_config conversion
	- Result building with stop_reason
	- Live progress observer for dashboard
	"""

	_shutdown_check: Optional[Callable[[], bool]]
	_log: Any  # Logger
	_dashboard_client: Any = None  # DashboardClient for live progress

	def set_dashboard_client(self, client) -> None:
		"""Set dashboard client for live progress reporting."""
		self._dashboard_client = client

	def _start_live_observer(self) -> Optional[LiveProgressObserver]:
		"""Start a live progress observer if dashboard client is available."""
		evaluator = getattr(self, '_cached_evaluator', None)
		client = self._dashboard_client
		experiment_id = getattr(self, '_tracker_experiment_id', None)
		if not evaluator:
			self._log("[LiveProgress] No evaluator, skipping observer")
		if not client:
			self._log("[LiveProgress] No dashboard client, skipping observer")
		if not experiment_id:
			self._log("[LiveProgress] No experiment_id, skipping observer")
		if evaluator and client and experiment_id:
			# Tell Rust which experiment this is for
			if hasattr(evaluator, 'set_experiment_context'):
				evaluator.set_experiment_context(experiment_id)
			# For BitwiseEvaluator/MultiStageEvaluator: reach the inner cache
			cache = getattr(evaluator, '_cache', None)
			if cache and hasattr(cache, 'set_experiment_context'):
				cache.set_experiment_context(experiment_id)
			observer = LiveProgressObserver(evaluator, client, experiment_id)
			observer.start()
			self._log(f"[LiveProgress] Observer started for experiment {experiment_id}")
			return observer
		return None

	def _stop_live_observer(self, observer: Optional[LiveProgressObserver]) -> None:
		"""Stop a live progress observer if one is active."""
		if observer:
			observer.stop()

	def _derive_phase_type(self) -> 'PhaseType':
		"""Derive PhaseType from optimize_* flags in ArchitectureConfig."""
		cfg = self._arch_config
		if cfg.optimize_connections and not cfg.optimize_bits and not cfg.optimize_neurons:
			return PhaseType.CONNECTIONS
		elif cfg.optimize_neurons and not cfg.optimize_bits:
			return PhaseType.NEURONS
		elif cfg.optimize_bits and not cfg.optimize_neurons:
			return PhaseType.BITS
		else:
			# Both bits+neurons or default: use Neurons phase (most general)
			return PhaseType.NEURONS

	def _cleanup_metal(self, iteration: int, log_interval: int = 10) -> None:
		"""Run GC and reset Metal evaluators to prevent buffer accumulation."""
		import gc
		gc.collect()
		try:
			import ram_accelerator
			ram_accelerator.reset_metal_evaluators()
			if iteration % log_interval == 0:
				import resource
				# macOS: ru_maxrss is in bytes
				rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
				self._log.info(f"[{self.name}] GC + Metal reset at iteration {iteration}, RSS: {rss_mb:.0f} MB")
		except Exception:
			pass  # Ignore if accelerator not available

	def _check_and_set_shutdown(self, shutdown_flag: list[bool]) -> bool:
		"""
		Check if shutdown is requested and update the shutdown flag.

		Args:
			shutdown_flag: List with single bool element (mutable reference)

		Returns:
			True if shutdown was requested
		"""
		if self._shutdown_check and self._shutdown_check():
			shutdown_flag[0] = True
			return True
		return False

	def _compute_threshold(self, progress: float = 0.0) -> float:
		"""Compute progressive accuracy threshold at given progress [0, 1].

		Shared across GA, TS, and Adaptation strategies. Config must provide:
		initial_threshold, progressive_threshold, threshold_delta, min_accuracy.
		"""
		cfg = self._config
		start = cfg.initial_threshold if cfg.initial_threshold is not None else getattr(cfg, 'min_accuracy', 0.0)
		if not getattr(cfg, 'progressive_threshold', True):
			return start
		progress = max(0.0, min(1.0, progress))
		return start + progress * cfg.threshold_delta

	def _genome_to_config_impl(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""
		Convert a ClusterGenome to a GenomeConfig for tracking.

		Finds contiguous runs of clusters with the same (neurons, mean_bits) config.
		This enables proper tracking of which cluster indices belong to which tier.

		The tier index is assigned based on the order of first appearance
		(earliest cluster index = tier 0), preserving the original tier config order.

		Per-neuron bits are averaged per cluster to produce a representative bits value.
		"""
		if not HAS_GENOME_TRACKING or GenomeConfig is None or TierConfig is None:
			return None

		# Find contiguous runs of clusters with same (neurons, mean_bits)
		runs: list[tuple[int, int, int, int]] = []  # (start, end, neurons, mean_bits)
		if len(genome.neurons_per_cluster) == 0:
			return GenomeConfig(tiers=[])

		neuron_offsets = genome.cluster_neuron_offsets

		def _cluster_mean_bits(c: int) -> int:
			"""Compute rounded mean bits for cluster c."""
			start_n = neuron_offsets[c]
			end_n = neuron_offsets[c + 1]
			if end_n == start_n:
				return 0
			return round(sum(genome.bits_per_neuron[start_n:end_n]) / (end_n - start_n))

		current_neurons = genome.neurons_per_cluster[0]
		current_bits = _cluster_mean_bits(0)
		run_start = 0

		for i in range(1, len(genome.neurons_per_cluster)):
			neurons = genome.neurons_per_cluster[i]
			bits = _cluster_mean_bits(i)
			if neurons != current_neurons or bits != current_bits:
				# End current run, start new one
				runs.append((run_start, i, current_neurons, current_bits))
				current_neurons = neurons
				current_bits = bits
				run_start = i

		# Don't forget the last run
		runs.append((run_start, len(genome.neurons_per_cluster), current_neurons, current_bits))

		# Assign tier indices based on first appearance of each (neurons, bits) config
		config_to_tier: dict[tuple[int, int], int] = {}
		next_tier = 0
		for _, _, neurons, bits in runs:
			key = (neurons, bits)
			if key not in config_to_tier:
				config_to_tier[key] = next_tier
				next_tier += 1

		# Create TierConfig for each contiguous run
		tiers = []
		for start, end, neurons, bits in runs:
			tier_idx = config_to_tier[(neurons, bits)]
			tiers.append(TierConfig(
				tier=tier_idx,
				clusters=end - start,
				neurons=neurons,
				bits=bits,
				start_cluster=start,
				end_cluster=end,
			))

		return GenomeConfig(tiers=tiers)

	def _determine_stop_reason(
		self,
		shutdown_requested: bool,
		early_stopper: Any,
	) -> Optional[StopReason]:
		"""Determine the stop reason based on shutdown flag and early stopper state."""
		if shutdown_requested:
			return StopReason.SHUTDOWN
		elif hasattr(early_stopper, 'patience_exhausted') and early_stopper.patience_exhausted:
			return StopReason.CONVERGENCE
		return None

	def _run_validation_summary(self, result: OptimizerResult) -> OptimizerResult:
		"""Run full-data validation on top genomes and update result.

		Shared between GA and TS architecture strategies.
		Evaluates top 20% of final population on full validation data,
		logs summary, records phase results, and returns updated result
		with best-CE genome.
		"""
		evaluator = self._cached_evaluator
		if evaluator is None or not result.final_population:
			return result

		self._log.info("")
		self._log.info("=" * 60)
		self._log.info(f"[{self.name}] VALIDATION SUMMARY (Full Dataset)")
		self._log.info("=" * 60)

		# Get top 20% for full evaluation
		top_k = max(1, int(len(result.final_population) * 0.2))
		top_genomes = result.final_population[:top_k]

		# Evaluate on full validation data
		full_results = evaluator.evaluate_batch_full(top_genomes)

		# Use fitness calculator to extract bests
		bests = self._fitness_calculator.bests(top_genomes, full_results)

		self._log.info(f"  Best by CE:       CE={bests.best_ce.metrics.ce:.4f}, Acc={bests.best_ce.metrics.acc:.4%}")
		self._log.info(f"  Best by Accuracy: CE={bests.best_acc.metrics.ce:.4f}, Acc={bests.best_acc.metrics.acc:.4%}")

		if bests.best_fitness.genome is bests.best_ce.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by CE)")
		elif bests.best_fitness.genome is bests.best_acc.genome:
			self._log.info(f"  Best by Fitness:  (same as Best by Accuracy)")
		else:
			self._log.info(f"  Best by Fitness:  CE={bests.best_fitness.metrics.ce:.4f}, Acc={bests.best_fitness.metrics.acc:.4%}")

		top_k_ce = sum(r.ce for r in full_results) / len(full_results)
		top_k_acc = sum(r.acc for r in full_results) / len(full_results)
		self._log.info(f"  Top-{top_k} Mean:    CE={top_k_ce:.4f}, Acc={top_k_acc:.4%}")
		self._log.info("=" * 60)

		# Record phase results via tracker
		if self._tracker and self._tracker_experiment_id:
			try:
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_ce",
					ce=bests.best_ce.metrics.ce,
					accuracy=bests.best_ce.metrics.acc,
					improvement_pct=(result.initial_fitness - bests.best_ce.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="best_acc",
					ce=bests.best_acc.metrics.ce,
					accuracy=bests.best_acc.metrics.acc,
					improvement_pct=(result.initial_fitness - bests.best_acc.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
				self._tracker.record_phase_result(
					experiment_id=self._tracker_experiment_id,
					metric_type="top_k_mean",
					ce=top_k_ce,
					accuracy=top_k_acc,
					improvement_pct=(result.initial_fitness - top_k_ce) / result.initial_fitness * 100 if result.initial_fitness else 0.0,
				)
			except Exception as e:
				self._log.debug(f"[{self.name}] Failed to record phase results: {e}")

		# Update result with validation bests
		improvement_pct = (result.initial_fitness - bests.best_ce.metrics.ce) / result.initial_fitness * 100 if result.initial_fitness != 0 else 0.0
		return OptimizerResult(
			initial_genome=result.initial_genome,
			best_genome=bests.best_ce.genome,
			initial_fitness=result.initial_fitness,
			final_fitness=bests.best_ce.metrics.ce,
			improvement_percent=improvement_pct,
			iterations_run=result.iterations_run,
			method_name=result.method_name,
			history=result.history,
			early_stopped=result.early_stopped,
			stop_reason=result.stop_reason,
			final_population=result.final_population,
			population_metrics=result.population_metrics,
			initial_accuracy=result.initial_accuracy,
			final_accuracy=bests.best_acc.metrics.acc,
			final_threshold=result.final_threshold,
		)

	def _apply_percentile_filter(
		self,
		offspring: list[tuple['ClusterGenome', float, Optional[float]]],
	) -> list[tuple['ClusterGenome', float, Optional[float]]]:
		"""Apply fitness percentile filter to offspring/neighbors (3-tuple format).

		Shared between GA (offspring) and TS (neighbors).
		"""
		cfg = self._config
		fitness_calculator = self._fitness_calculator

		if cfg.fitness_calculator_type == FitnessCalculatorType.CE:
			# CE mode: filter by CE only
			ce_filter = PercentileFilter(
				percentile=cfg.fitness_percentile,
				mode=FilterMode.LOWER_IS_BETTER,
				metric_name="CE",
			)
			offspring_2t = [(t[0], t[1]) for t in offspring]
			filter_result = ce_filter.apply(offspring_2t, key=lambda g, f: f)
			kept_ids = {id(g) for g, _ in filter_result.kept}
			offspring = [t for t in offspring if id(t[0]) in kept_ids]
		else:
			# HARMONIC_RANK or NORMALIZED: filter by fitness score
			offspring_metrics = [t[1] for t in offspring]
			fitness_scores = fitness_calculator.fitness(offspring_metrics)
			offspring_with_fitness = list(zip(offspring, fitness_scores))

			fitness_filter = PercentileFilter(
				percentile=cfg.fitness_percentile,
				mode=FilterMode.LOWER_IS_BETTER,
				metric_name=fitness_calculator.name,
			)
			filter_input = [(item, score) for item, score in offspring_with_fitness]
			filter_result = fitness_filter.apply(filter_input, key=lambda t, f: f)
			offspring = [t for t, _ in filter_result.kept]

		if filter_result.filtered:
			self._log.debug(
				f"[{self.name}] {filter_result.metric_name} filter: kept {filter_result.kept_count}/{filter_result.total_count} "
				f"(threshold={filter_result.threshold_value:.4f})"
			)

		return offspring


# =============================================================================
# Checkpoint System for Resume Support
# =============================================================================
