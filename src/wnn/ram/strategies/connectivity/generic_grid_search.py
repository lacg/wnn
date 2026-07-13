"""
GenericGridSearch — genome-agnostic grid-search core (template method).

Owns the ONE copy of the grid → evaluate → rank → top-K → expand → trim
pipeline that was previously duplicated between the IDS `GridSearchStrategy`
and the controller's hand-rolled `stage0_grid`. Subclasses inject the
strand-specific pieces via typed hooks; the invariant algebra (top-K slicing,
per-config expansion counts, fitness re-sort, trim) lives here and is reused by
inheritance, never copied.

Hooks a subclass MUST implement:
  _enumerate_points()               -> list of opaque "points" (one per grid cell)
  _make_genome(point)               -> a genome for that point (RNG-order-critical)
  _make_variant(point)              -> a fresh same-shape genome for expansion
  _evaluate(genomes, is_expansion)  -> list of metrics aligned to `genomes`
  _fitness(metrics)                 -> list[float] scores (lower = better)

Overridable no-op hooks (default do nothing):
  _on_ranked(ranked)                -> after the config ranking is known
  _on_final(outcome)                -> after the seed population is finalized

The base never inspects a point, genome, or metric — those stay opaque — so it
depends on neither `ClusterGenome`/`RecurrentArchGenome` nor a `Metrics` type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from wnn.ram.strategies.connectivity.grid_search_outcome import GridSearchOutcome


class GenericGridSearch(ABC):
	"""Template-method grid search shared by IDS and the controller."""

	def __init__(self, top_k: int, population_size: int,
	             log: Callable[[str], None] = print):
		self._top_k = top_k
		self._population_size = population_size
		self._log = log

	# ---- hooks a subclass MUST implement ---------------------------------
	@abstractmethod
	def _enumerate_points(self) -> list:
		"""Return the list of grid points (one per (neurons, bits, …) cell)."""
		...

	@abstractmethod
	def _make_genome(self, point: Any) -> Any:
		"""Build the genome for a point. Called in enumeration order — the RNG
		draw order here is part of parity, do not reorder."""
		...

	@abstractmethod
	def _make_variant(self, point: Any) -> Any:
		"""Build a fresh same-shape genome for the expansion phase."""
		...

	@abstractmethod
	def _evaluate(self, genomes: list, is_expansion: bool) -> list:
		"""Evaluate a batch of genomes; return metrics aligned to `genomes`.
		Owns any concurrency / streaming / tracker writes the strand needs."""
		...

	@abstractmethod
	def _fitness(self, metrics: list) -> list:
		"""Return fitness scores (lower = better) for `metrics`. Rank-based
		calculators depend on the whole set, so score exactly what is passed."""
		...

	# ---- overridable no-op hooks -----------------------------------------
	def _on_ranked(self, ranked: list) -> None:
		"""Called once the config ranking (list of (point, genome, metric,
		score), best first) is known. Default: nothing."""

	def _on_final(self, outcome: GridSearchOutcome) -> None:
		"""Called once the seed population is finalized. Default: nothing."""

	# ---- the invariant pipeline (implemented once) -----------------------
	def run(self) -> GridSearchOutcome:
		ranked = self._rank_configs()
		self._on_ranked(ranked)
		pop_points, pop_genomes, pop_metrics = self._expand_population(ranked)
		self._finalize_metrics(pop_genomes, pop_metrics)
		pop_points, pop_genomes, pop_metrics, _ = self._sort_and_trim(
			pop_points, pop_genomes, pop_metrics)
		outcome = GridSearchOutcome(
			seed_population=pop_genomes,
			seed_metrics=pop_metrics,
			seed_points=pop_points,
			best_genome=pop_genomes[0],
			best_point=pop_points[0],
			best_metrics=pop_metrics[0],
			ranked_points=[t[0] for t in ranked],
		)
		self._on_final(outcome)
		return outcome

	def _rank_configs(self) -> list:
		"""Evaluate every grid point once and rank by fitness (best first)."""
		points = self._enumerate_points()
		if not points:
			raise ValueError("Grid search produced no points")
		genomes = [self._make_genome(p) for p in points]   # enumeration order
		metrics = self._evaluate(genomes, is_expansion=False)
		scores = self._fitness(metrics)
		ranked = list(zip(points, genomes, metrics, scores))
		ranked.sort(key=lambda t: t[3])
		return ranked

	def _expand_population(self, ranked: list) -> tuple:
		"""Distribute population_size across the top-K configs: config original
		first, then fresh same-shape variants. Variants get placeholder (None)
		metrics, filled by `_finalize_metrics`."""
		top_k = min(self._top_k, len(ranked))
		target_total = max(top_k, int(self._population_size * 1.1))
		base_per_config = target_total // top_k
		remainder = target_total - base_per_config * top_k

		pop_points: list = []
		pop_genomes: list = []
		pop_metrics: list = []
		for i in range(top_k):
			point_i, genome_i, metric_i, _ = ranked[i]
			count = max(1, base_per_config + (1 if i < remainder else 0))
			pop_points.append(point_i)
			pop_genomes.append(genome_i)
			pop_metrics.append(metric_i)
			for _ in range(count - 1):
				pop_points.append(point_i)
				pop_genomes.append(self._make_variant(point_i))
				pop_metrics.append(None)
		return pop_points, pop_genomes, pop_metrics

	def _finalize_metrics(self, pop_genomes: list, pop_metrics: list) -> None:
		"""Evaluate the freshly-made variants (those with placeholder metrics)
		and fill them in place, preserving order."""
		new_indices = [i for i, m in enumerate(pop_metrics) if m is None]
		if not new_indices:
			return
		new_genomes = [pop_genomes[i] for i in new_indices]
		new_metrics = self._evaluate(new_genomes, is_expansion=True)
		for idx, metric in zip(new_indices, new_metrics):
			pop_metrics[idx] = metric

	def _sort_and_trim(self, pop_points: list, pop_genomes: list,
	                   pop_metrics: list) -> tuple:
		"""Re-rank the full population by fitness and trim to population_size."""
		scores = self._fitness(pop_metrics)
		order = sorted(range(len(pop_genomes)), key=lambda i: scores[i])
		pop_points = [pop_points[i] for i in order]
		pop_genomes = [pop_genomes[i] for i in order]
		pop_metrics = [pop_metrics[i] for i in order]
		scores = [scores[i] for i in order]
		if len(pop_genomes) > self._population_size:
			dropped = len(pop_genomes) - self._population_size
			self._log(f"  Trimming population: {len(pop_genomes)} → "
			          f"{self._population_size} (dropped {dropped} weakest)")
			pop_points = pop_points[:self._population_size]
			pop_genomes = pop_genomes[:self._population_size]
			pop_metrics = pop_metrics[:self._population_size]
			scores = scores[:self._population_size]
		return pop_points, pop_genomes, pop_metrics, scores
