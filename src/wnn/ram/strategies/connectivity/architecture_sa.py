"""
ArchitectureSAStrategy — simulated annealing over ClusterGenome on the generic SA core (Garcia-2003)

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

from wnn.ram.strategies.connectivity.framework import SAConfig
from wnn.ram.strategies.connectivity.generic_sa import GenericSAStrategy
from wnn.ram.strategies.connectivity.genome_tracking import HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole
from wnn.ram.strategies.connectivity.architecture_mixin import ArchitectureStrategyMixin
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig

if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

class ArchitectureSAStrategy(ArchitectureStrategyMixin, GenericSAStrategy['ClusterGenome']):
	"""
	Simulated Annealing for architecture (bits, neurons per cluster) optimization.

	Inherits the chained-Metropolis SA loop from GenericSAStrategy (Garcia 2003
	acceptance + cooling, batched chain proposals), implements ClusterGenome
	operations. Uses ArchitectureStrategyMixin for shared functionality
	(phase typing, genome→config tracking, Metal cleanup, shutdown).

	Unlike TS, SA needs no tabu move tracking — mutate_genome returns
	move_info=None.
	"""

	def __init__(
		self,
		arch_config: ArchitectureConfig,
		sa_config: Optional[SAConfig] = None,
		seed: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		batch_evaluator: Optional[Any] = None,
		cached_evaluator: Optional[Any] = None,
		shutdown_check: Optional[Callable[[], bool]] = None,
	):
		super().__init__(config=sa_config, seed=seed, logger=logger)
		self._arch_config = arch_config
		self._batch_evaluator = batch_evaluator
		self._cached_evaluator = cached_evaluator if cached_evaluator is not None else (
			batch_evaluator if batch_evaluator is not None and hasattr(batch_evaluator, 'search_neighbors') else None
		)
		self._shutdown_check = shutdown_check
		self._phase_type = self._derive_phase_type()

	@property
	def name(self) -> str:
		return "ArchitectureSA"

	def genome_to_config(self, genome: 'ClusterGenome') -> Optional['GenomeConfig']:
		"""Convert a ClusterGenome to a GenomeConfig for tracking."""
		return self._genome_to_config_impl(genome)

	def clone_genome(self, genome: 'ClusterGenome') -> 'ClusterGenome':
		return genome.clone()

	def mutate_genome(self, genome: 'ClusterGenome', mutation_rate: float) -> tuple['ClusterGenome', Any]:
		"""Phase-aware mutation via ClusterGenome.mutate(). No move tracking (SA has no tabu)."""
		from wnn.ram.strategies.connectivity.adaptive_cluster import AdaptiveClusterConfig
		self._ensure_rng()
		cfg = self._arch_config
		mutation_config = AdaptiveClusterConfig(
			min_bits=cfg.min_bits, max_bits=cfg.max_bits,
			min_neurons=cfg.min_neurons, max_neurons=cfg.max_neurons,
			neuron_bounds_per_cluster=cfg.neuron_bounds_per_cluster,
			bits_bounds_per_cluster=cfg.bits_bounds_per_cluster,
		)
		tib = cfg.total_input_bits or 64
		mutant = genome.mutate(self._phase_type, mutation_rate, mutation_config, tib, self._rng)
		return mutant, None
