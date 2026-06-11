"""
ArchitectureConfig — bounds + dimensions config for architecture search

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


if TYPE_CHECKING:
	from wnn.ram.strategies.connectivity.adaptive_cluster import (
		ClusterGenome,
		AdaptiveClusterConfig,
	)

@dataclass
class ArchitectureConfig:
	"""
	Configuration for architecture optimization.

	Controls both the search space bounds and what gets optimized.
	The optimizer is phase-agnostic - callers control what to optimize
	by setting the optimize_* flags.

	Example usage:
		# Phase 1: Optimize neurons only (bits fixed at default_bits)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=True,
			default_bits=8,  # All genomes start with 8 bits
		)

		# Phase 2: Optimize bits only (pass seed genome from Phase 1)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=True,
			optimize_neurons=False,
		)

		# Phase 3: Optimize connections only (pass seed genome from Phase 2)
		config = ArchitectureConfig(
			num_clusters=50257,
			optimize_bits=False,
			optimize_neurons=False,
			optimize_connections=True,
		)
	"""
	num_clusters: int
	min_bits: int = 4
	max_bits: int = 24
	min_neurons: int = 3
	max_neurons: int = 30
	# Explicit control over what gets optimized (no magic phase numbers)
	optimize_bits: bool = True
	optimize_neurons: bool = True
	optimize_connections: bool = False
	# Default values for dimensions not being optimized (used in random genome init)
	default_bits: int = 8
	default_neurons: int = 5
	# Token frequencies for frequency-scaled initialization
	token_frequencies: Optional[list[int]] = None
	# Total input bits for connection initialization/mutation
	total_input_bits: Optional[int] = None
	# Per-tier optimization: list of cluster indices that can be mutated (None = all clusters mutable)
	mutable_clusters: Optional[list[int]] = None
	# Cluster-level crossover ratio: 0.0 = all phase-specific, 1.0 = all cluster-level
	cluster_crossover_ratio: float = 0.0
	# Pool-and-shuffle crossover ratio: 0.0 = all uniform (2→2), 1.0 = all pool-and-shuffle (2→1)
	pool_shuffle_ratio: float = 0.0
	# Assortative mating ratio: 0.0 = random p2, 1.0 = always pick most similar p2 (NEAT-style)
	assortative_mating_ratio: float = 0.85
