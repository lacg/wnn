"""
Adaptive Cluster Architecture Search — Per-Neuron Bits

Each neuron owns its synapse count (bit count) and evolves independently.
The GA mutates both connectivity AND structure (bits per neuron, neurons per cluster).

Key data structures:
- bits_per_neuron: [total_neurons] — each neuron's synapse count
- neurons_per_cluster: [num_clusters] — structural grouping
- connections: flat list, sum(bits_per_neuron) entries

Key insight: Frequent tokens need different architectures than rare tokens.
Let the data decide rather than hand-tuning tier boundaries.
"""

from __future__ import annotations

import gzip
import json
import math
import random

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch import Tensor

from wnn.progress import ProgressTracker
from wnn.ram.strategies.connectivity.generic_strategies import OptimizationLogger
from wnn.ram.strategies.factory import (
	OptimizerStrategyFactory,
	OptimizerStrategyType,
)
from wnn.ram.strategies.connectivity.generic_strategies import (
	GAConfig,
	TSConfig,
	OptimizerResult,
)

if TYPE_CHECKING:
	pass

# Try to import Rust accelerator for fast connection generation
try:
	import ram_accelerator as _accel
	_HAS_RUST = True
except ImportError:
	_HAS_RUST = False

# ClusterGenome + its config/helpers moved to wnn.ram.genome (D6c, 11/06/2026)
# — core/ may import them without reaching into strategies/. Re-exported here
# so existing imports keep working.
from wnn.ram.genome import (  # noqa: F401
	ClusterGenome,
	AdaptiveClusterConfig,
	GenomeInitStrategy,
	generate_connections,
	enforce_unique_connections,
	OptimizationDimension,
	PhaseType,
)

# =============================================================================
# Helper Functions (module-level, internal)
# =============================================================================

# =============================================================================
# The LM-era runner stack that lived here (RustParallelEvaluator,
# AdaptiveRAMLMWrapper, run_architecture_search / run_architecture_tabu_search /
# run_connectivity_optimization, create_genome_evaluator) was DELETED on
# 11/06/2026 (D6d): the runners passed kwargs that no longer existed in the
# factory signature — they were unreachable, and kwargs had been hiding it.
# Git history preserves them. Production paths use the architecture/ evaluators
# + OptimizerStrategyFactory with typed configs.
# =============================================================================
