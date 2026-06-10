"""
Optimization Strategies (GA / TS / SA) for architecture + connectivity search.

Lineage: Garcia (2003) thesis on global optimization methods for choosing
connectivity patterns of weightless neural networks. The modern stack
operates on ClusterGenome (architecture + connectivity together):

- GenericGAStrategy / GenericTSStrategy / GenericSAStrategy — algorithm cores
  on the OptimizationTemplate framework (fitness calculators, early stopping,
  progressive threshold, checkpoint/tracker reporting)
- ArchitectureGAStrategy / ArchitectureTSStrategy / ArchitectureSAStrategy —
  ClusterGenome implementations (in architecture_strategies)

The LM-era connectivity-only stack (OptimizerStrategyBase,
GeneticAlgorithmStrategy, TabuSearchStrategy, SimulatedAnnealingStrategy,
AcceleratedOptimizer, ConnectivityOptimizer, per_cluster) was removed
10/06/2026 — see docs/ARCHITECTURE_REVIEW_2026-06.md §2.3. SA survives as
GenericSAStrategy (same Metropolis acceptance + cooling schedule).

Usage:
	from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType

	strategy = OptimizerStrategyFactory.create(
		OptimizerStrategyType.ARCHITECTURE_GA,
		num_clusters=2,
		population_size=30,
		generations=50,
		batch_evaluator=evaluator,
	)
"""

from wnn.ram.strategies.connectivity.generic_strategies import (
	StopReason,
	AdaptiveLevel,
	AdaptiveScaler,
	AdaptiveScalerConfig,
	ProgressiveThreshold,
	ProgressiveThresholdConfig,
	OptimizationConfig,
	GAConfig,
	TSConfig,
	SAConfig,
	GenericGAStrategy,
	GenericTSStrategy,
	GenericSAStrategy,
	OptimizerResult,
	# Logging
	OptimizationLogger,
	TRACE,
)
# Preferred: use enum instead of constants
from wnn.core.thresholds import OverfitThreshold, EarlyStopThreshold

# Import unified factory from strategies module
from wnn.ram.strategies.factory import OptimizerStrategyFactory, OptimizerStrategyType


__all__ = [
	# Results / control
	'OptimizerResult',
	'StopReason',
	# Adaptive scaling
	'AdaptiveLevel',
	'AdaptiveScaler',
	'AdaptiveScalerConfig',
	# Progressive threshold
	'ProgressiveThreshold',
	'ProgressiveThresholdConfig',
	# Logging
	'OptimizationLogger',
	'TRACE',
	# Threshold enums (preferred)
	'OverfitThreshold',
	'EarlyStopThreshold',
	# Configs
	'OptimizationConfig',
	'GAConfig',
	'TSConfig',
	'SAConfig',
	# Strategy cores
	'GenericGAStrategy',
	'GenericTSStrategy',
	'GenericSAStrategy',
	# Factory
	'OptimizerStrategyFactory',
	'OptimizerStrategyType',
]
