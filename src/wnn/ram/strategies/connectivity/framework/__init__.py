"""
Optimization framework primitives shared by every strategy (GA / TS / SA / grid).

Extracted from generic_strategies.py (D6a, 11/06/2026) to break the
generic_strategies ↔ optimization_template import cycle: the template imports
these primitives from here; strategy modules import the template. One
class-family per module (project class-per-file rule).
"""

from wnn.ram.strategies.connectivity.framework.logger import OptimizationLogger, TRACE
from wnn.ram.strategies.connectivity.framework.overfit import OverfitDetector
from wnn.ram.strategies.connectivity.framework.results import OptimizerResult, StopReason
from wnn.ram.strategies.connectivity.framework.early_stopping import EarlyStoppingConfig, EarlyStoppingTracker
from wnn.ram.strategies.connectivity.framework.adaptive_scaling import AdaptiveLevel, AdaptiveScalerConfig, AdaptiveScaler
from wnn.ram.strategies.connectivity.framework.progressive_threshold import ProgressiveThresholdConfig, ProgressiveThreshold
from wnn.ram.strategies.connectivity.framework.configs import OptimizationConfig, GAConfig, TSConfig, SAConfig

__all__ = [
	'OptimizationLogger', 'TRACE', 'OverfitDetector',
	'OptimizerResult', 'StopReason',
	'EarlyStoppingConfig', 'EarlyStoppingTracker',
	'AdaptiveLevel', 'AdaptiveScalerConfig', 'AdaptiveScaler',
	'ProgressiveThresholdConfig', 'ProgressiveThreshold',
	'OptimizationConfig', 'GAConfig', 'TSConfig', 'SAConfig',
]
