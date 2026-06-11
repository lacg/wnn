"""
Generic GA / TS / SA strategies — backward-compatibility re-export shim.

The classes were split class-per-file (D3, 11/06/2026):
- GenericGAStrategy → generic_ga.py
- GenericTSStrategy → generic_ts.py
- GenericSAStrategy → generic_sa.py

Framework primitives (configs, early stopping, logging, results) live in the
framework/ package since D6a. Import from the specific modules in new code;
this shim only keeps existing imports working.
"""

from typing import TypeVar

# Framework primitives (originally defined here, moved D6a).
from wnn.ram.strategies.connectivity.framework import (
	OptimizationLogger, TRACE, OverfitDetector,
	OptimizerResult, StopReason,
	EarlyStoppingConfig, EarlyStoppingTracker,
	AdaptiveLevel, AdaptiveScalerConfig, AdaptiveScaler,
	ProgressiveThresholdConfig, ProgressiveThreshold,
	OptimizationConfig, GAConfig, TSConfig, SAConfig,
)

from wnn.ram.strategies.connectivity.generic_ga import GenericGAStrategy, HAS_TRACKER
from wnn.ram.strategies.connectivity.generic_ts import GenericTSStrategy
from wnn.ram.strategies.connectivity.generic_sa import GenericSAStrategy

# Generic genome type (historically importable from this module)
T = TypeVar('T')

__all__ = [
	"OptimizationLogger", "TRACE", "OverfitDetector",
	"OptimizerResult", "StopReason",
	"EarlyStoppingConfig", "EarlyStoppingTracker",
	"AdaptiveLevel", "AdaptiveScalerConfig", "AdaptiveScaler",
	"ProgressiveThresholdConfig", "ProgressiveThreshold",
	"OptimizationConfig", "GAConfig", "TSConfig", "SAConfig",
	"GenericGAStrategy", "GenericTSStrategy", "GenericSAStrategy",
	"HAS_TRACKER", "T",
]
