"""
Architecture optimization strategies — backward-compatibility re-export shim.

The classes were split class-per-file (D3, 11/06/2026):
- LiveProgressObserver → live_progress.py
- ArchitectureStrategyMixin → architecture_mixin.py
- CheckpointConfig + CheckpointManager → checkpoint_manager.py
- ArchitectureConfig → architecture_config.py
- ArchitectureGAStrategy → architecture_ga.py
- ArchitectureTSStrategy → architecture_ts.py
- ArchitectureSAStrategy → architecture_sa.py
- GridSearchConfig + GridSearchStrategy → grid_search.py
- AdaptationConfig + AdaptationStrategy → adaptation.py
- tracker-optional names (HAS_GENOME_TRACKING, ...) → genome_tracking.py

Import from the specific modules in new code; this shim only keeps existing
imports working.
"""

from wnn.ram.strategies.connectivity.live_progress import LiveProgressObserver
from wnn.ram.strategies.connectivity.architecture_mixin import ArchitectureStrategyMixin
from wnn.ram.strategies.connectivity.checkpoint_manager import CheckpointConfig, CheckpointManager
from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
from wnn.ram.strategies.connectivity.architecture_ts import ArchitectureTSStrategy
from wnn.ram.strategies.connectivity.architecture_sa import ArchitectureSAStrategy
from wnn.ram.strategies.connectivity.grid_search import GridSearchConfig, GridSearchStrategy
from wnn.ram.strategies.connectivity.adaptation import AdaptationConfig, AdaptationStrategy
from wnn.ram.strategies.connectivity.genome_tracking import (
	HAS_GENOME_TRACKING, TierConfig, GenomeConfig, GenomeRole,
)

__all__ = [
	"LiveProgressObserver",
	"ArchitectureStrategyMixin",
	"CheckpointConfig", "CheckpointManager",
	"ArchitectureConfig",
	"ArchitectureGAStrategy", "ArchitectureTSStrategy", "ArchitectureSAStrategy",
	"GridSearchConfig", "GridSearchStrategy",
	"AdaptationConfig", "AdaptationStrategy",
	"HAS_GENOME_TRACKING", "TierConfig", "GenomeConfig", "GenomeRole",
]
