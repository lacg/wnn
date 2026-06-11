"""
Optional tracker integration for genome tracking.

Shared by the architecture strategy modules (split out of
architecture_strategies.py, D3 11/06/2026).
"""

try:
	from wnn.ram.experiments.tracker import TierConfig, GenomeConfig, GenomeRole
	HAS_GENOME_TRACKING = True
except ImportError:
	HAS_GENOME_TRACKING = False
	TierConfig = None
	GenomeConfig = None
	GenomeRole = None
