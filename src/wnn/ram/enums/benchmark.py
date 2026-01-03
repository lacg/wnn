"""Benchmark-related enumerations."""

from enum import IntEnum


class BenchmarkMode(IntEnum):
	"""Benchmark execution modes with different parameter scales.

	Controls the intensity of optimization algorithms:
	- FAST: Quick development tests with minimal parameters
	- FULL: Production-level optimization
	- OVERNIGHT: Extended thorough optimization for overnight runs
	"""
	FAST = 0      # Quick test: minimal params for development
	FULL = 1      # Standard: production-level optimization
	OVERNIGHT = 2 # Extended: thorough overnight optimization
