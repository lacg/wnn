"""Enums for connectivity optimization."""

from enum import IntEnum, auto


class OptimizationMethod(IntEnum):
	"""
	Optimization method for connectivity patterns.

	Based on Garcia (2003) thesis comparing global optimization methods
	for choosing connectivity patterns of weightless neural networks.
	"""
	TABU_SEARCH = auto()          # Best results: 17.27% error reduction, only 5 iterations
	SIMULATED_ANNEALING = auto()  # Good for escaping local minima, 600 iterations
	GENETIC_ALGORITHM = auto()    # Good for large search spaces, can reduce memory by 89%
