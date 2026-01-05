"""
Cost calculators for RAM constraint solving (EDRA).

This module provides different strategies for selecting solutions
during EDRA (Error Detection and Reconstruction Algorithm) backpropagation.
"""

from enum import IntEnum


class CostCalculatorType(IntEnum):
	"""Type of cost calculator for EDRA constraint solving."""
	ARGMIN = 0
	STOCHASTIC = 1
	VOTE = 2
	RAM = 3


from .CostCalculator import CostCalculator
from .CostCalculatorArgMin import CostCalculatorArgMin
from .CostCalculatorFactory import CostCalculatorFactory
from .CostCalculatorStochastic import CostCalculatorStochastic
from .CostCalculatorVote import CostCalculatorVote


__all__ = [
	# Enum
	"CostCalculatorType",
	# Base class
	"CostCalculator",
	# Implementations
	"CostCalculatorArgMin",
	"CostCalculatorStochastic",
	"CostCalculatorVote",
	# Factory
	"CostCalculatorFactory",
]
