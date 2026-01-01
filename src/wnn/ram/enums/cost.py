"""
Cost calculator enumerations.
"""

from enum import IntEnum


class CostCalculatorType(IntEnum):
	"""Type of cost calculator for EDRA constraint solving."""
	ARGMIN		= 0
	STOCHASTIC	= 1
	VOTE		= 2
	RAM			= 3
