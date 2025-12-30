from enum import IntEnum

class CostCalculatorType(IntEnum):
	ARGMIN			= 0
	STOCHASTIC	= 1
	VOTE				= 2
