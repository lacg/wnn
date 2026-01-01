"""
Memory-related enumerations.
"""

from enum import IntEnum


class MemoryVal(IntEnum):
	"""
	Ternary memory values stored as uint8:
	- FALSE = 0
	- TRUE  = 1
	- EMPTY = 2   (safe to overwrite; means "untrained")
	"""
	FALSE	= False
	TRUE	= True
	EMPTY	= 2
