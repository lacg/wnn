"""
Decoder-related enumerations.
"""

from enum import IntEnum


class OutputMode(IntEnum):
	"""How to decode output layer bits into predictions."""
	BITWISE			= 0
	HAMMING			= 1
	RAW				= 2
	TOKEN			= 3
	TOKEN_LIST		= 4


class PositionMode(IntEnum):
	"""How to encode sequence positions into bits."""
	NONE			= 0  # No position encoding
	BINARY			= 1  # Binary encoding: pos 5 → [1,0,1]
	RELATIVE		= 2  # Relative position: distance from query
	LEARNED			= 3  # Learned position embeddings via RAMLayer
