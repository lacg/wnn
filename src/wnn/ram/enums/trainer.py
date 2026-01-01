"""
Trainer-related enumerations.
"""

from enum import IntEnum


class LayerType(IntEnum):
	"""Types of layers in the RAM Transformer."""
	EMBEDDING = 0
	INPUT_PROJ = 1
	ATTENTION = 2
	FFN = 3
	OUTPUT_PROJ = 4
	TOKEN_MAPPER = 5
