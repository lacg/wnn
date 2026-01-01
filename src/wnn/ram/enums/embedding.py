"""
Embedding-related enumerations.
"""

from enum import IntEnum


class PositionEncoding(IntEnum):
	"""How to encode position information in embeddings."""
	NONE = 0       # No position encoding
	BINARY = 1     # Binary representation of position
	LEARNED = 2    # Learned position embeddings
