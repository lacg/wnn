"""
Encoders and Decoders for RAM Transformers.

Decoders: Interpret output bits as predictions
Position Encoders: Encode sequence positions into bits
"""

from enum import IntEnum


class OutputMode(IntEnum):
	"""How to decode output layer bits into predictions."""
	BITWISE = 0
	HAMMING = 1
	RAW = 2
	TOKEN = 3
	TOKEN_LIST = 4


class PositionMode(IntEnum):
	"""How to encode sequence positions into bits."""
	NONE = 0      # No position encoding
	BINARY = 1    # Binary encoding: pos 5 → [1,0,1]
	RELATIVE = 2  # Relative position: distance from query
	LEARNED = 3   # Learned position embeddings via RAMLayer


# Decoders
from .TransformerDecoder import TransformerDecoder
from .TransformerBitWiseDecoder import TransformerBitWiseDecoder
from .TransformerHammingDecoder import TransformerHammingDecoder
from .TransformerRawDecoder import TransformerRawDecoder
from .TransformerTokenDecoder import TransformerTokenDecoder
from .TransformerTokenListDecoder import TransformerTokenListDecoder
from .TransformerDecoderFactory import TransformerDecoderFactory

# Position encoders
from .PositionEncoder import PositionEncoder
from .BinaryPositionEncoder import BinaryPositionEncoder
from .RelativePositionEncoder import RelativePositionEncoder
from .LearnedPositionEncoder import LearnedPositionEncoder
from .PositionEncoderFactory import PositionEncoderFactory


__all__ = [
	# Enums
	"OutputMode",
	"PositionMode",
	# Decoders
	"TransformerDecoder",
	"TransformerBitWiseDecoder",
	"TransformerHammingDecoder",
	"TransformerRawDecoder",
	"TransformerTokenDecoder",
	"TransformerTokenListDecoder",
	"TransformerDecoderFactory",
	# Position encoders
	"PositionEncoder",
	"BinaryPositionEncoder",
	"RelativePositionEncoder",
	"LearnedPositionEncoder",
	"PositionEncoderFactory",
]
