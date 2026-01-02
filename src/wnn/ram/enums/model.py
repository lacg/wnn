"""
Model type enumeration for the unified factory.
"""

from enum import IntEnum, auto


class ModelType(IntEnum):
	"""
	High-level model architecture types.

	These represent the major model families in the RAM architecture.
	For transformer subtypes, see RAMTransformerType.
	"""
	# Recurrent models
	RECURRENT = auto()          # Basic RAMRecurrentNetwork
	KV_MEMORY = auto()          # RAMKVMemory with head-based routing

	# Transformer models
	TRANSFORMER = auto()        # RAMTransformer (use with RAMTransformerType)

	# Sequence-to-sequence models
	SEQ2SEQ = auto()            # RAMSeq2Seq
	ENCODER_DECODER = auto()    # RAMEncoderDecoder
