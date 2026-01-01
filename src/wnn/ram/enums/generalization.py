"""
Generalization-related enumerations for RAM networks.
"""

from enum import IntEnum


class ContextMode(IntEnum):
	"""How much context each bit sees in BitLevelMapper."""
	CUMULATIVE = 0  # bit i sees bits 0..i-1 (only LOWER bits for flip)
	FULL = 1        # each bit sees all bits
	LOCAL = 2       # each bit sees nearby bits (sliding window)


class BitMapperMode(IntEnum):
	"""
	What to learn in BitLevelMapper.

	Note: Previously named 'OutputMode' in RAMGeneralization.py,
	renamed to avoid conflict with decoder's OutputMode.
	"""
	OUTPUT = 0  # Learn the output bit value directly
	FLIP = 1    # Learn whether to flip (XOR) the input bit


# Backwards compatibility alias
OutputMode = BitMapperMode


class MapperStrategy(IntEnum):
	"""Generalization strategy for MapperFactory."""
	DIRECT = 0        # Standard RAMLayer (no generalization)
	BIT_LEVEL = 1     # Use BitLevelMapper
	COMPOSITIONAL = 2 # Use CompositionalMapper
	HYBRID = 3        # Combine compositional + bit-level
