"""
Generalization-related enumerations for RAM networks.
"""

from enum import IntEnum


class ContextMode(IntEnum):
	"""
	How much context each bit sees in BitLevelMapper.

	Different context modes enable different generalization patterns:
	- CUMULATIVE: Best for carry-chain operations (increment, add)
	- FULL: Best when all bits matter (arbitrary transforms)
	- LOCAL: Best when nearby bits matter most (local patterns)
	- BIDIRECTIONAL: Best when both before/after matter (symmetric ops)
	- CAUSAL: Best for autoregressive tasks (only sees past)
	"""
	CUMULATIVE = 0    # bit i sees bits 0..i-1 (only LOWER bits for flip)
	FULL = 1          # each bit sees all bits
	LOCAL = 2         # each bit sees nearby bits (sliding window)
	BIDIRECTIONAL = 3 # bit i sees bits before AND after (symmetric window)
	CAUSAL = 4        # bit i sees bits 0..i (autoregressive, includes self)


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
	"""
	Generalization strategy for MapperFactory.

	Strategies differ in how they reduce the pattern space:
	- DIRECT: No reduction (2^n patterns for n-bit input)
	- BIT_LEVEL: Per-bit learning reduces patterns exponentially
	- COMPOSITIONAL: Group-based reduces to k * 2^(n/k) patterns
	- HYBRID: Combines compositional + bit-level
	- HASH: Hash input to smaller lookup (loses precision but generalizes)
	- RESIDUAL: Learn corrections to identity (good for small changes)
	"""
	DIRECT = 0        # Standard RAMLayer (no generalization)
	BIT_LEVEL = 1     # Use BitLevelMapper
	COMPOSITIONAL = 2 # Use CompositionalMapper
	HYBRID = 3        # Combine compositional + bit-level
	HASH = 4          # Hash input to reduce lookup space
	RESIDUAL = 5      # Identity + learned correction
