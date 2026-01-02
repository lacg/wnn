"""
Feed-Forward Network (FFN) enumerations.
"""

from enum import IntEnum


class FFNMode(IntEnum):
	"""Feed-forward network mode."""
	STANDARD = 0     # Two RAMLayer projections
	GENERALIZED = 1  # Uses GeneralizingProjection for better generalization
	GATED = 2        # Gated variant: output = gate * up_proj


class FFNType(IntEnum):
	"""
	Type of feed-forward network in RAMTransformerBlock.

	Learned types may not generalize to unseen tokens.
	Computed types achieve 100% generalization with no training.
	"""
	# Learned FFN types (may not generalize to unseen tokens)
	NONE = 0            # No FFN (attention only)
	SINGLE = 1          # Single projection layer
	TWO_LAYER = 2       # Two-layer MLP (expand then contract)
	BIT_LEVEL = 3       # BIT_LEVEL generalization (partial)

	# Computed FFN types (100% generalization - no training needed)
	INCREMENT = 10      # Add 1 to value
	DECREMENT = 11      # Subtract 1 from value
	ADD_MOD = 12        # Add constant with modulo
	SUBTRACT_MOD = 13   # Subtract constant with modulo
	ROT13 = 14          # ROT13 cipher (add 13 mod 26)
	NEGATE = 15         # Bitwise complement (max - value)

	def is_computed(self) -> bool:
		"""Return True if this FFN type uses computed operations."""
		return self.value >= 10

	def is_learned(self) -> bool:
		"""Return True if this FFN type uses learned operations."""
		return self.value < 10 and self != FFNType.NONE


class ArithmeticOp(IntEnum):
	"""Arithmetic operations for ComputedArithmeticFFN."""
	INCREMENT = 0      # value + 1
	DECREMENT = 1      # value - 1
	ADD = 2            # value + constant
	SUBTRACT = 3       # value - constant
	ADD_MOD = 4        # (value + constant) mod N
	SUBTRACT_MOD = 5   # (value - constant) mod N
	ROT13 = 6          # (value + 13) mod 26
	NEGATE = 7         # max_value - value
