"""
Attention-related enumerations.
"""

from enum import IntEnum


class CrossAttentionMode(IntEnum):
	"""How to encode positions in cross-attention."""
	NONE = 0          # No position info (content-only)
	ENCODER_ONLY = 1  # Only encode key/value positions (encoder side)
	BOTH = 2          # Encode both query (decoder) and key (encoder) positions


class AttentionType(IntEnum):
    """Type of attention mechanism to use in RAMTransformerBlock."""
    SOFT_RAM = 0        # Standard SoftRAMAttention (configurable, learned)
    SORTING = 1         # SortingAttention (computed, 100% generalization)
    MIN_MAX = 2         # MinMaxAttention (computed, 100% generalization)
    POSITION_ONLY = 3   # Position-only attention (100% generalization)
    CONTENT_MATCH = 4   # Content-matching attention (XOR_EQUAL, etc.)


class ContentMatchMode(IntEnum):
    """Content-based attention matching modes (computed operations)."""
    NONE = 0           # No content matching (use learned attention)
    XOR_EQUAL = 1      # Attend if query == key (XOR is all zeros)
    HAMMING_1 = 2      # Attend if Hamming distance <= 1
    HAMMING_2 = 3      # Attend if Hamming distance <= 2
    LESS_THAN = 4      # Attend if key < query (for sorting)
    LESS_EQUAL = 5     # Attend if key <= query
    GREATER_THAN = 6   # Attend if key > query
    GREATER_EQUAL = 7  # Attend if key >= query


class AttentionCombineMode(IntEnum):
    """How to combine content matching with position patterns."""
    CONTENT_ONLY = 0    # Only use content matching (ignore position)
    POSITION_ONLY = 1   # Only use position patterns (ignore content)
    CONTENT_AND_POS = 2 # Attend if BOTH content AND position match
    CONTENT_OR_POS = 3  # Attend if EITHER content OR position match
    CONTENT_BIASED = 4  # Content match with position bias


class AggregationStrategy(IntEnum):
    """How to aggregate attention votes into final weights."""
    TOP_1 = 0       # Winner-take-all (best for retrieval)
    MAJORITY = 1    # Per-bit weighted voting (best for combining)
    THRESHOLD = 2   # Include only positions above 50% votes
    TOP_K = 3       # XOR top K highest-voted values
