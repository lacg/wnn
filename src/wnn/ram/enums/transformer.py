"""
Transformer-related enumerations.
"""

from enum import IntEnum, auto


class Step(IntEnum):
    """
    Steps for multi-step transformer pipelines.

    Each step represents an operation that can be composed
    in a multi-block transformer.
    """
    # Position-based operations (100% generalization)
    COPY = auto()       # Copy input to output (identity)
    SHIFT = auto()      # Shift right (causal)
    REVERSE = auto()    # Reverse sequence

    # Computed attention operations (100% generalization)
    SORT = auto()       # Sort by token value

    # Computed FFN operations (100% generalization)
    INCREMENT = auto()  # Add 1 to each token
    DECREMENT = auto()  # Subtract 1 from each token
    ROT13 = auto()      # Apply ROT13 cipher
    NEGATE = auto()     # Negate each token (max - value)


class RAMTransformerType(IntEnum):
    """
    Pre-configured RAM Transformer architectures.

    Each type represents a specific task with optimal configuration.
    """
    # Position-based transformers (100% generalization)
    COPY = auto()           # Copy task
    SHIFT = auto()          # Shift right
    REVERSE = auto()        # Reverse sequence

    # Computed attention transformers (100% generalization)
    SORTING = auto()        # Sort by value
    SELF_MATCHING = auto()  # Find matching tokens

    # Computed FFN transformers (100% generalization)
    INCREMENT = auto()      # Add 1 to each token
    DECREMENT = auto()      # Subtract 1 from each token
    ROT13 = auto()          # ROT13 cipher
    CAESAR = auto()         # Caesar cipher (configurable shift)
    NEGATE = auto()         # Negate each token
