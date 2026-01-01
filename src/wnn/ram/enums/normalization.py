"""
Normalization-related enumerations for RAM networks.
"""

from enum import IntEnum


class NormStrategy(IntEnum):
    """
    Discrete normalization strategies for RAM networks.

    Since RAM networks operate on boolean values, traditional layer
    normalization isn't applicable. These strategies provide discrete
    equivalents:

    - NONE: No normalization
    - ENSEMBLE_VOTE: Multiple sub-networks with majority voting
      (provides stability through redundancy)
    - BIT_BALANCE: Learn to transform toward ~50% ones
      (keeps information content high)
    """
    NONE = 0
    ENSEMBLE_VOTE = 1  # Majority voting across sub-networks
    BIT_BALANCE = 2    # Learn toward 50% ones (max entropy)
