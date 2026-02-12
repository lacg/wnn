"""
Learned Binary Representations for RAM-based Language Models.

Traditional embeddings (word2vec, GloVe) produce continuous vectors
that require neural networks to use effectively. This module learns
BINARY codes that are directly compatible with RAM lookup.

Key insight: RAM networks learn from co-occurrence statistics.
Embeddings fundamentally capture co-occurrence. Therefore, RAM
can learn embeddings!

The approach:
    Traditional: word → neural network → continuous vector → prediction
    RAM-native:  word → co-occurrence stats → binary code → RAM lookup

Available encoders:
- MutualInfoEncoder: Learn bits maximizing MI(bit, target | context)
- RAMBinaryEncoder: Use RAM to learn context → code mappings
- CooccurrenceCodes: SVD on co-occurrence matrix, binarized (baseline)

Usage:
    from wnn.representations import RAMBinaryEncoder, RepresentationType

    # Create encoder
    encoder = RAMBinaryEncoder(n_bits=12)
    encoder.train(tokens)

    # Encode words to binary codes
    code = encoder.encode("cat")  # Returns int (binary code)
    codes = encoder.encode_sequence(["the", "cat", "sat"])  # List of ints

    # Similar words should have similar codes (low Hamming distance)
    dist = encoder.hamming_distance("cat", "dog")  # Should be small
"""

from enum import IntEnum
from typing import Type

from wnn.representations.base import BinaryEncoder
from wnn.representations.mutual_info import MutualInfoEncoder
from wnn.representations.ram_encoder import RAMBinaryEncoder
from wnn.representations.cooccurrence import CooccurrenceCodes
from wnn.representations.token_bit_encoder import (
    TokenBitEncoder,
    BinaryTokenEncoder,
    GrayCodeTokenEncoder,
    TokenBitEncoderType,
    create_token_bit_encoder,
)


class RepresentationType(IntEnum):
    """Binary representation learning strategies."""
    HASH = 0            # Simple hash-based encoding (fast, no learning, no generalization)
    RAM_LEARNED = 1     # RAM-based context → code learning (fast, discrete)
    MUTUAL_INFO = 2     # Iterative bit selection maximizing MI (medium speed)
    COOCCURRENCE = 3    # SVD on co-occurrence matrix, binarized (SLOW, weighted NN baseline)


class RepresentationFactory:
    """Factory for creating binary encoders."""

    _TYPE_TO_CLASS: dict[RepresentationType, Type[BinaryEncoder]] = {
        # HASH doesn't need an encoder - uses simple hash function
        RepresentationType.RAM_LEARNED: RAMBinaryEncoder,
        RepresentationType.MUTUAL_INFO: MutualInfoEncoder,
        RepresentationType.COOCCURRENCE: CooccurrenceCodes,
    }

    @classmethod
    def create(cls, rep_type: RepresentationType, **kwargs) -> BinaryEncoder | None:
        """
        Create a binary encoder.

        Args:
            rep_type: Type of representation learning
            **kwargs: Encoder-specific parameters

        Returns:
            BinaryEncoder instance, or None for HASH type (no encoder needed)
        """
        # HASH type uses simple hash function - no encoder needed
        if rep_type == RepresentationType.HASH:
            return None

        encoder_class = cls._TYPE_TO_CLASS.get(rep_type)
        if encoder_class is None:
            raise ValueError(f"Unknown representation type: {rep_type}")

        return encoder_class(**kwargs)


def create_encoder(
    rep_type: RepresentationType = RepresentationType.RAM_LEARNED,
    **kwargs,
) -> BinaryEncoder | None:
    """Convenience function to create a binary encoder."""
    return RepresentationFactory.create(rep_type, **kwargs)


__all__ = [
    # Base
    "BinaryEncoder",
    # Implementations
    "MutualInfoEncoder",
    "RAMBinaryEncoder",
    "CooccurrenceCodes",
    # Factory
    "RepresentationFactory",
    "RepresentationType",
    "create_encoder",
    # Token Bit Encoders (Layer 2: token ID → bit vector)
    "TokenBitEncoder",
    "BinaryTokenEncoder",
    "GrayCodeTokenEncoder",
    "TokenBitEncoderType",
    "create_token_bit_encoder",
]
