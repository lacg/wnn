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


class RepresentationType(IntEnum):
    """Binary representation learning strategies."""
    COOCCURRENCE = 0    # SVD on co-occurrence matrix, binarized
    MUTUAL_INFO = 1     # Iterative bit selection maximizing MI
    RAM_LEARNED = 2     # RAM-based context → code learning


class RepresentationFactory:
    """Factory for creating binary encoders."""

    _TYPE_TO_CLASS: dict[RepresentationType, Type[BinaryEncoder]] = {
        RepresentationType.COOCCURRENCE: CooccurrenceCodes,
        RepresentationType.MUTUAL_INFO: MutualInfoEncoder,
        RepresentationType.RAM_LEARNED: RAMBinaryEncoder,
    }

    @classmethod
    def create(cls, rep_type: RepresentationType, **kwargs) -> BinaryEncoder:
        """
        Create a binary encoder.

        Args:
            rep_type: Type of representation learning
            **kwargs: Encoder-specific parameters

        Returns:
            BinaryEncoder instance
        """
        encoder_class = cls._TYPE_TO_CLASS.get(rep_type)
        if encoder_class is None:
            raise ValueError(f"Unknown representation type: {rep_type}")

        return encoder_class(**kwargs)


def create_encoder(
    rep_type: RepresentationType = RepresentationType.RAM_LEARNED,
    **kwargs,
) -> BinaryEncoder:
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
]
