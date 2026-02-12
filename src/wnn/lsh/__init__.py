"""
Locality-Sensitive Hashing (LSH) module for RAM networks.

Enables similar contexts to map to the same (or nearby) RAM addresses,
allowing generalization between semantically related patterns.

Key insight: With standard hashing, "cat ate fish" and "dog ate fish"
are completely unrelated addresses. With LSH, similar contexts hash
to nearby addresses → transfer learning.

Available strategies:
- RandomProjectionHasher: Learn embeddings from co-occurrence, project to bits
- SimHasher: Simple +1/-1 vectors, weighted sum, binarize
- PretrainedEmbeddingHasher: Use word2vec/GloVe embeddings

Usage:
	from wnn.lsh import RandomProjectionHasher

	hasher = RandomProjectionHasher(n_bits=12)
	hasher.train(tokens)

	addr = hasher.hash_context(["the", "cat", "ate"])
"""

from enum import IntEnum
from typing import Type

from wnn.lsh.base import ContextHasher, EmbeddingHasher
from wnn.lsh.random_projection import (
	RandomProjectionHasher,
	SimHasher,
	PretrainedEmbeddingHasher,
)


class LSHType(IntEnum):
	"""LSH strategy types."""
	RANDOM_PROJECTION = 0  # Learned embeddings + random projection
	SIMHASH = 1            # Simple +1/-1 vectors
	PRETRAINED = 2         # Pre-trained embeddings (word2vec, GloVe)


class LSHFactory:
	"""Factory for creating LSH context hashers."""

	_TYPE_TO_CLASS: dict[LSHType, Type[ContextHasher]] = {
		LSHType.RANDOM_PROJECTION: RandomProjectionHasher,
		LSHType.SIMHASH: SimHasher,
		LSHType.PRETRAINED: PretrainedEmbeddingHasher,
	}

	@classmethod
	def create(cls, lsh_type: LSHType, **kwargs) -> ContextHasher:
		"""
		Create an LSH context hasher.

		Args:
			lsh_type: Type of LSH strategy
			**kwargs: Strategy-specific parameters

		Returns:
			ContextHasher instance
		"""
		hasher_class = cls._TYPE_TO_CLASS.get(lsh_type)
		if hasher_class is None:
			raise ValueError(f"Unknown LSH type: {lsh_type}")

		return hasher_class(**kwargs)


def create_lsh(
	lsh_type: LSHType = LSHType.RANDOM_PROJECTION,
	**kwargs,
) -> ContextHasher:
	"""Convenience function to create an LSH hasher."""
	return LSHFactory.create(lsh_type, **kwargs)


__all__ = [
	# Base
	"ContextHasher",
	"EmbeddingHasher",
	# Implementations
	"RandomProjectionHasher",
	"SimHasher",
	"PretrainedEmbeddingHasher",
	# Factory
	"LSHFactory",
	"LSHType",
	"create_lsh",
]
