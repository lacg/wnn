"""
Base classes for Locality-Sensitive Hashing (LSH) in RAM networks.

LSH enables similar contexts to map to the same (or nearby) RAM addresses,
allowing generalization between semantically related patterns.

Key insight: Currently "the cat ate" and "the dog ate" are completely unrelated.
With LSH, similar contexts hash to nearby addresses → transfer learning.

Approaches:
1. SimHash: Hash each word, combine with XOR weighted by position
2. Random Projection: Project context embeddings onto random hyperplanes
3. MinHash: Jaccard similarity-preserving hash for sets

For RAM networks, we convert contexts to binary addresses using LSH.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class ContextHasher(ABC):
	"""
	Abstract base class for context hashing strategies.

	Converts a sequence of tokens into a fixed-size binary hash
	that preserves similarity: similar contexts → similar hashes.
	"""

	def __init__(self, n_bits: int = 12, name: str = "base"):
		"""
		Initialize context hasher.

		Args:
			n_bits: Number of output bits (RAM address size)
			name: Human-readable name
		"""
		self._n_bits = n_bits
		self._name = name
		self._is_trained = False

	@property
	def n_bits(self) -> int:
		return self._n_bits

	@property
	def name(self) -> str:
		return self._name

	@property
	def is_trained(self) -> bool:
		return self._is_trained

	@abstractmethod
	def train(self, tokens: list[str], **kwargs) -> None:
		"""
		Train the hasher on a corpus (e.g., learn word embeddings).

		Args:
			tokens: List of tokens from training corpus
			**kwargs: Strategy-specific parameters
		"""
		...

	@abstractmethod
	def hash_context(self, context: list[str]) -> int:
		"""
		Hash a context to a binary address.

		Args:
			context: List of tokens (e.g., ["the", "cat", "ate"])

		Returns:
			Integer address in range [0, 2^n_bits)
		"""
		...

	def hash_contexts_batch(self, contexts: list[list[str]]) -> list[int]:
		"""
		Hash multiple contexts (for efficiency).

		Default implementation calls hash_context() for each.
		Subclasses can override for vectorized operations.
		"""
		return [self.hash_context(ctx) for ctx in contexts]

	def similarity(self, ctx1: list[str], ctx2: list[str]) -> float:
		"""
		Compute hash similarity between two contexts.

		Returns fraction of matching bits (Hamming similarity).
		"""
		h1 = self.hash_context(ctx1)
		h2 = self.hash_context(ctx2)

		# XOR gives bits that differ
		xor = h1 ^ h2

		# Count differing bits
		diff_bits = bin(xor).count('1')

		# Return fraction of matching bits
		return 1.0 - (diff_bits / self._n_bits)

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(n_bits={self._n_bits}, name={self._name!r})"


class EmbeddingHasher(ContextHasher):
	"""
	Base class for hashers that use word embeddings.

	Converts tokens to embeddings, combines them, then projects
	to binary using random hyperplanes.
	"""

	def __init__(
		self,
		n_bits: int = 12,
		embedding_dim: int = 64,
		name: str = "embedding",
	):
		"""
		Initialize embedding-based hasher.

		Args:
			n_bits: Number of output bits
			embedding_dim: Dimension of word embeddings
			name: Human-readable name
		"""
		super().__init__(n_bits=n_bits, name=name)
		self._embedding_dim = embedding_dim

		# Word to embedding mapping (populated during training)
		self._word_to_idx: dict[str, int] = {}
		self._embeddings: Optional[np.ndarray] = None

		# Random projection matrix (n_bits × embedding_dim)
		# Initialized during training with consistent seed
		self._projection: Optional[np.ndarray] = None

	@property
	def embedding_dim(self) -> int:
		return self._embedding_dim

	@property
	def vocab_size(self) -> int:
		return len(self._word_to_idx)

	def _get_embedding(self, word: str) -> np.ndarray:
		"""Get embedding for a word (with fallback for OOV)."""
		if word in self._word_to_idx:
			idx = self._word_to_idx[word]
			return self._embeddings[idx]
		else:
			# OOV: use hash-based random embedding
			# This ensures consistent embedding for same OOV word
			rng = np.random.default_rng(hash(word) % (2**32))
			return rng.standard_normal(self._embedding_dim).astype(np.float32)

	def _combine_embeddings(self, context: list[str]) -> np.ndarray:
		"""
		Combine word embeddings into a single context vector.

		Default: weighted sum with position weights.
		Override for different combination strategies.
		"""
		if not context:
			return np.zeros(self._embedding_dim, dtype=np.float32)

		# Position weights: more recent words matter more
		n = len(context)
		weights = np.array([1.0 / (n - i) for i in range(n)], dtype=np.float32)
		weights /= weights.sum()

		# Weighted sum of embeddings
		result = np.zeros(self._embedding_dim, dtype=np.float32)
		for i, word in enumerate(context):
			result += weights[i] * self._get_embedding(word)

		return result

	def _project_to_bits(self, vector: np.ndarray) -> int:
		"""
		Project embedding to binary using random hyperplanes.

		Each hyperplane divides space in half.
		Point on positive side → 1, negative side → 0.
		"""
		if self._projection is None:
			raise RuntimeError("Hasher not trained - projection matrix not initialized")

		# Dot product with each hyperplane
		dots = self._projection @ vector

		# Convert to bits: positive → 1, negative → 0
		bits = (dots > 0).astype(np.int32)

		# Pack bits into integer
		result = 0
		for i, bit in enumerate(bits):
			if bit:
				result |= (1 << i)

		return result

	def hash_context(self, context: list[str]) -> int:
		"""Hash context using embedding + random projection."""
		if not self._is_trained:
			raise RuntimeError("Must call train() before hash_context()")

		# Get combined embedding
		embedding = self._combine_embeddings(context)

		# Project to bits
		return self._project_to_bits(embedding)
