"""
Random Projection LSH for context hashing.

Uses random hyperplanes to project context embeddings to binary codes.
Similar contexts will have similar (overlapping) bits.

Key properties:
- Preserves cosine similarity: P(hash collision) ∝ similarity
- Fast: O(d × n_bits) per hash, where d = embedding dimension
- Trainable: Can learn embeddings from corpus

Usage:
	hasher = RandomProjectionHasher(n_bits=12, embedding_dim=64)
	hasher.train(tokens)

	addr1 = hasher.hash_context(["the", "cat", "ate"])
	addr2 = hasher.hash_context(["the", "dog", "ate"])
	# addr1 and addr2 will share some bits due to similarity
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from wnn.lsh.base import EmbeddingHasher


class RandomProjectionHasher(EmbeddingHasher):
	"""
	LSH using random projections of learned embeddings.

	Training learns word embeddings from co-occurrence statistics,
	then initializes random hyperplanes for projection.
	"""

	def __init__(
		self,
		n_bits: int = 12,
		embedding_dim: int = 64,
		window_size: int = 5,
		min_count: int = 5,
		name: str = "random_projection",
	):
		"""
		Initialize random projection hasher.

		Args:
			n_bits: Number of output bits
			embedding_dim: Dimension of word embeddings
			window_size: Context window for co-occurrence
			min_count: Minimum word frequency to include
			name: Human-readable name
		"""
		super().__init__(n_bits=n_bits, embedding_dim=embedding_dim, name=name)
		self._window_size = window_size
		self._min_count = min_count

	def train(self, tokens: list[str], seed: int = 42, **kwargs) -> None:
		"""
		Train embeddings from corpus and initialize projections.

		Uses a simple but effective approach:
		1. Build co-occurrence matrix (PPMI-weighted)
		2. Apply SVD for dimensionality reduction
		3. Initialize random projection hyperplanes

		Args:
			tokens: Training corpus tokens
			seed: Random seed for reproducibility
		"""
		rng = np.random.default_rng(seed)

		# Build vocabulary
		word_counts = Counter(tokens)
		vocab = [w for w, c in word_counts.items() if c >= self._min_count]
		self._word_to_idx = {w: i for i, w in enumerate(vocab)}
		vocab_size = len(vocab)

		if vocab_size == 0:
			raise ValueError(f"No words with count >= {self._min_count}")

		# Build co-occurrence matrix
		cooc = defaultdict(Counter)
		for i, word in enumerate(tokens):
			if word not in self._word_to_idx:
				continue

			# Count co-occurrences in window
			start = max(0, i - self._window_size)
			end = min(len(tokens), i + self._window_size + 1)

			for j in range(start, end):
				if i != j and tokens[j] in self._word_to_idx:
					cooc[word][tokens[j]] += 1

		# Convert to dense matrix with PPMI weighting
		# PPMI = max(0, log(P(w,c) / P(w)P(c)))
		total_pairs = sum(sum(c.values()) for c in cooc.values())

		# Word probabilities
		word_probs = np.array([word_counts[w] for w in vocab], dtype=np.float32)
		word_probs /= word_probs.sum()

		# Build PPMI matrix (sparse → dense for SVD)
		ppmi_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

		for word, contexts in cooc.items():
			i = self._word_to_idx[word]
			for ctx, count in contexts.items():
				j = self._word_to_idx[ctx]

				# P(w, c) = count / total
				p_wc = count / total_pairs

				# PPMI
				pmi = np.log(p_wc / (word_probs[i] * word_probs[j]) + 1e-10)
				ppmi_matrix[i, j] = max(0, pmi)

		# SVD for dimensionality reduction
		# Use truncated SVD for efficiency
		actual_dim = min(self._embedding_dim, vocab_size - 1)

		try:
			from scipy.sparse.linalg import svds
			# scipy svds is efficient for sparse-ish matrices
			u, s, _ = svds(ppmi_matrix, k=actual_dim)
			# Weight by sqrt of singular values (common practice)
			self._embeddings = (u * np.sqrt(s)).astype(np.float32)
		except ImportError:
			# Fallback to full SVD
			u, s, _ = np.linalg.svd(ppmi_matrix, full_matrices=False)
			self._embeddings = (u[:, :actual_dim] * np.sqrt(s[:actual_dim])).astype(np.float32)

		# Pad if embedding_dim > actual_dim
		if actual_dim < self._embedding_dim:
			padding = np.zeros((vocab_size, self._embedding_dim - actual_dim), dtype=np.float32)
			self._embeddings = np.hstack([self._embeddings, padding])

		# Initialize random projection hyperplanes
		# Each row is a hyperplane normal vector
		self._projection = rng.standard_normal((self._n_bits, self._embedding_dim)).astype(np.float32)
		# Normalize for numerical stability
		norms = np.linalg.norm(self._projection, axis=1, keepdims=True)
		self._projection /= np.maximum(norms, 1e-8)

		self._is_trained = True

	def get_stats(self) -> dict:
		"""Get training statistics."""
		return {
			"vocab_size": len(self._word_to_idx),
			"embedding_dim": self._embedding_dim,
			"n_bits": self._n_bits,
			"window_size": self._window_size,
			"min_count": self._min_count,
		}


class SimHasher(EmbeddingHasher):
	"""
	SimHash for context hashing.

	SimHash is a specific LSH technique that:
	1. Maps each word to a random bit vector
	2. Weights vectors by TF-IDF or position
	3. Sums weighted vectors
	4. Binarizes by sign

	Simpler than random projection but effective for short contexts.
	"""

	def __init__(
		self,
		n_bits: int = 12,
		name: str = "simhash",
	):
		"""
		Initialize SimHash hasher.

		Args:
			n_bits: Number of output bits
			name: Human-readable name
		"""
		# SimHash uses n_bits as embedding dim (each word → n_bits vector)
		super().__init__(n_bits=n_bits, embedding_dim=n_bits, name=name)

		# Word to random vector mapping
		self._word_vectors: dict[str, np.ndarray] = {}

	def train(self, tokens: list[str], seed: int = 42, **kwargs) -> None:
		"""
		Train SimHash (just builds vocabulary and random vectors).

		Args:
			tokens: Training corpus tokens
			seed: Random seed for reproducibility
		"""
		rng = np.random.default_rng(seed)

		# Build vocabulary with random vectors
		vocab = set(tokens)
		self._word_to_idx = {w: i for i, w in enumerate(vocab)}

		# Each word gets a random vector of +1/-1
		for word in vocab:
			self._word_vectors[word] = rng.choice([-1, 1], size=self._n_bits).astype(np.float32)

		# For SimHash, we don't need a projection matrix
		# The word vectors ARE the hash functions
		self._embeddings = np.vstack([
			self._word_vectors[w] for w in sorted(self._word_to_idx.keys(), key=lambda x: self._word_to_idx[x])
		])

		# Identity projection (just take sign of sum)
		self._projection = np.eye(self._n_bits, dtype=np.float32)

		self._is_trained = True

	def _get_embedding(self, word: str) -> np.ndarray:
		"""Get SimHash vector for a word."""
		if word in self._word_vectors:
			return self._word_vectors[word]
		else:
			# OOV: consistent random vector based on word hash
			rng = np.random.default_rng(hash(word) % (2**32))
			return rng.choice([-1, 1], size=self._n_bits).astype(np.float32)

	def _combine_embeddings(self, context: list[str]) -> np.ndarray:
		"""
		Combine word vectors with position weighting.

		SimHash: weighted sum of +1/-1 vectors.
		"""
		if not context:
			return np.zeros(self._n_bits, dtype=np.float32)

		# Position weights: more recent words matter more
		n = len(context)
		weights = np.array([1.0 + i for i in range(n)], dtype=np.float32)

		# Weighted sum
		result = np.zeros(self._n_bits, dtype=np.float32)
		for i, word in enumerate(context):
			result += weights[i] * self._get_embedding(word)

		return result


class PretrainedEmbeddingHasher(EmbeddingHasher):
	"""
	LSH using pre-trained word embeddings (word2vec, GloVe, etc.).

	For best results, use embeddings trained on large corpora.
	Falls back to random embeddings for OOV words.
	"""

	def __init__(
		self,
		n_bits: int = 12,
		embedding_dim: int = 300,  # Common for GloVe/word2vec
		embedding_path: Optional[str] = None,
		name: str = "pretrained",
	):
		"""
		Initialize pretrained embedding hasher.

		Args:
			n_bits: Number of output bits
			embedding_dim: Expected dimension of embeddings
			embedding_path: Path to embedding file (optional, can load later)
			name: Human-readable name
		"""
		super().__init__(n_bits=n_bits, embedding_dim=embedding_dim, name=name)
		self._embedding_path = embedding_path

	def train(self, tokens: list[str], embedding_path: Optional[str] = None, seed: int = 42, **kwargs) -> None:
		"""
		Load pre-trained embeddings and initialize projections.

		Args:
			tokens: Training corpus (used to subset embeddings)
			embedding_path: Path to embedding file (word2vec/GloVe format)
			seed: Random seed for projection matrix
		"""
		path = embedding_path or self._embedding_path

		if path is None:
			raise ValueError("Must provide embedding_path either in __init__ or train()")

		rng = np.random.default_rng(seed)

		# Build vocabulary from corpus
		vocab_in_corpus = set(tokens)

		# Load embeddings
		self._word_to_idx = {}
		embeddings_list = []

		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < self._embedding_dim + 1:
					continue

				word = parts[0]
				if word in vocab_in_corpus:
					vec = np.array([float(x) for x in parts[1:self._embedding_dim + 1]], dtype=np.float32)
					if len(vec) == self._embedding_dim:
						self._word_to_idx[word] = len(embeddings_list)
						embeddings_list.append(vec)

		if not embeddings_list:
			raise ValueError(f"No embeddings loaded from {path}")

		self._embeddings = np.vstack(embeddings_list)

		# Initialize random projections
		self._projection = rng.standard_normal((self._n_bits, self._embedding_dim)).astype(np.float32)
		norms = np.linalg.norm(self._projection, axis=1, keepdims=True)
		self._projection /= np.maximum(norms, 1e-8)

		self._is_trained = True
