"""
Token Clustering — Balanced token-to-group assignment for two-stage prediction.

Splits vocabulary into K balanced groups using:
  - Round-robin: token_id % K (frequency-interleaved for GPT-2)
  - Semantic: K-means on GPT-2 wte embeddings (similar tokens share groups)
  - Semantic Bitwise: Hierarchical PCA bisection (bit positions have semantic meaning)

Each group gets ≈ vocab_size/K tokens. Stage 1 predicts the group,
Stage 2 predicts the token within the group.

Clustering strategies (GoF Strategy pattern):
  - BalancedStrategy      → round-robin, rust_stage_type varies
  - SemanticStrategy      → K-means, rust_stage_type = "tiered"
  - SemanticBitwiseStrategy → PCA bisection, rust_stage_type = "bitwise"
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _bits_needed(n: int) -> int:
	"""Bits needed to represent n distinct values (0..n-1)."""
	if n <= 1:
		return 1
	return (n - 1).bit_length()


@dataclass(frozen=True)
class TokenClustering:
	"""Precomputed balanced token-to-group assignment.

	Fields:
		k: Number of groups
		vocab_size: Total vocabulary size (e.g. 50257)
		cluster_of: [vocab_size] → group_id (0..K-1)
		index_in_cluster: [vocab_size] → within-group index (0..cluster_size-1)
		cluster_tokens: [K] lists of token_ids per group
		max_cluster_size: Maximum tokens in any group
		bits_per_cluster_id: ceil(log2(K))
		bits_per_within_index: ceil(log2(max_cluster_size))
	"""
	k: int
	vocab_size: int
	cluster_of: list[int]
	index_in_cluster: list[int]
	cluster_tokens: list[list[int]]
	max_cluster_size: int
	bits_per_cluster_id: int
	bits_per_within_index: int

	@classmethod
	def create_balanced(cls, vocab_size: int, k: int) -> 'TokenClustering':
		"""Create balanced clustering via frequency-interleaved round-robin.

		GPT-2 token IDs are roughly frequency-ordered (lower = more frequent).
		Round-robin assignment (token_id % K) distributes frequent tokens
		evenly across groups, balancing training data density.

		Args:
			vocab_size: Total vocabulary (e.g. 50257)
			k: Number of groups (e.g. 256 for bitwise, 225 for tiered)

		Returns:
			TokenClustering with all fields computed.
		"""
		if k < 2:
			raise ValueError(f"k must be >= 2, got {k}")
		if k > vocab_size:
			raise ValueError(f"k ({k}) > vocab_size ({vocab_size})")

		cluster_of = [0] * vocab_size
		index_in_cluster = [0] * vocab_size
		cluster_tokens: list[list[int]] = [[] for _ in range(k)]

		for token_id in range(vocab_size):
			group = token_id % k
			cluster_of[token_id] = group
			index_in_cluster[token_id] = len(cluster_tokens[group])
			cluster_tokens[group].append(token_id)

		max_cluster_size = max(len(g) for g in cluster_tokens)
		bits_per_cluster_id = _bits_needed(k)
		bits_per_within_index = _bits_needed(max_cluster_size)

		return cls(
			k=k,
			vocab_size=vocab_size,
			cluster_of=cluster_of,
			index_in_cluster=index_in_cluster,
			cluster_tokens=cluster_tokens,
			max_cluster_size=max_cluster_size,
			bits_per_cluster_id=bits_per_cluster_id,
			bits_per_within_index=bits_per_within_index,
		)

	@classmethod
	def create_semantic(
		cls, vocab_size: int, k: int, cache_dir: Optional[str] = None
	) -> 'TokenClustering':
		"""Create clustering via K-means on GPT-2 token embeddings.

		Semantically similar tokens (e.g. "cat", "dog") share groups,
		making group prediction from context much easier for S0.

		Args:
			vocab_size: Total vocabulary (e.g. 50257)
			k: Number of groups (e.g. 225 for tiered)
			cache_dir: Directory to cache clustering result. None = no caching.

		Returns:
			TokenClustering with semantically coherent groups.
		"""
		if k < 2:
			raise ValueError(f"k must be >= 2, got {k}")
		if k > vocab_size:
			raise ValueError(f"k ({k}) > vocab_size ({vocab_size})")

		# Check cache first
		cache_path = None
		if cache_dir is not None:
			cache_path = Path(cache_dir) / f"semantic_clusters_k{k}.json"
			if cache_path.exists():
				logger.info(f"Loading cached semantic clustering from {cache_path}")
				with open(cache_path) as f:
					labels = json.load(f)
				assert len(labels) == vocab_size, f"Cached labels size {len(labels)} != vocab_size {vocab_size}"
				return cls._from_labels(vocab_size, k, labels)

		# Load GPT-2 embeddings
		import numpy as np
		from transformers import GPT2Model
		logger.info("Loading GPT-2 embeddings for semantic clustering...")
		model = GPT2Model.from_pretrained("gpt2")
		embeddings = model.wte.weight.detach().numpy()[:vocab_size]  # (vocab_size, 768)

		# K-means clustering
		from sklearn.cluster import MiniBatchKMeans
		logger.info(f"Running MiniBatchKMeans (k={k}) on {vocab_size} embeddings...")
		kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
		labels = kmeans.fit_predict(embeddings).tolist()

		# Balance enforcement: reassign tokens in over-capacity clusters
		target_size = vocab_size / k
		max_size = int(target_size * 2)
		cluster_counts = [0] * k
		for lbl in labels:
			cluster_counts[lbl] += 1

		oversized = {c for c, cnt in enumerate(cluster_counts) if cnt > max_size}
		if oversized:
			logger.info(f"Rebalancing {len(oversized)} oversized clusters (max_size={max_size})...")
			centers = kmeans.cluster_centers_  # (k, 768)
			for token_id in range(vocab_size):
				if labels[token_id] not in oversized:
					continue
				# Find nearest under-capacity cluster
				emb = embeddings[token_id]
				dists = np.linalg.norm(centers - emb, axis=1)
				for nearest in np.argsort(dists):
					nearest = int(nearest)
					if cluster_counts[nearest] < max_size:
						old = labels[token_id]
						cluster_counts[old] -= 1
						labels[token_id] = nearest
						cluster_counts[nearest] += 1
						if cluster_counts[old] <= max_size:
							oversized.discard(old)
						break

		# Cache result
		if cache_path is not None:
			cache_path.parent.mkdir(parents=True, exist_ok=True)
			with open(cache_path, 'w') as f:
				json.dump(labels, f)
			logger.info(f"Cached semantic clustering to {cache_path}")

		sizes = [cluster_counts[c] for c in range(k)]
		logger.info(
			f"Semantic clustering: k={k}, min_size={min(sizes)}, max_size={max(sizes)}, "
			f"mean_size={sum(sizes)/len(sizes):.1f}"
		)

		return cls._from_labels(vocab_size, k, labels)

	@classmethod
	def _from_labels(cls, vocab_size: int, k: int, labels: list[int]) -> 'TokenClustering':
		"""Build a TokenClustering from pre-computed cluster labels."""
		cluster_of = [0] * vocab_size
		index_in_cluster = [0] * vocab_size
		cluster_tokens: list[list[int]] = [[] for _ in range(k)]

		for token_id in range(vocab_size):
			group = labels[token_id]
			cluster_of[token_id] = group
			index_in_cluster[token_id] = len(cluster_tokens[group])
			cluster_tokens[group].append(token_id)

		max_cluster_size = max(len(g) for g in cluster_tokens)
		bits_per_cluster_id = _bits_needed(k)
		bits_per_within_index = _bits_needed(max_cluster_size)

		return cls(
			k=k,
			vocab_size=vocab_size,
			cluster_of=cluster_of,
			index_in_cluster=index_in_cluster,
			cluster_tokens=cluster_tokens,
			max_cluster_size=max_cluster_size,
			bits_per_cluster_id=bits_per_cluster_id,
			bits_per_within_index=bits_per_within_index,
		)

	def encode_cluster_id_bits(self, token_id: int) -> list[int]:
		"""Encode the cluster_id for a token as a list of bits (LSB first)."""
		group = self.cluster_of[token_id]
		return self._int_to_bits(group, self.bits_per_cluster_id)

	def encode_within_index_bits(self, token_id: int) -> list[int]:
		"""Encode the within-group index for a token as a list of bits (LSB first)."""
		idx = self.index_in_cluster[token_id]
		return self._int_to_bits(idx, self.bits_per_within_index)

	@staticmethod
	def _int_to_bits(value: int, num_bits: int) -> list[int]:
		"""Convert integer to list of bits (LSB first)."""
		return [(value >> b) & 1 for b in range(num_bits)]

	def cluster_size(self, group_id: int) -> int:
		"""Number of tokens in the given group."""
		return len(self.cluster_tokens[group_id])


# ── Clustering Strategy (GoF Strategy pattern) ──────────────────────

class ClusteringStrategy(ABC):
	"""Abstract base for clustering strategies.

	Each strategy knows:
	  (a) how to compute cluster_of (via create())
	  (b) what Rust eval path to use (rust_stage_type)
	  (c) cache key for deduplication
	"""

	@abstractmethod
	def create(self, vocab_size: int, k: int, cache_dir: Optional[str] = None) -> TokenClustering:
		"""Create a TokenClustering using this strategy."""

	@property
	@abstractmethod
	def rust_stage_type(self) -> str:
		"""Rust stage type: 'tiered' or 'bitwise'."""

	@property
	@abstractmethod
	def cache_key(self) -> str:
		"""Unique key identifying this strategy for caching."""


class BalancedStrategy(ClusteringStrategy):
	"""Round-robin balanced clustering. rust_stage_type is configurable."""

	def __init__(self, eval_type: str = "bitwise"):
		self._eval_type = eval_type

	def create(self, vocab_size: int, k: int, cache_dir: Optional[str] = None) -> TokenClustering:
		return TokenClustering.create_balanced(vocab_size, k)

	@property
	def rust_stage_type(self) -> str:
		return self._eval_type

	@property
	def cache_key(self) -> str:
		return f"balanced_{self._eval_type}"


class SemanticStrategy(ClusteringStrategy):
	"""K-means on GPT-2 embeddings. Always uses tiered eval."""

	def create(self, vocab_size: int, k: int, cache_dir: Optional[str] = None) -> TokenClustering:
		return TokenClustering.create_semantic(vocab_size, k, cache_dir)

	@property
	def rust_stage_type(self) -> str:
		return "tiered"

	@property
	def cache_key(self) -> str:
		return "semantic"


class SemanticBitwiseStrategy(ClusteringStrategy):
	"""Hierarchical PCA bisection — bit positions have semantic meaning.

	For K=256 (8 bits), recursively splits embedding space:
	  1. PCA-1 on all embeddings → split at median → bit 0
	  2. Within each half, PCA-1 → split at median → bit 1
	  3. Continue for 8 levels → 256 leaves

	Cluster ID = path through tree (LSB-first). Tokens with similar
	embeddings share most bits → bitwise prediction from context
	is much easier than random bit assignment.

	K must be a power of 2.
	"""

	def create(self, vocab_size: int, k: int, cache_dir: Optional[str] = None) -> TokenClustering:
		if k < 2 or (k & (k - 1)) != 0:
			raise ValueError(f"SemanticBitwiseStrategy requires K to be a power of 2, got {k}")
		if k > vocab_size:
			raise ValueError(f"k ({k}) > vocab_size ({vocab_size})")

		num_bits = _bits_needed(k)

		# Check cache
		cache_path = None
		if cache_dir is not None:
			cache_path = Path(cache_dir) / f"semantic_bitwise_clusters_k{k}.json"
			if cache_path.exists():
				logger.info(f"Loading cached semantic-bitwise clustering from {cache_path}")
				with open(cache_path) as f:
					labels = json.load(f)
				assert len(labels) == vocab_size, f"Cached labels size {len(labels)} != {vocab_size}"
				return TokenClustering._from_labels(vocab_size, k, labels)

		# Load GPT-2 embeddings
		import numpy as np
		from transformers import GPT2Model
		logger.info("Loading GPT-2 embeddings for semantic-bitwise clustering...")
		model = GPT2Model.from_pretrained("gpt2")
		embeddings = model.wte.weight.detach().numpy()[:vocab_size]  # (vocab_size, 768)

		# Hierarchical PCA bisection
		labels = [0] * vocab_size
		# groups: list of (bit_prefix, token_indices)
		groups = [(0, list(range(vocab_size)))]

		for bit_level in range(num_bits):
			next_groups = []
			for prefix, indices in groups:
				if len(indices) < 3:
					# Too few tokens for PCA — split by token ID
					mid = len(indices) // 2
					sorted_idx = sorted(indices)
					for tok in sorted_idx[:mid]:
						labels[tok] = prefix  # bit=0
					for tok in sorted_idx[mid:]:
						labels[tok] = prefix | (1 << bit_level)  # bit=1
					next_groups.append((prefix, sorted_idx[:mid]))
					next_groups.append((prefix | (1 << bit_level), sorted_idx[mid:]))
					continue

				# PCA-1: find direction of maximum variance
				embs = embeddings[indices]  # (n, 768)
				mean = embs.mean(axis=0)
				centered = embs - mean

				# Power iteration for dominant eigenvector (faster than full SVD)
				v = np.random.RandomState(42 + bit_level).randn(centered.shape[1])
				for _ in range(20):
					v = centered.T @ (centered @ v)
					norm = np.linalg.norm(v)
					if norm > 0:
						v /= norm

				# Project onto principal component and split at median
				projections = centered @ v
				median = np.median(projections)

				low, high = [], []
				for j, tok in enumerate(indices):
					if projections[j] <= median:
						labels[tok] = prefix  # bit=0
						low.append(tok)
					else:
						labels[tok] = prefix | (1 << bit_level)  # bit=1
						high.append(tok)

				next_groups.append((prefix, low))
				next_groups.append((prefix | (1 << bit_level), high))

			groups = next_groups
			logger.info(f"  PCA bisection level {bit_level}: {len(groups)} groups")

		# Cache result
		if cache_path is not None:
			cache_path.parent.mkdir(parents=True, exist_ok=True)
			with open(cache_path, 'w') as f:
				json.dump(labels, f)
			logger.info(f"Cached semantic-bitwise clustering to {cache_path}")

		# Log stats
		cluster_counts = [0] * k
		for lbl in labels:
			cluster_counts[lbl] += 1
		sizes = [c for c in cluster_counts if c > 0]
		logger.info(
			f"Semantic-bitwise clustering: k={k}, num_bits={num_bits}, "
			f"groups_with_tokens={len(sizes)}, "
			f"min_size={min(sizes)}, max_size={max(sizes)}, "
			f"mean_size={sum(sizes)/len(sizes):.1f}"
		)

		return TokenClustering._from_labels(vocab_size, k, labels)

	@property
	def rust_stage_type(self) -> str:
		return "bitwise"

	@property
	def cache_key(self) -> str:
		return "semantic_bitwise"


def get_clustering_strategy(stage_type: str) -> ClusteringStrategy:
	"""Factory: stage_type → ClusteringStrategy.

	Args:
		stage_type: One of "bitwise", "tiered", "semantic", "semantic_bitwise"

	Returns:
		The appropriate ClusteringStrategy instance.
	"""
	strategies = {
		"bitwise": lambda: BalancedStrategy("bitwise"),
		"tiered": lambda: BalancedStrategy("tiered"),
		"semantic": lambda: SemanticStrategy(),
		"semantic_bitwise": lambda: SemanticBitwiseStrategy(),
	}
	factory = strategies.get(stage_type)
	if factory is None:
		raise ValueError(f"Unknown clustering strategy: {stage_type!r}. "
						 f"Valid: {list(strategies.keys())}")
	return factory()
