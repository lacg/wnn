"""
Token Clustering — Balanced token-to-group assignment for two-stage prediction.

Splits vocabulary into K balanced groups using frequency-interleaved round-robin:
  token rank i → group (i % K)

GPT-2 token IDs are roughly frequency-ordered, so round-robin on raw IDs
gives each group similar total training data density.

Each group gets ≈ vocab_size/K tokens. Stage 1 predicts the group,
Stage 2 predicts the token within the group.
"""

from dataclasses import dataclass
from math import ceil, log2


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
