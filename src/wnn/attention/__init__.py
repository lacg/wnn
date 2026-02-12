"""
Dynamic Attention module for RAM-based language models.

Traditional n-gram models use fixed context windows where all positions
have equal importance. Dynamic attention learns which positions matter
for prediction, enabling:

1. Variable-length effective context: Skip irrelevant positions
2. Position weighting: Important positions contribute more to voting
3. Content-based selection: Weight by word informativeness

Key insight: This is attention WITHOUT neural networks!
- Position weights: Learned from prediction accuracy statistics
- Content weights: TF-IDF-like word importance scoring
- No backprop needed: Pure frequency-based learning

Available strategies:
- PositionAttention: Learn position importance from accuracy correlation
- ContentAttention: Weight by word informativeness (IDF-like)
- HybridAttention: Combine position and content attention
- SparseAttention: Select top-k positions only

Usage:
	from wnn.attention import PositionAttention, AttentionType

	# Create and train
	attention = PositionAttention(max_context=6)
	attention.train(tokens)

	# Get weights for context
	weights = attention.get_weights(["the", "cat", "sat", "on"])
	# [0.15, 0.25, 0.20, 0.40]  # Most recent position most important

	# Use with RAM: weight votes by attention
	for pos, (word, weight) in enumerate(zip(context, weights)):
		vote_score *= weight
"""

from enum import IntEnum
from typing import Type

from wnn.attention.base import (
	AttentionMechanism,
	PositionAttention,
	ContentAttention,
	HybridAttention,
)
from wnn.attention.sparse import SparseAttention


class AttentionType(IntEnum):
	"""Attention strategy types."""
	NONE = 0           # No attention (uniform weights)
	POSITION = 1       # Position-based importance
	CONTENT = 2        # Content-based (TF-IDF-like)
	HYBRID = 3         # Combined position + content
	SPARSE = 4         # Top-k positions only


class AttentionFactory:
	"""Factory for creating attention mechanisms."""

	_TYPE_TO_CLASS: dict[AttentionType, Type[AttentionMechanism]] = {
		AttentionType.POSITION: PositionAttention,
		AttentionType.CONTENT: ContentAttention,
		AttentionType.HYBRID: HybridAttention,
		AttentionType.SPARSE: SparseAttention,
	}

	@classmethod
	def create(cls, attention_type: AttentionType, **kwargs) -> AttentionMechanism:
		"""
		Create an attention mechanism.

		Args:
			attention_type: Type of attention
			**kwargs: Strategy-specific parameters

		Returns:
			AttentionMechanism instance
		"""
		if attention_type == AttentionType.NONE:
			# Return a simple uniform attention
			return PositionAttention(recency_bias=0.0, **kwargs)

		attn_class = cls._TYPE_TO_CLASS.get(attention_type)
		if attn_class is None:
			raise ValueError(f"Unknown attention type: {attention_type}")

		return attn_class(**kwargs)


def create_attention(
	attention_type: AttentionType = AttentionType.HYBRID,
	**kwargs,
) -> AttentionMechanism:
	"""Convenience function to create an attention mechanism."""
	return AttentionFactory.create(attention_type, **kwargs)


__all__ = [
	# Base
	"AttentionMechanism",
	# Implementations
	"PositionAttention",
	"ContentAttention",
	"HybridAttention",
	"SparseAttention",
	# Factory
	"AttentionFactory",
	"AttentionType",
	"create_attention",
]
