"""
Base Attention Interface

Abstract base class defining the common interface for all RAM attention mechanisms.
This enables consistent usage across different attention implementations:
- RAMAttention (learned, hard attention)
- SoftRAMAttention (learned, soft/voting attention)
- ComputedSortingAttention (computed, sorting)
- ComputedMinMaxAttention (computed, min/max selection)
"""

from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Module


class AttentionBase(Module, ABC):
	"""
	Abstract base class for all attention mechanisms.

	All attention implementations should inherit from this class
	and implement the required methods.

	Attention types:
	- Self-attention: context=None, tokens attend to themselves
	- Cross-attention: context=encoder_output, queries attend to context
	"""

	@property
	@abstractmethod
	def is_learnable(self) -> bool:
		"""Whether this attention mechanism has learnable parameters."""
		...

	@property
	def is_computed(self) -> bool:
		"""Whether this attention mechanism is purely computed (no learning)."""
		return not self.is_learnable

	@abstractmethod
	def forward(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> list[Tensor]:
		"""
		Forward pass through attention.

		Args:
			tokens: Query tokens, list of [bits] tensors
			context: Key/value tokens for cross-attention, None for self-attention

		Returns:
			Transformed tokens, list of [bits] tensors (same length as tokens)
		"""
		...

	@abstractmethod
	def get_attention_weights(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> Tensor:
		"""
		Compute attention weights without applying them.

		Args:
			tokens: Query tokens
			context: Key/value tokens for cross-attention

		Returns:
			Attention weights [num_queries, num_keys]
			- For self-attention: [len(tokens), len(tokens)]
			- For cross-attention: [len(tokens), len(context)]
		"""
		...

	def visualize_attention(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
		query_labels: list[str] | None = None,
		key_labels: list[str] | None = None,
	) -> str:
		"""
		Generate ASCII visualization of attention pattern.

		Args:
			tokens: Query tokens
			context: Key/value tokens for cross-attention
			query_labels: Optional labels for query positions
			key_labels: Optional labels for key positions

		Returns:
			ASCII art string showing attention pattern
		"""
		weights = self.get_attention_weights(tokens, context)
		num_queries, num_keys = weights.shape

		# Default labels
		if query_labels is None:
			query_labels = [str(i) for i in range(num_queries)]
		if key_labels is None:
			key_labels = [str(i) for i in range(num_keys)]

		# Build visualization
		lines = []
		max_label = max(len(l) for l in query_labels)

		# Header
		header = " " * (max_label + 1) + " ".join(f"{l:>3}" for l in key_labels)
		lines.append(header)

		# Rows
		for i, (label, row) in enumerate(zip(query_labels, weights)):
			row_str = f"{label:>{max_label}} "
			for w in row:
				if w > 0.75:
					row_str += " ██ "
				elif w > 0.5:
					row_str += " ▓▓ "
				elif w > 0.25:
					row_str += " ░░ "
				elif w > 0:
					row_str += " ·· "
				else:
					row_str += "    "
			lines.append(row_str)

		return "\n".join(lines)


class LearnableAttention(AttentionBase):
	"""Base class for attention mechanisms with learnable parameters."""

	@property
	def is_learnable(self) -> bool:
		return True

	@abstractmethod
	def train_step(
		self,
		tokens: list[Tensor],
		targets: list[Tensor],
		context: list[Tensor] | None = None,
	) -> int:
		"""
		Single training step.

		Args:
			tokens: Input tokens
			targets: Target outputs
			context: Context for cross-attention

		Returns:
			Number of updates made
		"""
		...


class ComputedAttention(AttentionBase):
	"""Base class for computed (non-learnable) attention mechanisms."""

	@property
	def is_learnable(self) -> bool:
		return False

	def train_step(self, *args, **kwargs) -> int:
		"""Computed attention has no training - always returns 0."""
		return 0
