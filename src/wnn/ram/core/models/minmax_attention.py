"""
Computed MinMax Attention

Attention mechanism that finds minimum or maximum token.
Uses computed comparisons for 100% generalization.

This is a COMPUTED (non-learnable) attention mechanism - no training required.
"""

from torch import Tensor, zeros, uint8, float32

from wnn.ram.core.models.attention_base import ComputedAttention
from wnn.ram.core.models.computed_arithmetic import bits_to_int


class ComputedMinMaxAttention(ComputedAttention):
	"""
	Find minimum or maximum token using computed comparisons.

	Every position outputs the min (or max) token from the sequence.
	Generalizes 100% because comparison is computed, not learned.
	"""

	def __init__(
		self,
		input_bits: int,
		find_max: bool = False,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per token
			find_max: If True, find maximum instead of minimum
			rng: Random seed (unused, for API compatibility)
		"""
		super().__init__()

		self.input_bits = input_bits
		self.find_max = find_max

		print(f"[ComputedMinMaxAttention] input={input_bits}b, "
				f"mode={'max' if find_max else 'min'}")

	def get_attention_weights(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> Tensor:
		"""
		Get attention weights for min/max finding (implements AttentionBase interface).

		Returns a [num_queries, num_keys] tensor where every row has 1.0 at
		the position(s) of the min/max token.
		"""
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		n = len(tokens)

		# Find min or max value
		values = [bits_to_int(t) for t in tokens]
		if self.find_max:
			target_val = max(values)
		else:
			target_val = min(values)

		# Build attention weights (same for every query position)
		row_weights = zeros(n, dtype=float32)
		for j, val in enumerate(values):
			if val == target_val:
				row_weights[j] = 1.0

		# Replicate for all query positions
		weights = row_weights.unsqueeze(0).expand(n, n).clone()
		return weights

	def _get_weights_1d(self, tokens: list[Tensor]) -> list[float]:
		"""Get attention weights as a 1D list (internal helper)."""
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		values = [bits_to_int(t) for t in tokens]
		if self.find_max:
			target_val = max(values)
		else:
			target_val = min(values)

		weights = []
		for val in values:
			if val == target_val:
				weights.append(1.0)
			else:
				weights.append(0.0)
		return weights

	def forward(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> list[Tensor]:
		"""
		Output min (or max) at every position (implements AttentionBase interface).

		context parameter is ignored (min/max is always self-attention).
		"""
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		seq_len = len(tokens)

		weights = self._get_weights_1d(tokens)

		# Find the min/max token
		min_max_token = None
		for j, w in enumerate(weights):
			if w > 0:
				min_max_token = tokens[j].clone()
				break

		if min_max_token is None:
			min_max_token = zeros(self.input_bits, dtype=uint8)

		# Output same token at every position
		return [min_max_token.clone() for _ in range(seq_len)]


# Backwards-compatible alias
MinMaxAttention = ComputedMinMaxAttention
