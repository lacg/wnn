"""
RAM-based Aggregator for combining multiple attended values.

In standard transformers, aggregation is a weighted sum:
    output = Σ attention[i,j] * value[j]

With discrete/hard attention, we need a different approach.
This module learns how to combine multiple binary vectors
using vote counting (order-invariant) and a learned RAM.

Architecture:
    attended_values: [v0, v1, v2, ...]  (variable count, each K bits)
           │
           ▼
    ┌─────────────────────────────────┐
    │  Vote Counting (per bit)        │
    │  count[b] = Σ v[b] for all v    │
    │  (order-invariant summary)      │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  Encode counts as bits          │
    │  (each count needs log2 bits)   │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  RAMLayer (learned)             │
    │  Input: [count_bits...]         │
    │  Output: [output_bits]          │
    │  Learns: vote thresholds        │
    └─────────────────────────────────┘
           │
           ▼
    output: (K bits)
"""

from wnn.ram.core.RAMLayer import RAMLayer

from torch import Tensor, zeros, uint8, cat
from torch.nn import Module


class RAMAggregator(Module):
	"""
	Learnable order-invariant aggregation for hard attention.

	Instead of weighted sums (impossible with binary attention),
	this counts votes per bit position and learns aggregation rules.

	Example learned rules:
	  - Majority vote: output bit b = 1 if count[b] > N/2
	  - Any vote: output bit b = 1 if count[b] > 0
	  - Threshold: output bit b = 1 if count[b] >= k
	  - Complex: different thresholds for different bits
	"""

	def __init__(
		self,
		value_bits: int,
		max_attended: int,
		rng: int | None = None,
	):
		"""
		Args:
			value_bits: Number of bits per value vector
			max_attended: Maximum number of values that can be attended to
			rng: Random seed for RAM initialization
		"""
		super().__init__()

		self.value_bits = value_bits
		self.max_attended = max_attended

		# Bits needed to encode vote counts (0 to max_attended)
		# e.g., max_attended=7 needs 3 bits (0-7), max_attended=8 needs 4 bits (0-8)
		self.count_bits_per_position = (max_attended + 1).bit_length()

		# Total input bits: count for each bit position
		total_count_bits = value_bits * self.count_bits_per_position

		# RAM learns: vote_counts → output_bits
		self.aggregation_ram = RAMLayer(
			total_input_bits=total_count_bits,
			num_neurons=value_bits,
			n_bits_per_neuron=min(total_count_bits, 14),
			rng=rng,
		)

		# Also include number of attended values as context
		# This helps learn "if only 1 value, pass through; if many, use majority"
		self.n_attended_bits = (max_attended + 1).bit_length()

		# Combined RAM: [counts..., n_attended] → output
		self.aggregation_ram = RAMLayer(
			total_input_bits=total_count_bits + self.n_attended_bits,
			num_neurons=value_bits,
			n_bits_per_neuron=min(total_count_bits + self.n_attended_bits, 14),
			rng=rng,
		)

	def _encode_count(self, count: int, n_bits: int) -> Tensor:
		"""Encode an integer count as a binary tensor."""
		bits = zeros(n_bits, dtype=uint8)
		for i in range(n_bits - 1, -1, -1):
			bits[i] = count & 1
			count >>= 1
		return bits

	def _compute_vote_counts(self, attended_values: list[Tensor]) -> Tensor:
		"""
		Compute per-bit vote counts across all attended values.

		Args:
			attended_values: List of [value_bits] tensors

		Returns:
			counts: Tensor of shape [value_bits] with integer counts
		"""
		if len(attended_values) == 0:
			return zeros(self.value_bits, dtype=uint8)

		# Stack and sum along the value dimension
		# Each attended value is [value_bits], stack gives [N, value_bits]
		stacked = zeros(len(attended_values), self.value_bits, dtype=uint8)
		for i, v in enumerate(attended_values):
			stacked[i] = v

		# Sum gives count per bit position
		counts = stacked.sum(dim=0)  # [value_bits] with counts 0..N
		return counts

	def forward(self, attended_values: list[Tensor]) -> Tensor:
		"""
		Aggregate multiple attended values into a single output.

		Args:
			attended_values: List of [value_bits] tensors (can be empty)

		Returns:
			output: [value_bits] tensor
		"""
		n_attended = len(attended_values)

		# Edge case: no values attended
		if n_attended == 0:
			return zeros(self.value_bits, dtype=uint8)

		# Edge case: single value - could pass through or still use RAM
		# Using RAM allows learning even for single-value case

		# Compute vote counts per bit position
		counts = self._compute_vote_counts(attended_values)

		# Encode counts as bits
		count_bits_list = []
		for b in range(self.value_bits):
			count_val = min(int(counts[b].item()), self.max_attended)
			count_bits_list.append(
				self._encode_count(count_val, self.count_bits_per_position)
			)

		# Encode number of attended values
		n_attended_clamped = min(n_attended, self.max_attended)
		n_attended_enc = self._encode_count(n_attended_clamped, self.n_attended_bits)

		# Concatenate all: [count_bits..., n_attended_bits]
		all_bits = cat(count_bits_list + [n_attended_enc])

		# Learned aggregation
		output = self.aggregation_ram(all_bits.unsqueeze(0)).squeeze()

		return output

	def train_example(
		self,
		attended_values: list[Tensor],
		target_output: Tensor,
	) -> bool:
		"""
		Train the aggregator on a single example.

		Args:
			attended_values: List of attended value tensors
			target_output: Desired output tensor

		Returns:
			True if training updated the RAM, False if already correct
		"""
		n_attended = len(attended_values)

		if n_attended == 0:
			return False  # Can't train on empty case

		# Compute the input encoding
		counts = self._compute_vote_counts(attended_values)

		count_bits_list = []
		for b in range(self.value_bits):
			count_val = min(int(counts[b].item()), self.max_attended)
			count_bits_list.append(
				self._encode_count(count_val, self.count_bits_per_position)
			)

		n_attended_clamped = min(n_attended, self.max_attended)
		n_attended_enc = self._encode_count(n_attended_clamped, self.n_attended_bits)

		all_bits = cat(count_bits_list + [n_attended_enc]).unsqueeze(0)

		# Check current output
		current_output = self.aggregation_ram(all_bits)

		if (current_output.squeeze() == target_output).all():
			return False  # Already correct

		# Train
		self.aggregation_ram.commit(all_bits, target_output.unsqueeze(0))
		return True

	def __repr__(self):
		return (
			f"RAMAggregator(value_bits={self.value_bits}, "
			f"max_attended={self.max_attended}, "
			f"count_bits={self.count_bits_per_position})"
		)
