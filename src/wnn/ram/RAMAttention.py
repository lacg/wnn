"""
RAM-based Attention Mechanism

Traditional Transformer Attention:
  scores = softmax(Q·K^T / √d)    # Continuous weights [0,1]
  output = scores · V              # Weighted sum of all values

RAM Attention (Hard/Discrete):
  For each query position:
    1. Compute similarity to ALL keys (using RAM neurons)
    2. Select winner(s) via hard routing or voting
    3. Return selected value(s)

Key differences:
  - No continuous weights - discrete selection
  - No weighted sums - winner-take-all or voting
  - But: content-addressable, learnable, multi-head

This is closer to:
  - Slot attention (hard assignment)
  - Sparse attention (attend to subset)
  - Memory networks (discrete addressing)
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMAggregator import RAMAggregator
from wnn.ram.encoders_decoders import OutputMode
from wnn.ram.encoders_decoders import TransformerDecoderFactory
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.encoders_decoders import PositionEncoderFactory

from torch import Tensor, zeros, ones, uint8, stack, cat, tensor
from torch.nn import Module, ModuleList
from typing import Optional


class RAMAttention(Module):
	"""
	RAM-based attention layer.

	Instead of soft attention (weighted sum), uses hard attention:
	- Each query selects which keys to attend to
	- Selection is discrete (binary: attend or not)
	- Multiple heads allow different attention patterns

	Architecture:
	  Input: sequence of tokens [t0, t1, t2, ..., tN]

	  For each query position i:
	    q_i = input[i]

	    For each key position j:
	      k_j = input[j]
	      attend[i,j] = RAM_similarity(q_i, k_j)  # Binary: 0 or 1

	    attended_values = [input[j] for j where attend[i,j] == 1]
	    output[i] = aggregate(attended_values)  # Vote or first match
	"""

	def __init__(
		self,
		input_bits: int,
		num_heads: int = 4,
		neurons_per_head: int = 8,
		position_mode: PositionMode = PositionMode.BINARY,
		max_seq_len: int = 16,
		causal: bool = True,  # Only attend to past positions
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per token
			num_heads: Number of attention heads
			neurons_per_head: RAM neurons per head
			position_mode: How to encode positions (NONE, BINARY, RELATIVE)
			max_seq_len: Maximum sequence length
			causal: If True, position i can only attend to j <= i
			rng: Random seed

		Position modes:
			NONE: No position encoding (content-only attention)
			BINARY: Absolute positions [query_pos, key_pos] - 2 * pos_bits
			RELATIVE: Relative distance [key_pos - query_pos] - 1 * pos_bits
		"""
		super().__init__()

		self.input_bits = input_bits
		self.num_heads = num_heads
		self.neurons_per_head = neurons_per_head
		self.causal = causal
		self.max_seq_len = max_seq_len
		self.position_mode = position_mode

		# Position encoding
		if position_mode == PositionMode.NONE:
			self.position_encoder = None
			self.n_position_bits = 0
			self.n_similarity_position_bits = 0
		elif position_mode == PositionMode.BINARY:
			self.position_encoder = PositionEncoderFactory.create(
				PositionMode.BINARY,
				max_seq_len=max_seq_len,
			)
			self.n_position_bits = self.position_encoder.n_bits
			# Binary mode uses both query and key positions
			self.n_similarity_position_bits = 2 * self.n_position_bits
		elif position_mode == PositionMode.RELATIVE:
			# For relative, max_distance is max_seq_len - 1 (causal: only look back)
			self.position_encoder = PositionEncoderFactory.create(
				PositionMode.RELATIVE,
				max_distance=max_seq_len - 1,
			)
			self.n_position_bits = self.position_encoder.n_bits
			# Relative mode uses single distance encoding
			self.n_similarity_position_bits = self.n_position_bits
		else:
			raise ValueError(f"Unsupported position_mode: {position_mode}")

		# Each head learns: (query, key) -> attend?
		# Input to similarity network: [query_bits, key_bits, position_encoding]
		similarity_input_bits = 2 * input_bits + self.n_similarity_position_bits

		self.similarity_heads = ModuleList([
			RAMLayer(
				total_input_bits=similarity_input_bits,
				num_neurons=1,  # Binary output: attend or not
				n_bits_per_neuron=min(similarity_input_bits, 12),
				rng=rng + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Value projection per head (optional - can also use raw input)
		# Value projection uses absolute position (not relative) for the token itself
		value_position_bits = self.n_position_bits if position_mode != PositionMode.NONE else 0
		self.value_heads = ModuleList([
			RAMLayer(
				total_input_bits=input_bits + value_position_bits,
				num_neurons=input_bits,  # Project to same size
				n_bits_per_neuron=min(input_bits + value_position_bits, 10),
				rng=rng + num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Learned aggregation per head (replaces XOR)
		self.aggregators = ModuleList([
			RAMAggregator(
				value_bits=input_bits,
				max_attended=max_seq_len,
				rng=rng + 2 * num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Output combination (aggregate head outputs)
		self.output_layer = RAMLayer(
			total_input_bits=num_heads * input_bits,
			num_neurons=input_bits,
			n_bits_per_neuron=min(num_heads * input_bits, 12),
			rng=rng + 3 * num_heads if rng else None,
		)

		print(f"[RAMAttention] heads={num_heads}, input={input_bits}b, "
			  f"pos_mode={position_mode.name}, pos_bits={self.n_position_bits}, "
			  f"causal={causal}, aggregation=learned")

	def _compute_attention_pattern(
		self,
		queries: list[Tensor],  # List of [input_bits] tensors
		keys: list[Tensor],     # List of [input_bits] tensors
		head_idx: int,
	) -> Tensor:
		"""
		Compute binary attention pattern for one head.

		Args:
			queries: Query tokens
			keys: Key tokens
			head_idx: Which attention head

		Returns:
			attention: [num_queries, num_keys] binary tensor
		"""
		num_q = len(queries)
		num_k = len(keys)
		attention = zeros(num_q, num_k, dtype=uint8)

		for i, q in enumerate(queries):
			for j, k in enumerate(keys):
				# Causal mask: can only attend to past/current
				if self.causal and j > i:
					continue

				# Build similarity input based on position mode
				parts = [q, k]

				if self.position_mode == PositionMode.BINARY:
					# Absolute positions: [query_pos, key_pos]
					q_pos = self.position_encoder.encode(i)
					k_pos = self.position_encoder.encode(j)
					parts.extend([q_pos, k_pos])
				elif self.position_mode == PositionMode.RELATIVE:
					# Relative distance: key_pos - query_pos
					# For causal attention, this is always <= 0
					rel_dist = self.position_encoder.encode_relative(i, j)
					parts.append(rel_dist)
				# else: NONE - no position encoding

				similarity_input = cat(parts).unsqueeze(0)

				# RAM neuron decides: attend or not?
				attend = self.similarity_heads[head_idx](similarity_input)
				attention[i, j] = attend.squeeze()

		return attention

	def _aggregate_values(
		self,
		values: list[Tensor],
		attention: Tensor,
		head_idx: int,
	) -> Tensor:
		"""
		Aggregate attended values for a single query using learned aggregation.

		Uses RAMAggregator which:
		1. Counts votes per bit position (order-invariant)
		2. Learns aggregation rules via RAM

		Args:
			values: Projected value tensors
			attention: Attention pattern for this query [num_keys]
			head_idx: Which attention head (for selecting aggregator)

		Returns:
			aggregated: [input_bits] tensor
		"""
		attended_indices = (attention == 1).nonzero(as_tuple=True)[0]

		if len(attended_indices) == 0:
			# No attention - return zeros
			return zeros(self.input_bits, dtype=uint8)

		# Collect attended values
		attended_values = [values[idx.item()] for idx in attended_indices]

		# Use learned aggregator for this head
		return self.aggregators[head_idx](attended_values)

	def forward(self, tokens: list[Tensor]) -> list[Tensor]:
		"""
		Apply RAM attention to a sequence.

		Args:
			tokens: List of [input_bits] tensors, one per position

		Returns:
			outputs: List of [input_bits] tensors after attention
		"""
		seq_len = len(tokens)
		if seq_len > self.max_seq_len:
			raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

		# Normalize inputs
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		outputs = []

		for i in range(seq_len):
			head_outputs = []

			for h in range(self.num_heads):
				# Compute attention pattern for this head
				attention = self._compute_attention_pattern(
					tokens, tokens, head_idx=h
				)

				# Project values through this head
				projected_values = []
				for j, tok in enumerate(tokens):
					if self.position_mode != PositionMode.NONE:
						# For value projection, always use absolute position
						# (even in RELATIVE mode, the value itself has an absolute position)
						if self.position_mode == PositionMode.RELATIVE:
							# RelativePositionEncoder.encode() takes a signed distance
							# For absolute position, we need a different approach
							# Use a simple binary encoding for value position
							pos_bits = zeros(self.n_position_bits, dtype=uint8)
							pos_val = j
							for b in range(self.n_position_bits - 1, -1, -1):
								pos_bits[b] = pos_val & 1
								pos_val >>= 1
						else:
							pos_bits = self.position_encoder.encode(j)
						val_input = cat([tok, pos_bits]).unsqueeze(0)
					else:
						val_input = tok.unsqueeze(0)
					proj = self.value_heads[h](val_input).squeeze()
					projected_values.append(proj)

				# Aggregate attended values using learned aggregator
				aggregated = self._aggregate_values(
					projected_values,
					attention[i],
					head_idx=h
				)
				head_outputs.append(aggregated)

			# Combine heads
			combined_input = cat(head_outputs).unsqueeze(0)
			output = self.output_layer(combined_input).squeeze()
			outputs.append(output)

		return outputs

	def visualize_attention(self, tokens: list[Tensor], head_idx: int = 0) -> str:
		"""
		Visualize attention pattern for a head.

		Returns ASCII art of attention matrix.
		"""
		attention = self._compute_attention_pattern(tokens, tokens, head_idx)

		lines = [f"Attention pattern (Head {head_idx}):"]
		lines.append("    " + " ".join(f"{j:2d}" for j in range(len(tokens))))

		for i in range(len(tokens)):
			row = f"{i:2d}: "
			for j in range(len(tokens)):
				if self.causal and j > i:
					row += " - "
				elif attention[i, j] == 1:
					row += " # "
				else:
					row += " . "
			lines.append(row)

		return "\n".join(lines)

	def train_step(
		self,
		tokens: list[Tensor],
		targets: list[Tensor],
	) -> int:
		"""
		Train the attention layer on input/target pairs.

		This implements a form of EDRA through the attention mechanism:
		1. Forward pass to get current outputs
		2. For positions with errors, train output layer
		3. Backpropagate to train heads, aggregators, values

		Args:
			tokens: Input token sequence
			targets: Target output sequence (same length)

		Returns:
			Number of positions with errors (before training)
		"""
		seq_len = len(tokens)
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

		errors = 0

		for i in range(seq_len):
			target = targets[i]

			# Forward pass for this position
			head_outputs = []

			for h in range(self.num_heads):
				attention = self._compute_attention_pattern(tokens, tokens, head_idx=h)

				projected_values = []
				for j, tok in enumerate(tokens):
					if self.position_mode != PositionMode.NONE:
						if self.position_mode == PositionMode.RELATIVE:
							pos_bits = zeros(self.n_position_bits, dtype=uint8)
							pos_val = j
							for b in range(self.n_position_bits - 1, -1, -1):
								pos_bits[b] = pos_val & 1
								pos_val >>= 1
						else:
							pos_bits = self.position_encoder.encode(j)
						val_input = cat([tok, pos_bits]).unsqueeze(0)
					else:
						val_input = tok.unsqueeze(0)
					proj = self.value_heads[h](val_input).squeeze()
					projected_values.append(proj)

				aggregated = self._aggregate_values(
					projected_values, attention[i], head_idx=h
				)
				head_outputs.append(aggregated)

			# Combine heads
			combined_input = cat(head_outputs).unsqueeze(0)
			output = self.output_layer(combined_input).squeeze()

			# Check if output matches target
			if not (output == target).all():
				errors += 1

				# Train output layer
				self.output_layer.commit(combined_input, target.unsqueeze(0))

				# For deeper training, we could also train:
				# - aggregators to produce better head outputs
				# - value heads to project better
				# - similarity heads to attend to better positions
				# This is left as optional deeper training

		return errors

	def train_attention_pattern(
		self,
		tokens: list[Tensor],
		attention_targets: list[list[tuple[int, int, int]]],
		head_idx: int = 0,
	) -> int:
		"""
		Train specific attention patterns for a head.

		Args:
			tokens: Token sequence
			attention_targets: List of (query_idx, key_idx, should_attend) tuples
			head_idx: Which head to train

		Returns:
			Number of attention decisions corrected
		"""
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		corrections = 0

		for query_idx, key_idx, should_attend in attention_targets:
			if self.causal and key_idx > query_idx:
				continue  # Skip invalid causal positions

			q = tokens[query_idx]
			k = tokens[key_idx]

			# Build similarity input
			parts = [q, k]
			if self.position_mode == PositionMode.BINARY:
				q_pos = self.position_encoder.encode(query_idx)
				k_pos = self.position_encoder.encode(key_idx)
				parts.extend([q_pos, k_pos])
			elif self.position_mode == PositionMode.RELATIVE:
				rel_dist = self.position_encoder.encode_relative(query_idx, key_idx)
				parts.append(rel_dist)

			similarity_input = cat(parts).unsqueeze(0)

			# Check current attention
			current = self.similarity_heads[head_idx](similarity_input).item()

			if current != should_attend:
				corrections += 1
				target = tensor([[should_attend]], dtype=uint8)
				self.similarity_heads[head_idx].commit(similarity_input, target)

		return corrections

	def __repr__(self):
		return (
			f"RAMAttention(heads={self.num_heads}, "
			f"input={self.input_bits}b, "
			f"pos={self.position_mode.name}, "
			f"causal={self.causal})"
		)
