"""
RAM-based Cross-Attention Mechanism

In encoder-decoder architectures:
  - Self-attention: decoder attends to itself (causal)
  - Cross-attention: decoder attends to encoder output (non-causal)

Cross-Attention:
  Query: from decoder hidden state (what am I looking for?)
  Key: from encoder output (what's available to match?)
  Value: from encoder output (what content to retrieve?)

Architecture:
                     Decoder            Encoder
                     Hidden             Output
                       │                   │
                       ▼                   ▼
                    [Query]         [Keys, Values]
                       │                   │
                       └───────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Cross-Attention    │
                    │  (attend to encoder)│
                    └─────────────────────┘
                               │
                               ▼
                    Contextualized Output

Key differences from self-attention:
  - Non-causal: can attend to ANY encoder position
  - Different sequence lengths: decoder_len != encoder_len
  - Keys/Values are precomputed from encoder
"""

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMAggregator import RAMAggregator
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.encoders_decoders import PositionEncoderFactory

from torch import Tensor, zeros, uint8, cat, tensor
from torch.nn import Module, ModuleList
from enum import IntEnum


class CrossAttentionMode(IntEnum):
	"""How to encode positions in cross-attention."""
	NONE = 0          # No position info (content-only)
	ENCODER_ONLY = 1  # Only encode key/value positions (encoder side)
	BOTH = 2          # Encode both query (decoder) and key (encoder) positions


class RAMCrossAttention(Module):
	"""
	RAM-based cross-attention layer.

	Enables decoder to attend to encoder outputs:
	- Query from decoder state (current generation context)
	- Key/Value from encoder output (source sequence information)

	Unlike self-attention:
	- Non-causal: any decoder position can attend to any encoder position
	- Different sequences: queries and keys can have different lengths
	"""

	def __init__(
		self,
		decoder_bits: int,
		encoder_bits: int | None = None,
		num_heads: int = 4,
		position_mode: CrossAttentionMode = CrossAttentionMode.ENCODER_ONLY,
		max_encoder_len: int = 32,
		max_decoder_len: int = 32,
		rng: int | None = None,
	):
		"""
		Args:
			decoder_bits: Bits per decoder token (query dimension)
			encoder_bits: Bits per encoder token (key/value dimension), defaults to decoder_bits
			num_heads: Number of attention heads
			position_mode: How to encode positions (NONE, ENCODER_ONLY, BOTH)
			max_encoder_len: Maximum encoder sequence length
			max_decoder_len: Maximum decoder sequence length
			rng: Random seed
		"""
		super().__init__()

		self.decoder_bits = decoder_bits
		self.encoder_bits = encoder_bits or decoder_bits
		self.num_heads = num_heads
		self.position_mode = position_mode
		self.max_encoder_len = max_encoder_len
		self.max_decoder_len = max_decoder_len

		# Position encoding for encoder positions
		self.n_position_bits = max(max_encoder_len, max_decoder_len).bit_length()

		# Compute similarity input size based on position mode
		match position_mode:
			case CrossAttentionMode.NONE:
				self.n_similarity_position_bits = 0
			case CrossAttentionMode.ENCODER_ONLY:
				# Only encoder (key) position
				self.n_similarity_position_bits = self.n_position_bits
			case CrossAttentionMode.BOTH:
				# Both decoder (query) and encoder (key) positions
				self.n_similarity_position_bits = 2 * self.n_position_bits

		# Similarity network: [query, key, positions] -> attend?
		similarity_input_bits = self.decoder_bits + self.encoder_bits + self.n_similarity_position_bits

		self.similarity_heads = ModuleList([
			RAMLayer(
				total_input_bits=similarity_input_bits,
				num_neurons=1,  # Binary: attend or not
				n_bits_per_neuron=min(similarity_input_bits, 12),
				rng=rng + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Value projection per head (encoder tokens -> values)
		value_input_bits = self.encoder_bits + self.n_position_bits
		self.value_heads = ModuleList([
			RAMLayer(
				total_input_bits=value_input_bits,
				num_neurons=self.decoder_bits,  # Project to decoder dimension
				n_bits_per_neuron=min(value_input_bits, 10),
				rng=rng + num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Aggregators (combine attended values)
		self.aggregators = ModuleList([
			RAMAggregator(
				value_bits=self.decoder_bits,
				max_attended=max_encoder_len,
				rng=rng + 2 * num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Output combination layer
		self.output_layer = RAMLayer(
			total_input_bits=num_heads * self.decoder_bits,
			num_neurons=self.decoder_bits,
			n_bits_per_neuron=min(num_heads * self.decoder_bits, 12),
			rng=rng + 3 * num_heads if rng else None,
		)

		print(f"[RAMCrossAttention] heads={num_heads}, "
			  f"decoder={decoder_bits}b, encoder={self.encoder_bits}b, "
			  f"pos_mode={position_mode.name}")

	def _encode_position(self, pos: int) -> Tensor:
		"""Encode a position as binary bits."""
		pos_bits = zeros(self.n_position_bits, dtype=uint8)
		for b in range(self.n_position_bits - 1, -1, -1):
			pos_bits[b] = pos & 1
			pos >>= 1
		return pos_bits

	def _compute_cross_attention_pattern(
		self,
		queries: list[Tensor],     # From decoder
		keys: list[Tensor],        # From encoder
		head_idx: int,
	) -> Tensor:
		"""
		Compute binary cross-attention pattern for one head.

		Args:
			queries: Decoder hidden states [decoder_len x decoder_bits]
			keys: Encoder outputs [encoder_len x encoder_bits]
			head_idx: Which attention head

		Returns:
			attention: [decoder_len, encoder_len] binary tensor
		"""
		num_q = len(queries)
		num_k = len(keys)
		attention = zeros(num_q, num_k, dtype=uint8)

		for i, q in enumerate(queries):
			for j, k in enumerate(keys):
				# Build similarity input based on position mode
				parts = [q, k]

				match self.position_mode:
					case CrossAttentionMode.ENCODER_ONLY:
						# Only include encoder position
						k_pos = self._encode_position(j)
						parts.append(k_pos)
					case CrossAttentionMode.BOTH:
						# Include both query (decoder) and key (encoder) positions
						q_pos = self._encode_position(i)
						k_pos = self._encode_position(j)
						parts.extend([q_pos, k_pos])
					# NONE: no position info

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
		Aggregate attended encoder values.

		Args:
			values: Projected encoder values
			attention: Attention pattern for this query [encoder_len]
			head_idx: Which head

		Returns:
			aggregated: [decoder_bits] tensor
		"""
		attended_indices = (attention == 1).nonzero(as_tuple=True)[0]

		if len(attended_indices) == 0:
			# No attention - return zeros
			return zeros(self.decoder_bits, dtype=uint8)

		attended_values = [values[idx.item()] for idx in attended_indices]
		return self.aggregators[head_idx](attended_values)

	def forward(
		self,
		decoder_hidden: list[Tensor],  # Decoder states (queries)
		encoder_output: list[Tensor],  # Encoder outputs (keys/values)
	) -> list[Tensor]:
		"""
		Apply cross-attention: decoder attends to encoder.

		Args:
			decoder_hidden: List of decoder hidden states [decoder_bits]
			encoder_output: List of encoder outputs [encoder_bits]

		Returns:
			outputs: List of contextualized decoder states [decoder_bits]
		"""
		decoder_len = len(decoder_hidden)
		encoder_len = len(encoder_output)

		if decoder_len > self.max_decoder_len:
			raise ValueError(f"Decoder length {decoder_len} exceeds max {self.max_decoder_len}")
		if encoder_len > self.max_encoder_len:
			raise ValueError(f"Encoder length {encoder_len} exceeds max {self.max_encoder_len}")

		# Normalize inputs
		decoder_hidden = [t.squeeze() if t.ndim > 1 else t for t in decoder_hidden]
		encoder_output = [t.squeeze() if t.ndim > 1 else t for t in encoder_output]

		outputs = []

		for i in range(decoder_len):
			head_outputs = []

			for h in range(self.num_heads):
				# Compute cross-attention pattern
				attention = self._compute_cross_attention_pattern(
					decoder_hidden, encoder_output, head_idx=h
				)

				# Project encoder values through this head
				projected_values = []
				for j, enc_tok in enumerate(encoder_output):
					pos_bits = self._encode_position(j)
					val_input = cat([enc_tok, pos_bits]).unsqueeze(0)
					proj = self.value_heads[h](val_input).squeeze()
					projected_values.append(proj)

				# Aggregate attended encoder values
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

	def forward_single(
		self,
		query: Tensor,
		encoder_output: list[Tensor],
		query_position: int = 0,
	) -> Tensor:
		"""
		Cross-attention for a single query position.

		Useful for autoregressive decoding where we process one token at a time.

		Args:
			query: Single decoder hidden state [decoder_bits]
			encoder_output: Full encoder output sequence
			query_position: Position of query in decoder sequence

		Returns:
			output: Contextualized decoder state [decoder_bits]
		"""
		query = query.squeeze() if query.ndim > 1 else query
		encoder_output = [t.squeeze() if t.ndim > 1 else t for t in encoder_output]

		head_outputs = []

		for h in range(self.num_heads):
			# Compute attention for this single query against all encoder keys
			attention = zeros(len(encoder_output), dtype=uint8)

			for j, k in enumerate(encoder_output):
				parts = [query, k]

				match self.position_mode:
					case CrossAttentionMode.ENCODER_ONLY:
						k_pos = self._encode_position(j)
						parts.append(k_pos)
					case CrossAttentionMode.BOTH:
						q_pos = self._encode_position(query_position)
						k_pos = self._encode_position(j)
						parts.extend([q_pos, k_pos])

				similarity_input = cat(parts).unsqueeze(0)
				attend = self.similarity_heads[h](similarity_input)
				attention[j] = attend.squeeze()

			# Project encoder values
			projected_values = []
			for j, enc_tok in enumerate(encoder_output):
				pos_bits = self._encode_position(j)
				val_input = cat([enc_tok, pos_bits]).unsqueeze(0)
				proj = self.value_heads[h](val_input).squeeze()
				projected_values.append(proj)

			# Aggregate attended values
			aggregated = self._aggregate_values(
				projected_values, attention, head_idx=h
			)
			head_outputs.append(aggregated)

		# Combine heads
		combined_input = cat(head_outputs).unsqueeze(0)
		return self.output_layer(combined_input).squeeze()

	def train_step(
		self,
		decoder_hidden: list[Tensor],
		encoder_output: list[Tensor],
		targets: list[Tensor],
	) -> int:
		"""
		Train cross-attention on decoder/encoder pairs.

		Args:
			decoder_hidden: Decoder hidden states (queries)
			encoder_output: Encoder outputs (keys/values)
			targets: Target outputs for each decoder position

		Returns:
			Number of positions with errors
		"""
		decoder_hidden = [t.squeeze() if t.ndim > 1 else t for t in decoder_hidden]
		encoder_output = [t.squeeze() if t.ndim > 1 else t for t in encoder_output]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

		errors = 0

		for i in range(len(decoder_hidden)):
			head_outputs = []

			for h in range(self.num_heads):
				attention = self._compute_cross_attention_pattern(
					decoder_hidden, encoder_output, head_idx=h
				)

				projected_values = []
				for j, enc_tok in enumerate(encoder_output):
					pos_bits = self._encode_position(j)
					val_input = cat([enc_tok, pos_bits]).unsqueeze(0)
					proj = self.value_heads[h](val_input).squeeze()
					projected_values.append(proj)

				aggregated = self._aggregate_values(
					projected_values, attention[i], head_idx=h
				)
				head_outputs.append(aggregated)

			combined_input = cat(head_outputs).unsqueeze(0)
			output = self.output_layer(combined_input).squeeze()

			if not (output == targets[i]).all():
				errors += 1
				self.output_layer.commit(combined_input, targets[i].unsqueeze(0))

		return errors

	def visualize_cross_attention(
		self,
		decoder_hidden: list[Tensor],
		encoder_output: list[Tensor],
		head_idx: int = 0,
		decoder_labels: list[str] | None = None,
		encoder_labels: list[str] | None = None,
	) -> str:
		"""
		Visualize cross-attention pattern.

		Returns ASCII art showing which encoder positions each decoder position attends to.
		"""
		attention = self._compute_cross_attention_pattern(
			decoder_hidden, encoder_output, head_idx
		)

		dec_len = len(decoder_hidden)
		enc_len = len(encoder_output)

		# Use labels if provided, otherwise use indices
		dec_labels = decoder_labels or [str(i) for i in range(dec_len)]
		enc_labels = encoder_labels or [str(j) for j in range(enc_len)]

		lines = [f"Cross-Attention Pattern (Head {head_idx}):"]
		lines.append("Decoder \\ Encoder")

		# Header row (encoder positions)
		header = "      " + " ".join(f"{l[:3]:>3}" for l in enc_labels)
		lines.append(header)

		for i in range(dec_len):
			row = f"{dec_labels[i][:3]:>3}: "
			for j in range(enc_len):
				if attention[i, j] == 1:
					row += " # "
				else:
					row += " . "
			lines.append(row)

		return "\n".join(lines)

	def __repr__(self):
		return (
			f"RAMCrossAttention(heads={self.num_heads}, "
			f"decoder={self.decoder_bits}b, encoder={self.encoder_bits}b, "
			f"pos={self.position_mode.name})"
		)
