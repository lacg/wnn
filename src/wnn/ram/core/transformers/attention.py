"""
RAM-based Attention Mechanism (Unified Self + Cross Attention)

Traditional Transformer Attention:
  scores = softmax(Q·K^T / √d)    # Continuous weights [0,1]
  output = scores · V              # Weighted sum of all values

RAM Attention (Hard/Discrete):
  For each query position:
    1. Compute similarity to ALL keys (using RAM neurons)
    2. Select winner(s) via hard routing or voting
    3. Return selected value(s)

Supports both:
  - Self-attention: tokens attend to themselves (context=None)
  - Cross-attention: queries attend to context (context=encoder_output)
"""

from wnn.ram.core import RAMLayer
from wnn.ram.core import RAMAggregator
from wnn.ram.core.transformers.attention_base import LearnableAttention
from wnn.ram.core.transformers.attention_mask import AttentionMask, MaskStrategy
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.encoders_decoders import PositionEncoderFactory
from wnn.ram.enums import CrossAttentionMode

from torch import Tensor, zeros, uint8, cat, tensor
from torch.nn import ModuleList


class RAMAttention(LearnableAttention):
	"""
	Unified RAM-based attention layer supporting self-attention and cross-attention.

	Self-attention (context=None):
	  - Queries, keys, and values all come from same sequence
	  - Supports flexible masking strategies (causal, sliding window, etc.)
	  - Position modes: NONE, BINARY, RELATIVE

	Cross-attention (context=encoder_output):
	  - Queries from decoder, keys/values from encoder
	  - Always bidirectional (can attend to any encoder position)
	  - Position modes: NONE, ENCODER_ONLY, BOTH

	Architecture:
	  similarity_heads: [query, key, positions] -> attend? (binary)
	  value_heads: [value, position] -> projected_value
	  aggregators: [attended_values] -> aggregated (learned, order-invariant)
	  output_layer: [head_outputs] -> final_output
	"""

	def __init__(
		self,
		query_bits: int,
		key_bits: int | None = None,  # None = same as query_bits (self-attention)
		num_heads: int = 4,
		neurons_per_head: int = 8,
		position_mode: PositionMode | CrossAttentionMode = PositionMode.BINARY,
		max_seq_len: int = 16,
		max_context_len: int | None = None,  # For cross-attention
		causal: bool = True,  # Backwards compatibility (self-attention only)
		mask_strategy: MaskStrategy | None = None,  # Override causal with flexible strategy
		window_size: int = 3,  # For SLIDING_WINDOW strategy
		block_size: int = 4,  # For BLOCK strategy
		prefix_len: int = 2,  # For PREFIX strategy
		rng: int | None = None,
	):
		"""
		Args:
			query_bits: Bits per query token
			key_bits: Bits per key/value token (None = same as query_bits)
			num_heads: Number of attention heads
			neurons_per_head: RAM neurons per head (unused, kept for compatibility)
			position_mode: How to encode positions
			max_seq_len: Maximum sequence length for queries
			max_context_len: Maximum context length (for cross-attention)
			causal: If True, use CAUSAL mask (backwards compatible)
			mask_strategy: Explicit masking strategy (overrides causal)
			window_size: Window size for SLIDING_WINDOW strategy
			block_size: Block size for BLOCK strategy
			prefix_len: Prefix length for PREFIX strategy
			rng: Random seed
		"""
		super().__init__()

		# Store for potential future use (currently unused)
		_ = neurons_per_head

		self.query_bits = query_bits
		self.key_bits = key_bits or query_bits
		self.is_cross_attention = key_bits is not None
		self.num_heads = num_heads
		self.max_seq_len = max_seq_len
		self.max_context_len = max_context_len or max_seq_len
		self.position_mode = position_mode

		# Mask configuration
		self.window_size = window_size
		self.block_size = block_size
		self.prefix_len = prefix_len

		# Determine mask strategy
		if self.is_cross_attention:
			# Cross-attention is always bidirectional
			self.mask_strategy = MaskStrategy.BIDIRECTIONAL
		elif mask_strategy is not None:
			self.mask_strategy = mask_strategy
		else:
			# Backwards compatibility: causal=True -> CAUSAL, causal=False -> BIDIRECTIONAL
			self.mask_strategy = MaskStrategy.CAUSAL if causal else MaskStrategy.BIDIRECTIONAL

		# Keep causal attribute for backwards compatibility
		self.causal = self.mask_strategy == MaskStrategy.CAUSAL

		# Determine position encoding setup
		self._setup_position_encoding()

		# Similarity network: [query, key, positions] -> attend?
		similarity_input_bits = self.query_bits + self.key_bits + self.n_similarity_position_bits

		self.similarity_heads = ModuleList([
			RAMLayer(
				total_input_bits=similarity_input_bits,
				num_neurons=1,  # Binary output: attend or not
				n_bits_per_neuron=min(similarity_input_bits, 12),
				rng=rng + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Value projection per head
		value_input_bits = self.key_bits + self.n_position_bits
		self.value_heads = ModuleList([
			RAMLayer(
				total_input_bits=value_input_bits,
				num_neurons=self.query_bits,  # Project to query dimension
				n_bits_per_neuron=min(value_input_bits, 10),
				rng=rng + num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Learned aggregation per head
		self.aggregators = ModuleList([
			RAMAggregator(
				value_bits=self.query_bits,
				max_attended=self.max_context_len,
				rng=rng + 2 * num_heads + i if rng else None,
			)
			for i in range(num_heads)
		])

		# Output combination
		self.output_layer = RAMLayer(
			total_input_bits=num_heads * self.query_bits,
			num_neurons=self.query_bits,
			n_bits_per_neuron=min(num_heads * self.query_bits, 12),
			rng=rng + 3 * num_heads if rng else None,
		)

		attn_type = "cross" if self.is_cross_attention else "self"
		pos_name = position_mode.name if hasattr(position_mode, 'name') else str(position_mode)
		mask_name = self.mask_strategy.name
		print(f"[RAMAttention] type={attn_type}, heads={num_heads}, "
			  f"query={query_bits}b, key={self.key_bits}b, "
			  f"pos={pos_name}, mask={mask_name}")

	def _setup_position_encoding(self):
		"""Configure position encoding based on mode."""
		max_len = max(self.max_seq_len, self.max_context_len)

		if isinstance(self.position_mode, CrossAttentionMode):
			# Cross-attention position modes
			self.n_position_bits = max_len.bit_length()
			self.position_encoder = None  # Use manual encoding

			match self.position_mode:
				case CrossAttentionMode.NONE:
					self.n_similarity_position_bits = 0
				case CrossAttentionMode.ENCODER_ONLY:
					self.n_similarity_position_bits = self.n_position_bits
				case CrossAttentionMode.BOTH:
					self.n_similarity_position_bits = 2 * self.n_position_bits
		else:
			# Self-attention position modes (PositionMode)
			if self.position_mode == PositionMode.NONE:
				self.position_encoder = None
				self.n_position_bits = 0
				self.n_similarity_position_bits = 0
			elif self.position_mode == PositionMode.BINARY:
				self.position_encoder = PositionEncoderFactory.create(
					PositionMode.BINARY, max_seq_len=max_len
				)
				self.n_position_bits = self.position_encoder.n_bits
				self.n_similarity_position_bits = 2 * self.n_position_bits
			elif self.position_mode == PositionMode.RELATIVE:
				self.position_encoder = PositionEncoderFactory.create(
					PositionMode.RELATIVE, max_distance=max_len - 1
				)
				self.n_position_bits = self.position_encoder.n_bits
				self.n_similarity_position_bits = self.n_position_bits

	def _encode_position(self, pos: int) -> Tensor:
		"""Encode a position as binary bits."""
		pos_bits = zeros(self.n_position_bits, dtype=uint8)
		for b in range(self.n_position_bits - 1, -1, -1):
			pos_bits[b] = pos & 1
			pos >>= 1
		return pos_bits

	def _get_mask(self, query_len: int, key_len: int | None = None) -> Tensor:
		"""
		Get attention mask for the given sequence lengths.

		Args:
			query_len: Number of query positions
			key_len: Number of key positions (default: same as query_len)

		Returns:
			Boolean mask [query_len, key_len] where True = can attend
		"""
		key_len = key_len or query_len

		return AttentionMask.from_strategy(
			strategy=self.mask_strategy,
			seq_len=query_len,
			key_len=key_len,
			window_size=self.window_size,
			block_size=self.block_size,
			prefix_len=self.prefix_len,
		)

	def _can_attend(self, mask: Tensor, query_pos: int, key_pos: int) -> bool:
		"""Check if query position can attend to key position."""
		return bool(mask[query_pos, key_pos].item())

	def get_mask(self, query_len: int, key_len: int | None = None) -> Tensor:
		"""
		Public method to get attention mask.

		Args:
			query_len: Number of query positions
			key_len: Number of key positions (default: same as query_len)

		Returns:
			Boolean mask [query_len, key_len] where True = can attend
		"""
		return self._get_mask(query_len, key_len)

	def _compute_attention_pattern(
		self,
		queries: list[Tensor],
		keys: list[Tensor],
		head_idx: int,
		mask: Tensor | None = None,
	) -> Tensor:
		"""
		Compute binary attention pattern for one head.

		Args:
			queries: Query tokens
			keys: Key tokens (same as queries for self-attention)
			head_idx: Which attention head
			mask: Pre-computed attention mask (optional, computed if None)

		Returns:
			attention: [num_queries, num_keys] binary tensor
		"""
		num_q = len(queries)
		num_k = len(keys)
		attention = zeros(num_q, num_k, dtype=uint8)

		# Get mask if not provided
		if mask is None:
			mask = self._get_mask(num_q, num_k)

		for i, q in enumerate(queries):
			for j, k in enumerate(keys):
				# Check mask instead of inline causal check
				if not self._can_attend(mask, i, j):
					continue

				# Build similarity input based on position mode
				parts = [q, k]

				if isinstance(self.position_mode, CrossAttentionMode):
					# Cross-attention position encoding
					match self.position_mode:
						case CrossAttentionMode.ENCODER_ONLY:
							k_pos = self._encode_position(j)
							parts.append(k_pos)
						case CrossAttentionMode.BOTH:
							q_pos = self._encode_position(i)
							k_pos = self._encode_position(j)
							parts.extend([q_pos, k_pos])
				else:
					# Self-attention position encoding
					if self.position_mode == PositionMode.BINARY:
						q_pos = self.position_encoder.encode(i)
						k_pos = self.position_encoder.encode(j)
						parts.extend([q_pos, k_pos])
					elif self.position_mode == PositionMode.RELATIVE:
						rel_dist = self.position_encoder.encode_relative(i, j)
						parts.append(rel_dist)

				similarity_input = cat(parts).unsqueeze(0)
				attend = self.similarity_heads[head_idx](similarity_input)
				attention[i, j] = attend.squeeze()

		return attention

	def _aggregate_values(
		self,
		values: list[Tensor],
		attention: Tensor,
		head_idx: int,
	) -> Tensor:
		"""Aggregate attended values using learned aggregation."""
		attended_indices = (attention == 1).nonzero(as_tuple=True)[0]

		if len(attended_indices) == 0:
			return zeros(self.query_bits, dtype=uint8)

		attended_values = [values[idx.item()] for idx in attended_indices]
		return self.aggregators[head_idx](attended_values)

	def _project_values(
		self,
		tokens: list[Tensor],
		head_idx: int,
	) -> list[Tensor]:
		"""Project tokens through value head."""
		projected = []
		for j, tok in enumerate(tokens):
			if self.n_position_bits > 0:
				if isinstance(self.position_mode, CrossAttentionMode) or \
				   self.position_mode == PositionMode.RELATIVE:
					pos_bits = self._encode_position(j)
				else:
					pos_bits = self.position_encoder.encode(j)
				val_input = cat([tok, pos_bits]).unsqueeze(0)
			else:
				val_input = tok.unsqueeze(0)
			proj = self.value_heads[head_idx](val_input).squeeze()
			projected.append(proj)
		return projected

	def forward(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> list[Tensor]:
		"""
		Apply attention to a sequence.

		Args:
			tokens: Query tokens (list of [query_bits] tensors)
			context: Key/value tokens for cross-attention (None for self-attention)

		Returns:
			outputs: Transformed tokens (list of [query_bits] tensors)
		"""
		# Determine queries and keys
		queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		if context is None:
			# Self-attention: keys = queries
			keys = queries
		else:
			# Cross-attention: keys from context
			keys = [t.squeeze() if t.ndim > 1 else t for t in context]

		seq_len = len(queries)
		key_len = len(keys)

		if seq_len > self.max_seq_len:
			raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")
		if key_len > self.max_context_len:
			raise ValueError(f"Context length {key_len} exceeds max {self.max_context_len}")

		outputs = []

		# Pre-compute mask once for all heads
		mask = self._get_mask(seq_len, key_len)

		for i in range(seq_len):
			head_outputs = []

			for h in range(self.num_heads):
				# Compute attention pattern (pass pre-computed mask)
				attention = self._compute_attention_pattern(queries, keys, head_idx=h, mask=mask)

				# Project values
				projected_values = self._project_values(keys, head_idx=h)

				# Aggregate attended values
				aggregated = self._aggregate_values(
					projected_values, attention[i], head_idx=h
				)
				head_outputs.append(aggregated)

			# Combine heads
			combined_input = cat(head_outputs).unsqueeze(0)
			output = self.output_layer(combined_input).squeeze()
			outputs.append(output)

		return outputs

	def get_attention_weights(
		self,
		tokens: list[Tensor],
		context: list[Tensor] | None = None,
	) -> Tensor:
		"""
		Get attention weights (binary pattern).

		Returns:
			Tensor [num_queries, num_keys] with values 0 or 1
		"""
		queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		keys = queries if context is None else [t.squeeze() if t.ndim > 1 else t for t in context]

		# Return attention from head 0 (could aggregate across heads)
		return self._compute_attention_pattern(queries, keys, head_idx=0).float()

	def train_step(
		self,
		tokens: list[Tensor],
		targets: list[Tensor],
		context: list[Tensor] | None = None,
	) -> int:
		"""
		Single training step.

		Args:
			tokens: Input tokens (queries)
			targets: Target outputs
			context: Context for cross-attention

		Returns:
			Number of positions with errors
		"""
		queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]
		keys = queries if context is None else [t.squeeze() if t.ndim > 1 else t for t in context]

		# Pre-compute mask
		mask = self._get_mask(len(queries), len(keys))

		errors = 0

		for i in range(len(queries)):
			head_outputs = []

			for h in range(self.num_heads):
				attention = self._compute_attention_pattern(queries, keys, head_idx=h, mask=mask)
				projected_values = self._project_values(keys, head_idx=h)
				aggregated = self._aggregate_values(projected_values, attention[i], head_idx=h)
				head_outputs.append(aggregated)

			combined_input = cat(head_outputs).unsqueeze(0)
			output = self.output_layer(combined_input).squeeze()

			if not (output == targets[i]).all():
				errors += 1
				self.output_layer.commit(combined_input, targets[i].unsqueeze(0))

		return errors

	def train_attention_pattern(
		self,
		tokens: list[Tensor],
		attention_targets: list[tuple[int, int, int]],
		head_idx: int = 0,
		context: list[Tensor] | None = None,
	) -> int:
		"""
		Train specific attention patterns for a head.

		Args:
			tokens: Query tokens
			attention_targets: List of (query_idx, key_idx, should_attend) tuples
			head_idx: Which head to train
			context: Context for cross-attention

		Returns:
			Number of attention decisions corrected
		"""
		queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		keys = queries if context is None else [t.squeeze() if t.ndim > 1 else t for t in context]

		# Pre-compute mask
		mask = self._get_mask(len(queries), len(keys))

		corrections = 0

		for query_idx, key_idx, should_attend in attention_targets:
			# Use mask instead of inline causal check
			if not self._can_attend(mask, query_idx, key_idx):
				continue

			q = queries[query_idx]
			k = keys[key_idx]

			# Build similarity input
			parts = [q, k]

			if isinstance(self.position_mode, CrossAttentionMode):
				match self.position_mode:
					case CrossAttentionMode.ENCODER_ONLY:
						k_pos = self._encode_position(key_idx)
						parts.append(k_pos)
					case CrossAttentionMode.BOTH:
						q_pos = self._encode_position(query_idx)
						k_pos = self._encode_position(key_idx)
						parts.extend([q_pos, k_pos])
			else:
				if self.position_mode == PositionMode.BINARY:
					q_pos = self.position_encoder.encode(query_idx)
					k_pos = self.position_encoder.encode(key_idx)
					parts.extend([q_pos, k_pos])
				elif self.position_mode == PositionMode.RELATIVE:
					rel_dist = self.position_encoder.encode_relative(query_idx, key_idx)
					parts.append(rel_dist)

			similarity_input = cat(parts).unsqueeze(0)
			current = self.similarity_heads[head_idx](similarity_input).item()

			if current != should_attend:
				corrections += 1
				target = tensor([[should_attend]], dtype=uint8)
				self.similarity_heads[head_idx].commit(similarity_input, target)

		return corrections

	def __repr__(self):
		attn_type = "cross" if self.is_cross_attention else "self"
		pos_name = self.position_mode.name if hasattr(self.position_mode, 'name') else str(self.position_mode)
		mask_name = self.mask_strategy.name
		return (
			f"RAMAttention(type={attn_type}, heads={self.num_heads}, "
			f"query={self.query_bits}b, key={self.key_bits}b, "
			f"pos={pos_name}, mask={mask_name})"
		)


	# ------------------------------------------------------------------
	# Pre-training Methods for Position-Based Patterns
	# ------------------------------------------------------------------

	def pretrain_copy(self, seq_len: int, head_idx: int = 0) -> int:
		"""
		Pre-train attention to copy: position i attends to position i.

		This is the identity attention pattern - each position looks at itself.
		Achieves 100% generalization for copy task.

		Args:
			seq_len: Sequence length to train on
			head_idx: Which head to train (default: 0)

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				should_attend = 1 if i == j else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def pretrain_shift(self, seq_len: int, offset: int = 1, head_idx: int = 0) -> int:
		"""
		Pre-train attention to shift: position i attends to position i+offset.

		Args:
			seq_len: Sequence length to train on
			offset: Shift amount (positive = look right, negative = look left)
			head_idx: Which head to train

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				target_j = i + offset
				should_attend = 1 if j == target_j and 0 <= target_j < seq_len else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def pretrain_reverse(self, seq_len: int, head_idx: int = 0) -> int:
		"""
		Pre-train attention to reverse: position i attends to position (n-1-i).

		Args:
			seq_len: Sequence length to train on
			head_idx: Which head to train

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				target_j = seq_len - 1 - i
				should_attend = 1 if j == target_j else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def pretrain_first(self, seq_len: int, head_idx: int = 0) -> int:
		"""
		Pre-train attention: all positions attend to position 0 (first).

		Useful for tasks that need the first token (e.g., CLS token).

		Args:
			seq_len: Sequence length to train on
			head_idx: Which head to train

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				should_attend = 1 if j == 0 else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def pretrain_last(self, seq_len: int, head_idx: int = 0) -> int:
		"""
		Pre-train attention: all positions attend to position n-1 (last).

		Note: For causal attention, only the last position can attend to itself.

		Args:
			seq_len: Sequence length to train on
			head_idx: Which head to train

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				target_j = seq_len - 1
				should_attend = 1 if j == target_j else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def pretrain_causal_next(self, seq_len: int, head_idx: int = 0) -> int:
		"""
		Pre-train causal attention: position i attends to position i-1.

		This is the autoregressive pattern for next-token prediction.
		Position 0 attends to nothing (or itself if non-causal).

		Args:
			seq_len: Sequence length to train on
			head_idx: Which head to train

		Returns:
			Number of patterns written
		"""
		mask = self._get_mask(seq_len)
		patterns = 0
		for i in range(seq_len):
			for j in range(seq_len):
				# Position i attends to i-1 (previous token)
				target_j = i - 1 if i > 0 else 0
				should_attend = 1 if j == target_j else 0
				if not self._can_attend(mask, i, j):
					continue
				self._write_position_pattern(i, j, should_attend, head_idx)
				patterns += 1
		return patterns

	def _write_position_pattern(
		self,
		query_pos: int,
		key_pos: int,
		should_attend: int,
		head_idx: int,
	) -> None:
		"""
		Write a position-based attention pattern to memory.

		This writes directly to the similarity head's RAM, using
		dummy token content (zeros) since the pattern is position-only.
		"""
		# Create dummy tokens (all zeros - pattern is position-based)
		query = zeros(self.query_bits, dtype=uint8)
		key = zeros(self.key_bits, dtype=uint8)

		# Build similarity input with position encoding
		parts = [query, key]

		if isinstance(self.position_mode, CrossAttentionMode):
			match self.position_mode:
				case CrossAttentionMode.ENCODER_ONLY:
					k_pos = self._encode_position(key_pos)
					parts.append(k_pos)
				case CrossAttentionMode.BOTH:
					q_pos = self._encode_position(query_pos)
					k_pos = self._encode_position(key_pos)
					parts.extend([q_pos, k_pos])
		else:
			if self.position_mode == PositionMode.BINARY:
				q_pos = self.position_encoder.encode(query_pos)
				k_pos = self.position_encoder.encode(key_pos)
				parts.extend([q_pos, k_pos])
			elif self.position_mode == PositionMode.RELATIVE:
				rel_dist = self.position_encoder.encode_relative(query_pos, key_pos)
				parts.append(rel_dist)

		similarity_input = cat(parts).unsqueeze(0)
		target = tensor([[should_attend]], dtype=uint8)
		self.similarity_heads[head_idx].commit(similarity_input, target)

	def pretrain_identity_values(self, seq_len: int) -> int:
		"""
		Pre-train value heads to be identity (output = input).

		This ensures attended values pass through unchanged.

		Args:
			seq_len: Sequence length (for position encoding)

		Returns:
			Number of patterns written
		"""
		patterns = 0

		# For each possible position
		for pos in range(min(seq_len, self.max_context_len)):
			# For each head
			for h in range(self.num_heads):
				# Create identity mapping for common bit patterns
				for val in range(min(4, 2 ** self.key_bits)):  # Sample patterns
					# Create value bits
					value = zeros(self.key_bits, dtype=uint8)
					for b in range(self.key_bits):
						value[b] = (val >> b) & 1

					# Build input with position
					if self.n_position_bits > 0:
						if isinstance(self.position_mode, CrossAttentionMode) or \
						   self.position_mode == PositionMode.RELATIVE:
							pos_bits = self._encode_position(pos)
						else:
							pos_bits = self.position_encoder.encode(pos)
						val_input = cat([value, pos_bits]).unsqueeze(0)
					else:
						val_input = value.unsqueeze(0)

					# Target = same as input (identity)
					target = value.unsqueeze(0)
					self.value_heads[h].commit(val_input, target)
					patterns += 1

		return patterns


# Backwards-compatible alias (will be removed)
RAMCrossAttention = RAMAttention
