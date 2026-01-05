"""
RAM-based Embedding Layer

Provides learnable token embeddings for RAM Transformer models.

Traditional embeddings: token_id → dense vector (learned via gradients)
RAM embeddings: token_bits → embedding_bits (learned via EDRA)

Key differences:
- Discrete (binary) representations
- No gradients - learned through constraint satisfaction
- Can use generalization strategies for unseen tokens

Architecture:
					┌─────────────────────────────────────┐
					│         Token Input                 │
	Token bits ──────▶│  (e.g., 5-bit ASCII encoding)       │
					└─────────────────────────────────────┘
									 │
					┌────────────────▼────────────────────┐
					│         Embedding Projection        │
					│  RAMLayer: token_bits → embed_bits  │
					└─────────────────────────────────────┘
									 │
					┌────────────────▼────────────────────┐
					│    (Optional) Position Encoding     │
					│         XOR with position bits      │
					└─────────────────────────────────────┘
									 │
	Embedding ◀──────────────────────┘
"""

from wnn.ram.core import RAMLayer
# MapperFactory imported lazily to avoid circular import
from wnn.ram.core.models import PositionEncoding
from wnn.ram.core import MapperStrategy, ContextMode, BitMapperMode as OutputMode

from torch import Tensor, zeros, uint8, arange
from torch.nn import Module


class RAMEmbedding(Module):
	"""
	RAM-based token embedding layer.

	Maps discrete token representations to learned embedding vectors.
	Supports multiple embedding strategies and optional position encoding.
	"""

	def __init__(
		self,
		token_bits: int,
		embedding_bits: int,
		max_seq_len: int = 512,
		position_encoding: PositionEncoding = PositionEncoding.BINARY,
		strategy: MapperStrategy = MapperStrategy.DIRECT,
		rng: int | None = None,
	):
		"""
		Args:
			token_bits: Bits per input token (e.g., 5 for ASCII uppercase)
			embedding_bits: Bits per output embedding
			max_seq_len: Maximum sequence length (for position encoding)
			position_encoding: How to encode position information
			strategy: Generalization strategy for embedding projection
			rng: Random seed
		"""
		super().__init__()

		self.token_bits = token_bits
		self.embedding_bits = embedding_bits
		self.max_seq_len = max_seq_len
		self.position_encoding = position_encoding
		self.strategy = strategy

		# Bits needed for position encoding
		self.position_bits = max_seq_len.bit_length() if position_encoding != PositionEncoding.NONE else 0

		# Token embedding projection
		if strategy == MapperStrategy.DIRECT:
			self.token_embed = RAMLayer(
				total_input_bits=token_bits,
				num_neurons=embedding_bits,
				n_bits_per_neuron=min(token_bits, 12),
				rng=rng,
			)
		else:
			# Use MapperFactory for generalization strategies (lazy import to avoid circular)
			from wnn.ram.factories import MapperFactory
			# Note: requires token_bits == embedding_bits for some strategies
			self.token_embed = MapperFactory.create(
				strategy=strategy,
				n_bits=token_bits if token_bits == embedding_bits else embedding_bits,
				rng=rng,
			)
			# If dimensions differ, add a projection
			if token_bits != embedding_bits:
				self.dim_proj = RAMLayer(
					total_input_bits=token_bits,
					num_neurons=embedding_bits,
					n_bits_per_neuron=min(token_bits, 12),
					rng=rng + 500 if rng else None,
				)
			else:
				self.dim_proj = None

		# Position embedding (if learned)
		if position_encoding == PositionEncoding.LEARNED:
			self.position_embed = RAMLayer(
				total_input_bits=self.position_bits,
				num_neurons=embedding_bits,
				n_bits_per_neuron=self.position_bits,
				rng=rng + 1000 if rng else None,
			)
		else:
			self.position_embed = None

		print(f"[RAMEmbedding] {token_bits}→{embedding_bits} bits, "
			  f"pos={position_encoding.name}, strategy={strategy.name}")

	def _get_position_bits(self, position: int) -> Tensor:
		"""Get binary position encoding."""
		pos_bits = zeros(self.position_bits, dtype=uint8)
		for i in range(self.position_bits):
			pos_bits[self.position_bits - 1 - i] = (position >> i) & 1
		return pos_bits

	def _apply_position_encoding(
		self,
		embedding: Tensor,
		position: int,
	) -> Tensor:
		"""Apply position encoding to embedding."""
		match self.position_encoding:
			case PositionEncoding.NONE:
				return embedding

			case PositionEncoding.BINARY:
				# XOR with binary position bits (extended to embedding size)
				pos_bits = self._get_position_bits(position)
				# Repeat position bits to match embedding size
				if self.position_bits < self.embedding_bits:
					repeats = self.embedding_bits // self.position_bits + 1
					pos_extended = pos_bits.repeat(repeats)[:self.embedding_bits]
				else:
					pos_extended = pos_bits[:self.embedding_bits]
				return embedding ^ pos_extended

			case PositionEncoding.LEARNED:
				# Learned position embedding (added via XOR)
				pos_bits = self._get_position_bits(position)
				pos_embed = self.position_embed(pos_bits.unsqueeze(0)).squeeze()
				return embedding ^ pos_embed

			case PositionEncoding.SINUSOIDAL:
				# Discrete approximation of sinusoidal encoding
				# Use different bit patterns based on position
				pos_bits = zeros(self.embedding_bits, dtype=uint8)
				for i in range(self.embedding_bits):
					# Different frequencies for different dimensions
					freq = 2 ** (i // 2)
					if i % 2 == 0:
						# "sin" - based on position
						pos_bits[i] = 1 if (position // freq) % 2 == 0 else 0
					else:
						# "cos" - phase shifted
						pos_bits[i] = 1 if ((position + freq // 2) // freq) % 2 == 0 else 0
				return embedding ^ pos_bits

		return embedding

	def forward(
		self,
		tokens: list[Tensor],
		add_position: bool = True,
	) -> list[Tensor]:
		"""
		Embed a sequence of tokens.

		Args:
			tokens: List of token tensors, each [token_bits]
			add_position: Whether to add position encoding

		Returns:
			List of embedding tensors, each [embedding_bits]
		"""
		embeddings = []

		for pos, token in enumerate(tokens):
			token = token.squeeze()

			# Token embedding
			if self.strategy == MapperStrategy.DIRECT:
				embed = self.token_embed(token.unsqueeze(0)).squeeze()
			else:
				if self.dim_proj is not None:
					# Project to embedding dimension first
					projected = self.dim_proj(token.unsqueeze(0)).squeeze()
					embed = self.token_embed(projected)
				else:
					embed = self.token_embed(token)

			# Position encoding
			if add_position and self.position_encoding != PositionEncoding.NONE:
				embed = self._apply_position_encoding(embed, pos)

			embeddings.append(embed)

		return embeddings

	def forward_single(
		self,
		token: Tensor,
		position: int = 0,
		add_position: bool = True,
	) -> Tensor:
		"""
		Embed a single token.

		Args:
			token: Token tensor [token_bits]
			position: Position in sequence
			add_position: Whether to add position encoding

		Returns:
			Embedding tensor [embedding_bits]
		"""
		token = token.squeeze()

		# Token embedding
		if self.strategy == MapperStrategy.DIRECT:
			embed = self.token_embed(token.unsqueeze(0)).squeeze()
		else:
			if self.dim_proj is not None:
				projected = self.dim_proj(token.unsqueeze(0)).squeeze()
				embed = self.token_embed(projected)
			else:
				embed = self.token_embed(token)

		# Position encoding
		if add_position and self.position_encoding != PositionEncoding.NONE:
			embed = self._apply_position_encoding(embed, position)

		return embed

	def train_embedding(
		self,
		token: Tensor,
		target_embedding: Tensor,
	) -> int:
		"""
		Train embedding for a specific token.

		Args:
			token: Token tensor [token_bits]
			target_embedding: Desired embedding [embedding_bits]

		Returns:
			1 if updated, 0 if already correct
		"""
		token = token.squeeze()
		target_embedding = target_embedding.squeeze()

		if self.strategy == MapperStrategy.DIRECT:
			current = self.token_embed(token.unsqueeze(0)).squeeze()
			if not (current == target_embedding).all():
				self.token_embed.commit(token.unsqueeze(0), target_embedding.unsqueeze(0))
				return 1
		else:
			if hasattr(self.token_embed, 'train_mapping'):
				if self.dim_proj is not None:
					projected = self.dim_proj(token.unsqueeze(0)).squeeze()
					return self.token_embed.train_mapping(projected, target_embedding)
				return self.token_embed.train_mapping(token, target_embedding)

		return 0

	def train_position_embedding(
		self,
		position: int,
		target_embedding: Tensor,
	) -> int:
		"""
		Train position embedding (for LEARNED mode).

		Args:
			position: Position index
			target_embedding: Desired position embedding [embedding_bits]

		Returns:
			1 if updated, 0 if already correct
		"""
		if self.position_encoding != PositionEncoding.LEARNED:
			return 0

		pos_bits = self._get_position_bits(position)
		current = self.position_embed(pos_bits.unsqueeze(0)).squeeze()

		if not (current == target_embedding).all():
			self.position_embed.commit(pos_bits.unsqueeze(0), target_embedding.unsqueeze(0))
			return 1

		return 0

	def __repr__(self):
		return (f"RAMEmbedding({self.token_bits}→{self.embedding_bits}, "
				f"pos={self.position_encoding.name})")
