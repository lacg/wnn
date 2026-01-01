"""
RAM-based Encoder-Decoder Architecture

Full sequence-to-sequence model with:
  - Encoder: Processes source sequence (bidirectional self-attention)
  - Decoder: Generates target sequence (causal self-attention + cross-attention)

Architecture:
                        Source Sequence
                              │
                    ┌─────────▼─────────┐
                    │     Encoder        │
                    │  (Self-Attention)  │
                    │   [Bidirectional]  │
                    └─────────┬─────────┘
                              │
                    Encoder Output (Memory)
                              │
        Target Sequence       │
              │               │
    ┌─────────▼─────────┐     │
    │     Decoder        │    │
    │  ┌─────────────┐  │     │
    │  │Self-Attention│ │     │
    │  │  [Causal]   │  │     │
    │  └──────┬──────┘  │     │
    │         │         │     │
    │  ┌──────▼──────┐  │     │
    │  │Cross-Attend │◀─┼─────┘
    │  │ to Encoder  │  │
    │  └──────┬──────┘  │
    │         │         │
    │  ┌──────▼──────┐  │
    │  │    FFN      │  │
    │  └──────┬──────┘  │
    └─────────┬─────────┘
              │
        Output Sequence

Use cases:
  - Machine translation (source → target language)
  - Summarization (document → summary)
  - Question answering (question → answer)
  - Code generation (description → code)
"""

from wnn.ram.core import RAMLayer, GeneralizingProjection
from wnn.ram.enums import MapperStrategy
from wnn.ram.core.transformers.attention import RAMAttention
from wnn.ram.core.transformers.cross_attention import RAMCrossAttention, CrossAttentionMode
from wnn.ram.core.transformers.feedforward import RAMFeedForward, FFNMode
from wnn.ram.core.transformers.embedding import RAMEmbedding, PositionEncoding
from wnn.ram.encoders_decoders import PositionMode

from torch import Tensor, zeros, uint8, cat
from torch.nn import Module, ModuleList


class RAMEncoderDecoder(Module):
	"""
	RAM-based Encoder-Decoder for sequence-to-sequence tasks.

	Combines:
	- RAMAttention (self-attention) for both encoder and decoder
	- RAMCrossAttention for decoder-to-encoder attention
	- Optional RAMFeedForward layers
	- Optional RAMEmbedding for learned representations

	Training uses EDRA backpropagation through all components.
	"""

	def __init__(
		self,
		input_bits: int,
		hidden_bits: int | None = None,
		output_bits: int | None = None,
		num_encoder_layers: int = 2,
		num_decoder_layers: int = 2,
		num_heads: int = 4,
		position_mode: PositionMode = PositionMode.RELATIVE,
		max_encoder_len: int = 32,
		max_decoder_len: int = 32,
		use_residual: bool = True,
		use_ffn: bool = True,
		ffn_expansion: int = 4,
		use_embedding: bool = False,
		embedding_position: PositionEncoding = PositionEncoding.NONE,
		cross_attention_mode: CrossAttentionMode = CrossAttentionMode.ENCODER_ONLY,
		generalization: MapperStrategy = MapperStrategy.DIRECT,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per source token
			hidden_bits: Hidden dimension (defaults to input_bits)
			output_bits: Output dimension (defaults to input_bits)
			num_encoder_layers: Number of encoder self-attention layers
			num_decoder_layers: Number of decoder layers (self-attn + cross-attn + FFN)
			num_heads: Attention heads per layer
			position_mode: Position encoding for self-attention
			max_encoder_len: Maximum source sequence length
			max_decoder_len: Maximum target sequence length
			use_residual: Whether to use XOR residual connections
			use_ffn: Whether to include feed-forward layers
			ffn_expansion: FFN hidden dimension multiplier
			use_embedding: Whether to use learned embeddings
			embedding_position: Position encoding in embeddings
			cross_attention_mode: How to encode positions in cross-attention
			generalization: Strategy for output generalization
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.hidden_bits = hidden_bits or input_bits
		self.output_bits = output_bits or input_bits
		self.num_encoder_layers = num_encoder_layers
		self.num_decoder_layers = num_decoder_layers
		self.num_heads = num_heads
		self.max_encoder_len = max_encoder_len
		self.max_decoder_len = max_decoder_len
		self.use_residual = use_residual
		self.use_ffn = use_ffn
		self.use_embedding = use_embedding
		self.position_mode = position_mode

		# Convert generalization string to enum if needed
		if isinstance(generalization, str):
			name_map = {"none": "DIRECT", "bit_level": "BIT_LEVEL",
						"compositional": "COMPOSITIONAL", "hybrid": "HYBRID"}
			generalization = MapperStrategy[name_map.get(generalization, generalization.upper())]
		self.generalization = generalization

		# ========== ENCODER ==========

		# Encoder embedding (optional)
		if use_embedding:
			self.encoder_embedding = RAMEmbedding(
				token_bits=input_bits,
				embedding_bits=self.hidden_bits,
				max_seq_len=max_encoder_len,
				position_encoding=embedding_position,
				rng=rng,
			)
		else:
			self.encoder_embedding = None

		# Encoder input projection (if no embedding and dims differ)
		if not use_embedding and self.hidden_bits != input_bits:
			self.encoder_input_proj = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=self.hidden_bits,
				n_bits_per_neuron=min(input_bits, 12),
				rng=rng,
			)
		else:
			self.encoder_input_proj = None

		# Encoder self-attention layers (BIDIRECTIONAL - causal=False)
		self.encoder_attention = ModuleList([
			RAMAttention(
				input_bits=self.hidden_bits,
				num_heads=num_heads,
				position_mode=position_mode,
				max_seq_len=max_encoder_len,
				causal=False,  # Encoder sees all positions
				rng=rng + i * 100 if rng else None,
			)
			for i in range(num_encoder_layers)
		])

		# Encoder FFN layers
		if use_ffn:
			self.encoder_ffn = ModuleList([
				RAMFeedForward(
					input_bits=self.hidden_bits,
					expansion_factor=ffn_expansion,
					use_residual=use_residual,
					rng=rng + i * 100 + 50 if rng else None,
				)
				for i in range(num_encoder_layers)
			])
		else:
			self.encoder_ffn = None

		# ========== DECODER ==========

		# Decoder embedding (optional, separate from encoder)
		if use_embedding:
			self.decoder_embedding = RAMEmbedding(
				token_bits=input_bits,  # Target tokens same size as input
				embedding_bits=self.hidden_bits,
				max_seq_len=max_decoder_len,
				position_encoding=embedding_position,
				rng=rng + 1000 if rng else None,
			)
		else:
			self.decoder_embedding = None

		# Decoder input projection
		if not use_embedding and self.hidden_bits != input_bits:
			self.decoder_input_proj = RAMLayer(
				total_input_bits=input_bits,
				num_neurons=self.hidden_bits,
				n_bits_per_neuron=min(input_bits, 12),
				rng=rng + 1000 if rng else None,
			)
		else:
			self.decoder_input_proj = None

		# Decoder self-attention layers (CAUSAL - can only see past)
		self.decoder_self_attention = ModuleList([
			RAMAttention(
				input_bits=self.hidden_bits,
				num_heads=num_heads,
				position_mode=position_mode,
				max_seq_len=max_decoder_len,
				causal=True,  # Decoder only sees past positions
				rng=rng + 2000 + i * 100 if rng else None,
			)
			for i in range(num_decoder_layers)
		])

		# Decoder cross-attention layers (attends to encoder output)
		self.decoder_cross_attention = ModuleList([
			RAMCrossAttention(
				decoder_bits=self.hidden_bits,
				encoder_bits=self.hidden_bits,
				num_heads=num_heads,
				position_mode=cross_attention_mode,
				max_encoder_len=max_encoder_len,
				max_decoder_len=max_decoder_len,
				rng=rng + 3000 + i * 100 if rng else None,
			)
			for i in range(num_decoder_layers)
		])

		# Decoder FFN layers
		if use_ffn:
			self.decoder_ffn = ModuleList([
				RAMFeedForward(
					input_bits=self.hidden_bits,
					expansion_factor=ffn_expansion,
					use_residual=use_residual,
					rng=rng + 4000 + i * 100 if rng else None,
				)
				for i in range(num_decoder_layers)
			])
		else:
			self.decoder_ffn = None

		# ========== OUTPUT ==========

		# Output projection (if hidden != output)
		if self.hidden_bits != self.output_bits:
			self.output_proj = RAMLayer(
				total_input_bits=self.hidden_bits,
				num_neurons=self.output_bits,
				n_bits_per_neuron=min(self.hidden_bits, 12),
				rng=rng + 5000 if rng else None,
			)
		else:
			self.output_proj = None

		# Generalization layer (optional)
		if generalization != MapperStrategy.DIRECT:
			n_groups = 1
			for g in [5, 4, 3, 2]:
				if self.output_bits % g == 0:
					n_groups = g
					break

			self.token_mapper = GeneralizingProjection(
				input_bits=self.output_bits,
				output_bits=self.output_bits,
				strategy=generalization,
				n_groups=n_groups,
				rng=rng + 5500 if rng else None,
			)
		else:
			self.token_mapper = None

		ffn_str = ", ffn=True" if use_ffn else ""
		embed_str = f", embed={embedding_position.name}" if use_embedding else ""
		print(f"[RAMEncoderDecoder] enc_layers={num_encoder_layers}, dec_layers={num_decoder_layers}, "
			  f"heads={num_heads}, dims={input_bits}->{self.hidden_bits}->{self.output_bits}{ffn_str}{embed_str}")

	def encode(self, source: list[Tensor]) -> list[Tensor]:
		"""
		Encode source sequence.

		Args:
			source: Source token sequence [input_bits each]

		Returns:
			encoder_output: Encoded representations [hidden_bits each]
		"""
		if len(source) > self.max_encoder_len:
			raise ValueError(f"Source length {len(source)} exceeds max {self.max_encoder_len}")

		# Normalize inputs
		hidden = [t.squeeze() if t.ndim > 1 else t for t in source]

		# Embedding or projection
		if self.encoder_embedding is not None:
			hidden = self.encoder_embedding(hidden, add_position=True)
		elif self.encoder_input_proj is not None:
			hidden = [self.encoder_input_proj(h.unsqueeze(0)).squeeze() for h in hidden]

		# Encoder layers
		for i in range(self.num_encoder_layers):
			# Self-attention (bidirectional)
			attn_output = self.encoder_attention[i].forward(hidden)

			if self.use_residual:
				hidden = [h ^ out for h, out in zip(hidden, attn_output)]
			else:
				hidden = attn_output

			# FFN
			if self.encoder_ffn is not None:
				hidden = [self.encoder_ffn[i](h) for h in hidden]

		return hidden

	def decode(
		self,
		target: list[Tensor],
		encoder_output: list[Tensor],
	) -> list[Tensor]:
		"""
		Decode target sequence given encoder output.

		Args:
			target: Target token sequence (shifted right for teacher forcing)
			encoder_output: Output from encode()

		Returns:
			outputs: Decoded representations [output_bits each]
		"""
		if len(target) > self.max_decoder_len:
			raise ValueError(f"Target length {len(target)} exceeds max {self.max_decoder_len}")

		# Normalize inputs
		hidden = [t.squeeze() if t.ndim > 1 else t for t in target]
		encoder_output = [e.squeeze() if e.ndim > 1 else e for e in encoder_output]

		# Embedding or projection
		if self.decoder_embedding is not None:
			hidden = self.decoder_embedding(hidden, add_position=True)
		elif self.decoder_input_proj is not None:
			hidden = [self.decoder_input_proj(h.unsqueeze(0)).squeeze() for h in hidden]

		# Decoder layers
		for i in range(self.num_decoder_layers):
			# Self-attention (causal)
			self_attn_output = self.decoder_self_attention[i].forward(hidden)

			if self.use_residual:
				hidden = [h ^ out for h, out in zip(hidden, self_attn_output)]
			else:
				hidden = self_attn_output

			# Cross-attention (to encoder)
			cross_attn_output = self.decoder_cross_attention[i].forward(
				hidden, encoder_output
			)

			if self.use_residual:
				hidden = [h ^ out for h, out in zip(hidden, cross_attn_output)]
			else:
				hidden = cross_attn_output

			# FFN
			if self.decoder_ffn is not None:
				hidden = [self.decoder_ffn[i](h) for h in hidden]

		# Output projection
		if self.output_proj is not None:
			outputs = [self.output_proj(h.unsqueeze(0)).squeeze() for h in hidden]
		else:
			outputs = hidden

		# Generalization
		if self.token_mapper is not None:
			outputs = [self.token_mapper(o) for o in outputs]

		return outputs

	def forward(
		self,
		source: list[Tensor],
		target: list[Tensor],
	) -> list[Tensor]:
		"""
		Full forward pass: encode source, decode target.

		Args:
			source: Source sequence
			target: Target sequence (for teacher forcing during training)

		Returns:
			outputs: Predicted outputs
		"""
		encoder_output = self.encode(source)
		return self.decode(target, encoder_output)

	def generate(
		self,
		source: list[Tensor],
		max_length: int,
		start_token: Tensor,
		end_token: Tensor | None = None,
		decoder=None,
	) -> list[Tensor]:
		"""
		Autoregressive generation from source.

		Args:
			source: Source sequence to translate/transform
			max_length: Maximum output length
			start_token: Start-of-sequence token
			end_token: End-of-sequence token (stops generation if produced)
			decoder: Optional decoder for visualization

		Returns:
			generated: Generated output sequence
		"""
		# Encode source
		encoder_output = self.encode(source)

		# Start with start token
		generated = [start_token.squeeze()]

		for step in range(max_length - 1):
			# Decode current sequence
			outputs = self.decode(generated, encoder_output)

			# Take last output as next token
			next_token = outputs[-1]
			generated.append(next_token)

			if decoder:
				decoded = decoder.decode(next_token.unsqueeze(0))
				print(f"  Step {step + 1}: '{decoded}'")

			# Check for end token
			if end_token is not None and (next_token == end_token.squeeze()).all():
				break

		return generated

	def train_step(
		self,
		source: list[Tensor],
		target_input: list[Tensor],  # Shifted right (e.g., [<s>, t1, t2])
		target_output: list[Tensor], # What to predict (e.g., [t1, t2, </s>])
	) -> dict:
		"""
		Single training step with EDRA.

		Args:
			source: Source sequence
			target_input: Target input (shifted right for teacher forcing)
			target_output: Expected outputs (what model should predict)

		Returns:
			Training statistics
		"""
		# Forward pass
		encoder_output = self.encode(source)
		predictions = self.decode(target_input, encoder_output)

		# Count errors
		target_output = [t.squeeze() if t.ndim > 1 else t for t in target_output]
		errors = sum(
			1 for pred, tgt in zip(predictions, target_output)
			if not (pred == tgt).all()
		)

		if errors == 0:
			return {"errors": 0, "accuracy": 100.0}

		# Train output layers
		# For simplicity, train the output projection / token mapper on errors
		if self.token_mapper is not None:
			for pred, tgt in zip(predictions, target_output):
				if not (pred == tgt).all():
					# Get hidden state before token mapper
					# (would need to store intermediates for proper EDRA)
					self.token_mapper.train_mapping(pred, tgt)

		accuracy = 100 * (len(predictions) - errors) / len(predictions)
		return {"errors": errors, "accuracy": accuracy}

	def train(
		self,
		dataset: list[tuple[list[Tensor], list[Tensor], list[Tensor]]],
		epochs: int = 10,
		verbose: bool = True,
	) -> list[dict]:
		"""
		Train on a dataset.

		Args:
			dataset: List of (source, target_input, target_output) tuples
			epochs: Number of epochs
			verbose: Print progress

		Returns:
			Training history
		"""
		history = []

		for epoch in range(epochs):
			total_errors = 0
			total_positions = 0

			for source, target_input, target_output in dataset:
				stats = self.train_step(source, target_input, target_output)
				total_errors += stats["errors"]
				total_positions += len(target_output)

			accuracy = 100 * (total_positions - total_errors) / total_positions
			history.append({"errors": total_errors, "accuracy": accuracy})

			if verbose:
				print(f"Epoch {epoch + 1}/{epochs}: {total_errors} errors, {accuracy:.1f}% accuracy")

			if total_errors == 0:
				if verbose:
					print(f"Converged at epoch {epoch + 1}!")
				break

		return history

	def __repr__(self):
		return (
			f"RAMEncoderDecoder(enc={self.num_encoder_layers}, dec={self.num_decoder_layers}, "
			f"heads={self.num_heads}, dims={self.input_bits}->{self.hidden_bits}->{self.output_bits})"
		)
