"""
RAM Transformer Block

A single transformer block using RAM-based attention and FFN.
Uses factory pattern for component creation.
"""

from torch import Tensor, zeros, uint8

from wnn.ram.core.base import RAMSequenceModel
from wnn.ram.core.models import (
	AttentionType,
	FFNType,
	ContentMatchMode,
	AttentionCombineMode,
	NormStrategy,
)
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.core.models.discrete_normalization import DiscreteNormalization
from wnn.ram.factories.ffn import FFNFactory
from wnn.ram.factories.attention import AttentionFactory
from wnn.ram.core.models.computed_arithmetic import bits_to_int


class RAMTransformerBlock(RAMSequenceModel):
	"""
	A single RAM transformer block.

	Architecture (without normalization):
		x -> Attention(x) -> x XOR attn_out -> FFN -> x XOR ffn_out -> output

	Architecture (with normalization):
		x -> Attention(x) -> x XOR attn_out -> Norm -> FFN -> x XOR ffn_out -> Norm -> output

	Where XOR is the discrete residual connection and Norm is discrete
	normalization using ensemble voting (multiple sub-networks vote per bit).
	"""

	def __init__(
		self,
		input_bits: int,
		# Attention config
		attention_type: AttentionType = AttentionType.POSITION_ONLY,
		num_heads: int = 8,
		content_match: ContentMatchMode = ContentMatchMode.NONE,
		attention_combine: AttentionCombineMode = AttentionCombineMode.CONTENT_ONLY,
		position_mode: PositionMode = PositionMode.RELATIVE,
		causal: bool = True,
		# FFN config
		ffn_type: FFNType = FFNType.BIT_LEVEL,
		ffn_hidden_bits: int | None = None,
		ffn_constant: int = 1,
		ffn_modulo: int | None = None,
		# Normalization
		use_normalization: bool = False,
		norm_strategy: NormStrategy = NormStrategy.ENSEMBLE_VOTE,
		norm_sub_networks: int = 4,
		# Other
		use_residual: bool = True,
		max_seq_len: int = 16,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per token
			attention_type: Type of attention mechanism
			num_heads: Number of attention heads
			content_match: Content matching mode
			attention_combine: How to combine content and position
			position_mode: Position encoding mode
			causal: Use causal attention mask
			ffn_type: Type of feed-forward network
			ffn_hidden_bits: Hidden dimension for TWO_LAYER FFN
			ffn_constant: Constant for computed arithmetic FFN
			ffn_modulo: Modulo for computed arithmetic FFN
			use_normalization: Apply discrete normalization after residuals
			norm_strategy: Normalization strategy (ENSEMBLE_VOTE or BIT_BALANCE)
			norm_sub_networks: Number of sub-networks for ensemble voting
			use_residual: Use XOR residual connections
			max_seq_len: Maximum sequence length
			rng: Random seed
		"""
		super().__init__()

		# Store config for serialization
		self.input_bits = input_bits
		self.attention_type = attention_type
		self.num_heads = num_heads
		self.content_match = content_match
		self.attention_combine = attention_combine
		self.position_mode = position_mode
		self.causal = causal
		self.ffn_type = ffn_type
		self.ffn_hidden_bits = ffn_hidden_bits
		self.ffn_constant = ffn_constant
		self.ffn_modulo = ffn_modulo
		self.use_normalization = use_normalization
		self.norm_strategy = norm_strategy
		self.norm_sub_networks = norm_sub_networks
		self.use_residual = use_residual
		self.max_seq_len = max_seq_len
		self.rng = rng

		# Build attention using factory
		self.attention = AttentionFactory.create(
			attention_type=attention_type,
			input_bits=input_bits,
			num_heads=num_heads,
			content_match=content_match,
			attention_combine=attention_combine,
			position_mode=position_mode,
			causal=causal,
			max_seq_len=max_seq_len,
			rng=rng,
		)

		# Build FFN using factory
		self.ffn = FFNFactory.create(
			ffn_type=ffn_type,
			input_bits=input_bits,
			hidden_bits=ffn_hidden_bits,
			constant=ffn_constant,
			modulo=ffn_modulo,
			rng=rng + 1000 if rng else None,
		)

		# Build normalization layers (one after attention, one after FFN)
		if use_normalization and norm_strategy != NormStrategy.NONE:
			self.norm_attn = DiscreteNormalization(
				input_bits=input_bits,
				strategy=norm_strategy,
				num_sub_networks=norm_sub_networks,
				rng=rng + 2000 if rng else None,
			)
			self.norm_ffn = DiscreteNormalization(
				input_bits=input_bits,
				strategy=norm_strategy,
				num_sub_networks=norm_sub_networks,
				rng=rng + 3000 if rng else None,
			)
		else:
			self.norm_attn = None
			self.norm_ffn = None

		# Summary
		attn_name = attention_type.name
		ffn_name = ffn_type.name
		residual_str = "+residual" if use_residual else ""
		norm_str = f"+norm({norm_strategy.name})" if use_normalization else ""
		print(f"[RAMTransformerBlock] {input_bits}b, attn={attn_name}, "
				f"ffn={ffn_name}{residual_str}{norm_str}")

	def forward(self, tokens: list[Tensor]) -> list[Tensor]:
		"""
		Forward pass through the transformer block.

		Args:
			tokens: List of token tensors

		Returns:
			outputs: Transformed tokens
		"""
		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]

		# Attention
		attn_out = self.attention.forward(tokens)
		attn_out = [t.squeeze() if t.ndim > 1 else t for t in attn_out]

		# Residual connection (XOR)
		if self.use_residual:
			attn_out = [t ^ r for t, r in zip(tokens, attn_out)]

		# Post-attention normalization
		if self.norm_attn is not None:
			attn_out = [self.norm_attn(t) for t in attn_out]

		# FFN
		if self.ffn is not None:
			ffn_out = []
			for t in attn_out:
				from wnn.ram.core import GeneralizingProjection
				if isinstance(self.ffn, GeneralizingProjection):
					out = self.ffn(t)
				else:
					out = self.ffn(t.unsqueeze(0)).squeeze()
				ffn_out.append(out)

			# Residual connection (XOR)
			if self.use_residual:
				ffn_out = [t ^ r for t, r in zip(attn_out, ffn_out)]

			# Post-FFN normalization
			if self.norm_ffn is not None:
				ffn_out = [self.norm_ffn(t) for t in ffn_out]

			return ffn_out

		return attn_out

	def train_block(
		self,
		input_tokens: list[Tensor],
		target_tokens: list[Tensor] | None = None,
		attention_pattern: str = "copy",
	) -> int:
		"""
		Train the transformer block.

		Args:
			input_tokens: Input sequence
			target_tokens: Target output (if None, inferred from pattern)
			attention_pattern: What attention pattern to train

		Returns:
			corrections: Number of corrections made
		"""
		input_tokens = [t.squeeze() if t.ndim > 1 else t for t in input_tokens]
		n = len(input_tokens)
		corrections = 0

		# Infer targets if not provided
		if target_tokens is None:
			match attention_pattern:
				case "copy":
					target_tokens = input_tokens
				case "shift":
					target_tokens = [input_tokens[0]] + input_tokens[:-1]
				case "reverse":
					target_tokens = input_tokens[::-1]
				case "sort":
					sorted_indices = sorted(range(n), key=lambda i: bits_to_int(input_tokens[i]))
					target_tokens = [input_tokens[i] for i in sorted_indices]
				case _:
					target_tokens = input_tokens

		target_tokens = [t.squeeze() if t.ndim > 1 else t for t in target_tokens]

		# Train attention (if trainable)
		if hasattr(self.attention, 'train_value_projection'):
			corrections += self.attention.train_value_projection(input_tokens)

			match attention_pattern:
				case "copy":
					for pos in range(n):
						weights = [0.0] * n
						weights[pos] = 1.0
						corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

				case "shift":
					for pos in range(n):
						weights = [0.0] * n
						if pos > 0:
							weights[pos - 1] = 1.0
						else:
							weights[0] = 1.0
						corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

				case "reverse":
					for pos in range(n):
						weights = [0.0] * n
						weights[n - 1 - pos] = 1.0
						corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

		# Train FFN (if trainable)
		if self.ffn is not None and hasattr(self.ffn, 'train_mapping'):
			for inp, tgt in zip(input_tokens, target_tokens):
				corrections += self.ffn.train_mapping(inp, tgt)

		return corrections

	def train_normalization(
		self,
		tokens: list[Tensor],
		targets: list[Tensor] | None = None,
	) -> int:
		"""
		Train normalization layers to preserve token identity.

		For ENSEMBLE_VOTE, trains all sub-networks to output the target.
		If targets is None, trains toward identity (input = output).

		Args:
			tokens: Input tokens
			targets: Target outputs (None = identity mapping)

		Returns:
			Total errors across normalization layers
		"""
		if targets is None:
			targets = tokens

		tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
		targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

		total_errors = 0

		for t, tgt in zip(tokens, targets):
			if self.norm_attn is not None:
				total_errors += self.norm_attn.commit_ensemble(t, tgt)
			if self.norm_ffn is not None:
				total_errors += self.norm_ffn.commit_ensemble(t, tgt)

		return total_errors

	# -------------------------
	# Serialization
	# -------------------------

	def get_config(self) -> dict:
		"""Get configuration for serialization."""
		return {
			'input_bits': self.input_bits,
			'attention_type': self.attention_type.value if hasattr(self.attention_type, 'value') else self.attention_type,
			'num_heads': self.num_heads,
			'content_match': self.content_match.value if hasattr(self.content_match, 'value') else self.content_match,
			'attention_combine': self.attention_combine.value if hasattr(self.attention_combine, 'value') else self.attention_combine,
			'position_mode': self.position_mode.value if hasattr(self.position_mode, 'value') else self.position_mode,
			'causal': self.causal,
			'ffn_type': self.ffn_type.value if hasattr(self.ffn_type, 'value') else self.ffn_type,
			'ffn_hidden_bits': self.ffn_hidden_bits,
			'ffn_constant': self.ffn_constant,
			'ffn_modulo': self.ffn_modulo,
			'use_normalization': self.use_normalization,
			'norm_strategy': self.norm_strategy.value if hasattr(self.norm_strategy, 'value') else self.norm_strategy,
			'norm_sub_networks': self.norm_sub_networks,
			'use_residual': self.use_residual,
			'max_seq_len': self.max_seq_len,
			'rng': self.rng,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMTransformerBlock":
		"""Create model from configuration."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)
