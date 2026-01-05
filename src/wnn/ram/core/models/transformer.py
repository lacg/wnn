"""
RAM Transformer

Full RAM Transformer with multiple stacked blocks.
"""

from torch import Tensor
from torch.nn import ModuleList

from wnn.ram.core.base import RAMSequenceModel
from wnn.ram.core.models import (
	AttentionType,
	FFNType,
	ContentMatchMode,
	AttentionCombineMode,
)
from wnn.ram.encoders_decoders import PositionMode
from wnn.ram.core.models.transformer_block import RAMTransformerBlock


class RAMTransformer(RAMSequenceModel):
	"""
	Full RAM Transformer with multiple stacked blocks.

	Architecture:
		Input -> Block1 -> Block2 -> ... -> BlockN -> Output

	Supports different block configurations for different layers.
	"""

	def __init__(
		self,
		input_bits: int,
		num_blocks: int = 2,
		# Default block config
		attention_type: AttentionType = AttentionType.POSITION_ONLY,
		num_heads: int = 8,
		content_match: ContentMatchMode = ContentMatchMode.NONE,
		position_mode: PositionMode = PositionMode.RELATIVE,
		causal: bool = True,
		ffn_type: FFNType = FFNType.BIT_LEVEL,
		ffn_constant: int = 1,
		ffn_modulo: int | None = None,
		use_residual: bool = True,
		max_seq_len: int = 16,
		# Per-block overrides
		block_configs: list[dict] | None = None,
		rng: int | None = None,
	):
		"""
		Args:
			input_bits: Bits per token
			num_blocks: Number of transformer blocks
			attention_type: Default attention type
			num_heads: Default number of heads
			content_match: Default content matching mode
			position_mode: Default position encoding
			causal: Default causal mask setting
			ffn_type: Default FFN type
			ffn_constant: Default constant for computed arithmetic FFN
			ffn_modulo: Default modulo for computed arithmetic FFN
			use_residual: Default residual connection setting
			max_seq_len: Maximum sequence length
			block_configs: List of per-block config overrides
			rng: Random seed
		"""
		super().__init__()

		self.input_bits = input_bits
		self.num_blocks = num_blocks
		self.max_seq_len = max_seq_len
		# Store defaults for serialization
		self.attention_type = attention_type
		self.num_heads = num_heads
		self.content_match = content_match
		self.position_mode = position_mode
		self.causal = causal
		self.ffn_type = ffn_type
		self.ffn_constant = ffn_constant
		self.ffn_modulo = ffn_modulo
		self.use_residual = use_residual
		self.block_configs = block_configs

		# Build blocks
		self.blocks = ModuleList()

		for i in range(num_blocks):
			# Get per-block config or use defaults
			cfg = block_configs[i] if block_configs and i < len(block_configs) else {}

			block = RAMTransformerBlock(
				input_bits=input_bits,
				attention_type=cfg.get('attention_type', attention_type),
				num_heads=cfg.get('num_heads', num_heads),
				content_match=cfg.get('content_match', content_match),
				attention_combine=cfg.get('attention_combine', AttentionCombineMode.CONTENT_ONLY),
				position_mode=cfg.get('position_mode', position_mode),
				causal=cfg.get('causal', causal),
				ffn_type=cfg.get('ffn_type', ffn_type),
				ffn_constant=cfg.get('ffn_constant', ffn_constant),
				ffn_modulo=cfg.get('ffn_modulo', ffn_modulo),
				use_residual=cfg.get('use_residual', use_residual),
				max_seq_len=max_seq_len,
				rng=rng + i * 10000 if rng else None,
			)
			self.blocks.append(block)

		print(f"[RAMTransformer] {num_blocks} blocks, {input_bits}b tokens")

	def forward(self, tokens: list[Tensor]) -> list[Tensor]:
		"""Forward pass through all blocks."""
		x = tokens
		for block in self.blocks:
			x = block.forward(x)
		return x

	def train_transformer(
		self,
		input_tokens: list[Tensor],
		target_tokens: list[Tensor] | None = None,
		attention_pattern: str = "copy",
	) -> int:
		"""Train all blocks."""
		corrections = 0
		for block in self.blocks:
			corrections += block.train_block(input_tokens, target_tokens, attention_pattern)
		return corrections

	# Serialization support
	def get_config(self) -> dict:
		"""Get configuration dict for model recreation."""
		return {
			'input_bits': self.input_bits,
			'num_blocks': self.num_blocks,
			'attention_type': self.attention_type,
			'num_heads': self.num_heads,
			'content_match': self.content_match,
			'position_mode': self.position_mode,
			'causal': self.causal,
			'ffn_type': self.ffn_type,
			'ffn_constant': self.ffn_constant,
			'ffn_modulo': self.ffn_modulo,
			'use_residual': self.use_residual,
			'max_seq_len': self.max_seq_len,
			'block_configs': self.block_configs,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMTransformer":
		"""Create model from configuration dict."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)

	@classmethod
	def load(cls, path: str, device: str = 'cpu') -> "RAMTransformer":
		"""Load model from file."""
		from wnn.ram.core.serialization import load_model
		return load_model(path, model_class=cls, device=device)
