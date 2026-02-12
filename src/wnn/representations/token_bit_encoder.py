"""
Token Bit Encoders — map token IDs to bit vectors for RAM neuron addressing.

This is the Layer 2 encoding in the pipeline:
	Layer 1 (Tokenizer): "the cat sat" → [262, 3797, 3332]  (text → token IDs)
	Layer 2 (BitEncoder): [262, 3797, 3332] → [[0,1,0,...], ...]  (token IDs → bits)

Layer 1 is fixed for reproducibility (tiktoken/GPT-2 BPE).
Layer 2 is the research variable — different encodings change how
token IDs map to RAM neuron addresses, affecting generalization.

Available encoders:
- BinaryTokenEncoder: Standard binary (token_id → binary bits). Default.
- GrayCodeTokenEncoder: Gray code (adjacent IDs differ by 1 bit).

The encoder is injected into RAMLM/BitwiseRAMLM via composition:
	model = BitwiseRAMLM(encoder=GrayCodeTokenEncoder(vocab_size=50257))

Usage:
	from wnn.representations import BinaryTokenEncoder, GrayCodeTokenEncoder

	enc = GrayCodeTokenEncoder(vocab_size=50257)
	bits = enc.encode_token(262)          # [16] bool tensor
	bits = enc.encode_tokens_batch(ids)   # [N, 16] bool tensor
"""

from enum import IntEnum
from math import ceil, log2
from typing import Protocol, runtime_checkable

from torch import arange, float32, long, Tensor
from torch.nn import Module


def bits_needed(vocab_size: int) -> int:
	"""Minimum bits to represent vocab_size distinct values."""
	if vocab_size <= 1:
		return 1
	return ceil(log2(vocab_size))


@runtime_checkable
class TokenBitEncoder(Protocol):
	"""Protocol for token ID → bit vector encoding.

	Any class that implements these methods can be used as an encoder
	in RAMLM and BitwiseRAMLM. Concrete implementations should be
	torch.nn.Module subclasses so they participate in .to(device).
	"""

	@property
	def bits_per_token(self) -> int:
		"""Number of bits per encoded token."""
		...

	def encode_token(self, token_id: int) -> Tensor:
		"""Encode a single token ID to a bit vector.

		Args:
			token_id: Integer token ID (0 to vocab_size-1)

		Returns:
			[bits_per_token] boolean tensor
		"""
		...

	def encode_tokens_batch(self, token_ids: Tensor) -> Tensor:
		"""Vectorized encoding of multiple token IDs.

		Args:
			token_ids: [N] int64 tensor of token IDs

		Returns:
			[N, bits_per_token] boolean tensor
		"""
		...


class BinaryTokenEncoder(Module):
	"""Standard binary encoding: token_id → binary representation.

	This is the default encoder, equivalent to the inline encoding
	that RAMLM/BitwiseRAMLM have always used.

	Properties:
		- Minimal bits: ceil(log2(vocab_size))
		- Arbitrary Hamming distance between adjacent IDs
		- Deterministic, reproducible, zero training cost
	"""

	def __init__(self, vocab_size: int = 50257):
		super().__init__()
		self._bits_per_token = bits_needed(vocab_size)
		self.register_buffer(
			"_bit_positions",
			arange(self._bits_per_token - 1, -1, -1, dtype=long),
		)

	@property
	def bits_per_token(self) -> int:
		return self._bits_per_token

	def encode_token(self, token_id: int) -> Tensor:
		return ((token_id >> self._bit_positions) & 1).bool()

	def encode_tokens_batch(self, token_ids: Tensor) -> Tensor:
		# [N, 1] >> [bits_per_token] → [N, bits_per_token]
		return ((token_ids.unsqueeze(-1) >> self._bit_positions) & 1).bool()

	def __repr__(self) -> str:
		return f"BinaryTokenEncoder(bits={self._bits_per_token})"


class GrayCodeTokenEncoder(Module):
	"""Gray code encoding: adjacent token IDs differ by exactly 1 bit.

	Conversion: gray = token_id ^ (token_id >> 1)

	Properties:
		- Same bits as binary: ceil(log2(vocab_size))
		- Adjacent IDs always differ by exactly 1 bit
		- Smoother address space for GA/TS connectivity optimization
		- Deterministic, reproducible, zero training cost

	Why this matters for RAM neurons:
		Binary: token 127 (0111111) → 128 (10000000) = 8 bits flip
		Gray:   token 127 (1000000) → 128 (11000000) = 1 bit flips

		When similar contexts have adjacent token IDs, Gray code ensures
		they map to nearby addresses. A neuron that doesn't observe the
		differing bit treats them identically — free generalization.
	"""

	def __init__(self, vocab_size: int = 50257):
		super().__init__()
		self._bits_per_token = bits_needed(vocab_size)
		self.register_buffer(
			"_bit_positions",
			arange(self._bits_per_token - 1, -1, -1, dtype=long),
		)

	@property
	def bits_per_token(self) -> int:
		return self._bits_per_token

	def _to_gray(self, token_ids: Tensor) -> Tensor:
		"""Convert integer token IDs to Gray code."""
		return token_ids ^ (token_ids >> 1)

	def encode_token(self, token_id: int) -> Tensor:
		gray = token_id ^ (token_id >> 1)
		return ((gray >> self._bit_positions) & 1).bool()

	def encode_tokens_batch(self, token_ids: Tensor) -> Tensor:
		gray = self._to_gray(token_ids)
		return ((gray.unsqueeze(-1) >> self._bit_positions) & 1).bool()

	def __repr__(self) -> str:
		return f"GrayCodeTokenEncoder(bits={self._bits_per_token})"


class TokenBitEncoderType(IntEnum):
	"""Available token bit encoding strategies."""
	BINARY = 0     # Standard binary (default, backward compatible)
	GRAY_CODE = 1  # Gray code (adjacent IDs differ by 1 bit)


def create_token_bit_encoder(
	encoder_type: TokenBitEncoderType = TokenBitEncoderType.BINARY,
	vocab_size: int = 50257,
) -> Module:
	"""Factory function for creating token bit encoders.

	Args:
		encoder_type: Which encoding strategy to use
		vocab_size: Size of the token vocabulary

	Returns:
		A TokenBitEncoder-compatible Module
	"""
	if encoder_type == TokenBitEncoderType.BINARY:
		return BinaryTokenEncoder(vocab_size=vocab_size)
	elif encoder_type == TokenBitEncoderType.GRAY_CODE:
		return GrayCodeTokenEncoder(vocab_size=vocab_size)
	else:
		raise ValueError(f"Unknown encoder type: {encoder_type}")
