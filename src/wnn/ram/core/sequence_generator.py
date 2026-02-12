"""
Sequence Generator for RAM Models

Wraps any RAM model to provide autoregressive generation capabilities:
- Greedy decoding
- Beam search
- Sampling (with temperature)
- Top-k sampling
- Streaming versions of all methods

This separates generation concerns from model architecture, allowing
the same generation code to work with RAMSeq2Seq, RAMEncoderDecoder, etc.

Usage:
	# Decoder-only model (like RAMSeq2Seq)
	generator = SequenceGenerator(model)
	result = generator.decode(start_token, max_len=10)

	# Encoder-decoder model
	generator = SequenceGenerator(model, encoder_output=encoder_output)
	result = generator.decode(start_token, max_len=10)

	# Streaming
	for token in generator.stream_decode(start_token, max_len=10):
		print(f"Token: {token.token}")
"""

from typing import Callable, Generator
from dataclasses import dataclass
from torch import Tensor

from wnn.ram.core.generation import (
	GenerationResult,
	BeamCandidate,
	StreamToken,
	greedy_decode,
	beam_search,
	sample_decode,
	top_k_decode,
	stream_greedy_decode,
	stream_sample_decode,
	stream_top_k_decode,
	collect_stream,
)


@dataclass
class SequenceGenerator:
	"""
	Wrapper that provides generation methods for any RAM model.

	The model must have a callable interface that takes:
	- tokens: list[Tensor] - the current token sequence
	- context: list[Tensor] | None - encoder output (for encoder-decoder models)

	And returns:
	- list[Tensor] - output tokens for each position
	"""

	model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]]
	encoder_output: list[Tensor] | None = None

	@classmethod
	def from_seq2seq(cls, model: "RAMSeq2Seq") -> "SequenceGenerator":
		"""Create generator from a decoder-only model."""
		def model_fn(tokens, context=None):
			return model.forward(tokens)
		return cls(model=model_fn, encoder_output=None)

	@classmethod
	def from_encoder_decoder(
		cls,
		model: "RAMEncoderDecoder",
		encoder_output: list[Tensor],
	) -> "SequenceGenerator":
		"""Create generator from an encoder-decoder model with pre-computed encoder output."""
		def model_fn(tokens, context):
			return model.decode(tokens, context)
		return cls(model=model_fn, encoder_output=encoder_output)

	# =========================================================================
	# Non-streaming generation
	# =========================================================================

	def decode(
		self,
		start_token: Tensor,
		max_len: int,
		eos_value: int | None = None,
		get_score: Callable[[Tensor], float] | None = None,
	) -> GenerationResult:
		"""
		Greedy decoding: always pick the most likely next token.

		Args:
			start_token: Start-of-sequence token
			max_len: Maximum generation length
			eos_value: End-of-sequence token value (stops generation)
			get_score: Function to score tokens (for logging only)

		Returns:
			GenerationResult with generated sequence
		"""
		return greedy_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			eos_value=eos_value,
			get_score=get_score,
		)

	def search(
		self,
		start_token: Tensor,
		beam_width: int,
		max_len: int,
		vocab_size: int | None = None,
		eos_value: int | None = None,
		length_penalty: float = 0.0,
		score_fn: Callable[[Tensor, Tensor], float] | None = None,
	) -> GenerationResult:
		"""
		Beam search decoding: maintain multiple candidate sequences.

		Args:
			start_token: Start-of-sequence token
			beam_width: Number of beams to maintain
			max_len: Maximum generation length
			vocab_size: Vocabulary size (2^token_bits)
			eos_value: End-of-sequence token value
			length_penalty: Penalize shorter sequences (0.0 = no penalty)
			score_fn: Custom scoring function(output_token, all_outputs) -> score

		Returns:
			GenerationResult with best sequence and all candidates
		"""
		return beam_search(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			beam_width=beam_width,
			max_len=max_len,
			vocab_size=vocab_size,
			eos_value=eos_value,
			length_penalty=length_penalty,
			score_fn=score_fn,
		)

	def sample(
		self,
		start_token: Tensor,
		max_len: int,
		temperature: float = 1.0,
		eos_value: int | None = None,
	) -> GenerationResult:
		"""
		Sampling-based decoding with temperature.

		Args:
			start_token: Start token
			max_len: Maximum length
			temperature: Sampling temperature (higher = more random)
			eos_value: End-of-sequence value

		Returns:
			GenerationResult with sampled sequence
		"""
		return sample_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			temperature=temperature,
			eos_value=eos_value,
		)

	def top_k(
		self,
		start_token: Tensor,
		max_len: int,
		k: int = 5,
		eos_value: int | None = None,
	) -> GenerationResult:
		"""
		Top-k decoding: sample from top k most likely tokens.

		Args:
			start_token: Start token
			max_len: Maximum length
			k: Number of top candidates to sample from
			eos_value: End-of-sequence value

		Returns:
			GenerationResult
		"""
		return top_k_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			k=k,
			eos_value=eos_value,
		)

	# =========================================================================
	# Streaming generation
	# =========================================================================

	def stream_decode(
		self,
		start_token: Tensor,
		max_len: int,
		eos_value: int | None = None,
		get_score: Callable[[Tensor], float] | None = None,
	) -> Generator[StreamToken, None, None]:
		"""
		Streaming greedy decoding: yields tokens as they are generated.

		Args:
			start_token: Start-of-sequence token
			max_len: Maximum generation length
			eos_value: End-of-sequence token value (stops generation)
			get_score: Function to score tokens

		Yields:
			StreamToken for each generated token

		Example:
			for token_info in generator.stream_decode(start, max_len=10):
				print(f"Step {token_info.step}: {token_info.token}")
				if should_stop(token_info.token):
					break  # Early termination
		"""
		yield from stream_greedy_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			eos_value=eos_value,
			get_score=get_score,
		)

	def stream_sample(
		self,
		start_token: Tensor,
		max_len: int,
		temperature: float = 1.0,
		eos_value: int | None = None,
	) -> Generator[StreamToken, None, None]:
		"""
		Streaming sampling-based decoding.

		Args:
			start_token: Start token
			max_len: Maximum length
			temperature: Sampling temperature (higher = more random)
			eos_value: End-of-sequence value

		Yields:
			StreamToken for each sampled token
		"""
		yield from stream_sample_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			temperature=temperature,
			eos_value=eos_value,
		)

	def stream_top_k(
		self,
		start_token: Tensor,
		max_len: int,
		k: int = 5,
		eos_value: int | None = None,
	) -> Generator[StreamToken, None, None]:
		"""
		Streaming top-k decoding.

		Args:
			start_token: Start token
			max_len: Maximum length
			k: Number of top candidates to sample from
			eos_value: End-of-sequence value

		Yields:
			StreamToken for each generated token
		"""
		yield from stream_top_k_decode(
			model=self.model,
			encoder_output=self.encoder_output,
			start_token=start_token,
			max_len=max_len,
			k=k,
			eos_value=eos_value,
		)

	def collect(
		self,
		stream: Generator[StreamToken, None, None],
	) -> GenerationResult:
		"""
		Collect all tokens from a streaming generator into a GenerationResult.

		Args:
			stream: A streaming generation generator

		Returns:
			GenerationResult with all collected tokens
		"""
		return collect_stream(stream)
