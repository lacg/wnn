"""
Tokenizers module for WNN language models.

Provides a unified interface for different tokenization strategies:
- WordTokenizer: Standard word-level tokenization (WikiText-2 compatible)
- BPETokenizer: Byte-Pair Encoding (trainable, handles OOV)
- GPT2Tokenizer: Pre-trained GPT-2 tokenizer (50,257 vocab)
- CharacterTokenizer: Character-level tokenization

Usage:
	from wnn.tokenizers import TokenizerFactory, TokenizerType

	# Create a tokenizer
	tokenizer = TokenizerFactory.create(TokenizerType.BPE, vocab_size=32000)

	# Train on corpus
	tokenizer.train(texts)

	# Use
	ids = tokenizer.encode("Hello world")
	text = tokenizer.decode(ids)
"""

from enum import IntEnum
from typing import Optional, Type

from wnn.tokenizers.base import Tokenizer
from wnn.tokenizers.word import WordTokenizer, SimpleWordTokenizer
from wnn.tokenizers.bpe import BPETokenizer, GPT2Tokenizer, CharacterTokenizer


class TokenizerType(IntEnum):
	"""
	Tokenizer type enumeration.

	Published perplexity benchmarks for comparison:
	- WORD (WikiText-2): LSTM ~65-100, AWD-LSTM ~57
	- BPE (GPT-2 style): GPT-2 Small ~29, GPT-2 Large ~22

	Note: Word-level and BPE perplexities are NOT directly comparable!
	"""
	SIMPLE_WORD = 0   # Basic regex word splitting (testing only)
	WORD = 1          # Standard WikiText-2 word-level (~33K vocab)
	BPE = 2           # Trainable BPE (configurable vocab size)
	GPT2 = 3          # Pre-trained GPT-2 BPE (50,257 vocab)
	CHARACTER = 4     # Character-level (small vocab, long sequences)


class TokenizerFactory:
	"""
	Factory for creating tokenizers.

	Provides a unified interface for creating different tokenizer types
	with sensible defaults.

	Usage:
		# Create word tokenizer
		tokenizer = TokenizerFactory.create(TokenizerType.WORD)

		# Create BPE with custom vocab size
		tokenizer = TokenizerFactory.create(TokenizerType.BPE, vocab_size=16000)

		# Load pre-trained
		tokenizer = TokenizerFactory.load(TokenizerType.BPE, "path/to/tokenizer.json")
	"""

	# Map types to classes
	_TYPE_TO_CLASS: dict[TokenizerType, Type[Tokenizer]] = {
		TokenizerType.SIMPLE_WORD: SimpleWordTokenizer,
		TokenizerType.WORD: WordTokenizer,
		TokenizerType.BPE: BPETokenizer,
		TokenizerType.GPT2: GPT2Tokenizer,
		TokenizerType.CHARACTER: CharacterTokenizer,
	}

	@classmethod
	def create(
		cls,
		tokenizer_type: TokenizerType,
		**kwargs,
	) -> Tokenizer:
		"""
		Create a new tokenizer instance.

		Args:
			tokenizer_type: Type of tokenizer to create
			**kwargs: Tokenizer-specific arguments

		Returns:
			Tokenizer instance (may need training before use)

		Examples:
			# Word tokenizer with WikiText-2 style
			tok = TokenizerFactory.create(TokenizerType.WORD)

			# BPE with 16k vocabulary
			tok = TokenizerFactory.create(TokenizerType.BPE, vocab_size=16000)

			# GPT-2 (pre-trained, ready to use)
			tok = TokenizerFactory.create(TokenizerType.GPT2)
		"""
		tokenizer_class = cls._TYPE_TO_CLASS.get(tokenizer_type)
		if tokenizer_class is None:
			raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

		return tokenizer_class(**kwargs)

	@classmethod
	def load(
		cls,
		tokenizer_type: TokenizerType,
		path: str,
	) -> Tokenizer:
		"""
		Load a pre-trained tokenizer from disk.

		Args:
			tokenizer_type: Type of tokenizer to load
			path: Path to saved tokenizer

		Returns:
			Loaded tokenizer instance
		"""
		tokenizer_class = cls._TYPE_TO_CLASS.get(tokenizer_type)
		if tokenizer_class is None:
			raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

		return tokenizer_class.load(path)

	@classmethod
	def get_class(cls, tokenizer_type: TokenizerType) -> Type[Tokenizer]:
		"""
		Get the tokenizer class for a type.

		Args:
			tokenizer_type: Tokenizer type

		Returns:
			Tokenizer class
		"""
		tokenizer_class = cls._TYPE_TO_CLASS.get(tokenizer_type)
		if tokenizer_class is None:
			raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
		return tokenizer_class


# Convenience function for quick tokenizer creation
def create_tokenizer(
	tokenizer_type: TokenizerType = TokenizerType.WORD,
	**kwargs,
) -> Tokenizer:
	"""
	Convenience function to create a tokenizer.

	Args:
		tokenizer_type: Type of tokenizer
		**kwargs: Tokenizer-specific arguments

	Returns:
		Tokenizer instance
	"""
	return TokenizerFactory.create(tokenizer_type, **kwargs)


__all__ = [
	# Base
	"Tokenizer",
	# Implementations
	"WordTokenizer",
	"SimpleWordTokenizer",
	"BPETokenizer",
	"GPT2Tokenizer",
	"CharacterTokenizer",
	# Factory
	"TokenizerFactory",
	"TokenizerType",
	"create_tokenizer",
]
