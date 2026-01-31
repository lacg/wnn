"""
Base tokenizer interface for WNN language models.

This module provides an abstract interface for tokenizers, enabling
consistent handling of different tokenization strategies (word-level,
BPE, character-level, etc.) across the codebase.

Usage:
	from wnn.text_tokenizers import TokenizerFactory, TokenizerType

	# Create a tokenizer
	tokenizer = TokenizerFactory.create(TokenizerType.BPE)

	# Train on corpus (if needed)
	tokenizer.train(texts)

	# Encode/decode
	ids = tokenizer.encode("Hello world")
	text = tokenizer.decode(ids)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


class Tokenizer(ABC):
	"""
	Abstract base class for all tokenizers.

	A tokenizer converts text to token IDs and back. Different implementations
	handle different tokenization strategies (word-level, BPE, etc.).

	Key properties:
	- vocab_size: Number of unique tokens in vocabulary
	- unk_token_id: ID for unknown/OOV tokens (None if no UNK)
	- pad_token_id: ID for padding (None if no padding)

	Subclasses must implement:
	- encode(): Convert text to token IDs
	- decode(): Convert token IDs back to text
	- vocab_size property
	- _train_impl(): Training logic (if trainable)
	"""

	def __init__(self, name: str = "base"):
		"""
		Initialize tokenizer.

		Args:
			name: Human-readable name for this tokenizer
		"""
		self._name = name
		self._is_trained = False

	@property
	def name(self) -> str:
		"""Human-readable name for this tokenizer."""
		return self._name

	@property
	def is_trained(self) -> bool:
		"""Whether the tokenizer has been trained or loaded."""
		return self._is_trained

	@property
	@abstractmethod
	def vocab_size(self) -> int:
		"""Number of unique tokens in vocabulary."""
		...

	@property
	def unk_token_id(self) -> Optional[int]:
		"""ID for unknown/OOV tokens. None if tokenizer handles all inputs."""
		return None

	@property
	def pad_token_id(self) -> Optional[int]:
		"""ID for padding token. None if no padding support."""
		return None

	@abstractmethod
	def encode(self, text: str) -> list[int]:
		"""
		Convert text to a list of token IDs.

		Args:
			text: Input text string

		Returns:
			List of integer token IDs
		"""
		...

	@abstractmethod
	def decode(self, ids: list[int]) -> str:
		"""
		Convert token IDs back to text.

		Args:
			ids: List of integer token IDs

		Returns:
			Decoded text string
		"""
		...

	def encode_batch(self, texts: list[str]) -> list[list[int]]:
		"""
		Encode multiple texts. Override for optimized batch processing.

		Args:
			texts: List of input texts

		Returns:
			List of token ID lists
		"""
		return [self.encode(text) for text in texts]

	def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
		"""
		Decode multiple token sequences. Override for optimized batch processing.

		Args:
			ids_batch: List of token ID lists

		Returns:
			List of decoded texts
		"""
		return [self.decode(ids) for ids in ids_batch]

	def token_to_id(self, token: str) -> Optional[int]:
		"""
		Get ID for a single token. Returns None if not in vocabulary.

		Args:
			token: Token string

		Returns:
			Token ID or None if unknown
		"""
		# Default implementation - subclasses may override for efficiency
		ids = self.encode(token)
		return ids[0] if len(ids) == 1 else None

	def id_to_token(self, id: int) -> Optional[str]:
		"""
		Get token string for an ID. Returns None if invalid ID.

		Args:
			id: Token ID

		Returns:
			Token string or None if invalid
		"""
		# Default implementation - subclasses may override for efficiency
		try:
			return self.decode([id])
		except (IndexError, KeyError):
			return None

	def train(self, texts: list[str], **kwargs) -> None:
		"""
		Train the tokenizer on a corpus of texts.

		Not all tokenizers are trainable (e.g., pre-trained BPE).
		Call is_trainable to check before calling.

		Args:
			texts: List of training texts
			**kwargs: Tokenizer-specific training parameters
		"""
		if not self.is_trainable:
			raise NotImplementedError(f"{self.name} tokenizer is not trainable")
		self._train_impl(texts, **kwargs)
		self._is_trained = True

	@property
	def is_trainable(self) -> bool:
		"""Whether this tokenizer can be trained on a corpus."""
		return False

	def _train_impl(self, texts: list[str], **kwargs) -> None:
		"""
		Internal training implementation. Override in trainable subclasses.

		Args:
			texts: List of training texts
			**kwargs: Tokenizer-specific parameters
		"""
		raise NotImplementedError

	def save(self, path: Union[str, Path]) -> None:
		"""
		Save tokenizer state to disk.

		Args:
			path: Directory or file path to save to
		"""
		raise NotImplementedError(f"{self.name} does not support saving")

	@classmethod
	def load(cls, path: Union[str, Path]) -> "Tokenizer":
		"""
		Load tokenizer state from disk.

		Args:
			path: Directory or file path to load from

		Returns:
			Loaded tokenizer instance
		"""
		raise NotImplementedError(f"{cls.__name__} does not support loading")

	def get_vocab(self) -> dict[str, int]:
		"""
		Get the full vocabulary as a dict mapping tokens to IDs.

		Returns:
			Dictionary of token -> ID mappings
		"""
		raise NotImplementedError(f"{self.name} does not expose vocabulary")

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(name={self.name!r}, vocab_size={self.vocab_size})"
