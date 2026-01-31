"""
BPE (Byte-Pair Encoding) tokenizer for WNN language models.

BPE is the tokenization method used by GPT-2, GPT-3, and many modern LLMs.
Key advantages over word-level tokenization:
- Fixed vocabulary size regardless of corpus
- Handles OOV words by decomposition ("unfamiliar" → "un" + "familiar")
- Subword units capture morphological structure

This module provides:
- BPETokenizer: Trainable BPE tokenizer using HuggingFace tokenizers
- GPT2Tokenizer: Pre-trained GPT-2 tokenizer for comparison
"""

from pathlib import Path
from typing import Optional, Union

from wnn.text_tokenizers.base import Tokenizer


class BPETokenizer(Tokenizer):
	"""
	Trainable BPE tokenizer using HuggingFace tokenizers library.

	This tokenizer learns subword units from a corpus using the BPE algorithm.
	Once trained, it can tokenize any text without OOV issues.

	Features:
	- Configurable vocabulary size (typically 8k-50k)
	- Special tokens support (<unk>, <pad>, <eos>, etc.)
	- Fast training and inference via Rust backend
	- Handles any language/domain

	Usage:
		tokenizer = BPETokenizer(vocab_size=32000)
		tokenizer.train(texts)

		ids = tokenizer.encode("Hello world")
		text = tokenizer.decode(ids)
	"""

	# Special tokens
	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"
	EOS_TOKEN = "<eos>"
	BOS_TOKEN = "<bos>"

	def __init__(
		self,
		name: str = "bpe",
		vocab_size: int = 32000,
		min_frequency: int = 2,
		show_progress: bool = True,
	):
		"""
		Initialize BPE tokenizer.

		Args:
			name: Tokenizer name
			vocab_size: Target vocabulary size
			min_frequency: Minimum frequency for a token pair to be merged
			show_progress: Show progress during training
		"""
		super().__init__(name=name)
		self._vocab_size = vocab_size
		self._min_frequency = min_frequency
		self._show_progress = show_progress
		self._tokenizer = None  # HuggingFace tokenizer instance

	@property
	def vocab_size(self) -> int:
		"""Number of unique tokens in vocabulary."""
		if self._tokenizer is None:
			return self._vocab_size  # Target size before training
		return self._tokenizer.get_vocab_size()

	@property
	def unk_token_id(self) -> Optional[int]:
		"""ID for unknown token."""
		if self._tokenizer is None:
			return None
		return self._tokenizer.token_to_id(self.UNK_TOKEN)

	@property
	def pad_token_id(self) -> Optional[int]:
		"""ID for padding token."""
		if self._tokenizer is None:
			return None
		return self._tokenizer.token_to_id(self.PAD_TOKEN)

	@property
	def eos_token_id(self) -> Optional[int]:
		"""ID for end-of-sequence token."""
		if self._tokenizer is None:
			return None
		return self._tokenizer.token_to_id(self.EOS_TOKEN)

	@property
	def is_trainable(self) -> bool:
		"""BPE tokenizer is trainable."""
		return True

	def encode(self, text: str) -> list[int]:
		"""
		Convert text to token IDs.

		Args:
			text: Input text

		Returns:
			List of token IDs
		"""
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer not trained. Call train() first.")
		encoding = self._tokenizer.encode(text)
		return encoding.ids

	def decode(self, ids: list[int]) -> str:
		"""
		Convert token IDs back to text.

		Args:
			ids: List of token IDs

		Returns:
			Decoded text
		"""
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer not trained. Call train() first.")
		return self._tokenizer.decode(ids)

	def encode_batch(self, texts: list[str]) -> list[list[int]]:
		"""Encode multiple texts efficiently."""
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer not trained. Call train() first.")
		encodings = self._tokenizer.encode_batch(texts)
		return [e.ids for e in encodings]

	def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
		"""Decode multiple token sequences efficiently."""
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer not trained. Call train() first.")
		return self._tokenizer.decode_batch(ids_batch)

	def token_to_id(self, token: str) -> Optional[int]:
		"""Get ID for a token."""
		if self._tokenizer is None:
			return None
		return self._tokenizer.token_to_id(token)

	def id_to_token(self, id: int) -> Optional[str]:
		"""Get token for an ID."""
		if self._tokenizer is None:
			return None
		return self._tokenizer.id_to_token(id)

	def _train_impl(self, texts: list[str], **kwargs) -> None:
		"""
		Train BPE tokenizer on corpus.

		Args:
			texts: List of training texts
			**kwargs: Additional training parameters
		"""
		try:
			from tokenizers import Tokenizer as HFTokenizer
			from tokenizers.models import BPE
			from tokenizers.trainers import BpeTrainer
			from tokenizers.pre_tokenizers import Whitespace
			from tokenizers.processors import TemplateProcessing
		except ImportError:
			raise ImportError(
				"BPE tokenizer requires 'tokenizers' library. "
				"Install with: pip install tokenizers"
			)

		# Override vocab_size if provided
		vocab_size = kwargs.get("vocab_size", self._vocab_size)
		min_frequency = kwargs.get("min_frequency", self._min_frequency)

		# Create BPE tokenizer
		self._tokenizer = HFTokenizer(BPE(unk_token=self.UNK_TOKEN))

		# Pre-tokenizer: split on whitespace
		self._tokenizer.pre_tokenizer = Whitespace()

		# Trainer configuration
		trainer = BpeTrainer(
			vocab_size=vocab_size,
			min_frequency=min_frequency,
			special_tokens=[self.UNK_TOKEN, self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN],
			show_progress=self._show_progress,
		)

		# Train from iterator (memory efficient)
		self._tokenizer.train_from_iterator(texts, trainer=trainer)

		# Post-processor for adding special tokens
		self._tokenizer.post_processor = TemplateProcessing(
			single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
			special_tokens=[
				(self.BOS_TOKEN, self._tokenizer.token_to_id(self.BOS_TOKEN)),
				(self.EOS_TOKEN, self._tokenizer.token_to_id(self.EOS_TOKEN)),
			],
		)

	def save(self, path: Union[str, Path]) -> None:
		"""
		Save tokenizer to disk.

		Args:
			path: File path to save to (will add .json extension)
		"""
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer not trained. Call train() first.")

		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)

		# Save as single JSON file (HuggingFace format)
		self._tokenizer.save(str(path))

	@classmethod
	def load(cls, path: Union[str, Path]) -> "BPETokenizer":
		"""
		Load tokenizer from disk.

		Args:
			path: File path to load from

		Returns:
			Loaded BPETokenizer instance
		"""
		try:
			from tokenizers import Tokenizer as HFTokenizer
		except ImportError:
			raise ImportError(
				"BPE tokenizer requires 'tokenizers' library. "
				"Install with: pip install tokenizers"
			)

		path = Path(path)
		tokenizer = cls(name=path.stem)
		tokenizer._tokenizer = HFTokenizer.from_file(str(path))
		tokenizer._is_trained = True
		return tokenizer

	def get_vocab(self) -> dict[str, int]:
		"""Get the full vocabulary mapping."""
		if self._tokenizer is None:
			return {}
		return self._tokenizer.get_vocab()


class GPT2Tokenizer(Tokenizer):
	"""
	Pre-trained GPT-2 BPE tokenizer.

	This uses the exact tokenizer from GPT-2, enabling direct comparison
	with published GPT-2 perplexity numbers.

	Note: GPT-2 tokenizer has 50,257 tokens and uses byte-level BPE.
	"""

	def __init__(self, name: str = "gpt2"):
		"""Initialize GPT-2 tokenizer."""
		super().__init__(name=name)
		self._encoder = None

		try:
			import tiktoken
			self._encoder = tiktoken.get_encoding("gpt2")
			self._is_trained = True  # Pre-trained
		except ImportError:
			raise ImportError(
				"GPT-2 tokenizer requires 'tiktoken' library. "
				"Install with: pip install tiktoken"
			)

	@property
	def vocab_size(self) -> int:
		"""GPT-2 vocabulary size."""
		return 50257  # Fixed for GPT-2

	@property
	def is_trainable(self) -> bool:
		"""GPT-2 tokenizer is pre-trained, not trainable."""
		return False

	def encode(self, text: str) -> list[int]:
		"""Encode text to token IDs."""
		return self._encoder.encode(text)

	def decode(self, ids: list[int]) -> str:
		"""Decode token IDs to text."""
		return self._encoder.decode(ids)

	def encode_batch(self, texts: list[str]) -> list[list[int]]:
		"""Encode multiple texts."""
		return self._encoder.encode_batch(texts)

	def decode_batch(self, ids_batch: list[list[int]]) -> list[str]:
		"""Decode multiple token sequences."""
		return self._encoder.decode_batch(ids_batch)


class CharacterTokenizer(Tokenizer):
	"""
	Character-level tokenizer.

	Useful for character-level language modeling or as a baseline.
	Very small vocabulary (~100-300 chars) but long sequences.
	"""

	def __init__(self, name: str = "char"):
		"""Initialize character tokenizer."""
		super().__init__(name=name)
		self._char_to_id: dict[str, int] = {}
		self._id_to_char: dict[int, str] = {}

	@property
	def vocab_size(self) -> int:
		"""Number of unique characters."""
		return len(self._char_to_id)

	@property
	def is_trainable(self) -> bool:
		"""Character tokenizer is trainable."""
		return True

	def encode(self, text: str) -> list[int]:
		"""Encode text to character IDs."""
		return [self._char_to_id.get(c, 0) for c in text]

	def decode(self, ids: list[int]) -> str:
		"""Decode character IDs to text."""
		return "".join(self._id_to_char.get(i, "?") for i in ids)

	def _train_impl(self, texts: list[str], **kwargs) -> None:
		"""Build character vocabulary from texts."""
		chars = set()
		for text in texts:
			chars.update(text)

		# Sort for deterministic ordering
		self._char_to_id = {c: i for i, c in enumerate(sorted(chars))}
		self._id_to_char = {i: c for c, i in self._char_to_id.items()}

	def get_vocab(self) -> dict[str, int]:
		"""Get character vocabulary."""
		return self._char_to_id.copy()
