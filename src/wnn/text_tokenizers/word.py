"""
Word-level tokenizer for WNN language models.

This tokenizer splits text into words and maps each word to an integer ID.
It follows the WikiText-2 standard tokenization for comparability with
published benchmarks.

Two modes:
- SIMPLE: Basic regex word splitting (for quick testing)
- WIKITEXT: Standard WikiText-2 preprocessing (for benchmark comparability)
"""

import re
from collections import Counter
from pathlib import Path
from typing import Optional, Union
import json

from wnn.text_tokenizers.base import Tokenizer


class WordTokenizer(Tokenizer):
	"""
	Word-level tokenizer that maps words to integer IDs.

	This is a trainable tokenizer - call train() with a corpus to build
	the vocabulary, or load a pre-trained vocabulary.

	Features:
	- Handles OOV words with <unk> token
	- Optional frequency threshold for rare words
	- Preserves word boundaries for accurate decoding

	Usage:
		tokenizer = WordTokenizer()
		tokenizer.train(texts, min_freq=5)

		ids = tokenizer.encode("Hello world")
		text = tokenizer.decode(ids)
	"""

	# Special tokens
	UNK_TOKEN = "<unk>"
	PAD_TOKEN = "<pad>"
	EOS_TOKEN = "<eos>"

	def __init__(
		self,
		name: str = "word",
		lowercase: bool = False,
		min_freq: int = 1,
		wikitext_style: bool = True,
	):
		"""
		Initialize word tokenizer.

		Args:
			name: Tokenizer name
			lowercase: Whether to lowercase all text
			min_freq: Minimum frequency for a word to be in vocabulary
			wikitext_style: Use WikiText-2 standard preprocessing
		"""
		super().__init__(name=name)
		self._lowercase = lowercase
		self._min_freq = min_freq
		self._wikitext_style = wikitext_style

		# Vocabulary mappings (populated during training)
		self._word_to_id: dict[str, int] = {}
		self._id_to_word: dict[int, str] = {}
		self._word_counts: Counter = Counter()

		# Initialize with special tokens
		self._init_special_tokens()

	def _init_special_tokens(self) -> None:
		"""Initialize special token mappings."""
		self._word_to_id = {
			self.UNK_TOKEN: 0,
			self.PAD_TOKEN: 1,
			self.EOS_TOKEN: 2,
		}
		self._id_to_word = {v: k for k, v in self._word_to_id.items()}

	@property
	def vocab_size(self) -> int:
		"""Number of unique tokens in vocabulary."""
		return len(self._word_to_id)

	@property
	def unk_token_id(self) -> int:
		"""ID for unknown/OOV tokens."""
		return self._word_to_id[self.UNK_TOKEN]

	@property
	def pad_token_id(self) -> int:
		"""ID for padding token."""
		return self._word_to_id[self.PAD_TOKEN]

	@property
	def eos_token_id(self) -> int:
		"""ID for end-of-sequence token."""
		return self._word_to_id[self.EOS_TOKEN]

	@property
	def is_trainable(self) -> bool:
		"""Word tokenizer is trainable."""
		return True

	def _preprocess(self, text: str) -> str:
		"""
		Preprocess text before tokenization.

		Args:
			text: Raw input text

		Returns:
			Preprocessed text
		"""
		if self._wikitext_style:
			# WikiText-2 standard preprocessing
			# Replace newlines with <eos>
			text = text.replace("\n", f" {self.EOS_TOKEN} ")
			# Normalize whitespace
			text = " ".join(text.split())

		if self._lowercase:
			text = text.lower()

		return text

	def _tokenize(self, text: str) -> list[str]:
		"""
		Split text into word tokens.

		Args:
			text: Preprocessed text

		Returns:
			List of word tokens
		"""
		if self._wikitext_style:
			# WikiText standard: split on whitespace, keep punctuation attached
			return text.split()
		else:
			# Simple regex: split on non-alphanumeric, filter empty
			tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
			return [t for t in tokens if t.strip()]

	def encode(self, text: str) -> list[int]:
		"""
		Convert text to token IDs.

		Args:
			text: Input text

		Returns:
			List of token IDs
		"""
		text = self._preprocess(text)
		tokens = self._tokenize(text)
		return [self._word_to_id.get(token, self.unk_token_id) for token in tokens]

	def decode(self, ids: list[int]) -> str:
		"""
		Convert token IDs back to text.

		Args:
			ids: List of token IDs

		Returns:
			Decoded text
		"""
		tokens = [self._id_to_word.get(id, self.UNK_TOKEN) for id in ids]
		# Join with spaces (WikiText style)
		text = " ".join(tokens)
		# Post-process: convert <eos> back to newlines
		text = text.replace(f" {self.EOS_TOKEN} ", "\n")
		text = text.replace(self.EOS_TOKEN, "\n")
		return text

	def token_to_id(self, token: str) -> Optional[int]:
		"""Get ID for a token, or None if not in vocabulary."""
		return self._word_to_id.get(token)

	def id_to_token(self, id: int) -> Optional[str]:
		"""Get token for an ID, or None if invalid."""
		return self._id_to_word.get(id)

	def _train_impl(self, texts: list[str], min_freq: Optional[int] = None, **kwargs) -> None:
		"""
		Build vocabulary from training texts.

		Args:
			texts: List of training texts
			min_freq: Override minimum frequency (uses init value if None)
		"""
		if min_freq is not None:
			self._min_freq = min_freq

		# Count word frequencies
		self._word_counts = Counter()
		for text in texts:
			text = self._preprocess(text)
			tokens = self._tokenize(text)
			self._word_counts.update(tokens)

		# Build vocabulary from frequent words
		self._init_special_tokens()  # Reset to special tokens only

		for word, count in self._word_counts.most_common():
			if count >= self._min_freq and word not in self._word_to_id:
				idx = len(self._word_to_id)
				self._word_to_id[word] = idx
				self._id_to_word[idx] = word

	def add_tokens(self, tokens: list[str]) -> int:
		"""
		Add tokens to vocabulary (for handling OOV in test set).

		Args:
			tokens: List of tokens to add

		Returns:
			Number of new tokens added
		"""
		added = 0
		for token in tokens:
			if token not in self._word_to_id:
				idx = len(self._word_to_id)
				self._word_to_id[token] = idx
				self._id_to_word[idx] = token
				added += 1
		return added

	def get_vocab(self) -> dict[str, int]:
		"""Get the full vocabulary mapping."""
		return self._word_to_id.copy()

	def get_word_counts(self) -> Counter:
		"""Get word frequency counts from training."""
		return self._word_counts.copy()

	def save(self, path: Union[str, Path]) -> None:
		"""
		Save tokenizer to disk.

		Args:
			path: Directory to save to
		"""
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)

		# Save config
		config = {
			"name": self._name,
			"lowercase": self._lowercase,
			"min_freq": self._min_freq,
			"wikitext_style": self._wikitext_style,
			"vocab_size": self.vocab_size,
		}
		with open(path / "config.json", "w") as f:
			json.dump(config, f, indent=2)

		# Save vocabulary
		with open(path / "vocab.json", "w") as f:
			json.dump(self._word_to_id, f, indent=2)

		# Save word counts if available
		if self._word_counts:
			with open(path / "word_counts.json", "w") as f:
				json.dump(dict(self._word_counts), f, indent=2)

	@classmethod
	def load(cls, path: Union[str, Path]) -> "WordTokenizer":
		"""
		Load tokenizer from disk.

		Args:
			path: Directory to load from

		Returns:
			Loaded WordTokenizer instance
		"""
		path = Path(path)

		# Load config
		with open(path / "config.json") as f:
			config = json.load(f)

		# Create instance
		tokenizer = cls(
			name=config["name"],
			lowercase=config["lowercase"],
			min_freq=config["min_freq"],
			wikitext_style=config["wikitext_style"],
		)

		# Load vocabulary
		with open(path / "vocab.json") as f:
			tokenizer._word_to_id = json.load(f)
			tokenizer._id_to_word = {int(v): k for k, v in tokenizer._word_to_id.items()}

		# Load word counts if available
		counts_path = path / "word_counts.json"
		if counts_path.exists():
			with open(counts_path) as f:
				tokenizer._word_counts = Counter(json.load(f))

		tokenizer._is_trained = True
		return tokenizer


class SimpleWordTokenizer(WordTokenizer):
	"""
	Simple word tokenizer using regex splitting.

	This is a simplified version for quick testing, not for benchmark comparisons.
	"""

	def __init__(self, name: str = "simple_word", lowercase: bool = True):
		super().__init__(name=name, lowercase=lowercase, wikitext_style=False)
