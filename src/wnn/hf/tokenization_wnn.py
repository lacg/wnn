"""
WNN Tokenizer wrapper for HuggingFace integration.

WNN uses tiktoken's GPT-2 BPE tokenizer internally. This module provides
a thin wrapper that makes it compatible with HuggingFace's tokenizer
interface for save_pretrained/from_pretrained and pipeline support.

Note: WNN doesn't use a custom tokenizer — it reuses GPT-2's tokenization.
The "tokenizer" here is the BinaryTokenEncoder that maps token IDs to
bit vectors for RAM neuron addressing. This wrapper makes the encoding
config serializable via the HF interface.
"""

from __future__ import annotations

import json
from pathlib import Path


class WNNTokenizerConfig:
	"""Serializable configuration for the WNN token encoding pipeline.

	This is saved alongside the model to ensure reproducible encoding:
	- Which tokenizer (always GPT-2 BPE via tiktoken)
	- Which bit encoder (binary or gray_code)
	- Vocabulary size
	"""

	def __init__(
		self,
		tokenizer_name: str = "gpt2",
		encoding_type: str = "binary",
		vocab_size: int = 50257,
	):
		self.tokenizer_name = tokenizer_name
		self.encoding_type = encoding_type
		self.vocab_size = vocab_size

	def save_pretrained(self, save_directory: str | Path):
		"""Save tokenizer config to a directory."""
		save_directory = Path(save_directory)
		save_directory.mkdir(parents=True, exist_ok=True)

		config = {
			"tokenizer_class": "WNNTokenizer",
			"tokenizer_name": self.tokenizer_name,
			"encoding_type": self.encoding_type,
			"vocab_size": self.vocab_size,
		}

		config_path = save_directory / "tokenizer_config.json"
		with open(config_path, "w") as f:
			json.dump(config, f, indent=2)

		# Standard HF special tokens map
		special_tokens = {
			"eos_token": "<|endoftext|>",
			"bos_token": "<|endoftext|>",
			"unk_token": "<|endoftext|>",
		}
		tokens_path = save_directory / "special_tokens_map.json"
		with open(tokens_path, "w") as f:
			json.dump(special_tokens, f, indent=2)

	@classmethod
	def from_pretrained(cls, pretrained_path: str | Path) -> WNNTokenizerConfig:
		"""Load tokenizer config from a directory."""
		config_path = Path(pretrained_path) / "tokenizer_config.json"
		with open(config_path) as f:
			config = json.load(f)

		return cls(
			tokenizer_name=config.get("tokenizer_name", "gpt2"),
			encoding_type=config.get("encoding_type", "binary"),
			vocab_size=config.get("vocab_size", 50257),
		)

	def get_tokenizer(self):
		"""Get the tiktoken tokenizer instance."""
		import tiktoken
		return tiktoken.get_encoding(self.tokenizer_name)

	def get_bit_encoder(self):
		"""Get the appropriate bit encoder."""
		from wnn.representations.token_bit_encoder import (
			BinaryTokenEncoder,
			GrayCodeTokenEncoder,
		)
		if self.encoding_type == "gray_code":
			return GrayCodeTokenEncoder(self.vocab_size)
		return BinaryTokenEncoder(self.vocab_size)
