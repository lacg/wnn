"""Tokenizer type enums for language model benchmarks."""

from enum import IntEnum


class TokenizerType(IntEnum):
	"""
	Tokenizer options for WikiText-2 benchmarks.

	Published perplexity benchmarks for comparison:
	- WIKITEXT_WORD: LSTM ~65-100, AWD-LSTM ~57
	- GPT2_BPE: GPT-2 Small ~29, GPT-2 Large ~22

	Note: Word-level and BPE perplexities are NOT directly comparable!
	"""
	SIMPLE = 0        # Our original: regex word-level (not standard)
	WIKITEXT_WORD = 1 # Standard WikiText-2 word-level (~33K vocab)
	GPT2_BPE = 2      # GPT-2 BPE tokenization (50,257 vocab)
