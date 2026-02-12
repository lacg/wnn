"""
EvalTask — Evaluation task specification for language model benchmarking.

Defines WHAT to evaluate (dataset + tokenizer), not HOW (model architecture).
The same EvalTask can compare any model: RAMLM, BitwiseRAMLM, GPT-2, etc.

Usage:
	from wnn.eval import EvalTask, WIKITEXT2_TEST

	# Use predefined task
	task = WIKITEXT2_TEST
	tokens = task.load_tokens()

	# Or create custom
	task = EvalTask(dataset_name="wikitext", split="validation")
	tokens = task.load_tokens()

	# Serialize for reproducibility
	d = task.to_dict()
	task2 = EvalTask.from_dict(d)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass(frozen=True)
class EvalTask:
	"""Evaluation task specification for language model benchmarking.

	This is the 'what' not the 'how' — it specifies the dataset and tokenizer
	for evaluation, but not the model architecture or hyperparameters.

	Frozen (immutable) so it can be used as a dict key and shared safely.
	"""

	# Dataset
	dataset_name: str = "wikitext"
	dataset_config: str = "wikitext-2-raw-v1"
	split: str = "test"

	# Tokenizer
	tokenizer_name: str = "gpt2"
	vocab_size: int = 50257
	pad_token_id: int = 50256

	# Optional limits
	max_tokens: Optional[int] = None

	def load_tokens(self) -> list[int]:
		"""Load and tokenize the evaluation dataset.

		Returns:
			List of token IDs from the specified dataset and split.
		"""
		from datasets import load_dataset

		ds = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
		text = "\n\n".join(row["text"] for row in ds if row["text"].strip())

		if self.tokenizer_name == "gpt2":
			import tiktoken
			enc = tiktoken.get_encoding("gpt2")
			tokens = enc.encode(text)
		else:
			raise ValueError(f"Unknown tokenizer: {self.tokenizer_name}")

		if self.max_tokens is not None:
			tokens = tokens[:self.max_tokens]

		return tokens

	def load_tokens_tensor(self, device: str = "cpu"):
		"""Load tokens as a PyTorch tensor.

		Args:
			device: Target device for the tensor.

		Returns:
			1-D int64 tensor of token IDs.
		"""
		from torch import tensor, long
		return tensor(self.load_tokens(), dtype=long, device=device)

	def describe(self) -> str:
		"""Human-readable description of this task."""
		parts = [
			f"{self.dataset_name}/{self.dataset_config}",
			f"split={self.split}",
			f"tokenizer={self.tokenizer_name}",
			f"vocab={self.vocab_size:,}",
		]
		if self.max_tokens is not None:
			parts.append(f"max_tokens={self.max_tokens:,}")
		return " | ".join(parts)

	def to_dict(self) -> dict:
		"""Serialize to a plain dict (JSON-safe)."""
		d = asdict(self)
		# Remove None values for cleaner JSON
		return {k: v for k, v in d.items() if v is not None}

	@classmethod
	def from_dict(cls, d: dict) -> EvalTask:
		"""Deserialize from a dict."""
		# Filter to only known fields
		known = {f.name for f in cls.__dataclass_fields__.values()}
		return cls(**{k: v for k, v in d.items() if k in known})

	def __repr__(self) -> str:
		return f"EvalTask({self.describe()})"


# ── Predefined tasks ──────────────────────────────────────────────────────

WIKITEXT2_TEST = EvalTask(
	dataset_name="wikitext",
	dataset_config="wikitext-2-raw-v1",
	split="test",
)

WIKITEXT2_VAL = EvalTask(
	dataset_name="wikitext",
	dataset_config="wikitext-2-raw-v1",
	split="validation",
)

WIKITEXT2_TRAIN = EvalTask(
	dataset_name="wikitext",
	dataset_config="wikitext-2-raw-v1",
	split="train",
)
