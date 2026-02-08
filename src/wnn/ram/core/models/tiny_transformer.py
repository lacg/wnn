"""
TinyTransformerLM - Small standard transformer for fair comparison.

A minimal causal language model using standard PyTorch transformer
components, trained from scratch on the same data as RAMLM.

This provides a fair comparison point: same vocab, same tokenizer,
same training data, but using gradient-based learning.

Usage:
	from wnn.ram.core.models.tiny_transformer import TinyTransformerLM

	model = TinyTransformerLM(vocab_size=50257, d_model=128, n_layers=4)
	model.train_epoch(train_tokens, batch_size=32, lr=1e-4)
	stats = model.evaluate(eval_tokens)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TinyTransformerLM(nn.Module):
	"""Standard small transformer language model for fair comparison.

	Architecture:
		Token embedding + Positional embedding
		→ N × TransformerDecoderLayer (causal self-attention + FFN)
		→ Linear projection to vocab

	Attributes:
		vocab_size: Vocabulary size.
		d_model: Model dimension.
		n_heads: Number of attention heads.
		n_layers: Number of transformer layers.
		max_context: Maximum sequence length.
	"""

	def __init__(
		self,
		vocab_size: int = 50257,
		d_model: int = 128,
		n_heads: int = 4,
		n_layers: int = 4,
		d_ff: int = 512,
		max_context: int = 128,
		dropout: float = 0.1,
	):
		"""Initialize TinyTransformerLM.

		Args:
			vocab_size: Number of tokens in vocabulary.
			d_model: Embedding and hidden dimension.
			n_heads: Number of attention heads.
			n_layers: Number of transformer decoder layers.
			d_ff: Feed-forward inner dimension.
			max_context: Maximum sequence length.
			dropout: Dropout rate.
		"""
		super().__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.max_context = max_context

		# Embeddings
		self.token_embedding = nn.Embedding(vocab_size, d_model)
		self.position_embedding = nn.Embedding(max_context, d_model)
		self.dropout = nn.Dropout(dropout)

		# Transformer layers
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=d_ff,
			dropout=dropout,
			batch_first=True,
			norm_first=True,
		)
		self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

		# Output projection
		self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

		# Weight tying: share embedding and output weights
		self.output_proj.weight = self.token_embedding.weight

		# Initialize weights
		self._init_weights()

		# Move to available device
		self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		self.to(self.device)

	def _init_weights(self):
		"""Initialize weights with Xavier uniform."""
		nn.init.normal_(self.token_embedding.weight, std=0.02)
		nn.init.normal_(self.position_embedding.weight, std=0.02)

	def _causal_mask(self, seq_len: int) -> Tensor:
		"""Create causal attention mask."""
		mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
		return mask.bool()

	def forward(self, token_ids: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			token_ids: [batch, seq_len] token indices.

		Returns:
			[batch, seq_len, vocab_size] logits.
		"""
		batch_size, seq_len = token_ids.shape
		assert seq_len <= self.max_context, f"Sequence length {seq_len} > max_context {self.max_context}"

		# Embeddings
		positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
		x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
		x = x + self.position_embedding(positions)
		x = self.dropout(x)

		# Causal mask
		mask = self._causal_mask(seq_len)

		# Transformer (using decoder as autoregressive model)
		# memory=x with no cross-attention effectively makes it a decoder-only model
		memory = torch.zeros(batch_size, 1, self.d_model, device=self.device)
		x = self.transformer(x, memory, tgt_mask=mask)

		# Output projection
		logits = self.output_proj(x)
		return logits

	def train_epoch(
		self,
		train_tokens: list[int],
		batch_size: int = 32,
		seq_len: int = 64,
		lr: float = 1e-4,
		verbose: bool = True,
	) -> dict:
		"""Train for one epoch on token sequence.

		Args:
			train_tokens: Training token IDs.
			batch_size: Batch size.
			seq_len: Sequence length for training windows.
			lr: Learning rate.
			verbose: Print progress.

		Returns:
			Dict with avg_loss, num_batches.
		"""
		self.train()
		optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
		criterion = nn.CrossEntropyLoss()

		# Create training sequences
		tokens_tensor = torch.tensor(train_tokens, dtype=torch.long)
		num_sequences = (len(train_tokens) - 1) // seq_len
		total_loss = 0.0
		num_batches = 0

		# Shuffle sequence starts
		starts = torch.randperm(num_sequences) * seq_len

		for batch_start in range(0, len(starts), batch_size):
			batch_indices = starts[batch_start:batch_start + batch_size]
			if len(batch_indices) == 0:
				break

			# Gather batch sequences
			input_seqs = []
			target_seqs = []
			for idx in batch_indices:
				idx = idx.item()
				if idx + seq_len + 1 > len(train_tokens):
					continue
				input_seqs.append(tokens_tensor[idx:idx + seq_len])
				target_seqs.append(tokens_tensor[idx + 1:idx + seq_len + 1])

			if not input_seqs:
				continue

			inputs = torch.stack(input_seqs).to(self.device)
			targets = torch.stack(target_seqs).to(self.device)

			# Forward
			logits = self.forward(inputs)  # [batch, seq_len, vocab]
			loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

			# Backward
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
			optimizer.step()

			total_loss += loss.item()
			num_batches += 1

			if verbose and num_batches % 100 == 0:
				avg = total_loss / num_batches
				print(f"    Batch {num_batches}: avg_loss={avg:.4f}")

		avg_loss = total_loss / max(num_batches, 1)
		if verbose:
			print(f"  Epoch complete: avg_loss={avg_loss:.4f}, {num_batches} batches")

		return {"avg_loss": avg_loss, "num_batches": num_batches}

	def evaluate(
		self,
		eval_tokens: list[int],
		batch_size: int = 32,
		seq_len: int = 64,
		verbose: bool = True,
	) -> dict:
		"""Evaluate on token sequence.

		Args:
			eval_tokens: Evaluation token IDs.
			batch_size: Batch size.
			seq_len: Sequence length.
			verbose: Print progress.

		Returns:
			Dict with cross_entropy, perplexity, accuracy.
		"""
		self.eval()
		criterion = nn.CrossEntropyLoss(reduction='sum')

		tokens_tensor = torch.tensor(eval_tokens, dtype=torch.long)
		num_sequences = (len(eval_tokens) - 1) // seq_len
		total_loss = 0.0
		total_correct = 0
		total_tokens = 0

		with torch.no_grad():
			for batch_start in range(0, num_sequences, batch_size):
				input_seqs = []
				target_seqs = []

				for seq_idx in range(batch_start, min(batch_start + batch_size, num_sequences)):
					start = seq_idx * seq_len
					if start + seq_len + 1 > len(eval_tokens):
						continue
					input_seqs.append(tokens_tensor[start:start + seq_len])
					target_seqs.append(tokens_tensor[start + 1:start + seq_len + 1])

				if not input_seqs:
					continue

				inputs = torch.stack(input_seqs).to(self.device)
				targets = torch.stack(target_seqs).to(self.device)

				logits = self.forward(inputs)
				loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

				predicted = logits.argmax(dim=-1)
				total_correct += (predicted == targets).sum().item()

				total_loss += loss.item()
				total_tokens += targets.numel()

		ce = total_loss / max(total_tokens, 1)
		ppl = math.exp(min(ce, 20))  # Cap to prevent overflow
		accuracy = total_correct / max(total_tokens, 1)

		stats = {
			"cross_entropy": ce,
			"perplexity": ppl,
			"accuracy": accuracy,
			"total_tokens": total_tokens,
		}

		if verbose:
			print(f"  CE: {ce:.4f}, PPL: {ppl:.2f}, Acc: {accuracy:.2%}")

		return stats

	def predict_next(self, context_tokens: list[int]) -> Tensor:
		"""Predict next token probabilities given context.

		Args:
			context_tokens: Context token IDs.

		Returns:
			[vocab_size] probability tensor.
		"""
		self.eval()
		with torch.no_grad():
			tokens = torch.tensor([context_tokens[-self.max_context:]], dtype=torch.long, device=self.device)
			logits = self.forward(tokens)
			# Get last position's logits
			return torch.softmax(logits[0, -1], dim=-1).cpu()

	def num_parameters(self) -> int:
		"""Count total trainable parameters."""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
