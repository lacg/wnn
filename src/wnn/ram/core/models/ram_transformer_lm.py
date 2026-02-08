"""
RAMTransformerLM - Language model using RAM features as transformer input.

Architecture:
	Context tokens → RAMFeatureExtractor (frozen RAM + trainable projection)
	→ Transformer decoder layers (trainable)
	→ Linear output head → next token prediction

The RAM neurons act as learned feature detectors (via GA/TS optimization).
Only the projection layer and transformer are trained via gradient descent.

Training protocol:
	1. Pre-train RAMLM with GA/TS (existing process)
	2. Freeze all RAM parameters (connectivity + memory)
	3. Train projection + transformer via Adam
	4. Cross-entropy loss on next-token prediction

Usage:
	from wnn.ram.core.models.ram_transformer_lm import RAMTransformerLM

	ram = RAMLM(...)  # Pre-trained
	model = RAMTransformerLM(ram, feature_dim=256, n_layers=4)
	model.train_epoch(train_tokens)
	stats = model.evaluate(eval_tokens)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RAMTransformerLM(nn.Module):
	"""Language model: RAM features -> transformer decoder -> next token.

	Uses a pre-trained RAMLM as a frozen feature extractor. The RAM
	features (per-cluster scores) are projected to an embedding space,
	then processed by a standard transformer decoder.

	The key hypothesis: RAM features, shaped by connectivity optimization,
	encode useful pattern information that helps the transformer converge
	faster than random-init embeddings.

	Attributes:
		ram_model: Frozen RAMLM (stays on CPU).
		feature_extractor: RAMFeatureExtractor with trainable projection.
		transformer_layers: Standard transformer decoder layers.
		output_head: Linear projection to vocab size.
	"""

	@property
	def ram_model(self):
		"""Access the frozen RAM model (stored outside nn.Module hierarchy)."""
		return self._ram_model[0]

	def __init__(
		self,
		ram_model: 'RAMLM',
		feature_dim: int = 256,
		n_heads: int = 4,
		n_layers: int = 4,
		d_ff: int = 512,
		max_seq_len: int = 512,
		dropout: float = 0.1,
		freeze_ram: bool = True,
	):
		"""Initialize RAMTransformerLM.

		Args:
			ram_model: Pre-trained RAMLM model.
			feature_dim: Dimension of projected RAM features.
			n_heads: Number of attention heads.
			n_layers: Number of transformer decoder layers.
			d_ff: Feed-forward inner dimension.
			max_seq_len: Maximum sequence length.
			dropout: Dropout rate.
			freeze_ram: Whether to freeze RAM parameters (recommended True).
		"""
		super().__init__()

		from wnn.ram.core.models.ram_embedding import RAMFeatureExtractor

		# Store RAM outside nn.Module hierarchy to keep it on CPU
		self._ram_model = [ram_model]
		self.vocab_size = ram_model.vocab_size
		self.context_size = ram_model.context_size
		self.feature_dim = feature_dim
		self.max_seq_len = max_seq_len

		if freeze_ram:
			# RAM is not a nn.Module, but ensure we don't accidentally modify it
			self._freeze_ram = True

		# RAM feature extractor (projection is trainable)
		self.feature_extractor = RAMFeatureExtractor(
			ram_model, feature_dim=feature_dim, trainable_projection=True,
		)

		# Position embeddings for sequences
		self.position_embedding = nn.Embedding(max_seq_len, feature_dim)
		self.dropout_layer = nn.Dropout(dropout)

		# Transformer decoder layers
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=feature_dim,
			nhead=n_heads,
			dim_feedforward=d_ff,
			dropout=dropout,
			batch_first=True,
			norm_first=True,
		)
		self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

		# Output head
		self.output_head = nn.Linear(feature_dim, self.vocab_size)

		# Move to device
		self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		self.to(self.device)

	def _causal_mask(self, seq_len: int) -> Tensor:
		"""Create causal attention mask."""
		mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
		return mask.bool()

	def forward(self, token_ids: Tensor) -> Tensor:
		"""Forward pass from token sequences.

		For each position in the sequence, extracts RAM features from
		the preceding context window and feeds through transformer.

		Args:
			token_ids: [batch, seq_len] token indices.

		Returns:
			[batch, seq_len, vocab_size] logits.
		"""
		batch_size, seq_len = token_ids.shape

		# Extract RAM features for each position
		# For position i, context is tokens[max(0, i-ctx+1):i+1]
		features_list = []
		for pos in range(seq_len):
			# Context for this position
			ctx_start = max(0, pos - self.context_size + 1)
			ctx_tokens = token_ids[:, ctx_start:pos + 1]  # [batch, ctx_len]

			# Pad context if needed
			ctx_len = ctx_tokens.shape[1]
			if ctx_len < self.context_size:
				pad = torch.full(
					(batch_size, self.context_size - ctx_len),
					self.ram_model.pad_token_id,
					dtype=torch.long, device=token_ids.device,
				)
				ctx_tokens = torch.cat([pad, ctx_tokens], dim=1)

			# Extract features: [batch, feature_dim]
			features = self.feature_extractor(ctx_tokens)
			features_list.append(features)

		# Stack: [batch, seq_len, feature_dim]
		x = torch.stack(features_list, dim=1)

		# Add position embeddings
		positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
		x = x + self.position_embedding(positions)
		x = self.dropout_layer(x)

		# Transformer with causal mask
		mask = self._causal_mask(seq_len)
		memory = torch.zeros(batch_size, 1, self.feature_dim, device=self.device)
		x = self.transformer(x, memory, tgt_mask=mask)

		# Output logits
		logits = self.output_head(x)
		return logits

	def train_epoch(
		self,
		train_tokens: list[int],
		batch_size: int = 16,
		seq_len: int = 32,
		lr: float = 1e-4,
		verbose: bool = True,
	) -> dict:
		"""Train for one epoch.

		Args:
			train_tokens: Training token IDs.
			batch_size: Batch size.
			seq_len: Training sequence length.
			lr: Learning rate.
			verbose: Print progress.

		Returns:
			Dict with avg_loss, num_batches.
		"""
		self.train()

		# Only optimize non-RAM parameters
		trainable_params = [p for p in self.parameters() if p.requires_grad]
		optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
		criterion = nn.CrossEntropyLoss()

		tokens_tensor = torch.tensor(train_tokens, dtype=torch.long)
		num_sequences = (len(train_tokens) - 1) // seq_len
		total_loss = 0.0
		num_batches = 0

		starts = torch.randperm(num_sequences) * seq_len

		for batch_start in range(0, len(starts), batch_size):
			batch_indices = starts[batch_start:batch_start + batch_size]

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

			logits = self.forward(inputs)
			loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
			optimizer.step()

			total_loss += loss.item()
			num_batches += 1

			if verbose and num_batches % 50 == 0:
				avg = total_loss / num_batches
				print(f"    Batch {num_batches}: avg_loss={avg:.4f}")

		avg_loss = total_loss / max(num_batches, 1)
		if verbose:
			print(f"  Epoch complete: avg_loss={avg_loss:.4f}, {num_batches} batches")

		return {"avg_loss": avg_loss, "num_batches": num_batches}

	def evaluate(
		self,
		eval_tokens: list[int],
		batch_size: int = 16,
		seq_len: int = 32,
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
		ppl = math.exp(min(ce, 20))
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

	def num_parameters(self) -> int:
		"""Count trainable parameters (excludes frozen RAM)."""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
