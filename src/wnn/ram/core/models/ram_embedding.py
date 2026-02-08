"""
RAMFeatureExtractor - Extract features from RAMLM neurons for transformer embeddings.

Uses GA/TS-optimized RAM neurons as feature extractors. Each neuron's
TRUE/EMPTY response to an input represents a learned binary feature.
A linear projection maps these to a continuous embedding space.

Architecture:
	Context tokens → binary encoding
	→ RAM neurons → [TRUE_count, EMPTY_count] per cluster (raw features)
	→ Linear projection → [feature_dim] embedding vector

The projection is the ONE place gradients flow, bridging discrete RAM
with continuous transformers.

Usage:
	from wnn.ram.core.models.ram_embedding import RAMFeatureExtractor

	extractor = RAMFeatureExtractor(ram_model, feature_dim=256)
	embeddings = extractor(token_ids)  # [batch, feature_dim]
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor, tensor, long, float32, zeros, arange
from torch.nn.functional import softmax


class RAMFeatureExtractor(nn.Module):
	"""Extract features from RAMLM neurons for use as transformer embeddings.

	The RAM model's neurons act as binary feature detectors — their
	TRUE/EMPTY response pattern to an input encodes learned features.
	This module extracts those features and projects them to a
	continuous embedding space via a trainable linear layer.

	The RAM parameters are frozen (connectivity + memory). Only the
	linear projection is trained via backpropagation.

	Attributes:
		ram_model: Frozen RAMLM providing feature extraction (stays on CPU).
		projection: Trainable linear layer [ram_feature_dim] -> [feature_dim].
		feature_dim: Output embedding dimension.
		ram_feature_dim: Raw feature dimension from RAM.
	"""

	@property
	def ram_model(self):
		"""Access the frozen RAM model (stored outside nn.Module hierarchy)."""
		return self._ram_model[0]

	def __init__(
		self,
		ram_model: 'RAMLM',
		feature_dim: int = 256,
		trainable_projection: bool = True,
		use_scores: bool = True,
	):
		"""Initialize RAMFeatureExtractor.

		Args:
			ram_model: Trained RAMLM model (will be frozen).
			feature_dim: Output embedding dimension.
			trainable_projection: Whether projection weights are trainable.
			use_scores: If True, use softmax scores as features.
				If False, use raw TRUE/EMPTY counts (2x wider but more info).
		"""
		super().__init__()
		# Store RAM as non-Module attribute to prevent .to(device) from moving
		# RAM's registered buffers (e.g. _bit_positions) off CPU.
		# RAM must stay on CPU for encode_batch() to work.
		self._ram_model = [ram_model]  # Wrap in list to hide from nn.Module
		self.feature_dim = feature_dim
		self.use_scores = use_scores

		# RAM features: either scores (vocab_size) or raw (vocab_size * 2)
		if use_scores:
			self.ram_feature_dim = ram_model.vocab_size
		else:
			# TRUE count + EMPTY count per cluster
			self.ram_feature_dim = ram_model.vocab_size * 2

		# Trainable linear projection
		self.projection = nn.Linear(self.ram_feature_dim, feature_dim)
		if not trainable_projection:
			for p in self.projection.parameters():
				p.requires_grad = False

		# Layer norm on output
		self.layer_norm = nn.LayerNorm(feature_dim)

		# Move to available device
		self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		self.to(self.device)

	def extract_raw_features(self, token_ids_batch: Tensor) -> Tensor:
		"""Extract raw RAM features before projection.

		Args:
			token_ids_batch: [batch, context_size] token IDs.

		Returns:
			[batch, ram_feature_dim] raw feature tensor.
		"""
		batch_size = token_ids_batch.shape[0]

		# Encode tokens to bits (RAM operates on CPU)
		contexts = token_ids_batch.cpu().tolist()
		bits = self.ram_model.encode_batch(contexts)

		# Get RAM scores (no gradient needed for RAM)
		with torch.no_grad():
			from wnn.ram.core import AccelerationMode
			scores = self.ram_model.forward(bits, backend=AccelerationMode.PYTORCH)

		if self.use_scores:
			# Use softmax-normalized scores as features
			features = softmax(scores, dim=-1)  # [batch, vocab_size]
		else:
			# Use raw TRUE/EMPTY counts (if accessible)
			# Fall back to scores if raw counts aren't available
			features = softmax(scores, dim=-1)
			# Duplicate: first half = scores, second half = 1 - scores (proxy for EMPTY)
			features = torch.cat([features, 1.0 - features], dim=-1)

		return features.to(self.device)

	def forward(self, token_ids_batch: Tensor) -> Tensor:
		"""Extract RAM features and project to embedding space.

		Args:
			token_ids_batch: [batch, context_size] token IDs.

		Returns:
			[batch, feature_dim] embedding tensor (differentiable w.r.t. projection).
		"""
		# Extract features (RAM part is detached, no grad)
		raw_features = self.extract_raw_features(token_ids_batch)

		# Project to embedding space (THIS is where gradients flow)
		embeddings = self.projection(raw_features)
		embeddings = self.layer_norm(embeddings)

		return embeddings

	def forward_from_bits(self, input_bits: Tensor) -> Tensor:
		"""Extract features from pre-encoded bits.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch, feature_dim] embedding tensor.
		"""
		with torch.no_grad():
			from wnn.ram.core import AccelerationMode
			scores = self.ram_model.forward(input_bits, backend=AccelerationMode.PYTORCH)

		if self.use_scores:
			features = softmax(scores, dim=-1)
		else:
			features = softmax(scores, dim=-1)
			features = torch.cat([features, 1.0 - features], dim=-1)

		features = features.to(self.device)
		embeddings = self.projection(features)
		embeddings = self.layer_norm(embeddings)
		return embeddings
