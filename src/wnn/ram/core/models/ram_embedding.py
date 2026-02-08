"""
RAMFeatureExtractor - Extract features from RAMLM neurons for transformer embeddings.

Uses GA/TS-optimized RAM neurons as feature extractors. Each neuron's
TRUE/EMPTY response to an input represents a learned binary feature.
A linear projection maps these to a continuous embedding space.

Architecture:
	Context tokens → binary encoding (CPU)
	→ RAM neurons → scores per cluster (CPU, AccelerationMode.AUTO)
	→ device transfer → GPU
	→ Linear projection → [feature_dim] embedding vector (GPU)

Device management:
	RAM stays on CPU (its Memory uses bit-packed int64 arrays, Rust/Metal
	accelerator manages its own CPU+GPU). Trainable components (projection,
	layer_norm) go to the compute device (MPS if available).

	The _apply() override ensures .to(device) skips the RAM model,
	so it stays on CPU regardless of where the parent module moves.

Usage:
	from wnn.ram.core.models.ram_embedding import RAMFeatureExtractor

	extractor = RAMFeatureExtractor(ram_model, feature_dim=256)
	embeddings = extractor(token_ids)  # [batch, feature_dim]
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax


def _best_device() -> torch.device:
	"""Select the best available compute device."""
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


class RAMFeatureExtractor(nn.Module):
	"""Extract features from RAMLM neurons for use as transformer embeddings.

	The RAM model's neurons act as binary feature detectors — their
	TRUE/EMPTY response pattern to an input encodes learned features.
	This module extracts those features and projects them to a
	continuous embedding space via a trainable linear layer.

	The RAM parameters are frozen (connectivity + memory) and stay on CPU.
	Only the linear projection is trained via backpropagation on GPU.

	Attributes:
		ram_model: Frozen RAMLM providing feature extraction (always on CPU).
		projection: Trainable linear layer [ram_feature_dim] -> [feature_dim].
		feature_dim: Output embedding dimension.
		ram_feature_dim: Raw feature dimension from RAM.
	"""

	def __init__(
		self,
		ram_model: 'RAMLM',
		feature_dim: int = 256,
		trainable_projection: bool = True,
		use_scores: bool = True,
	):
		"""Initialize RAMFeatureExtractor.

		Args:
			ram_model: Trained RAMLM model (stays on CPU).
			feature_dim: Output embedding dimension.
			trainable_projection: Whether projection weights are trainable.
			use_scores: If True, use softmax scores as features.
				If False, use raw TRUE/EMPTY counts (2x wider but more info).
		"""
		super().__init__()
		self.ram_model = ram_model
		self.feature_dim = feature_dim
		self.use_scores = use_scores

		# RAM features: either scores (vocab_size) or raw (vocab_size * 2)
		if use_scores:
			self.ram_feature_dim = ram_model.vocab_size
		else:
			self.ram_feature_dim = ram_model.vocab_size * 2

		# Trainable linear projection
		self.projection = nn.Linear(self.ram_feature_dim, feature_dim)
		if not trainable_projection:
			for p in self.projection.parameters():
				p.requires_grad = False

		# Layer norm on output
		self.layer_norm = nn.LayerNorm(feature_dim)

		# Move trainable components to best device; RAM stays on CPU
		self._compute_device = _best_device()
		self.projection.to(self._compute_device)
		self.layer_norm.to(self._compute_device)

	def _apply(self, fn, recurse=True):
		"""Override to keep ram_model on CPU when .to(device) is called.

		PyTorch's .to(), .cuda(), .cpu(), .float() etc. all call _apply()
		internally. By temporarily removing ram_model from the module tree
		during the apply, we prevent it from being moved off CPU.

		The RAM model manages its own acceleration via AccelerationMode
		(Rust CPU + Metal GPU hybrid) — it must not be moved by PyTorch.
		"""
		ram = self._modules.pop('ram_model', None)
		result = super()._apply(fn, recurse)
		if ram is not None:
			self._modules['ram_model'] = ram
		# Update compute device from projection (which DID move)
		self._compute_device = self.projection.weight.device
		return result

	@property
	def device(self) -> torch.device:
		"""Device of trainable components (projection, layer_norm)."""
		return self._compute_device

	def extract_raw_features(self, token_ids_batch: Tensor) -> Tensor:
		"""Extract raw RAM features before projection.

		Args:
			token_ids_batch: [batch, context_size] token IDs (any device).

		Returns:
			[batch, ram_feature_dim] raw feature tensor on compute device.
		"""
		from wnn.ram.core import AccelerationMode

		# RAM operates on CPU — encode on CPU
		contexts = token_ids_batch.cpu().tolist()
		bits = self.ram_model.encode_batch(contexts)

		# RAM forward with AUTO (uses HYBRID CPU+GPU via Rust/Metal if available)
		with torch.no_grad():
			scores = self.ram_model.forward(bits, backend=AccelerationMode.AUTO)

		if self.use_scores:
			features = softmax(scores, dim=-1)
		else:
			features = softmax(scores, dim=-1)
			features = torch.cat([features, 1.0 - features], dim=-1)

		# Transfer to compute device for projection
		return features.to(self._compute_device)

	def forward(self, token_ids_batch: Tensor) -> Tensor:
		"""Extract RAM features and project to embedding space.

		Args:
			token_ids_batch: [batch, context_size] token IDs (any device).

		Returns:
			[batch, feature_dim] embedding tensor on compute device.
		"""
		raw_features = self.extract_raw_features(token_ids_batch)

		# Projection + norm (on compute device, gradients flow here)
		embeddings = self.projection(raw_features)
		embeddings = self.layer_norm(embeddings)
		return embeddings

	def forward_from_bits(self, input_bits: Tensor) -> Tensor:
		"""Extract features from pre-encoded bits.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor (CPU).

		Returns:
			[batch, feature_dim] embedding tensor on compute device.
		"""
		from wnn.ram.core import AccelerationMode

		with torch.no_grad():
			scores = self.ram_model.forward(
				input_bits.cpu(), backend=AccelerationMode.AUTO,
			)

		if self.use_scores:
			features = softmax(scores, dim=-1)
		else:
			features = softmax(scores, dim=-1)
			features = torch.cat([features, 1.0 - features], dim=-1)

		features = features.to(self._compute_device)
		embeddings = self.projection(features)
		embeddings = self.layer_norm(embeddings)
		return embeddings
