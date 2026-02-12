"""
Standalone GatingTrainer for architecture-agnostic gating training.

Works with both tiered (CachedEvaluator) and bitwise (BitwiseEvaluator) architectures
via the GatingMode enum. The trainer creates and trains gating model(s) without
depending on any specific evaluator class.

GatingMode determines what gets trained:
  TOKEN_LEVEL (0): Universal — vocab_size gates, suppresses unlikely tokens after reconstruction
  BIT_LEVEL (1): Bitwise-specific — 16 gates, gates unreliable bit predictions
  DUAL_STAGE (2): Both — bit confidence → reconstruct → token pruning

Usage:
	config = GatingConfig(enabled=True, mode=GatingMode.TOKEN_LEVEL)
	trainer = GatingTrainer(config, logger=print)
	result = trainer.train(
		total_input_bits=64,
		train_tokens=tokens,
		vocab_size=50257,
		context_size=4,
		cluster_order=order,  # Optional, for tiered encoding
	)
	# result.token_gating and/or result.bit_gating populated per mode
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable

from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.core.gating import (
	GatingModel,
	create_gating,
	compute_beneficial_gates,
)


class GatingMode(IntEnum):
	"""Gating mode determining what gets trained and how gates are applied.

	TOKEN_LEVEL: Universal — works for any architecture. num_clusters = vocab_size.
		Train: target_gates[target_token] = 1, rest = 0.
		Eval: gated = log_probs + log(gates + eps).

	BIT_LEVEL: Bitwise-specific. num_clusters = bits_needed(vocab_size) = 16.
		Train: target_gates = bit_encoding(target_token).
		Eval: gated = gates * bit_scores + (1-gates) * 0.5, then reconstruct.

	DUAL_STAGE: Both TOKEN_LEVEL + BIT_LEVEL (bitwise-specific).
		Train: both models independently.
		Eval: bit gating → reconstruct → token gating.
	"""
	TOKEN_LEVEL = 0
	BIT_LEVEL = 1
	DUAL_STAGE = 2


@dataclass
class GatingConfig:
	"""Architecture-agnostic gating configuration."""
	enabled: bool = False
	neurons_per_gate: int = 8
	bits_per_neuron: int = 12
	threshold: float = 0.5
	num_epochs: int = 1
	batch_size: int = 256
	mode: GatingMode = GatingMode.TOKEN_LEVEL


@dataclass
class GatingResult:
	"""Result of gating training."""
	token_gating: Optional[GatingModel] = None
	bit_gating: Optional[GatingModel] = None
	mode: GatingMode = GatingMode.TOKEN_LEVEL
	stats: dict = field(default_factory=dict)


class GatingTrainer:
	"""Standalone gating trainer for any evaluator/architecture.

	Creates and trains gating model(s) based on GatingConfig.mode:
	- TOKEN_LEVEL: one model with vocab_size clusters
	- BIT_LEVEL: one model with bits_needed(vocab_size) clusters
	- DUAL_STAGE: both models

	Encoding adapts to architecture:
	- With cluster_order: maps token→cluster→bits (tiered)
	- Without: maps token→bits directly (bitwise)
	"""

	def __init__(self, config: GatingConfig, logger: Optional[Callable[[str], None]] = None):
		self.config = config
		self.log = logger or (lambda x: None)

	def train(
		self,
		total_input_bits: int,
		train_tokens: list[int],
		vocab_size: int,
		context_size: int,
		cluster_order: Optional[list[int]] = None,
		rng_seed: Optional[int] = None,
	) -> GatingResult:
		"""Train gating model(s) based on config.mode.

		Args:
			total_input_bits: Input dimensionality (context_size * bits_per_token)
			train_tokens: Token sequence for training
			vocab_size: Vocabulary size
			context_size: Context window size
			cluster_order: Token→cluster mapping for tiered encoding (None for bitwise)
			rng_seed: Random seed for gating model connectivity

		Returns:
			GatingResult with populated model(s) and stats
		"""
		import torch

		mode = self.config.mode
		bits_per_token = bits_needed(vocab_size)
		num_bits = bits_per_token

		self.log(f"{'='*60}")
		self.log(f"  Gating Training Phase (mode={mode.name})")
		self.log(f"{'='*60}")

		# Encode training data
		self.log(f"  Encoding training data ({len(train_tokens):,} tokens, context={context_size})...")
		samples = self._encode_training_data(
			train_tokens, context_size, vocab_size, cluster_order,
		)
		self.log(f"  Training samples: {len(samples):,}")

		result = GatingResult(mode=mode)
		t0 = time.time()

		# Create and train model(s) based on mode
		if mode in (GatingMode.TOKEN_LEVEL, GatingMode.DUAL_STAGE):
			self.log(f"  Training TOKEN_LEVEL gating ({vocab_size} clusters)...")
			token_model = create_gating(
				total_input_bits=total_input_bits,
				num_clusters=vocab_size,
				neurons_per_gate=self.config.neurons_per_gate,
				bits_per_neuron=self.config.bits_per_neuron,
				threshold=self.config.threshold,
				rng=rng_seed,
				prefer_rust=True,
			)
			token_stats = self._train_token_gating(token_model, samples, vocab_size)
			result.token_gating = token_model
			result.stats["token_gating"] = token_stats

		if mode in (GatingMode.BIT_LEVEL, GatingMode.DUAL_STAGE):
			self.log(f"  Training BIT_LEVEL gating ({num_bits} clusters)...")
			bit_model = create_gating(
				total_input_bits=total_input_bits,
				num_clusters=num_bits,
				neurons_per_gate=self.config.neurons_per_gate,
				bits_per_neuron=self.config.bits_per_neuron,
				threshold=self.config.threshold,
				rng=(rng_seed + 7919) if rng_seed is not None else None,
				prefer_rust=True,
			)
			bit_stats = self._train_bit_gating(bit_model, samples, num_bits)
			result.bit_gating = bit_model
			result.stats["bit_gating"] = bit_stats

		elapsed = time.time() - t0
		result.stats["mode"] = mode.name
		result.stats["training_time"] = elapsed
		result.stats["num_samples"] = len(samples)

		self.log(f"  Gating training complete in {elapsed:.1f}s")
		return result

	def _encode_training_data(
		self,
		train_tokens: list[int],
		context_size: int,
		vocab_size: int,
		cluster_order: Optional[list[int]] = None,
	) -> list[tuple[list[bool], int]]:
		"""Encode context tokens to input bits + target token.

		Args:
			train_tokens: Full token sequence
			context_size: Number of context tokens
			vocab_size: Vocabulary size
			cluster_order: If provided, maps token_id→cluster_id for tiered encoding

		Returns:
			List of (input_bits, target_token) tuples
		"""
		bits_per_token = bits_needed(vocab_size)

		# Build mapping if tiered
		token_to_cluster = None
		if cluster_order is not None:
			token_to_cluster = {token: cluster for cluster, token in enumerate(cluster_order)}

		samples = []
		for i in range(context_size, len(train_tokens)):
			context_tokens = train_tokens[i - context_size:i]
			target_token = train_tokens[i]

			# Map target through cluster order if tiered
			target = target_token
			if token_to_cluster is not None:
				target = token_to_cluster.get(target_token, target_token)

			# Encode context to bits
			input_bits = []
			for token in context_tokens:
				val = token
				if token_to_cluster is not None:
					val = token_to_cluster.get(token, token)
				for b in range(bits_per_token):
					input_bits.append(bool((val >> b) & 1))

			samples.append((input_bits, target))

		return samples

	def _train_token_gating(
		self,
		model: GatingModel,
		samples: list[tuple[list[bool], int]],
		vocab_size: int,
	) -> dict:
		"""Train TOKEN_LEVEL gating: gate[target_token] = 1, rest = 0.

		Uses train_from_targets() if available (RustRAMGating), otherwise
		falls back to compute_beneficial_gates() + train_step().
		"""
		import torch

		total_modified = 0
		batch_size = self.config.batch_size
		num_samples = len(samples)
		num_batches = (num_samples + batch_size - 1) // batch_size

		for epoch in range(self.config.num_epochs):
			epoch_modified = 0
			for batch_idx in range(num_batches):
				start = batch_idx * batch_size
				end = min(start + batch_size, num_samples)
				batch = samples[start:end]

				input_bits = torch.tensor([s[0] for s in batch], dtype=torch.bool)
				targets = torch.tensor([s[1] for s in batch], dtype=torch.long)

				if hasattr(model, 'train_from_targets'):
					modified = model.train_from_targets(input_bits, targets)
				else:
					target_gates = compute_beneficial_gates(
						torch.zeros(len(batch), vocab_size),
						targets,
						top_k=1,
					)
					modified = model.train_step(input_bits, target_gates)

				epoch_modified += modified

				if batch_idx > 0 and batch_idx % max(1, num_batches // 10) == 0:
					pct = 100 * batch_idx / num_batches
					self.log(f"    Token gating epoch {epoch+1}: {pct:.0f}%")

			total_modified += epoch_modified
			self.log(f"  Token gating epoch {epoch+1}/{self.config.num_epochs}: "
					f"{epoch_modified:,} cells modified")

		return {"total_modified": total_modified, "num_clusters": vocab_size}

	def _train_bit_gating(
		self,
		model: GatingModel,
		samples: list[tuple[list[bool], int]],
		num_bits: int,
	) -> dict:
		"""Train BIT_LEVEL gating: gate[b] = (target_token >> b) & 1.

		Each gate learns whether bit b of the target token is 1.
		"""
		import torch

		total_modified = 0
		batch_size = self.config.batch_size
		num_samples = len(samples)
		num_batches = (num_samples + batch_size - 1) // batch_size

		# Precompute bit positions for target encoding
		bit_positions = torch.arange(num_bits, dtype=torch.long)

		for epoch in range(self.config.num_epochs):
			epoch_modified = 0
			for batch_idx in range(num_batches):
				start = batch_idx * batch_size
				end = min(start + batch_size, num_samples)
				batch = samples[start:end]

				input_bits = torch.tensor([s[0] for s in batch], dtype=torch.bool)
				targets = torch.tensor([s[1] for s in batch], dtype=torch.long)

				# Target gates: bit_encoding of target token [B, num_bits]
				target_gates = ((targets.unsqueeze(-1) >> bit_positions) & 1).float()

				modified = model.train_step(input_bits, target_gates)
				epoch_modified += modified

				if batch_idx > 0 and batch_idx % max(1, num_batches // 10) == 0:
					pct = 100 * batch_idx / num_batches
					self.log(f"    Bit gating epoch {epoch+1}: {pct:.0f}%")

			total_modified += epoch_modified
			self.log(f"  Bit gating epoch {epoch+1}/{self.config.num_epochs}: "
					f"{epoch_modified:,} cells modified")

		return {"total_modified": total_modified, "num_clusters": num_bits}
