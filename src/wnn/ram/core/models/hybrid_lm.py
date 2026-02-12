"""
HybridRAMTransformerLM - RAM Cache + Transformer Fallback.

RAM predicts first (O(1) lookup). If confident (low entropy), use its
prediction. Otherwise, invoke a transformer for the uncertain examples.

Blending modes:
- select: Binary routing — RAM if confident, else transformer (cleanest)
- interpolate: alpha * p_ram + (1-alpha) * p_transformer weighted by confidence
- cascade: RAM top-5 predictions as context hints to transformer

Usage:
	from wnn.ram.core.models.hybrid_lm import HybridRAMTransformerLM
	from wnn.ram.core.models.ramlm import RAMLM

	ram = RAMLM(vocab_size=50257, context_size=4, ...)
	# Train ram...

	hybrid = HybridRAMTransformerLM(
		ram_model=ram,
		transformer_model=transformer,  # HuggingFace or TinyTransformerLM
		entropy_threshold=2.0,
		blending_mode='select',
	)
	stats = hybrid.evaluate(token_ids)
"""

import time
from typing import Optional, Union, Literal

import torch
from torch import Tensor, tensor, long, float32, zeros, arange
from torch.nn.functional import softmax

from wnn.ram.core.base import RAMComponent
from wnn.ram.core import AccelerationMode
from wnn.ram.strategies.confidence import ConfidenceAnalyzer


class HybridRAMTransformerLM(RAMComponent):
	"""Hybrid Language Model: RAM fast path + transformer fallback.

	Routes predictions through RAM first. If RAM is confident
	(low entropy), its prediction is used directly. Otherwise,
	the transformer processes the uncertain examples.

	This achieves:
	- Fast predictions for easy/cached patterns (RAM, O(1))
	- High quality for hard patterns (transformer)
	- Reduced transformer compute (only processes uncertain inputs)

	Attributes:
		ram_model: Trained RAMLM for fast path.
		transformer_model: Transformer model for fallback.
		entropy_threshold: Entropy cutoff for RAM confidence.
		blending_mode: How to combine RAM and transformer predictions.
	"""

	def __init__(
		self,
		ram_model: 'RAMLM',
		transformer_model,
		tokenizer=None,
		entropy_threshold: float = 2.0,
		blending_mode: Literal['select', 'interpolate', 'cascade'] = 'select',
		interpolation_alpha: float = 0.3,
	):
		"""Initialize HybridRAMTransformerLM.

		Args:
			ram_model: Trained RAMLM model.
			transformer_model: Transformer model. Can be:
				- TinyTransformerLM (from this package)
				- HuggingFace PreTrainedModel (e.g., distilgpt2)
				- Any model with predict_next(context) -> [vocab] probs
			tokenizer: HuggingFace tokenizer (needed for HF models).
			entropy_threshold: Max entropy for RAM to handle (lower = stricter).
			blending_mode: 'select', 'interpolate', or 'cascade'.
			interpolation_alpha: Weight for RAM probs in 'interpolate' mode.
		"""
		super().__init__()
		self.ram_model = ram_model
		self.transformer_model = transformer_model
		self.tokenizer = tokenizer
		self.entropy_threshold = entropy_threshold
		self.blending_mode = blending_mode
		self.interpolation_alpha = interpolation_alpha

		# Detect transformer type
		self._is_hf_model = hasattr(transformer_model, 'generate')
		self._is_tiny_transformer = hasattr(transformer_model, 'predict_next')

	def forward(self, input_bits: Tensor, backend: AccelerationMode = AccelerationMode.AUTO) -> Tensor:
		"""Forward pass (RAM only, for RAMComponent compatibility).

		For hybrid evaluation, use forward_batch or evaluate instead.
		"""
		return self.ram_model.forward(input_bits, backend=backend)

	def _get_transformer_probs(self, context_tokens: list[int]) -> Tensor:
		"""Get next-token probabilities from transformer.

		Handles both HuggingFace and TinyTransformerLM models.

		Args:
			context_tokens: Context token IDs.

		Returns:
			[vocab_size] probability tensor on CPU.
		"""
		if self._is_tiny_transformer:
			return self.transformer_model.predict_next(context_tokens)

		if self._is_hf_model:
			import torch as th
			device = next(self.transformer_model.parameters()).device
			input_ids = th.tensor([context_tokens], dtype=th.long, device=device)
			with th.no_grad():
				outputs = self.transformer_model(input_ids)
				logits = outputs.logits[0, -1]  # Last position
				probs = th.softmax(logits, dim=-1)
			return probs.cpu()

		raise ValueError(f"Unknown transformer model type: {type(self.transformer_model)}")

	def _get_transformer_probs_batch(self, contexts_batch: list[list[int]]) -> Tensor:
		"""Get next-token probabilities for a batch of contexts.

		Args:
			contexts_batch: List of context token ID lists.

		Returns:
			[batch, vocab_size] probability tensor on CPU.
		"""
		if not contexts_batch:
			return zeros(0, self.ram_model.vocab_size)

		if self._is_tiny_transformer:
			# Process one at a time for TinyTransformerLM
			probs_list = [self.transformer_model.predict_next(ctx) for ctx in contexts_batch]
			return torch.stack(probs_list)

		if self._is_hf_model:
			import torch as th
			device = next(self.transformer_model.parameters()).device

			# Pad contexts to same length
			max_len = max(len(ctx) for ctx in contexts_batch)
			padded = []
			attention_masks = []
			for ctx in contexts_batch:
				pad_len = max_len - len(ctx)
				padded.append([self.ram_model.pad_token_id] * pad_len + ctx)
				attention_masks.append([0] * pad_len + [1] * len(ctx))

			input_ids = th.tensor(padded, dtype=th.long, device=device)
			attention_mask = th.tensor(attention_masks, dtype=th.long, device=device)

			with th.no_grad():
				outputs = self.transformer_model(input_ids, attention_mask=attention_mask)
				logits = outputs.logits[:, -1]  # Last position
				probs = th.softmax(logits, dim=-1)
			return probs.cpu()

		raise ValueError(f"Unknown transformer model type: {type(self.transformer_model)}")

	def forward_batch(
		self,
		token_ids_batch: list[list[int]],
		backend: AccelerationMode = AccelerationMode.AUTO,
	) -> tuple[Tensor, dict]:
		"""Hybrid forward pass with routing.

		1. RAM forward -> probs + entropy
		2. Partition: confident (entropy < threshold) vs uncertain
		3. Confident -> use RAM probs
		4. Uncertain -> transformer forward
		5. Return combined probs + routing stats

		Args:
			token_ids_batch: List of context token ID lists.
				Each context should be of length >= ram_model.context_size.
			backend: Acceleration backend for RAM.

		Returns:
			Tuple of:
				- [batch, vocab_size] combined probability tensor
				- dict with routing stats
		"""
		batch_size = len(token_ids_batch)
		vocab_size = self.ram_model.vocab_size

		# Step 1: RAM forward on all examples
		ram_contexts = [ctx[-self.ram_model.context_size:] for ctx in token_ids_batch]
		ram_bits = self.ram_model.encode_batch(ram_contexts)
		ram_scores = self.ram_model.forward(ram_bits, backend=backend)
		ram_probs = softmax(ram_scores, dim=-1)

		# Compute entropy
		entropy = ConfidenceAnalyzer.compute_entropy(ram_probs)

		# Step 2: Partition by confidence
		confident_mask = entropy < self.entropy_threshold
		num_confident = confident_mask.sum().item()
		num_uncertain = batch_size - num_confident

		# Step 3-4: Combine based on blending mode
		combined_probs = zeros(batch_size, vocab_size)

		if self.blending_mode == 'select':
			# Binary routing
			combined_probs[confident_mask] = ram_probs[confident_mask]

			if num_uncertain > 0:
				uncertain_indices = (~confident_mask).nonzero(as_tuple=True)[0]
				uncertain_contexts = [token_ids_batch[i] for i in uncertain_indices.tolist()]
				transformer_probs = self._get_transformer_probs_batch(uncertain_contexts)
				combined_probs[~confident_mask] = transformer_probs

		elif self.blending_mode == 'interpolate':
			# Weighted interpolation
			transformer_probs = self._get_transformer_probs_batch(token_ids_batch)
			alpha = self.interpolation_alpha
			# Weight RAM more for confident predictions
			confidence_weight = (1.0 - entropy / entropy.max().clamp(min=1e-6)).unsqueeze(-1)
			effective_alpha = alpha * confidence_weight
			combined_probs = effective_alpha * ram_probs + (1.0 - effective_alpha) * transformer_probs

		elif self.blending_mode == 'cascade':
			# RAM top-5 as hints (always use transformer but with RAM context)
			transformer_probs = self._get_transformer_probs_batch(token_ids_batch)
			# Boost RAM's top predictions in transformer output
			top5_vals, top5_idx = ram_probs.topk(5, dim=-1)
			boost = zeros_like(transformer_probs)
			boost.scatter_(1, top5_idx, top5_vals * 0.1)
			combined_probs = transformer_probs + boost
			combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)

		stats = {
			"total": batch_size,
			"ram_handled": num_confident,
			"transformer_handled": num_uncertain,
			"ram_fraction": num_confident / max(batch_size, 1),
			"avg_entropy": entropy.mean().item(),
			"threshold": self.entropy_threshold,
		}

		return combined_probs, stats

	def evaluate(
		self,
		token_ids: list[int],
		batch_size: int = 100,
		backend: AccelerationMode = AccelerationMode.AUTO,
		verbose: bool = True,
	) -> dict:
		"""Evaluate hybrid model on a token sequence.

		Args:
			token_ids: Token sequence to evaluate on.
			batch_size: Examples per batch.
			backend: Acceleration backend for RAM.
			verbose: Print progress.

		Returns:
			Dict with overall CE/PPL/accuracy + per-path breakdown + latency.
		"""
		from wnn.ram.strategies.perplexity import PerplexityCalculator

		total_examples = len(token_ids) - self.ram_model.context_size

		if verbose:
			print(f"Hybrid evaluation on {total_examples:,} examples "
				  f"(mode={self.blending_mode}, threshold={self.entropy_threshold:.2f})...")

		calc = PerplexityCalculator(vocab_size=self.ram_model.vocab_size)
		ram_calc = PerplexityCalculator(vocab_size=self.ram_model.vocab_size)
		transformer_calc = PerplexityCalculator(vocab_size=self.ram_model.vocab_size)

		total_ram_handled = 0
		total_transformer_handled = 0
		ram_time = 0.0
		transformer_time = 0.0

		num_batches = (total_examples + batch_size - 1) // batch_size

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			# Build contexts (full context for transformer, truncated for RAM)
			contexts = []
			for i in range(start, end):
				# Provide longer context for transformer (up to 128 tokens)
				ctx_start = max(0, i - 128 + self.ram_model.context_size)
				contexts.append(token_ids[ctx_start:i + self.ram_model.context_size])

			targets_list = token_ids[start + self.ram_model.context_size:end + self.ram_model.context_size]
			batch_targets = tensor(targets_list, dtype=long)

			# RAM part (timed)
			t0 = time.time()
			ram_contexts = [ctx[-self.ram_model.context_size:] for ctx in contexts]
			ram_bits = self.ram_model.encode_batch(ram_contexts)
			ram_scores = self.ram_model.forward(ram_bits, backend=backend)
			ram_probs = softmax(ram_scores, dim=-1)
			entropy = ConfidenceAnalyzer.compute_entropy(ram_probs)
			ram_time += time.time() - t0

			confident_mask = entropy < self.entropy_threshold

			# Transformer part (timed, only for uncertain)
			t0 = time.time()
			combined_probs = ram_probs.clone()

			if self.blending_mode == 'select':
				uncertain_indices = (~confident_mask).nonzero(as_tuple=True)[0]
				if len(uncertain_indices) > 0:
					uncertain_contexts = [contexts[i] for i in uncertain_indices.tolist()]
					t_probs = self._get_transformer_probs_batch(uncertain_contexts)
					combined_probs[~confident_mask] = t_probs
			elif self.blending_mode == 'interpolate':
				t_probs = self._get_transformer_probs_batch(contexts)
				alpha = self.interpolation_alpha
				confidence_weight = (1.0 - entropy / entropy.max().clamp(min=1e-6)).unsqueeze(-1)
				effective_alpha = alpha * confidence_weight
				combined_probs = effective_alpha * ram_probs + (1.0 - effective_alpha) * t_probs

			transformer_time += time.time() - t0

			# Accumulate stats
			calc.add_from_scores_batch(combined_probs, batch_targets, normalize=False)

			# Per-path stats
			num_confident = confident_mask.sum().item()
			total_ram_handled += num_confident
			total_transformer_handled += (end - start) - num_confident

			if num_confident > 0:
				ram_calc.add_from_scores_batch(
					ram_probs[confident_mask],
					batch_targets[confident_mask],
					normalize=False,
				)
			if num_confident < (end - start):
				transformer_calc.add_from_scores_batch(
					combined_probs[~confident_mask],
					batch_targets[~confident_mask],
					normalize=False,
				)

			if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
				pct = (end / total_examples) * 100
				stats = calc.get_stats()
				print(f"    {pct:5.1f}% - CE: {stats['cross_entropy']:.4f}, "
					  f"RAM: {total_ram_handled}/{end} ({total_ram_handled/max(end,1):.1%})")

		overall = calc.get_stats()
		ram_stats = ram_calc.get_stats()
		transformer_stats = transformer_calc.get_stats()

		result = {
			"overall": overall,
			"ram_path": {
				**ram_stats,
				"examples_handled": total_ram_handled,
				"fraction": total_ram_handled / max(total_examples, 1),
				"latency_s": round(ram_time, 3),
			},
			"transformer_path": {
				**transformer_stats,
				"examples_handled": total_transformer_handled,
				"fraction": total_transformer_handled / max(total_examples, 1),
				"latency_s": round(transformer_time, 3),
			},
			"config": {
				"blending_mode": self.blending_mode,
				"entropy_threshold": self.entropy_threshold,
				"interpolation_alpha": self.interpolation_alpha,
			},
			"compute_savings": {
				"transformer_calls_avoided": total_ram_handled,
				"fraction_avoided": total_ram_handled / max(total_examples, 1),
				"total_time_s": round(ram_time + transformer_time, 3),
			},
		}

		if verbose:
			print(f"\n  Overall: CE={overall['cross_entropy']:.4f}, PPL={overall['perplexity']:.2f}, Acc={overall['accuracy']:.2%}")
			print(f"  RAM path: CE={ram_stats['cross_entropy']:.4f}, handled {total_ram_handled:,} ({total_ram_handled/total_examples:.1%})")
			print(f"  Transformer path: CE={transformer_stats['cross_entropy']:.4f}, handled {total_transformer_handled:,}")
			print(f"  Transformer calls avoided: {total_ram_handled:,} ({total_ram_handled/total_examples:.1%})")
			print(f"  Latency: RAM={ram_time:.2f}s, Transformer={transformer_time:.2f}s")

		return result
