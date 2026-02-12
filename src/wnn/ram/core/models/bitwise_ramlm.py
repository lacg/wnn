"""
BitwiseRAMLM - Per-bit output language model.

Instead of 50K clusters (one per token), uses 16 clusters (one per output bit).
Each cluster predicts P(bit_i=1 | context). Token probabilities are reconstructed
via log-product over bits:

	log P(token=t) = Σ_i [b_i(t)·log(P_i) + (1-b_i(t))·log(1-P_i)]

where b_i(t) is the i-th bit of token t's binary encoding.

Key advantage: Every neuron sees ALL training examples (not just ~20 for rare
tokens), providing orders-of-magnitude better data density per classifier.

Architecture:
	INPUT: context_size tokens × bits_per_token bits
	         ↓
	[BitwiseRAMClusterLayer] - 16 clusters × neurons_per_cluster
	         ↓
	16 per-bit probabilities P(bit_i=1)
	         ↓
	Log-product reconstruction → vocab_size log-probabilities

Invalid token masking: 16 bits = 65536 patterns but only vocab_size valid tokens.
The precomputed token_bits [vocab_size, 16] matrix only has rows for valid tokens,
so invalid patterns are never scored.
"""

from typing import Optional

from torch import (
	arange,
	bool as torch_bool,
	clamp,
	float32,
	log,
	long,
	tensor,
	Tensor,
	zeros,
)

from wnn.ram.core.base import RAMComponent
from wnn.ram.core.BitwiseRAMClusterLayer import BitwiseRAMClusterLayer
from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.representations.token_bit_encoder import (
	TokenBitEncoder,
	BinaryTokenEncoder,
)


def reconstruct_logprobs(bit_scores: Tensor, token_bits: Tensor) -> Tensor:
	"""Reconstruct token log-probabilities from per-bit scores.

	Computes log P(token=t) = Σ_i [b_i(t)·log(P_i) + (1-b_i(t))·log(1-P_i)]
	as a matmul for efficiency.

	Args:
		bit_scores: [batch, num_bits] float tensor of P(bit=1) probabilities
		token_bits: [vocab_size, num_bits] binary matrix of token bit patterns

	Returns:
		[batch, vocab_size] float tensor of unnormalized log-probabilities
	"""
	eps = 1e-7
	p1 = clamp(bit_scores, eps, 1.0 - eps)
	log_p1 = log(p1)
	log_p0 = log(1.0 - p1)
	return log_p1 @ token_bits.T + log_p0 @ (1.0 - token_bits).T


class BitwiseRAMLM(RAMComponent):
	"""Per-bit output language model.

	Single BitwiseRAMClusterLayer with num_bits clusters (one per output bit).
	P(bit_i=1) = cluster_i score. Token probs via log-product.
	"""

	def __init__(
		self,
		vocab_size: int = 50257,
		context_size: int = 4,
		neurons_per_cluster: int = 1000,
		bits_per_neuron: int = 10,
		pad_token_id: int = 50256,
		rng: Optional[int] = None,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
		encoder: Optional[TokenBitEncoder] = None,
	):
		super().__init__()

		self.vocab_size = vocab_size
		self.context_size = context_size
		self.pad_token_id = pad_token_id
		self.memory_mode = memory_mode  # 0=TERNARY, 1=QUAD_BINARY, 2=QUAD_WEIGHTED
		self.neuron_sample_rate = neuron_sample_rate

		# Token bit encoder (defaults to standard binary)
		self.encoder = encoder if encoder is not None else BinaryTokenEncoder(vocab_size=vocab_size)
		self.bits_per_token = self.encoder.bits_per_token
		self.num_bits = self.bits_per_token  # 16 for GPT-2

		# Total input bits
		self.total_input_bits = context_size * self.bits_per_token

		# The layer: 16 clusters (one per output bit), shared connections
		# Force DENSE backend — our Rust+Metal train_and_eval path uses dense
		# bit-packed memory directly, so the Python Memory must have the full
		# address space allocated (not SPARSE/LSH which use hash tables).
		from wnn.ram.core.RAMClusterLayer import MemoryBackend
		self.layer = BitwiseRAMClusterLayer(
			total_input_bits=self.total_input_bits,
			num_clusters=self.num_bits,
			neurons_per_cluster=neurons_per_cluster,
			bits_per_neuron=bits_per_neuron,
			rng=rng,
			backend=MemoryBackend.DENSE,
		)

		# Pre-compute token bit patterns via encoder: [vocab_size, num_bits]
		# This matrix is used by reconstruct_logprobs for the log-product
		token_ids_all = arange(vocab_size, dtype=long)
		self.register_buffer(
			"token_bits",
			self.encoder.encode_tokens_batch(token_ids_all).float()
		)

		# Memory mode names for display
		self._mode_names = {0: "TERNARY", 1: "QUAD_BINARY", 2: "QUAD_WEIGHTED"}

	def __repr__(self) -> str:
		mode_name = self._mode_names.get(self.memory_mode, f"UNKNOWN({self.memory_mode})")
		rate_str = f", sample_rate={self.neuron_sample_rate}" if self.neuron_sample_rate < 1.0 else ""
		return (
			f"BitwiseRAMLM("
			f"vocab={self.vocab_size}, "
			f"context={self.context_size}, "
			f"num_bits={self.num_bits}, "
			f"neurons_per_cluster={self.layer.neurons_per_cluster}, "
			f"bits_per_neuron={self.layer.bits_per_neuron}, "
			f"total_neurons={self.layer.total_neurons}, "
			f"encoder={self.encoder!r}, "
			f"mode={mode_name}{rate_str})"
		)

	# =========================================================================
	# Encoding (delegated to self.encoder)
	# =========================================================================

	def encode_token(self, token_id: int) -> Tensor:
		"""Encode a single token ID to bits via the encoder."""
		return self.encoder.encode_token(token_id)

	def encode_tokens_batch(self, token_ids: Tensor) -> Tensor:
		"""Vectorized encoding of multiple token IDs → [N, bits_per_token]."""
		return self.encoder.encode_tokens_batch(token_ids)

	def encode_context(self, token_ids: list[int]) -> Tensor:
		"""Encode a context of token IDs to bits, with padding."""
		if len(token_ids) < self.context_size:
			padded = [self.pad_token_id] * (self.context_size - len(token_ids)) + list(token_ids)
		else:
			padded = list(token_ids[-self.context_size:])

		tokens_tensor = tensor(padded, dtype=long)
		bits_2d = self.encode_tokens_batch(tokens_tensor)
		return bits_2d.flatten()

	def encode_sequence(self, token_ids: list[int]) -> Tensor:
		"""Encode all sliding windows from a token sequence."""
		n = len(token_ids)
		num_examples = n - self.context_size

		if num_examples <= 0:
			return zeros(0, self.total_input_bits, dtype=torch_bool)

		tokens = tensor(token_ids, dtype=long)
		indices = arange(num_examples).unsqueeze(1) + arange(self.context_size)
		contexts = tokens[indices]

		# Encode via encoder: flatten → encode → reshape
		flat_ids = contexts.flatten()
		flat_bits = self.encode_tokens_batch(flat_ids)
		return flat_bits.view(num_examples, self.context_size, -1).reshape(num_examples, -1)

	# =========================================================================
	# Token-to-bits conversion
	# =========================================================================

	def _target_to_bits(self, target_ids: Tensor) -> Tensor:
		"""Convert token IDs to per-bit labels via the encoder.

		Args:
			target_ids: [N] int64 tensor of token IDs

		Returns:
			[N, num_bits] int tensor (0/1 per bit)
		"""
		return self.encoder.encode_tokens_batch(target_ids).to(dtype=long)

	# =========================================================================
	# Forward pass
	# =========================================================================

	def forward_bits(self, input_bits: Tensor) -> Tensor:
		"""Forward pass returning per-bit P(bit=1) scores.

		Dispatches based on memory_mode:
		  TERNARY (0): inherited RAMClusterLayer forward
		  QUAD_BINARY (1): count(cell >= 2) / neurons
		  QUAD_WEIGHTED (2): sum(weight[cell]) / neurons

		Args:
			input_bits: [batch, total_input_bits] boolean tensor

		Returns:
			[batch, num_bits] float tensor of per-bit probabilities
		"""
		if self.memory_mode == 0:
			return self.layer(input_bits)

		# Quad modes: use Rust forward functions
		import ram_accelerator
		import numpy as np

		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		num_examples = input_bits.shape[0]
		input_bits_np = input_bits.flatten().bool().numpy().astype(np.uint8)
		connections_np = self.layer.memory.connections.flatten().numpy().astype(np.int64)
		memory_words_np = self.layer.memory.memory_words.flatten().numpy().astype(np.int64)

		if self.memory_mode == 1:
			probs_flat = ram_accelerator.ramlm_forward_batch_quad_binary_numpy(
				input_bits_np, connections_np, memory_words_np,
				num_examples, self.total_input_bits, self.layer.total_neurons,
				self.layer.bits_per_neuron, self.layer.neurons_per_cluster,
				self.num_bits, self.layer.memory.words_per_neuron,
			)
		else:
			probs_flat = ram_accelerator.ramlm_forward_batch_quad_weighted_numpy(
				input_bits_np, connections_np, memory_words_np,
				num_examples, self.total_input_bits, self.layer.total_neurons,
				self.layer.bits_per_neuron, self.layer.neurons_per_cluster,
				self.num_bits, self.layer.memory.words_per_neuron,
			)

		return tensor(probs_flat, dtype=float32).view(num_examples, self.num_bits)

	def forward(self, input_bits: Tensor) -> Tensor:
		"""Forward pass returning token log-probabilities.

		Reconstructs log P(token=t) via matmul:
			log_probs = log(p1) @ token_bits.T + log(1-p1) @ (1-token_bits).T

		Args:
			input_bits: [batch, total_input_bits] boolean tensor

		Returns:
			[batch, vocab_size] float tensor of log-probabilities
		"""
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		bit_scores = self.forward_bits(input_bits)  # [batch, num_bits]
		return reconstruct_logprobs(bit_scores, self.token_bits)

	def forward_tokens(self, token_ids: list[int]) -> Tensor:
		"""Forward pass from token IDs → [vocab_size] log-probabilities."""
		bits = self.encode_context(token_ids).unsqueeze(0)
		log_probs = self.forward(bits)
		return log_probs[0]

	# =========================================================================
	# Training
	# =========================================================================

	def train_epoch_fast(
		self,
		token_ids: list[int],
		batch_size: int = 1000,
		allow_override: bool = False,
		verbose: bool = True,
	) -> dict:
		"""Batch training on a sequence of token IDs.

		Encodes all contexts, converts targets to per-bit labels,
		and trains via BitwiseRAMClusterLayer.train().

		Args:
			token_ids: Training token IDs (full sequence)
			batch_size: Number of examples per batch
			allow_override: Whether FALSE can override non-EMPTY cells
			verbose: Print progress

		Returns:
			Training statistics dict
		"""
		total_examples = len(token_ids) - self.context_size

		if verbose:
			print(f"Training on {total_examples:,} examples (batch_size={batch_size})...")

		# Pre-encode all contexts
		if verbose:
			print("  Encoding contexts...")
		all_bits = self.encode_sequence(token_ids)  # [N, total_input_bits]

		# Get all targets and convert to per-bit labels
		targets = tensor(token_ids[self.context_size:], dtype=long)
		all_target_bits = self._target_to_bits(targets)  # [N, num_bits]

		total_modified = 0
		num_batches = (total_examples + batch_size - 1) // batch_size

		if verbose:
			print("  Training batches...")

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			batch_bits = all_bits[start:end]
			batch_targets = all_target_bits[start:end]

			# Pass memory mode and sampling config
			rng_seed = 42 + batch_idx * 997
			modified = self.layer.train(
				batch_bits,
				batch_targets,
				allow_override=allow_override,
				memory_mode=self.memory_mode,
				neuron_sample_rate=self.neuron_sample_rate,
				rng_seed=rng_seed,
			)
			total_modified += modified

			if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
				pct = (end / total_examples) * 100
				print(f"    {pct:5.1f}% ({end:,}/{total_examples:,})")

		if verbose:
			print(f"Training complete. Modified {total_modified:,} memory cells.")

		return {
			"examples": total_examples,
			"modified": total_modified,
		}

	# =========================================================================
	# Evaluation
	# =========================================================================

	def evaluate_fast(
		self,
		token_ids: list[int],
		batch_size: int = 5000,
		verbose: bool = True,
		per_bit: bool = False,
	) -> dict:
		"""Evaluate on a token sequence.

		Computes cross-entropy, perplexity, and accuracy using the
		log-product reconstruction.

		Args:
			token_ids: Token IDs to evaluate on
			batch_size: Batch size for forward pass
			verbose: Print progress
			per_bit: If True, include per-bit accuracy breakdown

		Returns:
			Dict with cross_entropy, perplexity, accuracy, etc.
		"""
		from math import exp as math_exp
		from torch import logsumexp

		total_examples = len(token_ids) - self.context_size

		if verbose:
			print(f"Evaluating on {total_examples:,} examples (batch_size={batch_size})...")
			print("  Encoding contexts...")

		all_bits = self.encode_sequence(token_ids)
		targets = tensor(token_ids[self.context_size:], dtype=long)

		total_ce = 0.0
		total_correct = 0
		num_batches = (total_examples + batch_size - 1) // batch_size

		# Per-bit tracking
		bit_correct = zeros(self.num_bits)
		bit_total = 0

		if verbose:
			print("  Evaluating batches...")

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)
			batch_len = end - start

			batch_bits = all_bits[start:end]
			batch_targets = targets[start:end]

			# Get log-probs (unnormalized)
			log_probs = self.forward(batch_bits)  # [batch, vocab_size]

			# CE = logsumexp(log_probs) - log_prob[target]
			# This is numerically stable (avoids softmax float32 precision loss)
			lse = logsumexp(log_probs, dim=-1)  # [batch]
			target_lp = log_probs[arange(batch_len), batch_targets]  # [batch]
			batch_ce = (lse - target_lp).sum().item()
			total_ce += batch_ce

			# Accuracy
			predicted = log_probs.argmax(dim=-1)
			total_correct += (predicted == batch_targets).sum().item()

			# Per-bit accuracy
			if per_bit:
				bit_scores = self.forward_bits(batch_bits)  # [batch, num_bits]
				bit_preds = (bit_scores > 0.5).long()
				target_bits = self._target_to_bits(batch_targets)
				bit_correct += (bit_preds == target_bits).float().sum(dim=0).cpu()
				bit_total += batch_len

			if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
				pct = (end / total_examples) * 100
				running_ce = total_ce / end
				running_acc = total_correct / end
				print(f"    {pct:5.1f}% - CE: {running_ce:.4f}, Acc: {running_acc:.2%}")

		ce = total_ce / total_examples
		ppl = math_exp(min(ce, 100))  # Cap to avoid overflow
		accuracy = total_correct / total_examples

		stats = {
			"cross_entropy": ce,
			"perplexity": ppl,
			"accuracy": accuracy,
			"total": total_examples,
			"correct": total_correct,
		}

		if per_bit and bit_total > 0:
			per_bit_acc = (bit_correct / bit_total).tolist()
			stats["per_bit_accuracy"] = per_bit_acc
			stats["mean_bit_accuracy"] = sum(per_bit_acc) / len(per_bit_acc)

		if verbose:
			print(f"  Cross-entropy: {ce:.4f}")
			print(f"  Perplexity: {ppl:.2f}")
			print(f"  Accuracy: {accuracy:.2%}")
			if per_bit and bit_total > 0:
				print(f"  Mean bit accuracy: {stats['mean_bit_accuracy']:.2%}")
				for i, acc in enumerate(per_bit_acc):
					label = "easy" if acc > 0.9 else "hard" if acc < 0.6 else ""
					print(f"    bit {i:2d}: {acc:.2%} {label}")

		return stats

	def train_and_eval_metal(
		self,
		token_ids: list[int],
		verbose: bool = True,
		per_bit: bool = True,
	) -> dict:
		"""Complete train + eval in one Rust+Metal call.

		Much faster than train_epoch_fast() + evaluate_fast() because:
		1. Training uses neuron-parallel (no CAS atomics)
		2. Forward uses rayon-parallel
		3. Reconstruction + CE uses Metal GPU
		4. Zero PyTorch overhead for the hot path

		Args:
			token_ids: Full token sequence (train portion, or use separate train/eval)
			verbose: Print progress
			per_bit: Include per-bit accuracy

		Returns:
			Dict with cross_entropy, perplexity, accuracy, per_bit_accuracy, etc.
		"""
		import ram_accelerator
		import numpy as np
		from math import exp as math_exp
		from time import time as _time

		# Split into train/eval (same tokens for now — caller handles the split)
		# Actually this method takes separate train/test sequences
		raise NotImplementedError("Use train_and_eval_metal_split() instead")

	def train_and_eval_metal_split(
		self,
		train_tokens: list[int],
		test_tokens: list[int],
		verbose: bool = True,
		per_bit: bool = True,
	) -> tuple[dict, dict]:
		"""Complete train + eval in one Rust+Metal call.

		Returns: (train_stats, eval_stats)
		"""
		import ram_accelerator
		import numpy as np
		from math import exp as math_exp
		from time import time as _time

		t0 = _time()

		# Encode train
		if verbose:
			print("  Encoding train contexts...")
		train_all_bits = self.encode_sequence(train_tokens)
		train_targets = tensor(train_tokens[self.context_size:], dtype=long)
		train_target_bits = self._target_to_bits(train_targets)

		num_train = len(train_tokens) - self.context_size

		# Encode eval
		if verbose:
			print("  Encoding eval contexts...")
		eval_all_bits = self.encode_sequence(test_tokens)
		eval_targets = tensor(test_tokens[self.context_size:], dtype=long)
		num_eval = len(test_tokens) - self.context_size

		# Prepare numpy arrays
		train_input_np = train_all_bits.flatten().bool().numpy().astype(np.uint8)
		train_target_np = train_target_bits.flatten().numpy().astype(np.uint8)
		eval_input_np = eval_all_bits.flatten().bool().numpy().astype(np.uint8)
		eval_targets_np = eval_targets.numpy().astype(np.uint32)

		# Token bits table
		token_bits_np = self.token_bits.numpy().astype(np.uint8).flatten()

		connections_np = self.layer.memory.connections.flatten().numpy().astype(np.int64)

		# Reset memory before training
		self.reset_memory()
		memory_np = self.layer.memory.memory_words.flatten().numpy().astype(np.int64)

		encode_time = _time() - t0
		if verbose:
			print(f"  Encoding done ({encode_time:.1f}s). Training + eval in Rust+Metal...")

		t1 = _time()
		ce, acc, per_bit_acc, updated_memory = ram_accelerator.ramlm_bitwise_train_and_eval_numpy(
			train_input_np, train_target_np,
			eval_input_np, eval_targets_np,
			token_bits_np, connections_np, memory_np,
			num_train, num_eval,
			self.total_input_bits,
			self.layer.bits_per_neuron,
			self.layer.neurons_per_cluster,
			self.num_bits,
			self.layer.memory.words_per_neuron,
			self.vocab_size,
			self.memory_mode,
			self.neuron_sample_rate,
			42,  # rng_seed
		)
		rust_time = _time() - t1

		# Update memory in model
		import torch
		self.layer.memory.memory_words = torch.tensor(
			updated_memory, dtype=torch.int64
		).view(self.layer.memory.memory_words.shape)

		ppl = math_exp(min(ce, 100))

		train_stats = {
			"examples": num_train,
			"modified": 0,  # not tracked in full path
		}

		eval_stats = {
			"cross_entropy": ce,
			"perplexity": ppl,
			"accuracy": acc,
			"total": num_eval,
		}

		if per_bit:
			eval_stats["per_bit_accuracy"] = per_bit_acc
			eval_stats["mean_bit_accuracy"] = sum(per_bit_acc) / len(per_bit_acc)

		if verbose:
			print(f"  Rust+Metal time: {rust_time:.1f}s (encode: {encode_time:.1f}s)")
			print(f"  Cross-entropy: {ce:.4f}")
			print(f"  Perplexity: {ppl:.2f}")
			print(f"  Accuracy: {acc:.2%}")
			if per_bit:
				print(f"  Mean bit accuracy: {eval_stats['mean_bit_accuracy']:.2%}")
				for i, ba in enumerate(per_bit_acc):
					label = "easy" if ba > 0.9 else "hard" if ba < 0.6 else ""
					print(f"    bit {i:2d}: {ba:.2%} {label}")

		return train_stats, eval_stats

	# =========================================================================
	# Connectivity (for GA/TS optimization)
	# =========================================================================

	@property
	def connections(self) -> Tensor:
		return self.layer.connections

	@connections.setter
	def connections(self, value: Tensor) -> None:
		self.layer.connections = value

	def reset_memory(self) -> None:
		"""Reset memory to initial state based on memory mode."""
		if self.memory_mode == 0:
			# TERNARY: all cells = EMPTY (2)
			self.layer.reset_memory()
		else:
			# QUAD modes: all cells = WEAK_FALSE (1)
			from torch import int64 as torch_int64
			empty_word = 0
			for i in range(31):
				empty_word |= (1 << (i * 2))  # QUAD_WEAK_FALSE = 1
			self.layer.memory.memory_words.fill_(empty_word)

	# =========================================================================
	# Serialization
	# =========================================================================

	def get_config(self) -> dict:
		return {
			"vocab_size": self.vocab_size,
			"context_size": self.context_size,
			"neurons_per_cluster": self.layer.neurons_per_cluster,
			"bits_per_neuron": self.layer.bits_per_neuron,
			"pad_token_id": self.pad_token_id,
			"memory_mode": self.memory_mode,
			"neuron_sample_rate": self.neuron_sample_rate,
		}

	@classmethod
	def from_config(cls, config: dict) -> "BitwiseRAMLM":
		return cls(**config)
