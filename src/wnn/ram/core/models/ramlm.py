"""
RAMLM - RAM-based Language Model.

A single-layer RAM WNN language model that predicts the next token
given a context of previous tokens.

Architecture:
	INPUT: context_size tokens × bits_per_token bits
	         ↓
	[RAMClusterLayer] - vocab_size clusters × neurons_per_cluster
	         ↓
	OUTPUT: vocab_size probabilities

Key features:
	- Direct binary encoding of token IDs (no embedding layer)
	- Single RAMClusterLayer for prediction
	- Hybrid top-k training: TRUE for target, FALSE for confusing alternatives
	- Connectivity optimization via GA/TS for generalization

Usage:
	from wnn.ram.core.models import RAMLM

	# Create model
	model = RAMLM(
		vocab_size=50257,
		context_size=6,
		neurons_per_cluster=7,
		bits_per_neuron=10,
	)

	# Train
	model.train_epoch(train_tokens, global_top_k=1000)

	# Predict
	probs = model.forward_tokens([464, 3797, 3332, 319, 262])
	next_token, confidence = model.predict_tokens([464, 3797, 3332, 319, 262])
"""

from collections import Counter
from typing import Optional, Union

from torch import arange
from torch import bool as torch_bool
from torch import float32
from torch import long
from torch import tensor
from torch import Tensor
from torch import zeros

from wnn.ram.core.base import RAMComponent
from wnn.ram.core.RAMClusterLayer import RAMClusterLayer, bits_needed
from wnn.ram.core import AccelerationMode


class RAMLM(RAMComponent):
	"""
	RAM-based Language Model.

	A single-layer architecture where a RAMClusterLayer directly maps
	context bits to next-token probabilities.

	The model learns via:
	1. Memory writes: TRUE for correct next token, FALSE for alternatives
	2. Connectivity optimization: GA/TS to find good partial connectivity

	Attributes:
		vocab_size: Number of tokens in vocabulary
		context_size: Number of context tokens (n-gram order)
		bits_per_token: Bits used to encode each token ID
		pad_token_id: Token ID used for padding short contexts
		layer: The RAMClusterLayer that does the prediction
	"""

	def __init__(
		self,
		vocab_size: int = 50257,
		context_size: int = 6,
		neurons_per_cluster: int = 7,
		bits_per_neuron: int = 10,
		pad_token_id: int = 50256,
		tokenizer: Optional[str] = None,
		rng: Optional[int] = None,
	):
		"""
		Initialize RAMLM.

		Args:
			vocab_size: Size of vocabulary (default 50257 for GPT-2)
			context_size: Number of context tokens (default 6)
			neurons_per_cluster: Neurons per output cluster (default 7, odd for majority)
			bits_per_neuron: Bits each neuron observes (default 10)
			pad_token_id: Token ID for padding (default 50256 for GPT-2)
			tokenizer: Optional tokenizer name ("gpt2") or None for token-ID-only mode
			rng: Random seed for reproducible connectivity initialization
		"""
		super().__init__()

		self.vocab_size = vocab_size
		self.context_size = context_size
		self.bits_per_token = bits_needed(vocab_size)
		self.pad_token_id = pad_token_id

		# Total input bits
		self.total_input_bits = context_size * self.bits_per_token

		# Create the RAMClusterLayer
		self.layer = RAMClusterLayer(
			total_input_bits=self.total_input_bits,
			num_clusters=vocab_size,
			neurons_per_cluster=neurons_per_cluster,
			bits_per_neuron=bits_per_neuron,
			rng=rng,
		)

		# Optional tokenizer
		self._tokenizer = None
		if tokenizer == "gpt2":
			from wnn.tokenizers import GPT2Tokenizer
			self._tokenizer = GPT2Tokenizer()

		# Global top-k token IDs (set during training)
		self._global_top_k: Optional[list[int]] = None

		# Pre-compute bit positions for vectorized encoding [bits_per_token-1, ..., 0]
		self.register_buffer(
			"_bit_positions",
			arange(self.bits_per_token - 1, -1, -1, dtype=long)
		)

	def __repr__(self) -> str:
		return (
			f"RAMLM("
			f"vocab={self.vocab_size}, "
			f"context={self.context_size}, "
			f"bits_per_token={self.bits_per_token}, "
			f"total_input_bits={self.total_input_bits}, "
			f"neurons_per_cluster={self.layer.neurons_per_cluster}, "
			f"bits_per_neuron={self.layer.bits_per_neuron})"
		)

	def __str__(self) -> str:
		lines = [
			"=== RAMLM (RAM Language Model) ===",
			f"  Vocabulary size: {self.vocab_size:,}",
			f"  Context size: {self.context_size} tokens",
			f"  Bits per token: {self.bits_per_token}",
			f"  Total input bits: {self.total_input_bits}",
			f"  Neurons per cluster: {self.layer.neurons_per_cluster}",
			f"  Bits per neuron: {self.layer.bits_per_neuron}",
			f"  Total neurons: {self.layer.total_neurons:,}",
			f"  PAD token ID: {self.pad_token_id}",
			f"  Tokenizer: {'GPT-2' if self._tokenizer else 'None (token IDs only)'}",
			f"  Global top-k: {len(self._global_top_k) if self._global_top_k else 'Not set'}",
		]
		return "\n".join(lines)

	# =========================================================================
	# Encoding (Vectorized for speed)
	# =========================================================================

	def encode_token(self, token_id: int) -> Tensor:
		"""
		Encode a single token ID to binary.

		Args:
			token_id: Token ID (0 to vocab_size-1)

		Returns:
			[bits_per_token] boolean tensor
		"""
		# Vectorized: (token_id >> bit_positions) & 1
		return ((token_id >> self._bit_positions) & 1).bool()

	def encode_tokens_batch(self, token_ids: Tensor) -> Tensor:
		"""
		Vectorized encoding of multiple token IDs.

		Args:
			token_ids: [N] int64 tensor of token IDs

		Returns:
			[N, bits_per_token] boolean tensor
		"""
		# token_ids: [N], _bit_positions: [bits_per_token]
		# Broadcast: [N, 1] >> [bits_per_token] -> [N, bits_per_token]
		return ((token_ids.unsqueeze(-1) >> self._bit_positions) & 1).bool()

	def encode_context(self, token_ids: list[int]) -> Tensor:
		"""
		Encode a context of token IDs to binary, with padding.

		Right-aligns tokens (most recent at the end), pads with pad_token_id.

		Args:
			token_ids: List of token IDs (can be shorter than context_size)

		Returns:
			[total_input_bits] boolean tensor
		"""
		# Pad or truncate to context_size
		if len(token_ids) < self.context_size:
			padded = [self.pad_token_id] * (self.context_size - len(token_ids)) + list(token_ids)
		else:
			padded = list(token_ids[-self.context_size:])

		# Vectorized encoding of all tokens at once
		tokens_tensor = tensor(padded, dtype=long)
		bits_2d = self.encode_tokens_batch(tokens_tensor)  # [context_size, bits_per_token]
		return bits_2d.flatten()  # [total_input_bits]

	def encode_batch(self, contexts: list[list[int]]) -> Tensor:
		"""
		Encode a batch of contexts (vectorized).

		Args:
			contexts: List of token ID lists

		Returns:
			[batch, total_input_bits] boolean tensor
		"""
		batch_size = len(contexts)

		# Pad all contexts to context_size and stack into tensor
		padded = []
		for ctx in contexts:
			if len(ctx) < self.context_size:
				p = [self.pad_token_id] * (self.context_size - len(ctx)) + list(ctx)
			else:
				p = list(ctx[-self.context_size:])
			padded.append(p)

		# [batch, context_size]
		tokens_tensor = tensor(padded, dtype=long)

		# Vectorized encoding: [batch, context_size, bits_per_token]
		bits_3d = ((tokens_tensor.unsqueeze(-1) >> self._bit_positions) & 1).bool()

		# Flatten last two dims: [batch, total_input_bits]
		return bits_3d.view(batch_size, -1)

	def encode_sequence(self, token_ids: list[int]) -> Tensor:
		"""
		Encode all sliding windows from a token sequence (fully vectorized).

		This is the fastest way to prepare data for training/evaluation.

		Args:
			token_ids: Full token sequence

		Returns:
			[num_examples, total_input_bits] boolean tensor

		Where num_examples = len(token_ids) - context_size
		"""
		n = len(token_ids)
		num_examples = n - self.context_size

		if num_examples <= 0:
			return zeros(0, self.total_input_bits, dtype=torch_bool)

		# Convert to tensor once
		tokens = tensor(token_ids, dtype=long)

		# Create sliding window indices: [num_examples, context_size]
		# Row i contains indices [i, i+1, ..., i+context_size-1]
		indices = arange(num_examples).unsqueeze(1) + arange(self.context_size)

		# Gather contexts: [num_examples, context_size]
		contexts = tokens[indices]

		# Vectorized encoding: [num_examples, context_size, bits_per_token]
		bits_3d = ((contexts.unsqueeze(-1) >> self._bit_positions) & 1).bool()

		# Flatten: [num_examples, total_input_bits]
		return bits_3d.view(num_examples, -1)

	# =========================================================================
	# Forward / Predict
	# =========================================================================

	def forward_tokens(self, token_ids: list[int]) -> Tensor:
		"""
		Forward pass from token IDs.

		Args:
			token_ids: Context token IDs

		Returns:
			[vocab_size] probability tensor
		"""
		bits = self.encode_context(token_ids).unsqueeze(0)  # [1, total_input_bits]
		probs = self.layer(bits)  # [1, vocab_size]
		return probs[0]  # [vocab_size]

	def forward_batch(self, contexts: list[list[int]]) -> Tensor:
		"""
		Batch forward pass from token IDs.

		Args:
			contexts: List of context token ID lists

		Returns:
			[batch, vocab_size] probability tensor
		"""
		bits = self.encode_batch(contexts)  # [batch, total_input_bits]
		return self.layer(bits)  # [batch, vocab_size]

	def forward(
		self,
		input_bits: Tensor,
		backend: AccelerationMode = AccelerationMode.AUTO,
	) -> Tensor:
		"""
		Forward pass from pre-encoded bits.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor
			backend: Acceleration backend to use (default AUTO):
				- AUTO: Auto-select best backend based on batch size
				- PYTORCH: Pure PyTorch (best for small batches < 1000)
				- CPU: Rust CPU via rayon (16 cores)
				- METAL: Metal GPU (40 cores on M4 Max)
				- HYBRID: CPU+GPU combined (56 cores, best for large batches)

		Returns:
			[batch, vocab_size] probability tensor
		"""
		match backend:
			case AccelerationMode.AUTO:
				return self.layer.forward_auto(input_bits)
			case AccelerationMode.PYTORCH:
				return self.layer(input_bits)
			case AccelerationMode.CPU:
				return self.layer.forward_rust(input_bits)
			case AccelerationMode.METAL:
				return self.layer.forward_metal(input_bits)
			case AccelerationMode.HYBRID:
				return self.layer.forward_hybrid(input_bits)
			case _:
				return self.layer.forward_auto(input_bits)

	def predict_tokens(self, token_ids: list[int]) -> tuple[int, float]:
		"""
		Predict next token from context.

		Args:
			token_ids: Context token IDs

		Returns:
			Tuple of (predicted_token_id, confidence)
		"""
		probs = self.forward_tokens(token_ids)
		confidence, predicted = probs.max(dim=-1)
		return int(predicted.item()), float(confidence.item())

	def predict_batch(self, contexts: list[list[int]]) -> tuple[Tensor, Tensor]:
		"""
		Batch prediction.

		Args:
			contexts: List of context token ID lists

		Returns:
			Tuple of (predicted_token_ids [batch], confidences [batch])
		"""
		probs = self.forward_batch(contexts)
		confidences, predicted = probs.max(dim=-1)
		return predicted, confidences

	# =========================================================================
	# Training
	# =========================================================================

	def compute_global_top_k(self, token_ids: list[int], k: int = 1000) -> list[int]:
		"""
		Compute global top-k most frequent tokens from training data.

		Args:
			token_ids: Training token IDs
			k: Number of top tokens to return

		Returns:
			List of k most frequent token IDs
		"""
		counts = Counter(token_ids)
		top_k = [token_id for token_id, _ in counts.most_common(k)]
		self._global_top_k = top_k
		return top_k

	def train_example(
		self,
		context: list[int],
		target: int,
		context_negatives: Optional[list[int]] = None,
		allow_override: bool = False,
	) -> int:
		"""
		Train on a single example.

		Args:
			context: Context token IDs
			target: Target (next) token ID
			context_negatives: Optional context-specific negative token IDs
			allow_override: Whether to override existing memory cells

		Returns:
			Number of memory cells modified
		"""
		# Encode context
		input_bits = self.encode_context(context).unsqueeze(0)  # [1, total_input_bits]
		true_clusters = tensor([target], dtype=long)

		# Build false clusters: global top-k + context-specific
		false_set = set()
		if self._global_top_k:
			false_set.update(self._global_top_k)
		if context_negatives:
			false_set.update(context_negatives)
		# Remove target from negatives
		false_set.discard(target)

		if false_set:
			false_clusters = tensor([list(false_set)], dtype=long)  # [1, k]
		else:
			false_clusters = None

		return self.layer.train_batch(
			input_bits,
			true_clusters,
			false_clusters,
			allow_override=allow_override,
		)

	def train_epoch(
		self,
		token_ids: list[int],
		global_top_k: int = 1000,
		context_top_k: int = 10,
		smoothing=None,
		allow_override: bool = False,
		verbose: bool = True,
	) -> dict:
		"""
		Train on a sequence of token IDs for one epoch.

		For each position, extracts context and target, then trains:
		- TRUE for target cluster
		- FALSE for global top-k + context-specific alternatives

		Args:
			token_ids: Training token IDs (full sequence)
			global_top_k: Number of global most-frequent tokens for FALSE training
			context_top_k: Number of context-specific alternatives (from smoothing)
			smoothing: Optional smoothing model for context-specific negatives
			allow_override: Whether to override existing memory cells
			verbose: Print progress

		Returns:
			Training statistics dict
		"""
		# Compute global top-k if not already set
		if self._global_top_k is None or len(self._global_top_k) != global_top_k:
			if verbose:
				print(f"Computing global top-{global_top_k} tokens...")
			self.compute_global_top_k(token_ids, global_top_k)

		total_examples = len(token_ids) - self.context_size
		total_modified = 0

		if verbose:
			print(f"Training on {total_examples:,} examples...")

		for i in range(total_examples):
			context = token_ids[i:i + self.context_size]
			target = token_ids[i + self.context_size]

			# Get context-specific negatives from smoothing
			context_negatives = None
			if smoothing is not None and context_top_k > 0:
				# Get top-k predictions from smoothing model
				# (This is a placeholder - actual implementation depends on smoothing API)
				pass

			modified = self.train_example(
				context,
				target,
				context_negatives,
				allow_override=allow_override,
			)
			total_modified += modified

			if verbose and (i + 1) % 10000 == 0:
				print(f"  Trained {i + 1:,}/{total_examples:,} examples...")

		if verbose:
			print(f"Training complete. Modified {total_modified:,} memory cells.")

		return {
			"examples": total_examples,
			"modified": total_modified,
		}

	def train_epoch_fast(
		self,
		token_ids: list[int],
		global_top_k: int = 1000,
		batch_size: int = 1000,
		allow_override: bool = False,
		verbose: bool = True,
	) -> dict:
		"""
		Fast batch training on a sequence of token IDs.

		Uses vectorized encoding and batch processing for 10-50x speedup
		over train_epoch().

		Args:
			token_ids: Training token IDs (full sequence)
			global_top_k: Number of global most-frequent tokens for FALSE training
			batch_size: Number of examples to process per batch
			allow_override: Whether to override existing memory cells
			verbose: Print progress

		Returns:
			Training statistics dict
		"""
		# Compute global top-k if not already set
		if self._global_top_k is None or len(self._global_top_k) != global_top_k:
			if verbose:
				print(f"Computing global top-{global_top_k} tokens...")
			self.compute_global_top_k(token_ids, global_top_k)

		total_examples = len(token_ids) - self.context_size

		if verbose:
			print(f"Training on {total_examples:,} examples (batch_size={batch_size})...")

		# Pre-encode all contexts at once (vectorized)
		if verbose:
			print("  Encoding contexts...")
		all_bits = self.encode_sequence(token_ids)  # [num_examples, total_input_bits]

		# Get all targets
		targets = tensor(token_ids[self.context_size:], dtype=long)  # [num_examples]

		# Prepare global negatives tensor (same for all examples)
		false_clusters_base = tensor(self._global_top_k, dtype=long)  # [global_top_k]

		total_modified = 0
		num_batches = (total_examples + batch_size - 1) // batch_size

		if verbose:
			print("  Training batches...")

		# Prepare false_clusters for all examples: [num_examples, global_top_k]
		# (We broadcast the same negatives to all examples; conflicts are handled
		# by the layer - TRUE writes happen before FALSE, so target wins)
		false_clusters_all = false_clusters_base.unsqueeze(0).expand(total_examples, -1)

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			# Get batch data
			batch_bits = all_bits[start:end]  # [batch_len, total_input_bits]
			batch_targets = targets[start:end]  # [batch_len]
			batch_false = false_clusters_all[start:end]  # [batch_len, global_top_k]

			# Fully vectorized training for the entire batch
			modified = self.layer.train_multi_examples(
				batch_bits,
				batch_targets,
				batch_false,
				allow_override=allow_override,
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

	def train_epoch_fast_rust(
		self,
		token_ids: list[int],
		global_top_k: int = 1000,
		batch_size: int = 1000,
		allow_override: bool = False,
		verbose: bool = True,
	) -> dict:
		"""
		Rust-accelerated batch training on a sequence of token IDs.

		Uses Rust + rayon for parallel training. Typically faster than
		train_epoch_fast() for large batches.

		Args:
			token_ids: Training token IDs (full sequence)
			global_top_k: Number of global most-frequent tokens for FALSE training
			batch_size: Number of examples to process per batch
			allow_override: Whether to override existing memory cells
			verbose: Print progress

		Returns:
			Training statistics dict
		"""
		# Compute global top-k if not already set
		if self._global_top_k is None or len(self._global_top_k) != global_top_k:
			if verbose:
				print(f"Computing global top-{global_top_k} tokens...")
			self.compute_global_top_k(token_ids, global_top_k)

		total_examples = len(token_ids) - self.context_size

		if verbose:
			print(f"Training on {total_examples:,} examples (batch_size={batch_size}, backend=Rust)...")

		# Pre-encode all contexts at once (vectorized)
		if verbose:
			print("  Encoding contexts...")
		all_bits = self.encode_sequence(token_ids)  # [num_examples, total_input_bits]

		# Get all targets
		targets = tensor(token_ids[self.context_size:], dtype=long)  # [num_examples]

		# Prepare global negatives tensor (same for all examples)
		false_clusters_base = tensor(self._global_top_k, dtype=long)  # [global_top_k]

		total_modified = 0
		num_batches = (total_examples + batch_size - 1) // batch_size

		if verbose:
			print("  Training batches (Rust)...")

		# Prepare false_clusters for all examples: [num_examples, global_top_k]
		false_clusters_all = false_clusters_base.unsqueeze(0).expand(total_examples, -1).contiguous()

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			# Get batch data
			batch_bits = all_bits[start:end]  # [batch_len, total_input_bits]
			batch_targets = targets[start:end]  # [batch_len]
			batch_false = false_clusters_all[start:end]  # [batch_len, global_top_k]

			# Use Rust training
			modified = self.layer.train_multi_examples_rust_numpy(
				batch_bits,
				batch_targets,
				batch_false,
				allow_override=allow_override,
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

	def train_epoch_fast_auto(
		self,
		token_ids: list[int],
		global_top_k: int = 1000,
		batch_size: int = 1000,
		allow_override: bool = False,
		verbose: bool = True,
	) -> dict:
		"""
		Auto-optimized batch training that picks the best backend.

		Tries Rust first, falls back to PyTorch if unavailable.

		Args:
			token_ids: Training token IDs (full sequence)
			global_top_k: Number of global most-frequent tokens for FALSE training
			batch_size: Number of examples to process per batch
			allow_override: Whether to override existing memory cells
			verbose: Print progress

		Returns:
			Training statistics dict
		"""
		try:
			import ram_accelerator
			return self.train_epoch_fast_rust(
				token_ids, global_top_k, batch_size, allow_override, verbose
			)
		except ImportError:
			if verbose:
				print("Rust accelerator not available, using PyTorch...")
			return self.train_epoch_fast(
				token_ids, global_top_k, batch_size, allow_override, verbose
			)

	# =========================================================================
	# Evaluation
	# =========================================================================

	def evaluate(
		self,
		token_ids: list[int],
		verbose: bool = True,
	) -> dict:
		"""
		Evaluate on a sequence of token IDs.

		Args:
			token_ids: Token IDs to evaluate on
			verbose: Print progress

		Returns:
			Dict with cross_entropy, perplexity, accuracy, etc.
		"""
		from wnn.ram.strategies.perplexity import PerplexityCalculator

		calc = PerplexityCalculator(vocab_size=self.vocab_size)
		total_examples = len(token_ids) - self.context_size

		if verbose:
			print(f"Evaluating on {total_examples:,} examples...")

		for i in range(total_examples):
			context = token_ids[i:i + self.context_size]
			target = token_ids[i + self.context_size]

			probs = self.forward_tokens(context)
			target_prob = float(probs[target].item())
			predicted = int(probs.argmax().item())

			calc.add_from_probability(target_prob, is_correct=(predicted == target))

		stats = calc.get_stats()

		if verbose:
			print(f"  Cross-entropy: {stats['cross_entropy']:.4f}")
			print(f"  Perplexity: {stats['perplexity']:.2f}")
			print(f"  Accuracy: {stats['accuracy']:.2%}")

		return stats

	def evaluate_fast(
		self,
		token_ids: list[int],
		batch_size: int = 5000,
		backend: AccelerationMode = AccelerationMode.AUTO,
		verbose: bool = True,
	) -> dict:
		"""
		Fast batch evaluation on a sequence of token IDs.

		Uses vectorized encoding and accelerated forward pass for significant speedup.
		With AUTO backend, uses Hybrid CPU+GPU for ~2x speedup on large batches.

		Args:
			token_ids: Token IDs to evaluate on
			batch_size: Number of examples to process per batch
			backend: Acceleration backend (AUTO, PYTORCH, CPU, METAL, HYBRID)
			verbose: Print progress

		Returns:
			Dict with cross_entropy, perplexity, accuracy, etc.
		"""
		from math import log

		total_examples = len(token_ids) - self.context_size

		backend_name = backend.name if hasattr(backend, 'name') else str(backend)
		if verbose:
			print(f"Evaluating on {total_examples:,} examples (batch_size={batch_size}, backend={backend_name})...")

		# Pre-encode all contexts at once (vectorized)
		if verbose:
			print("  Encoding contexts...")
		all_bits = self.encode_sequence(token_ids)  # [num_examples, total_input_bits]

		# Get all targets
		targets = tensor(token_ids[self.context_size:], dtype=long)  # [num_examples]

		# Use PerplexityCalculator for consistent normalized perplexity
		from wnn.ram.strategies.perplexity import PerplexityCalculator
		calc = PerplexityCalculator(vocab_size=self.vocab_size)

		num_batches = (total_examples + batch_size - 1) // batch_size

		if verbose:
			print("  Evaluating batches...")

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)

			# Get batch data
			batch_bits = all_bits[start:end]  # [batch_len, total_input_bits]
			batch_targets = targets[start:end]  # [batch_len]

			# Batch forward pass with selected backend
			scores = self.forward(batch_bits, backend=backend)  # [batch_len, vocab_size]

			# Add to calculator - it handles softmax normalization internally
			calc.add_from_scores_batch(scores, batch_targets, normalize=True)

			if verbose and (batch_idx + 1) % max(1, num_batches // 5) == 0:
				pct = (end / total_examples) * 100
				stats = calc.get_stats()
				print(f"    {pct:5.1f}% - CE: {stats['cross_entropy']:.4f}, PPL: {stats['perplexity']:.2f}, Acc: {stats['accuracy']:.2%}")

		# Get final stats from calculator
		stats = calc.get_stats()

		if verbose:
			print(f"  Cross-entropy: {stats['cross_entropy']:.4f}")
			print(f"  Perplexity: {stats['perplexity']:.2f}")
			print(f"  Accuracy: {stats['accuracy']:.2%}")

		return stats

	# =========================================================================
	# Connectivity (for GA/TS optimization)
	# =========================================================================

	@property
	def connections(self) -> Tensor:
		"""Get connectivity matrix [total_neurons, bits_per_neuron]."""
		return self.layer.connections

	@connections.setter
	def connections(self, value: Tensor) -> None:
		"""Set connectivity matrix."""
		self.layer.connections = value

	def reset_memory(self) -> None:
		"""
		Reset all memory cells to EMPTY, preserving connectivity.

		Clears all learned mappings while keeping the connectivity pattern.
		Useful for retraining after connectivity optimization.
		"""
		self.layer.reset_memory()

	# =========================================================================
	# Serialization
	# =========================================================================

	def get_config(self) -> dict:
		"""Get configuration for model recreation."""
		return {
			"vocab_size": self.vocab_size,
			"context_size": self.context_size,
			"neurons_per_cluster": self.layer.neurons_per_cluster,
			"bits_per_neuron": self.layer.bits_per_neuron,
			"pad_token_id": self.pad_token_id,
		}

	@classmethod
	def from_config(cls, config: dict) -> "RAMLM":
		"""Create RAMLM from configuration dict."""
		return cls(**config)

	def save(self, path: str) -> None:
		"""Save model to file."""
		from wnn.ram.core.serialization import save_model
		save_model(self, path)

	@classmethod
	def load(cls, path: str, device: str = "cpu") -> "RAMLM":
		"""Load model from file."""
		from wnn.ram.core.serialization import load_model
		return load_model(path, model_class=cls, device=device)
