"""
Bitwise Evaluator - Evaluates BitwiseRAMLM genomes for GA/TS optimization.

Uses full Rust+Metal pipeline for maximum performance:
- Token encoding: done once at init (in Rust)
- Training: CPU (rayon parallel over genomes)
- Forward pass: CPU (rayon parallel)
- Reconstruction + CE: Metal GPU (50K vocab × 16 bits matmul)

Falls back to Python-only evaluation if ram_accelerator is unavailable.

Usage:
	evaluator = BitwiseEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=50257,
		context_size=4,
		neurons_per_cluster=1000,
		bits_per_neuron=10,
	)

	# Evaluate genomes (same interface as CachedEvaluator)
	results = evaluator.evaluate_batch(genomes)
	# → [(ce, acc), (ce, acc), ...]
"""

import time
from typing import Optional, Callable

from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


class BitwiseEvaluator:
	"""
	Evaluator for BitwiseRAMLM genomes.

	Primary path: Rust+Metal (all 50 genomes evaluated in parallel).
	Fallback: Python (sequential, creates BitwiseRAMLM per genome).

	Supports data rotation (subset training) for GA/TS diversity,
	matching CachedEvaluator's interface.
	"""

	def __init__(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int = 50257,
		context_size: int = 4,
		neurons_per_cluster: int = 1000,
		bits_per_neuron: int = 10,
		num_parts: int = 3,
		num_eval_parts: int = 1,
		seed: Optional[int] = None,
		pad_token_id: int = 50256,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
	):
		self._vocab_size = vocab_size
		self._context_size = context_size
		self._neurons_per_cluster = neurons_per_cluster
		self._bits_per_neuron = bits_per_neuron
		self._num_parts = num_parts
		self._num_eval_parts = num_eval_parts
		self._pad_token_id = pad_token_id
		self._memory_mode = memory_mode
		self._neuron_sample_rate = neuron_sample_rate

		self._train_tokens = train_tokens
		self._eval_tokens = eval_tokens

		# Compute input dimensions
		self._bits_per_token = bits_needed(vocab_size)
		self._total_input_bits = context_size * self._bits_per_token

		# Subset rotation (for Python fallback)
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)
		self._seed = seed
		self._train_rotation_idx = 0
		self._eval_rotation_idx = 0

		# Try Rust+Metal backend
		self._rust_cache = None
		try:
			from ram_accelerator import BitwiseCacheWrapper
			self._rust_cache = BitwiseCacheWrapper(
				train_tokens=list(train_tokens),
				eval_tokens=list(eval_tokens),
				vocab_size=vocab_size,
				context_size=context_size,
				num_parts=num_parts,
				num_eval_parts=num_eval_parts,
				pad_token_id=pad_token_id,
			)
			print(f"[BitwiseEvaluator] Rust+Metal backend active "
				  f"(train: {num_parts} subsets, eval: {num_eval_parts} subsets)")
		except (ImportError, Exception) as e:
			print(f"[BitwiseEvaluator] Rust backend unavailable ({e}), using Python fallback")
			# Pre-split training data for Python fallback
			n = len(train_tokens)
			part_size = n // num_parts
			self._train_parts = []
			for i in range(num_parts):
				start = i * part_size
				end = start + part_size if i < num_parts - 1 else n
				self._train_parts.append(train_tokens[start:end])

	def next_train_idx(self) -> int:
		"""Advance and return next train subset index."""
		if self._rust_cache is not None:
			return self._rust_cache.next_train_idx()
		idx = self._train_rotation_idx % self._num_parts
		self._train_rotation_idx += 1
		return idx

	def next_eval_idx(self) -> int:
		"""Advance and return next eval subset index."""
		if self._rust_cache is not None:
			return self._rust_cache.next_eval_idx()
		idx = self._eval_rotation_idx % self._num_eval_parts
		self._eval_rotation_idx += 1
		return idx

	@property
	def vocab_size(self) -> int:
		return self._vocab_size

	@property
	def total_input_bits(self) -> int:
		return self._total_input_bits

	@property
	def num_parts(self) -> int:
		return self._num_parts

	@property
	def num_eval_parts(self) -> int:
		return self._num_eval_parts

	# =========================================================================
	# Rust+Metal evaluation (primary path)
	# =========================================================================

	def _flatten_genomes_heterogeneous(
		self,
		genomes: list[ClusterGenome],
	) -> tuple[list[int], list[int], list[int]]:
		"""
		Flatten per-cluster arrays from genomes for Rust.

		Returns:
			(bits_flat, neurons_flat, connections_flat) where:
			- bits_flat: [num_genomes * num_clusters]
			- neurons_flat: [num_genomes * num_clusters]
			- connections_flat: variable total
		"""
		import random
		num_clusters = bits_needed(self._vocab_size)

		bits_flat = []
		neurons_flat = []
		connections_flat = []

		for g in genomes:
			bits_flat.extend(g.bits_per_cluster)
			neurons_flat.extend(g.neurons_per_cluster)
			if g.connections is not None:
				connections_flat.extend(g.connections)
			else:
				# Generate random connections based on per-cluster config
				for c in range(num_clusters):
					n_neurons = g.neurons_per_cluster[c]
					n_bits = g.bits_per_cluster[c]
					for _ in range(n_neurons * n_bits):
						connections_flat.append(random.randint(0, self._total_input_bits - 1))

		return bits_flat, neurons_flat, connections_flat

	def _evaluate_batch_rust(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: int,
		eval_subset_idx: int,
	) -> list[tuple[float, float]]:
		"""Evaluate using Rust+Metal backend with per-cluster heterogeneous configs."""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes_heterogeneous(genomes)
		return self._rust_cache.evaluate_genomes(
			bits_per_cluster_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			train_subset_idx=train_subset_idx,
			eval_subset_idx=eval_subset_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

	def _evaluate_batch_full_rust(
		self,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float]]:
		"""Evaluate with full data using Rust+Metal backend."""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes_heterogeneous(genomes)
		return self._rust_cache.evaluate_genomes_full(
			bits_per_cluster_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

	# =========================================================================
	# Python fallback evaluation
	# =========================================================================

	def _evaluate_batch_python(
		self,
		genomes: list[ClusterGenome],
		train_tokens: list[int],
		eval_tokens: list[int],
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
	) -> list[tuple[float, float]]:
		"""Fallback: evaluate using Python BitwiseRAMLM (sequential)."""
		from torch import tensor, long as torch_long
		from wnn.ram.core.models import BitwiseRAMLM

		log = logger if logger is not None else lambda x: None

		results = []
		for i, genome in enumerate(genomes):
			start = time.time()

			model = BitwiseRAMLM(
				vocab_size=self._vocab_size,
				context_size=self._context_size,
				neurons_per_cluster=self._neurons_per_cluster,
				bits_per_neuron=self._bits_per_neuron,
				pad_token_id=self._pad_token_id,
				memory_mode=self._memory_mode,
				neuron_sample_rate=self._neuron_sample_rate,
			)

			if genome.connections is not None:
				total_neurons = model.layer.total_neurons
				bits_per_neuron = model.layer.bits_per_neuron
				conn_tensor = tensor(genome.connections, dtype=torch_long).view(
					total_neurons, bits_per_neuron
				)
				model.connections = conn_tensor

			model.reset_memory()
			model.train_epoch_fast(token_ids=train_tokens, batch_size=2000, verbose=False)
			stats = model.evaluate_fast(token_ids=eval_tokens, batch_size=5000, verbose=False)
			ce, acc = stats["cross_entropy"], stats["accuracy"]
			elapsed = time.time() - start
			results.append((ce, acc))

			if generation is not None:
				gen = generation + 1
				total = total_generations or len(genomes)
				log(f"[Gen {gen:02d}/{total:02d}] Genome {i+1}/{len(genomes)}: CE={ce:.4f}, Acc={acc:.2%} ({elapsed:.1f}s)")

		return results

	# =========================================================================
	# Public interface (matches CachedEvaluator)
	# =========================================================================

	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		min_accuracy: Optional[float] = None,
		streaming: bool = True,
		stream_batch_size: int = 1,
	) -> list[tuple[float, float]]:
		"""Evaluate multiple genomes using subset rotation (both train and eval)."""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = self.next_eval_idx()

		if self._rust_cache is not None:
			start = time.time()
			results = self._evaluate_batch_rust(genomes, train_subset_idx, eval_subset_idx)
			elapsed = time.time() - start
			log = logger if logger is not None else lambda x: None
			if generation is not None:
				gen = generation + 1
				total = total_generations or "?"
				best_ce = min(r[0] for r in results) if results else 0.0
				best_acc = max(r[1] for r in results) if results else 0.0
				log(f"[Gen {gen:02d}/{total}] {len(genomes)} genomes in {elapsed:.1f}s "
					f"(best CE={best_ce:.4f}, Acc={best_acc:.2%})")
			return results

		# Python fallback
		train_data = self._train_parts[train_subset_idx % self._num_parts]
		return self._evaluate_batch_python(
			genomes, train_data, self._eval_tokens,
			logger, generation, total_generations,
		)

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[tuple[float, float]]:
		"""Evaluate genomes using full train + eval data."""
		if self._rust_cache is not None:
			start = time.time()
			results = self._evaluate_batch_full_rust(genomes)
			elapsed = time.time() - start
			log = logger if logger is not None else lambda x: None
			log(f"[Full] {len(genomes)} genomes in {elapsed:.1f}s")
			return results

		# Python fallback
		return self._evaluate_batch_python(
			genomes, self._train_tokens, self._eval_tokens, logger,
		)

	def evaluate_single(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> float:
		"""Evaluate a single genome, returning CE only."""
		ce, _ = self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]
		return ce

	def evaluate_single_with_accuracy(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		return self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]

	def evaluate_single_full(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome with full data."""
		return self.evaluate_batch_full([genome])[0]

	# =========================================================================
	# Gated evaluation (all 3 modes: TOKEN_LEVEL, BIT_LEVEL, DUAL_STAGE)
	# =========================================================================

	def evaluate_with_gating(
		self,
		genome: ClusterGenome,
		train_tokens: list[int],
		gating_result,  # GatingResult from gating_trainer
		logger: Optional[Callable[[str], None]] = None,
	) -> dict:
		"""Evaluate genome with and without gating for comparison.

		Uses Python BitwiseRAMLM for final evaluation (not Rust batch path)
		since gating requires per-example gate computation.

		Args:
			genome: Optimized genome to evaluate
			train_tokens: Token sequence for model training
			gating_result: GatingResult from GatingTrainer.train()
			logger: Optional logging function

		Returns:
			Dict with: ce, acc, gated_ce, gated_acc, gating_mode, gating_stats
		"""
		import torch
		from math import exp as math_exp
		from torch import tensor, long as torch_long, arange as torch_arange, logsumexp, float32 as torch_float32
		from wnn.ram.core.models import BitwiseRAMLM, reconstruct_logprobs
		from wnn.ram.core.gating_trainer import GatingMode

		log = logger or (lambda x: None)
		mode = gating_result.mode

		log(f"  Evaluating with gating (mode={mode.name})...")

		# Build and train model
		model = BitwiseRAMLM(
			vocab_size=self._vocab_size,
			context_size=self._context_size,
			neurons_per_cluster=self._neurons_per_cluster,
			bits_per_neuron=self._bits_per_neuron,
			pad_token_id=self._pad_token_id,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
		)

		if genome.connections is not None:
			total_neurons = model.layer.total_neurons
			bpn = model.layer.bits_per_neuron
			conn_tensor = tensor(genome.connections, dtype=torch_long).view(total_neurons, bpn)
			model.connections = conn_tensor

		model.reset_memory()
		model.train_epoch_fast(token_ids=train_tokens, batch_size=2000, verbose=False)

		# Evaluate on eval tokens
		eval_tokens = self._eval_tokens
		total_examples = len(eval_tokens) - self._context_size
		all_bits = model.encode_sequence(eval_tokens)
		targets = tensor(eval_tokens[self._context_size:], dtype=torch_long)

		batch_size = 5000
		num_batches = (total_examples + batch_size - 1) // batch_size

		# Accumulators for ungated and gated
		total_ce = 0.0
		total_correct = 0
		gated_ce = 0.0
		gated_correct = 0

		for batch_idx in range(num_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, total_examples)
			batch_len = end - start

			batch_bits = all_bits[start:end]
			batch_targets = targets[start:end]

			# Ungated scores
			log_probs = model.forward(batch_bits)  # [B, vocab_size]
			lse = logsumexp(log_probs, dim=-1)
			target_lp = log_probs[torch_arange(batch_len), batch_targets]
			total_ce += (lse - target_lp).sum().item()
			total_correct += (log_probs.argmax(dim=-1) == batch_targets).sum().item()

			# Gated scores
			eps = 1e-7
			if mode == GatingMode.TOKEN_LEVEL:
				gates = gating_result.token_gating.forward(batch_bits)  # [B, vocab_size]
				gated_lp = log_probs + torch.log(gates + eps)

			elif mode == GatingMode.BIT_LEVEL:
				bit_scores = model.forward_bits(batch_bits)  # [B, num_bits]
				bit_gates = gating_result.bit_gating.forward(batch_bits)  # [B, num_bits]
				gated_bits = bit_gates * bit_scores + (1 - bit_gates) * 0.5
				gated_lp = reconstruct_logprobs(gated_bits, model.token_bits)

			elif mode == GatingMode.DUAL_STAGE:
				# Stage 1: bit-level confidence
				bit_scores = model.forward_bits(batch_bits)
				bit_gates = gating_result.bit_gating.forward(batch_bits)
				gated_bits = bit_gates * bit_scores + (1 - bit_gates) * 0.5
				# Stage 2: token-level pruning
				token_lp = reconstruct_logprobs(gated_bits, model.token_bits)
				token_gates = gating_result.token_gating.forward(batch_bits)
				gated_lp = token_lp + torch.log(token_gates + eps)

			gated_lse = logsumexp(gated_lp, dim=-1)
			gated_target_lp = gated_lp[torch_arange(batch_len), batch_targets]
			gated_ce += (gated_lse - gated_target_lp).sum().item()
			gated_correct += (gated_lp.argmax(dim=-1) == batch_targets).sum().item()

		# Compute metrics
		ce = total_ce / total_examples
		acc = total_correct / total_examples
		g_ce = gated_ce / total_examples
		g_acc = gated_correct / total_examples

		results = {
			"ce": ce,
			"acc": acc,
			"perplexity": math_exp(min(ce, 100)),
			"gated_ce": g_ce,
			"gated_acc": g_acc,
			"gated_perplexity": math_exp(min(g_ce, 100)),
			"gating_mode": mode.name,
			"ce_improvement": ce - g_ce,
			"acc_improvement": g_acc - acc,
			**gating_result.stats,
		}

		log(f"  Ungated: CE={ce:.4f}, Acc={acc:.2%}, PPL={results['perplexity']:.0f}")
		log(f"  Gated:   CE={g_ce:.4f}, Acc={g_acc:.2%}, PPL={results['gated_perplexity']:.0f}")
		log(f"  Delta:   CE={results['ce_improvement']:+.4f}, Acc={results['acc_improvement']:+.2%}")

		return results

	def reset(self, seed: Optional[int] = None) -> None:
		"""Reset subset rotation (both train and eval)."""
		if self._rust_cache is not None:
			self._rust_cache.reset()
		if seed is not None:
			self._seed = seed
		self._train_rotation_idx = 0
		self._eval_rotation_idx = 0

	def __repr__(self) -> str:
		backend = "Rust+Metal" if self._rust_cache is not None else "Python"
		mode_names = {0: "TERNARY", 1: "QUAD_BINARY", 2: "QUAD_WEIGHTED"}
		mode = mode_names.get(self._memory_mode, f"UNKNOWN({self._memory_mode})")
		rate_str = f", rate={self._neuron_sample_rate}" if self._neuron_sample_rate < 1.0 else ""
		return (
			f"BitwiseEvaluator(vocab={self._vocab_size}, "
			f"context={self._context_size}, "
			f"neurons={self._neurons_per_cluster}, "
			f"bits={self._bits_per_neuron}, "
			f"train_parts={self._num_parts}, "
			f"eval_parts={self._num_eval_parts}, "
			f"mode={mode}{rate_str}, "
			f"backend={backend})"
		)
