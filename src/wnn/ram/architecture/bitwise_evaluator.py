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
	# → [(ce, acc, weighted_bit_acc), (ce, acc, weighted_bit_acc), ...]
	# weighted_bit_acc uses entropy-based weights (balanced bits matter more)
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.cached_evaluator import OffspringSearchResult


@dataclass
class AdaptationConfig:
	"""Configuration for training-time architecture adaptation."""

	# Synaptogenesis (connection level)
	synaptogenesis_enabled: bool = False
	prune_entropy_threshold: float = 0.05
	grow_fill_threshold: float = 0.8
	grow_error_threshold: float = 0.5
	min_bits: int = 4
	max_bits: int = 24

	# Neurogenesis (cluster level)
	neurogenesis_enabled: bool = False
	cluster_error_threshold: float = 0.5
	cluster_fill_threshold: float = 0.7
	neuron_uniqueness_threshold: float = 0.05
	min_neurons: int = 3
	max_neurons: int = 30
	max_neurons_per_pass: int = 3
	warmup_generations: int = 10
	cooldown_iterations: int = 5

	# Shared
	passes_per_eval: int = 1


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

		# Per-generation metrics history for correlation tracking
		# Each entry: (generation, best_ce, best_acc, best_bit_acc, mean_ce, mean_acc, mean_bit_acc)
		self._generation_log: list[tuple[int, float, float, float, float, float, float]] = []

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
		Flatten per-neuron arrays from genomes for Rust.

		Returns:
			(bits_flat, neurons_flat, connections_flat) where:
			- bits_flat: [num_genomes * total_neurons] per-neuron bit counts
			- neurons_flat: [num_genomes * num_clusters]
			- connections_flat: variable total (sum of bits_per_neuron per genome)
		"""
		import random

		bits_flat = []
		neurons_flat = []
		connections_flat = []

		for g in genomes:
			bits_flat.extend(g.bits_per_neuron)
			neurons_flat.extend(g.neurons_per_cluster)
			if g.connections is not None:
				connections_flat.extend(g.connections)
			else:
				# Generate random connections based on per-neuron config
				for b in g.bits_per_neuron:
					for _ in range(b):
						connections_flat.append(random.randint(0, self._total_input_bits - 1))

		return bits_flat, neurons_flat, connections_flat

	def _evaluate_batch_rust(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: int,
		eval_subset_idx: int,
	) -> list[tuple[float, float, float]]:
		"""Evaluate using Rust+Metal backend with per-cluster heterogeneous configs.

		Returns list of (ce, accuracy, weighted_bit_accuracy) tuples.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes_heterogeneous(genomes)
		results = self._rust_cache.evaluate_genomes(
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			train_subset_idx=train_subset_idx,
			eval_subset_idx=eval_subset_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)
		# Cache weighted bit accuracy on each genome for downstream logging
		for genome, (_, _, bit_acc) in zip(genomes, results):
			genome._cached_bit_acc = bit_acc
		return results

	def _evaluate_batch_full_rust(
		self,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float, float]]:
		"""Evaluate with full data using Rust+Metal backend.

		Returns list of (ce, accuracy, weighted_bit_accuracy) tuples.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes_heterogeneous(genomes)
		results = self._rust_cache.evaluate_genomes_full(
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)
		# Cache weighted bit accuracy on each genome for downstream logging
		for genome, (_, _, bit_acc) in zip(genomes, results):
			genome._cached_bit_acc = bit_acc
		return results

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
	) -> list[tuple[float, float, float]]:
		"""Evaluate multiple genomes using subset rotation (both train and eval).

		Returns list of (ce, accuracy, weighted_bit_accuracy) tuples.
		"""
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
				best_bit_acc = max(r[2] for r in results) if results else 0.0
				n = len(results)
				mean_ce = sum(r[0] for r in results) / n if n else 0.0
				mean_acc = sum(r[1] for r in results) / n if n else 0.0
				mean_bit_acc = sum(r[2] for r in results) / n if n else 0.0
				log(f"[Gen {gen:02d}/{total}] {len(genomes)} genomes in {elapsed:.1f}s "
					f"(best CE={best_ce:.4f}, Acc={best_acc:.2%}, BitAcc={best_bit_acc:.2%})")
				# Record for correlation tracking
				self._generation_log.append((
					generation, best_ce, best_acc, best_bit_acc,
					mean_ce, mean_acc, mean_bit_acc,
				))
			return results

		# Python fallback (no bit_acc available)
		train_data = self._train_parts[train_subset_idx % self._num_parts]
		py_results = self._evaluate_batch_python(
			genomes, train_data, self._eval_tokens,
			logger, generation, total_generations,
		)
		# Pad with 0.0 for bit_acc
		return [(ce, acc, 0.0) for ce, acc in py_results]

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[tuple[float, float, float]]:
		"""Evaluate genomes using full train + eval data.

		Returns list of (ce, accuracy, weighted_bit_accuracy) tuples.
		"""
		if self._rust_cache is not None:
			start = time.time()
			results = self._evaluate_batch_full_rust(genomes)
			elapsed = time.time() - start
			log = logger if logger is not None else lambda x: None
			log(f"[Full] {len(genomes)} genomes in {elapsed:.1f}s")
			return results

		# Python fallback
		py_results = self._evaluate_batch_python(
			genomes, self._train_tokens, self._eval_tokens, logger,
		)
		return [(ce, acc, 0.0) for ce, acc in py_results]

	def evaluate_single(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> float:
		"""Evaluate a single genome, returning CE only."""
		result = self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]
		return result[0]

	def evaluate_single_with_accuracy(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		result = self.evaluate_batch([genome], train_subset_idx, eval_subset_idx)[0]
		return (result[0], result[1])

	def evaluate_single_full(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome with full data."""
		result = self.evaluate_batch_full([genome])[0]
		return (result[0], result[1])

	def evaluate_with_adaptation(
		self,
		genome: ClusterGenome,
		adapt_config: AdaptationConfig,
		generation: int,
		cooldowns: Optional[list[int]] = None,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> tuple[ClusterGenome, float, float, dict]:
		"""Evaluate a genome with training-time adaptation (synaptogenesis + neurogenesis).

		Pipeline: Train → Stats → Adapt → Re-train → Eval

		Returns:
			(adapted_genome, ce, accuracy, adaptation_stats)
			adaptation_stats contains: pruned, grown, neurons_added, neurons_removed, cooldowns
		"""
		if self._rust_cache is None:
			# Python fallback: just evaluate without adaptation
			ce, acc = self.evaluate_single_with_accuracy(genome, train_subset_idx, eval_subset_idx)
			return genome, ce, acc, {"pruned": 0, "grown": 0, "added": 0, "removed": 0, "cooldowns": []}

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = self.next_eval_idx()

		num_clusters = len(genome.neurons_per_cluster)
		if cooldowns is None:
			cooldowns = [0] * num_clusters

		rng_seed = random.getrandbits(64)

		(adapted_bits, adapted_neurons, adapted_connections,
		 ce, acc, bit_acc,
		 pruned, grown, added, removed,
		 new_cooldowns) = self._rust_cache.train_adapt_eval(
			bits_per_neuron=list(genome.bits_per_neuron),
			neurons_per_cluster=list(genome.neurons_per_cluster),
			connections=list(genome.connections),
			train_subset_idx=train_subset_idx,
			eval_subset_idx=eval_subset_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=rng_seed,
			generation=generation,
			cooldowns=cooldowns,
			synaptogenesis_enabled=adapt_config.synaptogenesis_enabled,
			neurogenesis_enabled=adapt_config.neurogenesis_enabled,
			prune_entropy_threshold=adapt_config.prune_entropy_threshold,
			grow_fill_threshold=adapt_config.grow_fill_threshold,
			grow_error_threshold=adapt_config.grow_error_threshold,
			min_bits=adapt_config.min_bits,
			max_bits=adapt_config.max_bits,
			cluster_error_threshold=adapt_config.cluster_error_threshold,
			cluster_fill_threshold=adapt_config.cluster_fill_threshold,
			neuron_uniqueness_threshold=adapt_config.neuron_uniqueness_threshold,
			min_neurons=adapt_config.min_neurons,
			max_neurons=adapt_config.max_neurons,
			max_neurons_per_pass=adapt_config.max_neurons_per_pass,
			warmup_generations=adapt_config.warmup_generations,
			cooldown_iterations=adapt_config.cooldown_iterations,
			passes_per_eval=adapt_config.passes_per_eval,
		)

		# Build adapted genome
		adapted_genome = genome.clone()
		adapted_genome.bits_per_neuron = list(adapted_bits)
		adapted_genome.neurons_per_cluster = list(adapted_neurons)
		adapted_genome.connections = list(adapted_connections)

		stats = {
			"pruned": pruned,
			"grown": grown,
			"added": added,
			"removed": removed,
			"cooldowns": list(new_cooldowns),
		}

		return adapted_genome, ce, acc, stats

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

	# =========================================================================
	# Neighbor/offspring search (matches CachedEvaluator interface)
	# =========================================================================
	# Mutation is done in Python, evaluation via Rust+Metal batch path.
	# This enables ArchitectureTSStrategy and ArchitectureGAStrategy to use
	# the fast path with retry loops instead of falling back to generic Python.

	def _mutate_genome(
		self,
		genome: ClusterGenome,
		bits_mutation_rate: float,
		neurons_mutation_rate: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		mutable_clusters: Optional[list[int]],
		rng: random.Random,
	) -> ClusterGenome:
		"""Mutate a genome to create a neighbor (matches Rust mutate_genome logic).

		Operates on per-neuron bits. Mutation applies a per-cluster delta to all
		neurons in that cluster, then adjusts connections accordingly.
		"""
		num_clusters = len(genome.neurons_per_cluster)
		offsets = genome.cluster_neuron_offsets
		old_bits = genome.bits_per_neuron.copy()
		new_bits = genome.bits_per_neuron.copy()
		new_neurons = genome.neurons_per_cluster.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		# Delta ranges: 10% of (min + max), minimum 1
		bits_delta_max = max(1, round(0.1 * (min_bits + max_bits)))
		neurons_delta_max = max(1, round(0.1 * (min_neurons + max_neurons)))

		indices = mutable_clusters if mutable_clusters is not None else list(range(num_clusters))
		for i in indices:
			if i >= num_clusters:
				continue
			if rng.random() < bits_mutation_rate:
				delta = rng.randint(-bits_delta_max, bits_delta_max)
				# Apply same delta to all neurons in this cluster
				for n_idx in range(offsets[i], offsets[i + 1]):
					new_bits[n_idx] = max(min_bits, min(max_bits, new_bits[n_idx] + delta))
			if rng.random() < neurons_mutation_rate:
				delta = rng.randint(-neurons_delta_max, neurons_delta_max)
				new_neurons[i] = max(min_neurons, min(max_neurons, new_neurons[i] + delta))

		# Rebuild per-neuron bits for changed neuron counts
		final_bits = []
		for c in range(num_clusters):
			old_n = old_neurons[c]
			new_n = new_neurons[c]
			cluster_old_bits = new_bits[offsets[c]:offsets[c + 1]]
			for n in range(new_n):
				if n < old_n:
					final_bits.append(cluster_old_bits[n])
				else:
					# New neuron: copy bits from random existing neuron in cluster
					template = rng.randint(0, old_n - 1) if old_n > 0 else 0
					final_bits.append(cluster_old_bits[template] if old_n > 0 else min_bits)

		# Adjust connections for architecture changes
		new_conns = self._adjust_connections(
			genome.connections, old_bits, old_neurons, final_bits, new_neurons, rng,
		)

		return ClusterGenome(
			bits_per_neuron=final_bits,
			neurons_per_cluster=new_neurons,
			connections=new_conns,
		)

	def _adjust_connections(
		self,
		old_connections: Optional[list[int]],
		old_bits: list[int],
		old_neurons: list[int],
		new_bits: list[int],
		new_neurons: list[int],
		rng: random.Random,
	) -> Optional[list[int]]:
		"""Adjust connections when architecture changes.

		old_bits/new_bits: per-neuron bit counts (flat).
		old_neurons/new_neurons: per-cluster neuron counts.
		"""
		if old_connections is None or len(old_connections) == 0:
			return None

		total_input_bits = self._total_input_bits
		result = []

		# Build connection offsets for old per-neuron bits
		old_conn_offsets = [0]
		for b in old_bits:
			old_conn_offsets.append(old_conn_offsets[-1] + b)

		# Build neuron offsets for old clusters
		old_neuron_offsets = [0]
		for n in old_neurons:
			old_neuron_offsets.append(old_neuron_offsets[-1] + n)

		new_neuron_idx = 0
		for c in range(len(new_neurons)):
			o_n = old_neurons[c]
			n_n = new_neurons[c]
			old_cluster_start = old_neuron_offsets[c]

			for neuron in range(n_n):
				n_b = new_bits[new_neuron_idx]

				if neuron < o_n:
					old_n_idx = old_cluster_start + neuron
					o_b = old_bits[old_n_idx]
					old_start = old_conn_offsets[old_n_idx]

					for bit in range(n_b):
						if bit < o_b:
							conn = old_connections[old_start + bit]
							# 10% chance of small perturbation
							if rng.random() < 0.1:
								delta = rng.choice([-2, -1, 1, 2])
								conn = max(0, min(total_input_bits - 1, conn + delta))
							result.append(conn)
						else:
							result.append(rng.randint(0, total_input_bits - 1))
				else:
					# New neuron: copy from random existing with mutations
					if o_n > 0:
						template = rng.randint(0, o_n - 1)
						tmpl_n_idx = old_cluster_start + template
						o_b = old_bits[tmpl_n_idx]
						tmpl_start = old_conn_offsets[tmpl_n_idx]
						for bit in range(n_b):
							if bit < o_b:
								conn = old_connections[tmpl_start + bit]
								delta = rng.choice([-2, -1, 1, 2])
								conn = max(0, min(total_input_bits - 1, conn + delta))
								result.append(conn)
							else:
								result.append(rng.randint(0, total_input_bits - 1))
					else:
						for _ in range(n_b):
							result.append(rng.randint(0, total_input_bits - 1))

				new_neuron_idx += 1

		return result

	def search_neighbors(
		self,
		genome: ClusterGenome,
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.05,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
	) -> list[ClusterGenome]:
		"""Search for neighbor genomes above accuracy threshold.

		Mutation in Python, evaluation via Rust+Metal batch path.
		Implements retry loop matching Rust search_neighbors_best_n behavior.
		"""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = 0
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rng = random.Random(seed)
		passed: list[ClusterGenome] = []
		all_candidates: list[ClusterGenome] = []
		evaluated = 0
		batch_size = 50

		gen_str = f"{(generation or 0) + 1}/{total_generations or '?'}"

		while len(passed) < target_count and evaluated < max_attempts:
			remaining = target_count - len(passed)
			batch_n = min(remaining + 5, batch_size, max_attempts - evaluated)
			if batch_n <= 0:
				break

			# Generate mutations
			batch = []
			for _ in range(batch_n):
				mutant = self._mutate_genome(
					genome, bits_mutation_rate, neurons_mutation_rate,
					min_bits, max_bits, min_neurons, max_neurons,
					mutable_clusters, rng,
				)
				batch.append(mutant)

			# Evaluate via Rust+Metal
			results = self._evaluate_batch_rust(batch, train_subset_idx, eval_subset_idx)
			evaluated += len(results)

			for g, (ce, acc, bit_acc) in zip(batch, results):
				g._cached_fitness = (ce, acc)
				g._cached_bit_acc = bit_acc
				if acc >= accuracy_threshold:
					passed.append(g)
					if len(passed) >= target_count:
						break
				else:
					all_candidates.append(g)

		# Fallback: return best N by accuracy then CE
		if len(passed) < target_count and return_best_n:
			all_candidates.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
			need = target_count - len(passed)
			passed.extend(all_candidates[:need])

		return passed

	def search_offspring(
		self,
		population: list[tuple[ClusterGenome, float]],
		target_count: int,
		max_attempts: int,
		accuracy_threshold: float,
		min_bits: int,
		max_bits: int,
		min_neurons: int,
		max_neurons: int,
		bits_mutation_rate: float = 0.1,
		neurons_mutation_rate: float = 0.1,
		crossover_rate: float = 0.7,
		tournament_size: int = 3,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
	) -> OffspringSearchResult:
		"""Search for GA offspring above accuracy threshold.

		Tournament selection + crossover + mutation in Python,
		evaluation via Rust+Metal batch path. Retry loop included.
		"""
		if not population:
			return OffspringSearchResult(genomes=[], evaluated=0, viable=0)

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = 0
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rng = random.Random(seed)
		passed: list[ClusterGenome] = []
		all_candidates: list[ClusterGenome] = []
		evaluated = 0
		viable_count = 0
		batch_size = 50

		while len(passed) < target_count and evaluated < max_attempts:
			remaining = target_count - len(passed)
			batch_n = min(remaining + 5, batch_size, max_attempts - evaluated)
			if batch_n <= 0:
				break

			batch = []
			for _ in range(batch_n):
				# Tournament selection
				parent1 = self._tournament_select(population, tournament_size, rng)
				if rng.random() < crossover_rate and len(population) > 1:
					parent2 = self._tournament_select(population, tournament_size, rng)
					child = self._crossover(parent1, parent2, rng)
				else:
					child = parent1.clone()
				# Mutate
				child = self._mutate_genome(
					child, bits_mutation_rate, neurons_mutation_rate,
					min_bits, max_bits, min_neurons, max_neurons,
					mutable_clusters, rng,
				)
				batch.append(child)

			# Evaluate via Rust+Metal
			results = self._evaluate_batch_rust(batch, train_subset_idx, eval_subset_idx)
			evaluated += len(results)

			for g, (ce, acc, bit_acc) in zip(batch, results):
				g._cached_fitness = (ce, acc)
				g._cached_bit_acc = bit_acc
				if acc >= accuracy_threshold:
					viable_count += 1
					passed.append(g)
					if len(passed) >= target_count:
						break
				else:
					all_candidates.append(g)

		# Fallback: return best N by accuracy then CE
		if len(passed) < target_count and return_best_n:
			all_candidates.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
			need = target_count - len(passed)
			passed.extend(all_candidates[:need])

		return OffspringSearchResult(genomes=passed, evaluated=evaluated, viable=viable_count)

	def _tournament_select(
		self,
		population: list[tuple[ClusterGenome, float]],
		tournament_size: int,
		rng: random.Random,
	) -> ClusterGenome:
		"""Tournament selection: pick best (lowest fitness) from random subset."""
		contestants = rng.sample(population, min(tournament_size, len(population)))
		best = min(contestants, key=lambda x: x[1])
		return best[0].clone()

	def _crossover(
		self,
		parent1: ClusterGenome,
		parent2: ClusterGenome,
		rng: random.Random,
	) -> ClusterGenome:
		"""Uniform crossover: randomly pick each cluster from either parent."""
		n = len(parent1.neurons_per_cluster)
		p1_offsets = parent1.cluster_neuron_offsets
		p2_offsets = parent2.cluster_neuron_offsets

		new_bits = []
		new_neurons = []
		for i in range(n):
			if rng.random() < 0.5:
				# Take cluster i's neurons and per-neuron bits from parent1
				new_neurons.append(parent1.neurons_per_cluster[i])
				new_bits.extend(parent1.bits_per_neuron[p1_offsets[i]:p1_offsets[i + 1]])
			else:
				new_neurons.append(parent2.neurons_per_cluster[i])
				new_bits.extend(parent2.bits_per_neuron[p2_offsets[i]:p2_offsets[i + 1]])

		# Connections: take from parent1 and adjust for new architecture
		new_conns = self._adjust_connections(
			parent1.connections,
			parent1.bits_per_neuron, parent1.neurons_per_cluster,
			new_bits, new_neurons, rng,
		)
		return ClusterGenome(
			bits_per_neuron=new_bits,
			neurons_per_cluster=new_neurons,
			connections=new_conns,
		)

	@property
	def generation_log(self) -> list[tuple[int, float, float, float, float, float, float]]:
		"""Per-generation metrics from evaluate_batch calls.

		Each entry: (generation, best_ce, best_acc, best_bit_acc, mean_ce, mean_acc, mean_bit_acc)
		"""
		return self._generation_log

	def clear_generation_log(self) -> None:
		"""Clear per-generation metrics (call between phases)."""
		self._generation_log.clear()

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
