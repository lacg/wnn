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

	# Evaluate genomes (same interface as TieredEvaluator)
	results = evaluator.evaluate_batch(genomes)
	# → [EvalResult(ce, accuracy, bit_accuracy), ...]
	# bit_accuracy uses entropy-based weights (balanced bits matter more)
"""

import random
import time
from typing import Optional, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.base_evaluator import OffspringSearchResult
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult, AdaptationConfig


class BitwiseEvaluator(BaseEvaluator):
	"""
	Evaluator for BitwiseRAMLM genomes.

	Primary path: Rust+Metal (all 50 genomes evaluated in parallel).
	Fallback: Python (sequential, creates BitwiseRAMLM per genome).

	Supports data rotation (subset training) for GA/TS diversity,
	matching TieredEvaluator's interface.
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
		adapt_config: Optional[AdaptationConfig] = None,
	):
		super().__init__(
			train_tokens=train_tokens,
			eval_tokens=eval_tokens,
			vocab_size=vocab_size,
			context_size=context_size,
			num_parts=num_parts,
			num_eval_parts=num_eval_parts,
			seed=seed,
			memory_mode=memory_mode,
			neuron_sample_rate=neuron_sample_rate,
			adapt_config=adapt_config,
		)

		# BitwiseEvaluator-specific fields
		self._neurons_per_cluster = neurons_per_cluster
		self._bits_per_neuron = bits_per_neuron
		self._pad_token_id = pad_token_id

		# Subset rotation (for Python fallback)
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

	def get_live_progress(self):
		"""Forward live progress from Rust cache."""
		if self._rust_cache is not None and hasattr(self._rust_cache, 'get_live_progress'):
			return self._rust_cache.get_live_progress()
		return None

	def set_experiment_context(self, experiment_id: int):
		"""Forward experiment context to Rust cache."""
		if self._rust_cache is not None and hasattr(self._rust_cache, 'set_experiment_context'):
			self._rust_cache.set_experiment_context(experiment_id)

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

	def _evaluate_batch_adaptive_rust(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: int,
		eval_subset_idx: int,
	) -> list[tuple[float, float, float]]:
		"""Evaluate with adaptation (Baldwin effect). Updates genomes IN-PLACE.

		Each genome is adapted during evaluation: train → stats → adapt → retrain → eval.
		GA/TS sees adapted fitness, so evolution selects for adaptable architectures.

		Returns list of (ce, accuracy, weighted_bit_accuracy) tuples.
		"""
		config = self._adapt_config
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes_heterogeneous(genomes)
		results = self._rust_cache.evaluate_genomes_adaptive(
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			train_subset_idx=train_subset_idx,
			eval_subset_idx=eval_subset_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
			generation=self._generation,
			synaptogenesis_enabled=config.synaptogenesis_enabled,
			neurogenesis_enabled=config.neurogenesis_enabled,
			axonogenesis_enabled=config.axonogenesis_enabled,
			prune_entropy_ratio=config.prune_entropy_ratio,
			grow_fill_utilization=config.grow_fill_utilization,
			grow_error_baseline=config.grow_error_baseline,
			min_bits=config.min_bits,
			max_bits=config.max_bits,
			cluster_error_factor=config.cluster_error_factor,
			cluster_fill_utilization=config.cluster_fill_utilization,
			neuron_prune_percentile=config.neuron_prune_percentile,
			neuron_removal_factor=config.neuron_removal_factor,
			max_growth_ratio=config.max_growth_ratio,
			min_neurons=config.min_neurons,
			max_neurons_per_pass=config.max_neurons_per_pass,
			axon_entropy_threshold=config.axon_entropy_threshold,
			axon_improvement_factor=config.axon_improvement_factor,
			axon_rewire_count=config.axon_rewire_count,
			warmup_generations=config.warmup_generations,
			cooldown_iterations=config.cooldown_iterations,
			stabilize_fraction=config.stabilize_fraction,
			total_generations=config.total_generations,
			passes_per_eval=config.passes_per_eval,
			stats_sample_size=config.stats_sample_size,
		)
		scores = []
		for genome, (ce, acc, bit_acc, a_bits, a_neurons, a_conns, p, g, a, r, rw) in zip(genomes, results):
			# Update genome in-place with adapted architecture
			genome.bits_per_neuron = list(a_bits)
			genome.neurons_per_cluster = list(a_neurons)
			genome.connections = list(a_conns)
			genome._cached_bit_acc = bit_acc
			scores.append((ce, acc, bit_acc))
		return scores

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
	# Public interface
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
	) -> list[EvalResult]:
		"""Evaluate multiple genomes using subset rotation (both train and eval).

		Returns list of EvalResult (with bit_accuracy populated).
		"""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = self.next_eval_idx()

		if self._rust_cache is not None:
			start = time.time()
			if self._adapt_config is not None:
				raw_results = self._evaluate_batch_adaptive_rust(genomes, train_subset_idx, eval_subset_idx)
			else:
				raw_results = self._evaluate_batch_rust(genomes, train_subset_idx, eval_subset_idx)
			elapsed = time.time() - start
			log = logger if logger is not None else lambda x: None
			if generation is not None:
				gen = generation + 1
				total = total_generations or "?"
				best_ce = min(r[0] for r in raw_results) if raw_results else 0.0
				best_acc = max(r[1] for r in raw_results) if raw_results else 0.0
				best_bit_acc = max(r[2] for r in raw_results) if raw_results else 0.0
				n = len(raw_results)
				mean_ce = sum(r[0] for r in raw_results) / n if n else 0.0
				mean_acc = sum(r[1] for r in raw_results) / n if n else 0.0
				mean_bit_acc = sum(r[2] for r in raw_results) / n if n else 0.0
				log(f"[Gen {gen:02d}/{total}] {len(genomes)} genomes in {elapsed:.1f}s "
					f"(best CE={best_ce:.4f}, Acc={best_acc:.2%}, BitAcc={best_bit_acc:.2%})")
				# Record for correlation tracking
				self._generation_log.append((
					generation, best_ce, best_acc, best_bit_acc,
					mean_ce, mean_acc, mean_bit_acc,
				))
			return [EvalResult(ce=ce, accuracy=acc, bit_accuracy=bit_acc)
					for ce, acc, bit_acc in raw_results]

		# Python fallback (no bit_acc available)
		train_data = self._train_parts[train_subset_idx % self._num_parts]
		py_results = self._evaluate_batch_python(
			genomes, train_data, self._eval_tokens,
			logger, generation, total_generations,
		)
		# bit_accuracy=0.0 for Python fallback (not available)
		return [EvalResult(ce=ce, accuracy=acc, bit_accuracy=0.0)
				for ce, acc in py_results]

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[EvalResult]:
		"""Evaluate genomes using full train + eval data.

		Returns list of EvalResult (with bit_accuracy populated).
		"""
		if self._rust_cache is not None:
			start = time.time()
			raw_results = self._evaluate_batch_full_rust(genomes)
			elapsed = time.time() - start
			log = logger if logger is not None else lambda x: None
			log(f"[Full] {len(genomes)} genomes in {elapsed:.1f}s")
			return [EvalResult(ce=ce, accuracy=acc, bit_accuracy=bit_acc)
					for ce, acc, bit_acc in raw_results]

		# Python fallback
		py_results = self._evaluate_batch_python(
			genomes, self._train_tokens, self._eval_tokens, logger,
		)
		return [EvalResult(ce=ce, accuracy=acc, bit_accuracy=0.0)
				for ce, acc in py_results]

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
