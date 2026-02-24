"""
Multi-Stage Evaluator — Evaluates multi-stage RAM LM genomes.

Wraps MultiStageCacheWrapper from Rust. Supports:
- Stage-agnostic bitwise evaluation: predict target bits from context (+ previous stage outputs)
- Tiered evaluation: direct K-class prediction with softmax CE
- Combined CE: joint P(token) = P(group|ctx) × P(token|group,ctx)

Usage:
	evaluator = MultiStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=50257,
		context_size=4,
		stage_k=[256, 256],
		target_stage=0,  # 0-indexed: which stage GA/TS optimizes
	)

	# Evaluate Stage 0 genomes
	results = evaluator.evaluate_batch(genomes)

	# Combined CE (needs genomes for all stages)
	combined = evaluator.compute_combined_metrics([stage0_genome, stage1_genome])
"""

import logging
import math
import os
import random
import time
from typing import Optional, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult, AdaptationConfig


def compute_default_k(num_stages: int, stage_cluster_types: list[str], vocab_size: int = 50257) -> list[int]:
	"""Compute default K values per stage so that the product >= vocab_size.

	For 2 stages:
	  bitwise+bitwise: [256, 256]  (65536 > 50257)
	  tiered+tiered:   [225, 224]  (50400 ≈ 50257)
	  mixed:           tiered=197, bitwise=256 (50432 ≈ 50257)

	For N>2: distribute evenly in log-space, round up so product >= vocab_size.
	"""
	if num_stages == 2:
		t0, t1 = stage_cluster_types[0], stage_cluster_types[1]
		if t0 == "bitwise" and t1 == "bitwise":
			return [256, 256]
		elif t0 == "tiered" and t1 == "tiered":
			k = math.ceil(math.sqrt(vocab_size))
			return [k, math.ceil(vocab_size / k)]
		else:
			# Mixed: bitwise stages get 256 (power of 2), tiered gets remainder
			result = []
			bitwise_product = 1
			tiered_indices = []
			for i, ct in enumerate(stage_cluster_types):
				if ct == "bitwise":
					result.append(256)
					bitwise_product *= 256
				else:
					result.append(0)
					tiered_indices.append(i)
			if tiered_indices:
				remaining = math.ceil(vocab_size / bitwise_product)
				per_tiered = math.ceil(remaining ** (1.0 / len(tiered_indices)))
				for idx in tiered_indices:
					result[idx] = per_tiered
			return result

	# General N>2: distribute evenly in log-space
	per_stage_k = math.ceil(vocab_size ** (1.0 / num_stages))
	result = [per_stage_k] * num_stages
	# Adjust last stage to ensure product >= vocab_size
	product = 1
	for k in result[:-1]:
		product *= k
	result[-1] = math.ceil(vocab_size / product)
	return result


class MultiStageEvaluator(BaseEvaluator):
	"""Evaluator for multi-stage RAM LM genomes.

	Primary path: Rust+Metal via MultiStageCacheWrapper.
	target_stage (0-indexed) determines which stage evaluate_batch() operates on.
	"""

	def __init__(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int = 50257,
		context_size: int = 4,
		num_stages: int = 2,
		stage_k: Optional[list[int]] = None,
		stage_cluster_type: Optional[list[str]] = None,
		stage_mode: Optional[list[int]] = None,
		target_stage: int = 0,
		num_parts: int = 3,
		num_eval_parts: int = 1,
		seed: Optional[int] = None,
		pad_token_id: int = 50256,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
		adapt_config: Optional[AdaptationConfig] = None,
		sparse_threshold: Optional[int] = None,
	):
		# Defaults
		if stage_cluster_type is None:
			stage_cluster_type = ["bitwise", "bitwise"]
		if stage_k is None:
			stage_k = compute_default_k(num_stages, stage_cluster_type, vocab_size)
		if stage_mode is None:
			from wnn.ram.experiments.experiment import StageMode
			stage_mode = [StageMode.INPUT_CONCAT]

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

		self._num_stages = num_stages
		self._stage_k = stage_k
		self._stage_cluster_type = stage_cluster_type
		self._stage_mode = stage_mode
		self._target_stage = target_stage
		self._pad_token_id = pad_token_id
		self._sparse_threshold = sparse_threshold

		self._progress_callback = None

		# Create Rust backend
		from ram_accelerator import MultiStageCacheWrapper
		self._cache = MultiStageCacheWrapper(
			train_tokens=list(train_tokens),
			eval_tokens=list(eval_tokens),
			vocab_size=vocab_size,
			context_size=context_size,
			k=stage_k[0],  # Rust uses K from stage 0
			num_parts=num_parts,
			num_eval_parts=num_eval_parts,
			pad_token_id=pad_token_id,
			sparse_threshold=sparse_threshold,
			stage_cluster_types=stage_cluster_type,
		)

		# Cache stage dimensions from Rust (stage-agnostic)
		self._stage_input_bits = [self._cache.stage_input_bits(s) for s in range(num_stages)]
		self._bitwise_output_bits = [self._cache.bitwise_output_bits(s) for s in range(num_stages)]

		# Per-stage dimensions — polymorphic on stage type
		# Tiered: K_s (direct class count), Bitwise: ceil(log2(K_s)) (bit count)
		self._stage_num_clusters = [
			self._cache.stage_num_output_clusters(s) for s in range(num_stages)
		]

		stage_types = [stage_cluster_type[s] if s < len(stage_cluster_type) else "bitwise" for s in range(num_stages)]
		print(
			f"[MultiStageEvaluator] Rust backend active "
			f"(stages={num_stages}, k={stage_k}, target_stage={target_stage}, "
			f"types={'+'.join(stage_types)}, "
			f"clusters={self._stage_num_clusters}, "
			f"input_bits={self._stage_input_bits})"
		)

	# ── Rotation (delegated to Rust) ─────────────────────────────────

	def next_train_idx(self) -> int:
		return self._cache.next_train_idx()

	def next_eval_idx(self) -> int:
		return self._cache.next_eval_idx()

	# ── Stage-aware properties ───────────────────────────────────────

	@property
	def target_stage(self) -> int:
		return self._target_stage

	@target_stage.setter
	def target_stage(self, value: int) -> None:
		assert 0 <= value < self._num_stages, f"target_stage must be 0..{self._num_stages-1}, got {value}"
		self._target_stage = value

	@property
	def total_input_bits(self) -> int:
		"""Input bits for the active stage."""
		return self._stage_input_bits[self._target_stage]

	@property
	def num_clusters(self) -> int:
		"""Output clusters (bit-clusters) for the active stage."""
		return self._stage_num_clusters[self._target_stage]

	@property
	def num_stages(self) -> int:
		return self._num_stages

	@property
	def stage_k(self) -> list[int]:
		return self._stage_k

	@property
	def bits_per_cluster_id(self) -> int:
		return self._bitwise_output_bits[0]

	@property
	def bits_per_within_index(self) -> int:
		return self._bitwise_output_bits[1] if self._num_stages > 1 else 0

	@property
	def cluster_sizes(self) -> list[int]:
		return self._cache.cluster_sizes()

	def set_progress_callback(self, callback):
		"""Set callback(batch_num, total_batches, evaluated_so_far, total) for sub-batch progress."""
		self._progress_callback = callback

	# ── Genome flattening ────────────────────────────────────────────

	def _flatten_genomes(
		self,
		genomes: list[ClusterGenome],
	) -> tuple[list[int], list[int], list[int]]:
		"""Flatten per-neuron arrays for Rust, generating random connections if missing."""
		input_bits = self.total_input_bits
		bits_flat = []
		neurons_flat = []
		connections_flat = []

		for g in genomes:
			bits_flat.extend(g.bits_per_neuron)
			neurons_flat.extend(g.neurons_per_cluster)
			if g.connections is not None:
				connections_flat.extend(g.connections)
			else:
				for b in g.bits_per_neuron:
					for _ in range(b):
						connections_flat.append(random.randint(0, input_bits - 1))

		return bits_flat, neurons_flat, connections_flat

	# ── Bitwise evaluation (stage-agnostic) ──────────────────────────

	def _evaluate_bitwise_rust(
		self,
		stage: int,
		genomes: list[ClusterGenome],
		train_idx: int,
		eval_idx: int,
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_bitwise_genomes(
			stage=stage,
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			train_subset_idx=train_idx,
			eval_subset_idx=eval_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

	def _evaluate_bitwise_full_rust(
		self,
		stage: int,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_bitwise_genomes_full(
			stage=stage,
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

	# ── Tiered evaluation ────────────────────────────────────────────

	def _evaluate_tiered_rust(
		self,
		genomes: list[ClusterGenome],
		stage: int,
		train_idx: int,
		eval_idx: int,
	) -> list[tuple[float, float, float]]:
		max_sub = int(os.environ.get("WNN_MAX_TIERED_BATCH", "10"))
		if len(genomes) <= max_sub:
			return self._evaluate_tiered_rust_single(genomes, stage, train_idx, eval_idx)

		all_results = []
		n = len(genomes)
		total_batches = (n + max_sub - 1) // max_sub
		for i in range(0, n, max_sub):
			chunk = genomes[i:i + max_sub]
			t0 = time.time()
			results = self._evaluate_tiered_rust_single(chunk, stage, train_idx, eval_idx)
			elapsed = time.time() - t0
			batch_num = i // max_sub + 1
			logging.info(
				f"[tiered_rust] sub-batch {batch_num}/{total_batches}: "
				f"{len(chunk)} genomes in {elapsed:.1f}s ({elapsed/len(chunk):.2f}s/genome)"
			)
			all_results.extend(results)
			if self._progress_callback:
				self._progress_callback(batch_num, total_batches, len(all_results), n)
		return all_results

	def _evaluate_tiered_rust_single(
		self,
		genomes: list[ClusterGenome],
		stage: int,
		train_idx: int,
		eval_idx: int,
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		t0 = time.time()
		raw = self._cache.evaluate_tiered_genomes(
			stage=stage,
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			train_subset_idx=train_idx,
			eval_subset_idx=eval_idx,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)
		elapsed = time.time() - t0
		if len(genomes) > 0:
			logging.debug(f"[tiered_rust] {len(genomes)} genomes in {elapsed:.1f}s ({elapsed/len(genomes):.2f}s/genome)")
		# Tiered returns (ce, acc), add 0.0 for bit_acc to match interface
		return [(ce, acc, 0.0) for ce, acc in raw]

	def _evaluate_tiered_full_rust(
		self,
		genomes: list[ClusterGenome],
		stage: int,
	) -> list[tuple[float, float, float]]:
		max_sub = int(os.environ.get("WNN_MAX_TIERED_BATCH", "10"))
		if len(genomes) <= max_sub:
			return self._evaluate_tiered_full_rust_single(genomes, stage)

		all_results = []
		n = len(genomes)
		total_batches = (n + max_sub - 1) // max_sub
		for i in range(0, n, max_sub):
			chunk = genomes[i:i + max_sub]
			t0 = time.time()
			results = self._evaluate_tiered_full_rust_single(chunk, stage)
			elapsed = time.time() - t0
			batch_num = i // max_sub + 1
			logging.info(
				f"[tiered_full_rust] sub-batch {batch_num}/{total_batches}: "
				f"{len(chunk)} genomes in {elapsed:.1f}s ({elapsed/len(chunk):.2f}s/genome)"
			)
			all_results.extend(results)
			if self._progress_callback:
				self._progress_callback(batch_num, total_batches, len(all_results), n)
		return all_results

	def _evaluate_tiered_full_rust_single(
		self,
		genomes: list[ClusterGenome],
		stage: int,
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		raw = self._cache.evaluate_tiered_genomes_full(
			stage=stage,
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)
		return [(ce, acc, 0.0) for ce, acc in raw]

	# ── Dispatch helper ──────────────────────────────────────────────

	def _is_stage_tiered(self, stage: int) -> bool:
		return self._stage_cluster_type[stage] == "tiered"

	# ── Public interface (dispatches by target_stage) ────────────────

	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		**kwargs,
	) -> list[EvalResult]:
		"""Evaluate genomes for the active target_stage."""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = self.next_eval_idx()

		start = time.time()
		if self._is_stage_tiered(self._target_stage):
			raw = self._evaluate_tiered_rust(genomes, self._target_stage, train_subset_idx, eval_subset_idx)
		else:
			raw = self._evaluate_bitwise_rust(self._target_stage, genomes, train_subset_idx, eval_subset_idx)
		elapsed = time.time() - start

		# Cache bit accuracy on genomes
		for genome, (_, _, bit_acc) in zip(genomes, raw):
			genome._cached_bit_acc = bit_acc

		# Logging
		log = logger if logger is not None else lambda x: None
		if generation is not None:
			gen = generation + 1
			total = total_generations or "?"
			best_ce = min(r[0] for r in raw) if raw else 0.0
			best_acc = max(r[1] for r in raw) if raw else 0.0
			best_bit_acc = max(r[2] for r in raw) if raw else 0.0
			n = len(raw)
			mean_ce = sum(r[0] for r in raw) / n if n else 0.0
			mean_acc = sum(r[1] for r in raw) / n if n else 0.0
			mean_bit_acc = sum(r[2] for r in raw) / n if n else 0.0
			stage_label = f"S{self._target_stage}"
			log(
				f"[Gen {gen:02d}/{total}] [{stage_label}] {len(genomes)} genomes in {elapsed:.1f}s "
				f"(best CE={best_ce:.4f}, Acc={best_acc:.2%}, BitAcc={best_bit_acc:.2%})"
			)
			self._generation_log.append((
				generation, best_ce, best_acc, best_bit_acc,
				mean_ce, mean_acc, mean_bit_acc,
			))

		return [
			EvalResult(ce=ce, accuracy=acc, bit_accuracy=bit_acc)
			for ce, acc, bit_acc in raw
		]

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[EvalResult]:
		"""Evaluate genomes with full (non-rotated) data for the active target_stage."""
		start = time.time()
		if self._is_stage_tiered(self._target_stage):
			raw = self._evaluate_tiered_full_rust(genomes, self._target_stage)
		else:
			raw = self._evaluate_bitwise_full_rust(self._target_stage, genomes)
		elapsed = time.time() - start

		for genome, (_, _, bit_acc) in zip(genomes, raw):
			genome._cached_bit_acc = bit_acc

		log = logger if logger is not None else lambda x: None
		stage_label = f"S{self._target_stage}"
		log(f"[Full] [{stage_label}] {len(genomes)} genomes in {elapsed:.1f}s")

		return [
			EvalResult(ce=ce, accuracy=acc, bit_accuracy=bit_acc)
			for ce, acc, bit_acc in raw
		]

	# ── Combined CE ──────────────────────────────────────────────────

	def compute_combined_metrics(
		self,
		stage_genomes: list[ClusterGenome],
	) -> EvalResult:
		"""Compute combined CE over the full vocabulary.

		Args:
			stage_genomes: List of genomes, one per stage (0-indexed).

		Trains all stages independently, reconstructs the joint distribution
		P(token) = P(group|ctx) × P(token|group,ctx), and computes CE + accuracy.

		Returns EvalResult with ce, accuracy, plus cluster_ce and within_ce breakdown.
		"""
		assert len(stage_genomes) == self._num_stages, \
			f"Expected {self._num_stages} genomes, got {len(stage_genomes)}"

		sparse_threshold = self._sparse_threshold or 4096

		# Flatten all stages into concatenated arrays
		all_bits = []
		all_neurons = []
		all_connections = []
		stage_num_clusters = []
		for g in stage_genomes:
			all_bits.extend(g.bits_per_neuron)
			all_neurons.extend(g.neurons_per_cluster)
			all_connections.extend(g.connections or [])
			stage_num_clusters.append(len(g.neurons_per_cluster))

		combined_ce, combined_acc, s0_ce, s1_ce = self._cache.evaluate_combined_ce(
			all_bits_per_neuron=all_bits,
			all_neurons_per_cluster=all_neurons,
			all_connections=all_connections,
			stage_num_clusters=stage_num_clusters,
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
			sparse_threshold=sparse_threshold,
		)

		return EvalResult(
			ce=combined_ce,
			accuracy=combined_acc,
			cluster_ce=s0_ce,
			within_ce=s1_ce,
		)

	# ── Re-encode stage with predicted clusters ─────────────────────

	def recompute_stage_with_predictions(
		self,
		frozen_stage: int,
		target_stage: int,
		frozen_genome: ClusterGenome,
	) -> tuple[float, float]:
		"""Re-encode target_stage data using frozen_stage's actual predictions.

		Args:
			frozen_stage: Stage whose genome is frozen and will be used to predict.
			target_stage: Stage whose data will be re-encoded with predictions.
			frozen_genome: The frozen stage's genome.

		Returns (train_accuracy, eval_accuracy) for the frozen stage's predictions.
		"""
		sparse_threshold = self._sparse_threshold or 4096

		return self._cache.recompute_stage_with_predictions(
			frozen_stage=frozen_stage,
			target_stage=target_stage,
			bits_per_neuron=list(frozen_genome.bits_per_neuron),
			neurons_per_cluster=list(frozen_genome.neurons_per_cluster),
			connections=list(frozen_genome.connections or []),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
			sparse_threshold=sparse_threshold,
		)

	# ── Search (overrides BaseEvaluator for single-batch optimization) ───
	# search_neighbors and search_offspring use inherited base class methods.
	# search_neighbors_batch is overridden for single-eval-call optimization.

	def search_neighbors_batch(
		self,
		sources: list[tuple[ClusterGenome, int]],
		max_attempts_multiplier: int,
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
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
	) -> list[list[ClusterGenome]]:
		"""Search neighbors for multiple source genomes in a single eval call.

		Generates ALL candidates across all sources, then does a SINGLE
		evaluate_batch() call. This avoids per-source Rust pipeline overhead.

		Args:
			sources: List of (genome, target_count) pairs
			max_attempts_multiplier: Generate target_count * this many candidates per source
			Other args: same as search_neighbors

		Returns:
			List of neighbor lists, one per source (same order as input)
		"""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if eval_subset_idx is None:
			eval_subset_idx = 0
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rng = random.Random(seed)

		# Generate all candidates for all sources using ClusterGenome.mutate()
		all_candidates: list[ClusterGenome] = []
		source_ranges: list[tuple[int, int, int]] = []  # (start_idx, end_idx, target_count)

		for source_genome, target_count in sources:
			gen_count = min(target_count * max_attempts_multiplier, target_count * 5)
			start = len(all_candidates)
			for _ in range(gen_count):
				all_candidates.append(
					self._mutate_genome_phased(
						source_genome, phase_type,
						bits_mutation_rate, neurons_mutation_rate,
						min_bits, max_bits, min_neurons, max_neurons, rng,
					)
				)
			source_ranges.append((start, len(all_candidates), target_count))

		if not all_candidates:
			return [[] for _ in sources]

		# Single Rust evaluation call for ALL candidates
		if self._is_stage_tiered(self._target_stage):
			results = self._evaluate_tiered_rust(all_candidates, self._target_stage, train_subset_idx, eval_subset_idx)
		else:
			results = self._evaluate_bitwise_rust(self._target_stage, all_candidates, train_subset_idx, eval_subset_idx)

		# Cache fitness on genomes
		for g, (ce, acc, bit_acc) in zip(all_candidates, results):
			g._cached_fitness = (ce, acc)
			g._cached_bit_acc = bit_acc

		# Split results back by source and filter
		output: list[list[ClusterGenome]] = []
		for start, end, target_count in source_ranges:
			passed = []
			below = []
			for i in range(start, end):
				g = all_candidates[i]
				if g._cached_fitness[1] >= accuracy_threshold:
					passed.append(g)
				else:
					below.append(g)

			# Take best up to target_count
			if len(passed) >= target_count:
				passed.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
				output.append(passed[:target_count])
			elif return_best_n:
				below.sort(key=lambda g: (-g._cached_fitness[1], g._cached_fitness[0]))
				need = target_count - len(passed)
				passed.extend(below[:need])
				output.append(passed)
			else:
				output.append(passed)

		return output

	def reset(self, seed: Optional[int] = None) -> None:
		"""Reset subset rotation."""
		self._cache.reset()
		if seed is not None:
			self._seed = seed

	def __repr__(self) -> str:
		mode_names = {0: "TERNARY", 1: "QUAD_BINARY", 2: "QUAD_WEIGHTED"}
		mode = mode_names.get(self._memory_mode, f"UNKNOWN({self._memory_mode})")
		return (
			f"MultiStageEvaluator(vocab={self._vocab_size}, "
			f"context={self._context_size}, k={self._stage_k}, "
			f"target_stage={self._target_stage}, "
			f"clusters={self._stage_num_clusters}, "
			f"mode={mode})"
		)
