"""
Base Evaluator — Abstract base class for all genome evaluators.

Provides shared behavior:
- EvalResult: Standardized evaluation result with backward-compatible tuple unpacking
- AdaptationConfig: Configuration for training-time architecture adaptation
- BaseEvaluator: ABC with data rotation, seed management, generation tracking,
  adaptation config, and convenience methods

Subclasses implement the actual evaluation logic:
- BitwiseEvaluator: Bitwise RAM with Rust+Metal pipeline
- CachedEvaluator: Tiered RAM with Rust hybrid CPU+GPU
- TwoStageEvaluator: Two-stage (group prediction + within-group) — future
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


@dataclass
class AdaptationConfig:
	"""Configuration for training-time architecture adaptation.

	Uses **relative** thresholds that scale with architecture size.
	See docs/ADAPTATION_MECHANISMS.md for design rationale.
	"""

	# Synaptogenesis (connection level)
	synaptogenesis_enabled: bool = False
	prune_entropy_ratio: float = 0.3       # Prune if entropy < median * ratio
	grow_fill_utilization: float = 0.5     # Grow if fill > expected_fill * factor
	grow_error_baseline: float = 0.35      # Grow if error > this (below random 0.5)
	min_bits: int = 4
	max_bits: int = 24

	# Neurogenesis (cluster level)
	neurogenesis_enabled: bool = False

	# Axonogenesis (connection rewiring)
	axonogenesis_enabled: bool = False
	axon_entropy_threshold: float = 0.3
	axon_improvement_factor: float = 1.5
	axon_rewire_count: int = 2
	cluster_error_factor: float = 0.7      # Add if error > 0.5 * factor
	cluster_fill_utilization: float = 0.5  # Add if mean_fill > expected * factor
	neuron_prune_percentile: float = 0.1   # Bottom 10% candidates for removal
	neuron_removal_factor: float = 0.5     # Only remove if score < mean * factor
	max_growth_ratio: float = 1.5          # Max neurons = initial * ratio
	min_neurons: int = 3
	max_neurons_per_pass: int = 3

	# Schedule (cosine annealing, warmup-excluded)
	warmup_generations: int = 10
	cooldown_iterations: int = 5
	stabilize_fraction: float = 0.25       # Last 25% of post-warmup: frozen
	total_generations: int = 250

	# Shared
	passes_per_eval: int = 1
	stats_sample_size: int = 10_000


@dataclass
class EvalResult:
	"""Standardized evaluation result for all evaluators.

	Supports backward-compatible tuple unpacking:
	- When bit_accuracy is None: unpacks as (ce, accuracy)
	- When bit_accuracy is set: unpacks as (ce, accuracy, bit_accuracy)

	Two-stage fields (cluster_ce, within_ce, etc.) are for future use.
	"""
	ce: float                              # Cross-entropy (lower = better)
	accuracy: float                        # Token accuracy (higher = better)
	bit_accuracy: Optional[float] = None   # Weighted bit accuracy (bitwise only)
	cluster_ce: Optional[float] = None     # Stage 1 CE (two-stage only)
	cluster_accuracy: Optional[float] = None  # Stage 1 accuracy (two-stage only)
	within_ce: Optional[float] = None      # Stage 2 CE (two-stage only)
	within_accuracy: Optional[float] = None  # Stage 2 accuracy (two-stage only)

	def __iter__(self):
		"""Yield (ce, accuracy) or (ce, accuracy, bit_accuracy) for tuple unpacking."""
		yield self.ce
		yield self.accuracy
		if self.bit_accuracy is not None:
			yield self.bit_accuracy

	def __getitem__(self, idx: int):
		"""Index-based access: result[0]=ce, result[1]=accuracy, result[2]=bit_accuracy."""
		if idx == 0:
			return self.ce
		elif idx == 1:
			return self.accuracy
		elif idx == 2:
			return self.bit_accuracy
		raise IndexError(f"EvalResult index {idx} out of range")

	def __len__(self) -> int:
		"""Length: 2 if bit_accuracy is None, 3 otherwise."""
		return 3 if self.bit_accuracy is not None else 2


class BaseEvaluator(ABC):
	"""Abstract base for all genome evaluators.

	Shared behavior:
	- Data rotation (train/eval subset cycling)
	- Seed management
	- Generation tracking and logging
	- Adaptation config (synaptogenesis, neurogenesis, axonogenesis)
	- Convenience methods (evaluate_single, evaluate_single_with_accuracy)

	Subclasses implement evaluation logic via evaluate_batch() and evaluate_batch_full().
	"""

	def __init__(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int,
		context_size: int,
		num_parts: int = 3,
		num_eval_parts: int = 1,
		seed: Optional[int] = None,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
		adapt_config: Optional[AdaptationConfig] = None,
	):
		self._vocab_size = vocab_size
		self._context_size = context_size
		self._num_parts = num_parts
		self._num_eval_parts = num_eval_parts
		self._memory_mode = memory_mode
		self._neuron_sample_rate = neuron_sample_rate
		self._adapt_config = adapt_config
		self._generation = 0

		if seed is None:
			seed = int(time.time() * 1000) % (2**32)
		self._seed = seed

		self._train_tokens = train_tokens
		self._eval_tokens = eval_tokens

		# Compute input dimensions
		self._bits_per_token = bits_needed(vocab_size)
		self._total_input_bits = context_size * self._bits_per_token

		# Per-generation metrics history for correlation tracking
		self._generation_log: list[tuple] = []

	# ── Rotation (abstract — each subclass delegates to its Rust cache) ───

	@abstractmethod
	def next_train_idx(self) -> int: ...

	@abstractmethod
	def next_eval_idx(self) -> int: ...

	# ── Core evaluation (abstract) ────────────────────────────────────────

	@abstractmethod
	def evaluate_batch(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		**kwargs,
	) -> list[EvalResult]: ...

	@abstractmethod
	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[EvalResult]: ...

	# ── Convenience (shared) ──────────────────────────────────────────────

	def evaluate_single(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> float:
		"""Evaluate a single genome, returning CE only."""
		return self.evaluate_batch(
			[genome], train_subset_idx, eval_subset_idx,
		)[0].ce

	def evaluate_single_with_accuracy(
		self,
		genome: ClusterGenome,
		train_subset_idx: Optional[int] = None,
		eval_subset_idx: Optional[int] = None,
	) -> tuple[float, float]:
		"""Evaluate a single genome, returning (CE, accuracy)."""
		r = self.evaluate_batch(
			[genome], train_subset_idx, eval_subset_idx,
		)[0]
		return (r.ce, r.accuracy)

	def evaluate_single_full(self, genome: ClusterGenome) -> tuple[float, float]:
		"""Evaluate a single genome with full data, returning (CE, accuracy)."""
		r = self.evaluate_batch_full([genome])[0]
		return (r.ce, r.accuracy)

	# ── Generation tracking (shared) ──────────────────────────────────────

	def set_generation(self, gen: int, total_generations: int | None = None) -> None:
		"""Set current generation for adaptive evaluation warmup tracking."""
		self._generation = gen
		if total_generations is not None and self._adapt_config is not None:
			self._adapt_config.total_generations = total_generations

	# ── Properties (shared) ───────────────────────────────────────────────

	@property
	def vocab_size(self) -> int:
		return self._vocab_size

	@property
	def context_size(self) -> int:
		return self._context_size

	@property
	def total_input_bits(self) -> int:
		return self._total_input_bits

	@property
	def num_parts(self) -> int:
		return self._num_parts

	@property
	def num_eval_parts(self) -> int:
		return self._num_eval_parts

	@property
	def adapt_config(self) -> Optional[AdaptationConfig]:
		return self._adapt_config

	@property
	def generation_log(self) -> list[tuple]:
		"""Per-generation metrics from evaluate_batch calls."""
		return self._generation_log

	def clear_generation_log(self) -> None:
		"""Clear per-generation metrics (call between phases)."""
		self._generation_log.clear()
