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
		seed: Optional[int] = None,
		pad_token_id: int = 50256,
	):
		self._vocab_size = vocab_size
		self._context_size = context_size
		self._neurons_per_cluster = neurons_per_cluster
		self._bits_per_neuron = bits_per_neuron
		self._num_parts = num_parts
		self._pad_token_id = pad_token_id

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
				pad_token_id=pad_token_id,
			)
			print(f"[BitwiseEvaluator] Rust+Metal backend active "
				  f"(eval={self._rust_cache.num_eval()} examples)")
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
		"""Eval always uses full data."""
		return 0

	@property
	def vocab_size(self) -> int:
		return self._vocab_size

	@property
	def total_input_bits(self) -> int:
		return self._total_input_bits

	@property
	def num_parts(self) -> int:
		return self._num_parts

	# =========================================================================
	# Rust+Metal evaluation (primary path)
	# =========================================================================

	def _flatten_genomes(self, genomes: list[ClusterGenome]) -> list[int]:
		"""Flatten all genomes' connections into a single list."""
		conns = []
		for g in genomes:
			if g.connections is not None:
				conns.extend(g.connections)
			else:
				# Generate random connections if genome has none
				num_bits = bits_needed(self._vocab_size)
				total_neurons = num_bits * self._neurons_per_cluster
				import random
				conns.extend([
					random.randint(0, self._total_input_bits - 1)
					for _ in range(total_neurons * self._bits_per_neuron)
				])
		return conns

	def _evaluate_batch_rust(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: int,
	) -> list[tuple[float, float]]:
		"""Evaluate using Rust+Metal backend."""
		connections_flat = self._flatten_genomes(genomes)
		return self._rust_cache.evaluate_genomes(
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			neurons_per_cluster=self._neurons_per_cluster,
			bits_per_neuron=self._bits_per_neuron,
			train_subset_idx=train_subset_idx,
		)

	def _evaluate_batch_full_rust(
		self,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float]]:
		"""Evaluate with full data using Rust+Metal backend."""
		connections_flat = self._flatten_genomes(genomes)
		return self._rust_cache.evaluate_genomes_full(
			connections_flat=connections_flat,
			num_genomes=len(genomes),
			neurons_per_cluster=self._neurons_per_cluster,
			bits_per_neuron=self._bits_per_neuron,
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
		"""Evaluate multiple genomes using subset rotation."""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		if self._rust_cache is not None:
			start = time.time()
			results = self._evaluate_batch_rust(genomes, train_subset_idx)
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

	def reset(self, seed: Optional[int] = None) -> None:
		"""Reset subset rotation."""
		if self._rust_cache is not None:
			self._rust_cache.reset()
		if seed is not None:
			self._seed = seed
		self._train_rotation_idx = 0

	def __repr__(self) -> str:
		backend = "Rust+Metal" if self._rust_cache is not None else "Python"
		return (
			f"BitwiseEvaluator(vocab={self._vocab_size}, "
			f"context={self._context_size}, "
			f"neurons={self._neurons_per_cluster}, "
			f"bits={self._bits_per_neuron}, "
			f"parts={self._num_parts}, "
			f"backend={backend})"
		)
