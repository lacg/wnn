"""
Bitwise Evaluator - Evaluates BitwiseRAMLM genomes for GA/TS optimization.

Each genome specifies connections for 16 clusters (one per output bit).
Evaluation = create BitwiseRAMLM → set connections → train → evaluate CE/accuracy.

Implements the same interface as CachedEvaluator so it plugs directly into
the existing GA/TS experiment framework.

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

from torch import tensor, long as torch_long

from wnn.ram.core.models import BitwiseRAMLM
from wnn.ram.core.RAMClusterLayer import bits_needed
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


class BitwiseEvaluator:
	"""
	Evaluator for BitwiseRAMLM genomes.

	Creates a BitwiseRAMLM per genome, trains on training tokens,
	evaluates CE/accuracy on eval tokens. With only 16 clusters,
	each train+eval cycle takes seconds (not minutes like 50K clusters).

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

		# Subset rotation
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)
		self._seed = seed
		self._train_rotation_idx = 0

		# Pre-split training data into parts
		n = len(train_tokens)
		part_size = n // num_parts
		self._train_parts = []
		for i in range(num_parts):
			start = i * part_size
			end = start + part_size if i < num_parts - 1 else n
			self._train_parts.append(train_tokens[start:end])

	def next_train_idx(self) -> int:
		"""Advance and return next train subset index."""
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

	def _create_model(self, rng: Optional[int] = None) -> BitwiseRAMLM:
		"""Create a fresh BitwiseRAMLM instance."""
		return BitwiseRAMLM(
			vocab_size=self._vocab_size,
			context_size=self._context_size,
			neurons_per_cluster=self._neurons_per_cluster,
			bits_per_neuron=self._bits_per_neuron,
			pad_token_id=self._pad_token_id,
			rng=rng,
		)

	def _apply_genome_connections(self, model: BitwiseRAMLM, genome: ClusterGenome) -> None:
		"""Set model connections from genome's flat connection list."""
		if genome.connections is not None:
			total_neurons = model.layer.total_neurons
			bits_per_neuron = model.layer.bits_per_neuron
			conn_tensor = tensor(genome.connections, dtype=torch_long).view(
				total_neurons, bits_per_neuron
			)
			model.connections = conn_tensor

	def _train_and_evaluate(
		self,
		genome: ClusterGenome,
		train_tokens: list[int],
		eval_tokens: list[int],
		batch_size: int = 2000,
	) -> tuple[float, float]:
		"""Create model, set connections, train, evaluate → (CE, accuracy)."""
		model = self._create_model()
		self._apply_genome_connections(model, genome)

		# Reset memory before training (connections may have been reused)
		model.reset_memory()

		# Train
		model.train_epoch_fast(
			token_ids=train_tokens,
			batch_size=batch_size,
			verbose=False,
		)

		# Evaluate
		stats = model.evaluate_fast(
			token_ids=eval_tokens,
			batch_size=5000,
			verbose=False,
		)

		return stats["cross_entropy"], stats["accuracy"]

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
		"""
		Evaluate multiple genomes using subset rotation.

		Same interface as CachedEvaluator.evaluate_batch.
		"""
		log = logger if logger is not None else lambda x: None

		# Determine train subset
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		train_data = self._train_parts[train_subset_idx % self._num_parts]
		eval_data = self._eval_tokens

		results = []
		for i, genome in enumerate(genomes):
			start = time.time()
			ce, acc = self._train_and_evaluate(genome, train_data, eval_data)
			elapsed = time.time() - start
			results.append((ce, acc))

			if generation is not None:
				gen = generation + 1
				total = total_generations or len(genomes)
				log(f"[Gen {gen:02d}/{total:02d}] Genome {i+1}/{len(genomes)}: CE={ce:.4f}, Acc={acc:.2%} ({elapsed:.1f}s)")

		return results

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[tuple[float, float]]:
		"""
		Evaluate genomes using full train + eval data.

		Same interface as CachedEvaluator.evaluate_batch_full.
		"""
		log = logger if logger is not None else lambda x: None

		results = []
		for i, genome in enumerate(genomes):
			ce, acc = self._train_and_evaluate(
				genome, self._train_tokens, self._eval_tokens
			)
			results.append((ce, acc))
			log(f"[Full] Genome {i+1}/{len(genomes)}: CE={ce:.4f}, Acc={acc:.2%}")

		return results

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
		if seed is not None:
			self._seed = seed
		self._train_rotation_idx = 0

	def __repr__(self) -> str:
		return (
			f"BitwiseEvaluator(vocab={self._vocab_size}, "
			f"context={self._context_size}, "
			f"neurons={self._neurons_per_cluster}, "
			f"bits={self._bits_per_neuron}, "
			f"parts={self._num_parts})"
		)
