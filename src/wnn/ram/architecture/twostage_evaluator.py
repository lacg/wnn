"""
Two-Stage Evaluator — Evaluates two-stage (group + within-group) RAM LM genomes.

Wraps TwoStageCacheWrapper from Rust. Supports:
- Stage 1 evaluation: predict cluster_id bits from context
- Stage 2 evaluation (input_concat): predict within-group index bits from context + cluster_id
- Combined CE: joint P(token) = P(group|ctx) × P(token|group,ctx)

Usage:
	evaluator = TwoStageEvaluator(
		train_tokens=train_tokens,
		eval_tokens=eval_tokens,
		vocab_size=50257,
		context_size=4,
		k=256,
		target_stage=1,  # which stage GA/TS optimizes
	)

	# Evaluate Stage 1 genomes
	results = evaluator.evaluate_batch(genomes)
	# → [EvalResult(ce, accuracy, bit_accuracy), ...]

	# Combined CE (needs genomes for both stages)
	combined = evaluator.compute_combined_metrics(stage1_genome, stage2_genome)
	# → EvalResult(ce, accuracy, cluster_ce=..., within_ce=...)
"""

import random
import time
from typing import Optional, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult, AdaptationConfig


class TwoStageEvaluator(BaseEvaluator):
	"""Evaluator for two-stage RAM LM genomes.

	Primary path: Rust+Metal via TwoStageCacheWrapper.
	target_stage determines which stage evaluate_batch() operates on.
	"""

	def __init__(
		self,
		train_tokens: list[int],
		eval_tokens: list[int],
		vocab_size: int = 50257,
		context_size: int = 4,
		k: int = 256,
		target_stage: int = 1,
		num_parts: int = 3,
		num_eval_parts: int = 1,
		seed: Optional[int] = None,
		pad_token_id: int = 50256,
		memory_mode: int = 0,
		neuron_sample_rate: float = 1.0,
		adapt_config: Optional[AdaptationConfig] = None,
		sparse_threshold: Optional[int] = None,
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

		self._k = k
		self._target_stage = target_stage
		self._pad_token_id = pad_token_id
		self._sparse_threshold = sparse_threshold

		# Create Rust backend
		from ram_accelerator import TwoStageCacheWrapper
		self._cache = TwoStageCacheWrapper(
			train_tokens=list(train_tokens),
			eval_tokens=list(eval_tokens),
			vocab_size=vocab_size,
			context_size=context_size,
			k=k,
			num_parts=num_parts,
			num_eval_parts=num_eval_parts,
			pad_token_id=pad_token_id,
			sparse_threshold=sparse_threshold,
		)

		# Cache stage dimensions from Rust
		self._bits_per_cluster_id = self._cache.bits_per_cluster_id()
		self._bits_per_within_index = self._cache.bits_per_within_index()
		self._stage1_input_bits = self._cache.stage1_input_bits()
		self._stage2_concat_input_bits = self._cache.stage2_concat_input_bits()

		print(
			f"[TwoStageEvaluator] Rust backend active "
			f"(k={k}, target_stage={target_stage}, "
			f"s1_clusters={self._bits_per_cluster_id}, "
			f"s2_clusters={self._bits_per_within_index}, "
			f"s1_input={self._stage1_input_bits}, "
			f"s2_input={self._stage2_concat_input_bits})"
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
		assert value in (1, 2), f"target_stage must be 1 or 2, got {value}"
		self._target_stage = value

	@property
	def total_input_bits(self) -> int:
		"""Input bits for the active stage."""
		if self._target_stage == 1:
			return self._stage1_input_bits
		return self._stage2_concat_input_bits

	@property
	def num_clusters(self) -> int:
		"""Output clusters (bit-clusters) for the active stage."""
		if self._target_stage == 1:
			return self._bits_per_cluster_id
		return self._bits_per_within_index

	@property
	def k(self) -> int:
		return self._k

	@property
	def bits_per_cluster_id(self) -> int:
		return self._bits_per_cluster_id

	@property
	def bits_per_within_index(self) -> int:
		return self._bits_per_within_index

	@property
	def cluster_sizes(self) -> list[int]:
		return self._cache.cluster_sizes()

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

	# ── Stage 1 evaluation ───────────────────────────────────────────

	def _evaluate_stage1_rust(
		self,
		genomes: list[ClusterGenome],
		train_idx: int,
		eval_idx: int,
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_stage1_genomes(
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

	def _evaluate_stage1_full_rust(
		self,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_stage1_genomes_full(
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

	# ── Stage 2 evaluation (input_concat) ────────────────────────────

	def _evaluate_stage2_concat_rust(
		self,
		genomes: list[ClusterGenome],
		train_idx: int,
		eval_idx: int,
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_stage2_concat_genomes(
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

	def _evaluate_stage2_concat_full_rust(
		self,
		genomes: list[ClusterGenome],
	) -> list[tuple[float, float, float]]:
		bits_flat, neurons_flat, conns_flat = self._flatten_genomes(genomes)
		return self._cache.evaluate_stage2_concat_genomes_full(
			bits_per_neuron_flat=bits_flat,
			neurons_per_cluster_flat=neurons_flat,
			connections_flat=conns_flat,
			num_genomes=len(genomes),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
		)

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
		if self._target_stage == 1:
			raw = self._evaluate_stage1_rust(genomes, train_subset_idx, eval_subset_idx)
		else:
			raw = self._evaluate_stage2_concat_rust(genomes, train_subset_idx, eval_subset_idx)
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
		if self._target_stage == 1:
			raw = self._evaluate_stage1_full_rust(genomes)
		else:
			raw = self._evaluate_stage2_concat_full_rust(genomes)
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
		stage1_genome: ClusterGenome,
		stage2_genome: ClusterGenome,
	) -> EvalResult:
		"""Compute combined CE over the full vocabulary.

		Trains both stages independently, reconstructs the joint distribution
		P(token) = P(group|ctx) × P(token|group,ctx), and computes CE + accuracy.

		Returns EvalResult with ce, accuracy, plus cluster_ce and within_ce breakdown.
		"""
		sparse_threshold = self._sparse_threshold or 4096

		combined_ce, combined_acc, s1_ce, s2_ce = self._cache.evaluate_combined_ce(
			s1_bits_per_neuron=list(stage1_genome.bits_per_neuron),
			s1_neurons_per_cluster=list(stage1_genome.neurons_per_cluster),
			s1_connections=list(stage1_genome.connections or []),
			s2_bits_per_neuron=list(stage2_genome.bits_per_neuron),
			s2_neurons_per_cluster=list(stage2_genome.neurons_per_cluster),
			s2_connections=list(stage2_genome.connections or []),
			memory_mode=self._memory_mode,
			neuron_sample_rate=self._neuron_sample_rate,
			rng_seed=self._seed,
			sparse_threshold=sparse_threshold,
		)

		return EvalResult(
			ce=combined_ce,
			accuracy=combined_acc,
			cluster_ce=s1_ce,
			within_ce=s2_ce,
		)

	# ── Mutation + search (same pattern as BitwiseEvaluator) ─────────

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
		"""Mutate a genome to create a neighbor."""
		num_clusters = len(genome.neurons_per_cluster)
		offsets = genome.cluster_neuron_offsets
		new_bits = genome.bits_per_neuron.copy()
		new_neurons = genome.neurons_per_cluster.copy()
		old_bits = genome.bits_per_neuron.copy()
		old_neurons = genome.neurons_per_cluster.copy()

		bits_delta_max = max(1, round(0.1 * (min_bits + max_bits)))
		neurons_delta_max = max(1, round(0.1 * (min_neurons + max_neurons)))

		indices = mutable_clusters if mutable_clusters is not None else list(range(num_clusters))
		for i in indices:
			if i >= num_clusters:
				continue
			if rng.random() < bits_mutation_rate:
				delta = rng.randint(-bits_delta_max, bits_delta_max)
				for n_idx in range(offsets[i], offsets[i + 1]):
					new_bits[n_idx] = max(min_bits, min(max_bits, new_bits[n_idx] + delta))
			if rng.random() < neurons_mutation_rate:
				delta = rng.randint(-neurons_delta_max, neurons_delta_max)
				new_neurons[i] = max(min_neurons, min(max_neurons, new_neurons[i] + delta))

		final_bits = []
		for c in range(num_clusters):
			old_n = old_neurons[c]
			new_n = new_neurons[c]
			cluster_old_bits = new_bits[offsets[c]:offsets[c + 1]]
			for n in range(new_n):
				if n < old_n:
					final_bits.append(cluster_old_bits[n])
				else:
					template = rng.randint(0, old_n - 1) if old_n > 0 else 0
					final_bits.append(cluster_old_bits[template] if old_n > 0 else min_bits)

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
		"""Adjust connections when architecture changes."""
		if old_connections is None or len(old_connections) == 0:
			return None

		input_bits = self.total_input_bits
		result = []

		old_conn_offsets = [0]
		for b in old_bits:
			old_conn_offsets.append(old_conn_offsets[-1] + b)

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
							if rng.random() < 0.1:
								delta = rng.choice([-2, -1, 1, 2])
								conn = max(0, min(input_bits - 1, conn + delta))
							result.append(conn)
						else:
							result.append(rng.randint(0, input_bits - 1))
				else:
					if o_n > 0:
						template = rng.randint(0, o_n - 1)
						tmpl_n_idx = old_cluster_start + template
						o_b = old_bits[tmpl_n_idx]
						tmpl_start = old_conn_offsets[tmpl_n_idx]
						for bit in range(n_b):
							if bit < o_b:
								conn = old_connections[tmpl_start + bit]
								delta = rng.choice([-2, -1, 1, 2])
								conn = max(0, min(input_bits - 1, conn + delta))
								result.append(conn)
							else:
								result.append(rng.randint(0, input_bits - 1))
					else:
						for _ in range(n_b):
							result.append(rng.randint(0, input_bits - 1))

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
		"""Search for neighbor genomes above accuracy threshold."""
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

		while len(passed) < target_count and evaluated < max_attempts:
			remaining = target_count - len(passed)
			batch_n = min(remaining + 5, batch_size, max_attempts - evaluated)
			if batch_n <= 0:
				break

			batch = [
				self._mutate_genome(
					genome, bits_mutation_rate, neurons_mutation_rate,
					min_bits, max_bits, min_neurons, max_neurons,
					mutable_clusters, rng,
				)
				for _ in range(batch_n)
			]

			if self._target_stage == 1:
				results = self._evaluate_stage1_rust(batch, train_subset_idx, eval_subset_idx)
			else:
				results = self._evaluate_stage2_concat_rust(batch, train_subset_idx, eval_subset_idx)
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
	):
		"""Search for GA offspring above accuracy threshold."""
		from wnn.ram.architecture.cached_evaluator import OffspringSearchResult

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
				parent1 = self._tournament_select(population, tournament_size, rng)
				if rng.random() < crossover_rate and len(population) > 1:
					parent2 = self._tournament_select(population, tournament_size, rng)
					child = self._crossover(parent1, parent2, rng)
				else:
					child = parent1.clone()
				child = self._mutate_genome(
					child, bits_mutation_rate, neurons_mutation_rate,
					min_bits, max_bits, min_neurons, max_neurons,
					mutable_clusters, rng,
				)
				batch.append(child)

			if self._target_stage == 1:
				results = self._evaluate_stage1_rust(batch, train_subset_idx, eval_subset_idx)
			else:
				results = self._evaluate_stage2_concat_rust(batch, train_subset_idx, eval_subset_idx)
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
		contestants = rng.sample(population, min(tournament_size, len(population)))
		best = min(contestants, key=lambda x: x[1])
		return best[0].clone()

	def _crossover(
		self,
		parent1: ClusterGenome,
		parent2: ClusterGenome,
		rng: random.Random,
	) -> ClusterGenome:
		n = len(parent1.neurons_per_cluster)
		p1_offsets = parent1.cluster_neuron_offsets
		p2_offsets = parent2.cluster_neuron_offsets

		new_bits = []
		new_neurons = []
		for i in range(n):
			if rng.random() < 0.5:
				new_neurons.append(parent1.neurons_per_cluster[i])
				new_bits.extend(parent1.bits_per_neuron[p1_offsets[i]:p1_offsets[i + 1]])
			else:
				new_neurons.append(parent2.neurons_per_cluster[i])
				new_bits.extend(parent2.bits_per_neuron[p2_offsets[i]:p2_offsets[i + 1]])

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

	def reset(self, seed: Optional[int] = None) -> None:
		"""Reset subset rotation."""
		self._cache.reset()
		if seed is not None:
			self._seed = seed

	def __repr__(self) -> str:
		mode_names = {0: "TERNARY", 1: "QUAD_BINARY", 2: "QUAD_WEIGHTED"}
		mode = mode_names.get(self._memory_mode, f"UNKNOWN({self._memory_mode})")
		return (
			f"TwoStageEvaluator(vocab={self._vocab_size}, "
			f"context={self._context_size}, k={self._k}, "
			f"target_stage={self._target_stage}, "
			f"s1_clusters={self._bits_per_cluster_id}, "
			f"s2_clusters={self._bits_per_within_index}, "
			f"mode={mode})"
		)
