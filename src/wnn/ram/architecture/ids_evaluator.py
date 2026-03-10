"""
IDS Evaluator — Genome evaluator for intrusion detection classification.

Parallel to TieredEvaluator but operates on pre-encoded binary features
instead of token sequences. Uses the Rust IDSCacheWrapper for zero-copy
stratified subset rotation and hybrid CPU+GPU evaluation.

Usage:
	from wnn.ids import load_unsw_nb15
	dataset = load_unsw_nb15(n_bits=8)

	evaluator = IDSEvaluator(dataset, classification="binary", num_parts=3)

	# Per iteration: evaluate genomes with subset rotation
	results = evaluator.evaluate_batch(genomes)

	# Final evaluation with full data
	final_results = evaluator.evaluate_batch_full(genomes)
"""

import time
from typing import Optional, Callable

from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome
from wnn.ram.architecture.base_evaluator import BaseEvaluator, EvalResult, OffspringSearchResult


class IDSEvaluator(BaseEvaluator):
	"""
	Rust-backed evaluator for IDS classification tasks.

	Holds all pre-encoded binary features in Rust memory. Per-iteration
	evaluations use zero-copy stratified subset selection via indices.
	"""

	def __init__(
		self,
		dataset,  # IDSDataset
		classification: str = "binary",  # "binary" or "multi"
		num_parts: int = 3,
		num_negatives: Optional[int] = None,  # None = num_classes - 1 (exhaustive)
		empty_value: float = 0.0,
		seed: Optional[int] = None,
		log_path: Optional[str] = None,
		neuron_sample_rate: float = 1.0,
	):
		if classification == "binary":
			y_train = dataset.y_train_binary
			y_test = dataset.y_test_binary
			num_classes = 2
		elif classification == "multi":
			y_train = dataset.y_train_multi
			y_test = dataset.y_test_multi
			num_classes = len(dataset.category_names)
		else:
			raise ValueError(f"classification must be 'binary' or 'multi', got '{classification}'")

		total_features = dataset.X_train.shape[1]

		if num_negatives is None:
			num_negatives = num_classes - 1

		# BaseEvaluator expects LM-style args; pass minimal values
		# context_size=1 and vocab_size=num_classes keeps base_evaluator happy
		super().__init__(
			train_tokens=[],
			eval_tokens=[],
			vocab_size=num_classes,
			context_size=1,
			num_parts=num_parts,
			seed=seed,
			neuron_sample_rate=neuron_sample_rate,
			log_path=log_path,
		)

		# Override the total_input_bits computed by BaseEvaluator
		self._total_input_bits = total_features
		self._empty_value = empty_value
		self._num_classes = num_classes
		self._classification = classification
		self._y_test = [int(y) for y in y_test]
		self._class_names = list(dataset.category_names) if hasattr(dataset, 'category_names') else None

		# Import and create Rust cache
		try:
			import ram_accelerator
		except ImportError:
			raise ImportError(
				"ram_accelerator not available. Build with: "
				"cd src/wnn/ram/strategies/accelerator && maturin develop --release"
			)

		# Flatten features to list[bool] for Rust
		train_features = dataset.X_train.ravel().tolist()
		train_labels = [int(y) for y in y_train]
		eval_features = dataset.X_test.ravel().tolist()
		eval_labels = [int(y) for y in y_test]

		self._cache = ram_accelerator.IDSCacheWrapper(
			train_features=train_features,
			train_labels=train_labels,
			eval_features=eval_features,
			eval_labels=eval_labels,
			num_classes=num_classes,
			total_features=total_features,
			num_parts=num_parts,
			num_negatives=num_negatives,
			seed=self._seed,
		)

		self._train_call_count = 0

	def next_train_idx(self) -> int:
		self._train_call_count += 1
		return self._cache.next_train_idx()

	def next_eval_idx(self) -> int:
		return 0  # IDS always uses full eval (no eval rotation)

	def _flatten_genomes(self, genomes: list[ClusterGenome]):
		"""Flatten genome arrays for Rust."""
		bits_flat = []
		neurons_flat = []
		connections_flat = []
		for g in genomes:
			bits_flat.extend(g.bits_per_neuron)
			neurons_flat.extend(g.neurons_per_cluster)
			if g.connections is not None:
				connections_flat.extend(g.connections)
		return bits_flat, neurons_flat, connections_flat

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
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		# If _adapt_config is set (by AdaptationStrategy for dedicated *genesis phases),
		# delegate to adaptive path — train, adapt architecture, retrain, eval.
		# Genomes are updated in-place with adapted architecture.
		if self._adapt_config is not None:
			config = self._adapt_config
			adaptive_results = self.evaluate_batch_adaptive(
				genomes,
				train_subset_idx=train_subset_idx,
				generation=self._generation,
				total_generations=config.total_generations,
				synaptogenesis=config.synaptogenesis_enabled,
				neurogenesis=config.neurogenesis_enabled,
				axonogenesis=config.axonogenesis_enabled,
				min_bits=config.min_bits,
				max_bits=config.max_bits,
				warmup_generations=config.warmup_generations,
				stats_sample_size=config.stats_sample_size,
				passes_per_eval=config.passes_per_eval,
			)
			# Update genomes in-place (like BitwiseEvaluator does)
			results = []
			for genome, (eval_result, adapted_genome) in zip(genomes, adaptive_results):
				genome.bits_per_neuron = adapted_genome.bits_per_neuron
				genome.neurons_per_cluster = adapted_genome.neurons_per_cluster
				genome.connections = adapted_genome.connections
				genome._cached_fitness = adapted_genome._cached_fitness
				results.append(eval_result)
			return results

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		raw_results = self._cache.evaluate_genomes_hybrid(
			bits_flat,
			neurons_flat,
			connections_flat,
			len(genomes),
			train_subset_idx,
			self._empty_value,
			self._neuron_sample_rate,
			0,  # rng_seed
		)

		return [EvalResult(ce=ce, accuracy=acc, f1_macro=f1, fpr=fpr) for ce, acc, f1, fpr in raw_results]

	def evaluate_batch_full(
		self,
		genomes: list[ClusterGenome],
		logger: Optional[Callable[[str], None]] = None,
	) -> list[EvalResult]:
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		raw_results = self._cache.evaluate_genomes_full_hybrid(
			bits_flat,
			neurons_flat,
			connections_flat,
			len(genomes),
			self._empty_value,
			self._neuron_sample_rate,
			0,  # rng_seed
		)

		return [EvalResult(ce=ce, accuracy=acc, f1_macro=f1, fpr=fpr) for ce, acc, f1, fpr in raw_results]

	def evaluate_batch_adaptive(
		self,
		genomes: list[ClusterGenome],
		train_subset_idx: Optional[int] = None,
		generation: int = 0,
		total_generations: int = 250,
		synaptogenesis: bool = True,
		neurogenesis: bool = False,
		axonogenesis: bool = False,
		min_bits: int = 4,
		max_bits: int = 24,
		warmup_generations: int = 10,
		stats_sample_size: int = 10000,
		passes_per_eval: int = 1,
		**kwargs,
	) -> list[tuple[EvalResult, ClusterGenome]]:
		"""Evaluate with training-time adaptation, returning adapted genomes.

		After initial training, computes per-neuron/cluster statistics and applies
		adaptation passes:
		  - synaptogenesis: prune/grow connections per neuron
		  - neurogenesis: add/remove neurons per cluster
		  - axonogenesis: rewire low-value connections to high-MI inputs

		If the genome was modified, it is retrained and the adapted architecture
		is returned alongside the scores.

		Returns list of (EvalResult, adapted_ClusterGenome) tuples.
		"""
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()

		bits_flat, neurons_flat, connections_flat = self._flatten_genomes(genomes)

		raw_results = self._cache.evaluate_genomes_hybrid_adaptive(
			bits_flat, neurons_flat, connections_flat,
			len(genomes), train_subset_idx,
			self._empty_value, self._neuron_sample_rate, 0,
			synaptogenesis_enabled=synaptogenesis,
			neurogenesis_enabled=neurogenesis,
			axonogenesis_enabled=axonogenesis,
			min_bits=min_bits,
			max_bits=max_bits,
			warmup_generations=warmup_generations,
			total_generations=total_generations,
			generation=generation,
			total_input_bits=self._total_input_bits,
			stats_sample_size=stats_sample_size,
			passes_per_eval=passes_per_eval,
			**kwargs,
		)

		results = []
		for ce, acc, f1, fpr, adapted_bits, adapted_neurons, adapted_conns, pruned, grown, added, removed, rewired in raw_results:
			eval_result = EvalResult(ce=ce, accuracy=acc, f1_macro=f1, fpr=fpr)
			adapted_genome = ClusterGenome(
				bits_per_neuron=list(adapted_bits),
				neurons_per_cluster=list(adapted_neurons),
				connections=list(adapted_conns) if adapted_conns else None,
			)
			adapted_genome._cached_fitness = (ce, acc, f1, fpr)
			results.append((eval_result, adapted_genome))
		return results

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
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
	) -> list[ClusterGenome]:
		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		effective_log_path = log_path if log_path is not None else self._log_path

		results = self._cache.search_neighbors(
			base_bits=genome.bits_per_neuron,
			base_neurons=genome.neurons_per_cluster,
			base_connections=genome.connections if genome.connections else [],
			target_count=target_count,
			max_attempts=max_attempts,
			accuracy_threshold=accuracy_threshold,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			bits_mutation_rate=bits_mutation_rate,
			neurons_mutation_rate=neurons_mutation_rate,
			train_subset_idx=train_subset_idx,
			empty_value=self._empty_value,
			seed=seed,
			log_path=effective_log_path,
			generation=generation,
			total_generations=total_generations,
			return_best_n=return_best_n,
			mutable_clusters=mutable_clusters,
			phase_type=phase_type,
		)

		genomes = []
		for bits, neurons, connections, ce, acc, f1, fpr in results:
			g = ClusterGenome(
				bits_per_neuron=list(bits),
				neurons_per_cluster=list(neurons),
				connections=list(connections) if connections else None,
			)
			g._cached_fitness = (ce, acc, f1, fpr)
			genomes.append(g)
		return genomes

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
		logger: Optional[Callable[[str], None]] = None,
		generation: Optional[int] = None,
		total_generations: Optional[int] = None,
		return_best_n: bool = True,
		mutable_clusters: Optional[list[int]] = None,
		phase_type: int = 0,
		fitness_scores: Optional[list[float]] = None,
	) -> OffspringSearchResult:
		if not population:
			return OffspringSearchResult(genomes=[], evaluated=0, viable=0)

		if train_subset_idx is None:
			train_subset_idx = self.next_train_idx()
		if seed is None:
			seed = int(time.time() * 1000) % (2**32)

		rust_population = []
		for genome, fitness in population:
			rust_population.append((
				genome.bits_per_neuron,
				genome.neurons_per_cluster,
				genome.connections if genome.connections else [],
				fitness,
			))

		effective_log_path = log_path if log_path is not None else self._log_path

		candidates, evaluated, viable = self._cache.search_offspring(
			population=rust_population,
			target_count=target_count,
			max_attempts=max_attempts,
			accuracy_threshold=accuracy_threshold,
			min_bits=min_bits,
			max_bits=max_bits,
			min_neurons=min_neurons,
			max_neurons=max_neurons,
			bits_mutation_rate=bits_mutation_rate,
			neurons_mutation_rate=neurons_mutation_rate,
			crossover_rate=crossover_rate,
			tournament_size=tournament_size,
			train_subset_idx=train_subset_idx,
			empty_value=self._empty_value,
			seed=seed,
			log_path=effective_log_path,
			generation=generation,
			total_generations=total_generations,
			return_best_n=return_best_n,
			mutable_clusters=mutable_clusters,
			phase_type=phase_type,
		)

		genomes = []
		for bits, neurons, connections, ce, acc, f1, fpr in candidates:
			g = ClusterGenome(
				bits_per_neuron=list(bits),
				neurons_per_cluster=list(neurons),
				connections=list(connections) if connections else None,
			)
			g._cached_fitness = (ce, acc, f1, fpr)
			genomes.append(g)

		return OffspringSearchResult(genomes=genomes, evaluated=evaluated, viable=viable)

	def predict(
		self,
		genome: ClusterGenome,
		rng_seed: int = 0,
	) -> list[int]:
		"""Train genome on full training data and return per-example test predictions.

		Returns list of predicted class indices for each eval example.
		Uses the Rust accelerator with GPU for both training and inference.
		"""
		bits_flat, neurons_flat, connections_flat = self._flatten_genomes([genome])
		return self._cache.predict_examples(
			bits_flat,
			neurons_flat,
			connections_flat,
			self._empty_value,
			self._neuron_sample_rate,
			rng_seed,
		)

	def reset(self, seed: Optional[int] = None) -> None:
		self._cache.reset(seed)
		self._train_call_count = 0

	@property
	def num_train_subsets(self) -> int:
		return self._cache.num_train_subsets()

	@property
	def num_classes(self) -> int:
		return self._num_classes

	@property
	def y_test(self) -> list[int]:
		return self._y_test

	@property
	def class_names(self) -> list[str] | None:
		return self._class_names

	def predict_classes(self, genome: ClusterGenome, rng_seed: int = 0) -> list[int]:
		"""Alias for predict() — returns per-example class predictions on eval set."""
		return self.predict(genome, rng_seed)

	def __repr__(self) -> str:
		return (
			f"IDSEvaluator(classes={self._num_classes}, "
			f"features={self._total_input_bits}, "
			f"classification='{self._classification}', "
			f"parts={self._num_parts})"
		)
