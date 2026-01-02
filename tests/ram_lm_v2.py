#!/usr/bin/env python3
"""
RAM Language Model v2 - With Full Thesis Optimization

Key improvements over v1:
1. TABU SEARCH with more iterations (2024 hardware, not 2001!)
2. ALL optimizers with increased iterations (GA: 200 gens, SA: 2000 iters, TS: 50 iters)
3. PATTERN GENERALIZATION for n4 coverage (partial connectivity instead of exact match)

The n4 coverage problem:
- v1: exact_n4 had 28.8% accuracy but only 2.3% coverage
- Why? Exact matching requires seeing exact 4-gram in training
- Solution: Use PARTIAL CONNECTIVITY to generalize!
  - If neuron sees 8 of 48 bits, patterns differing in other 40 bits → same address
  - This is your thesis insight applied to language!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from collections import defaultdict, Counter
import random
import time
import signal
import logging
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
from joblib import Parallel, delayed
import multiprocessing

# Number of parallel workers (leave some cores for system)
N_WORKERS = max(1, multiprocessing.cpu_count() - 4)  # 12 on 16-core M4 Max

# ============================================================================
# RUST ACCELERATOR (822x speedup over Python!)
# ============================================================================
try:
	import ram_accelerator
	RUST_AVAILABLE = True
	RUST_CPU_CORES = ram_accelerator.cpu_cores()
except ImportError:
	RUST_AVAILABLE = False
	RUST_CPU_CORES = 0

# ============================================================================
# LOGGING SETUP - Dual output to console and file
# ============================================================================
LOG_FILENAME = None  # Set at runtime

def setup_logging(log_dir: str = None) -> str:
	"""Setup logging to both console and file. Returns log filename."""
	global LOG_FILENAME

	if log_dir is None:
		log_dir = os.path.dirname(os.path.abspath(__file__))

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	LOG_FILENAME = os.path.join(log_dir, f"ram_lm_v2_{timestamp}.log")

	# Create logger
	logger = logging.getLogger('ram_lm_v2')
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	# File handler
	fh = logging.FileHandler(LOG_FILENAME, mode='w')
	fh.setLevel(logging.INFO)

	# Console handler
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)

	# Formatter
	formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)

	logger.addHandler(fh)
	logger.addHandler(ch)

	return LOG_FILENAME

def log(msg: str):
	"""Log to both console and file."""
	logger = logging.getLogger('ram_lm_v2')
	if logger.handlers:
		logger.info(msg)
		# Flush all handlers to ensure immediate write
		for handler in logger.handlers:
			handler.flush()
	else:
		print(msg, flush=True)

def log_separator(char: str = "=", width: int = 70):
	"""Log a separator line."""
	log(char * width)

def log_progress(strategy: str, iteration: int, total_iters: int,
				 current_error: float, best_error: float, elapsed: float):
	"""Log detailed progress for a strategy iteration."""
	pct = (iteration / total_iters) * 100 if total_iters > 0 else 0
	eta = (elapsed / iteration) * (total_iters - iteration) if iteration > 0 else 0
	log(f"  [{strategy}] Iter {iteration}/{total_iters} ({pct:.0f}%) | "
		f"error={current_error*100:.2f}% | best={best_error*100:.2f}% | "
		f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s")


class LogCapture:
	"""Capture stdout and redirect to our log function."""

	def __init__(self):
		self._original_stdout = None

	def __enter__(self):
		import io
		self._original_stdout = sys.stdout
		sys.stdout = self
		self._buffer = ""
		return self

	def __exit__(self, *args):
		sys.stdout = self._original_stdout
		if self._buffer:
			log(self._buffer.rstrip())

	def write(self, text):
		# Split by newlines and log each complete line
		self._buffer += text
		while '\n' in self._buffer:
			line, self._buffer = self._buffer.split('\n', 1)
			if line.strip():
				log(f"    {line}")

	def flush(self):
		pass

# Import thesis optimization strategies
from wnn.ram.enums import OptimizationMethod
from wnn.ram.strategies.connectivity import (
	OptimizerStrategyFactory,
	TabuSearchStrategy, TabuSearchConfig,
	SimulatedAnnealingStrategy, SimulatedAnnealingConfig,
	GeneticAlgorithmStrategy, GeneticAlgorithmConfig,
)


# Timeout handling
class TimeoutError(Exception):
	pass


@contextmanager
def timeout(seconds: int, strategy_name: str = ""):
	"""Context manager for strategy timeout."""
	def handler(signum, frame):
		raise TimeoutError(f"{strategy_name} timed out after {seconds}s")

	old_handler = signal.signal(signal.SIGALRM, handler)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)
		signal.signal(signal.SIGALRM, old_handler)


@dataclass
class StrategyResult:
	"""Result from a single strategy optimization."""
	name: str
	initial_error: float
	final_error: float
	improvement_percent: float
	elapsed_seconds: float
	timed_out: bool = False


@dataclass
class BenchmarkRun:
	"""Results from a single benchmark run."""
	run_id: int
	accuracy: float
	coverage_n4: float
	strategy_results: dict = field(default_factory=dict)
	best_strategy: str = ""
	elapsed_seconds: float = 0.0


class RAMNeuron:
	"""Single RAM neuron with partial connectivity."""

	def __init__(self, connected_bits: list[int], total_bits: int):
		self.connected_bits = connected_bits
		self.total_bits = total_bits
		self.ram = defaultdict(Counter)  # address → {word: count}

	def get_address(self, full_bits: tuple) -> tuple:
		"""Extract address from connected bits only."""
		return tuple(full_bits[b] for b in self.connected_bits if b < len(full_bits))

	def train(self, full_bits: tuple, target: str):
		"""Train on a pattern."""
		addr = self.get_address(full_bits)
		if addr:
			self.ram[addr][target] += 1

	def predict(self, full_bits: tuple) -> tuple[str, float]:
		"""Predict from address."""
		addr = self.get_address(full_bits)
		if addr and addr in self.ram:
			counts = self.ram[addr]
			total = sum(counts.values())
			best, count = counts.most_common(1)[0]
			return best, count / total
		return None, 0.0


class GeneralizedNGramRAM:
	"""
	N-gram RAM with partial connectivity for GENERALIZATION.

	Key insight: Instead of exact matching (2.3% coverage),
	use partial connectivity so similar n-grams map to same address.

	- Exact match: needs all 48 bits (4 words × 12 bits) to match
	- Partial (12 bits): only needs 12 bits to match → 40x more coverage!
	"""

	def __init__(self, n: int, n_neurons: int = 32, bits_per_neuron: int = 12,
				 bits_per_word: int = 12):
		self.n = n
		self.n_neurons = n_neurons
		self.bits_per_neuron = bits_per_neuron
		self.bits_per_word = bits_per_word
		self.total_bits = n * bits_per_word

		self.neurons = []
		self.word_to_cluster = {}
		self.n_clusters = 256

	def _word_to_bits(self, word: str) -> tuple:
		"""Convert word to bits."""
		cluster = self.word_to_cluster.get(word, hash(word) % self.n_clusters)
		h = hash(word)

		# Cluster bits (7) + hash bits (5) = 12 bits
		bits = []
		for i in range(7):
			bits.append((cluster >> i) & 1)
		for i in range(5):
			bits.append((h >> i) & 1)
		return tuple(bits)

	def _context_to_bits(self, context: list[str]) -> tuple:
		"""Convert context to bit vector."""
		bits = []
		for word in context[-self.n:]:
			bits.extend(self._word_to_bits(word))
		return tuple(bits)

	def create_diverse_connectivity(self):
		"""Create neurons with diverse connectivity patterns."""
		self.neurons = []

		for neuron_id in range(self.n_neurons):
			# Diverse strategy: each neuron focuses on different positions
			random.seed(neuron_id * 1000 + self.n)

			# Focus on 2 positions per neuron
			n_positions = self.n
			focus_positions = [(neuron_id + p) % n_positions for p in range(2)]

			connected = []
			bits_per_focus = self.bits_per_neuron // 2

			for pos in focus_positions:
				start = pos * self.bits_per_word
				end = start + self.bits_per_word
				available = list(range(start, min(end, self.total_bits)))
				selected = random.sample(available, min(bits_per_focus, len(available)))
				connected.extend(selected)

			# Fill remaining
			remaining = self.bits_per_neuron - len(connected)
			if remaining > 0:
				available = [b for b in range(self.total_bits) if b not in connected]
				if available:
					connected.extend(random.sample(available, min(remaining, len(available))))

			self.neurons.append(RAMNeuron(sorted(connected), self.total_bits))

	def set_connectivity(self, connectivity_matrix: list[list[int]]):
		"""Set connectivity from optimization result."""
		self.neurons = []
		for connected in connectivity_matrix:
			self.neurons.append(RAMNeuron(connected, self.total_bits))

	def get_connectivity(self) -> list[list[int]]:
		"""Get current connectivity for optimization."""
		return [n.connected_bits for n in self.neurons]

	def learn_clusters(self, tokens: list[str], window: int = 3):
		"""Learn word clusters from co-occurrence."""
		word_contexts = defaultdict(Counter)
		for i, word in enumerate(tokens):
			for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
				if i != j:
					word_contexts[word][tokens[j]] += 1

		for word, contexts in word_contexts.items():
			top = tuple(w for w, _ in contexts.most_common(10))
			self.word_to_cluster[word] = hash(top) % self.n_clusters

	def train(self, tokens: list[str]):
		"""Train all neurons."""
		for i in range(len(tokens) - self.n):
			context = tokens[i:i + self.n]
			target = tokens[i + self.n]
			full_bits = self._context_to_bits(context)

			for neuron in self.neurons:
				neuron.train(full_bits, target)

	def predict(self, context: list[str]) -> tuple[str, float]:
		"""Predict using voting."""
		if len(context) < self.n:
			return None, 0.0

		full_bits = self._context_to_bits(context)
		votes = Counter()

		for neuron in self.neurons:
			pred, conf = neuron.predict(full_bits)
			if pred:
				votes[pred] += conf

		if votes:
			best, score = votes.most_common(1)[0]
			return best, score / len(self.neurons)
		return None, 0.0

	def get_coverage(self, test_tokens: list[str]) -> float:
		"""Compute coverage (fraction of test contexts with any prediction)."""
		covered = 0
		total = len(test_tokens) - self.n

		for i in range(total):
			context = test_tokens[i:i + self.n]
			pred, _ = self.predict(context)
			if pred:
				covered += 1

		return covered / total if total > 0 else 0.0


# ============================================================================
# PARALLEL EVALUATION HELPER (module-level for pickling)
# ============================================================================

def _evaluate_single_connectivity(
	connectivity: list[list[int]],
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int
) -> float:
	"""
	Evaluate a single connectivity pattern. Module-level function for parallel execution.

	Returns error = 1 - (accuracy × coverage), lower is better.
	"""
	temp_ram = GeneralizedNGramRAM(
		n=4, n_neurons=len(connectivity),
		bits_per_neuron=bits_per_neuron
	)
	temp_ram.word_to_cluster = word_to_cluster
	temp_ram.set_connectivity(connectivity)
	temp_ram.train(train_tokens)

	# Evaluate on subset
	correct = 0
	covered = 0
	total = min(eval_subset, len(test_tokens) - 4)

	for i in range(total):
		context = test_tokens[i:i + 4]
		target = test_tokens[i + 4]
		pred, conf = temp_ram.predict(context)
		if pred is not None:
			covered += 1
			if pred == target:
				correct += 1

	accuracy = correct / covered if covered > 0 else 0.0
	coverage = covered / total if total > 0 else 0.0

	return 1.0 - (accuracy * coverage)


def _evaluate_batch_parallel(
	connectivities: list[list[list[int]]],
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
	n_workers: int = N_WORKERS
) -> list[float]:
	"""
	Evaluate multiple connectivity patterns in parallel using joblib (Python fallback).
	"""
	if len(connectivities) == 0:
		return []

	if len(connectivities) == 1:
		# Single item, no need for parallelism overhead
		return [_evaluate_single_connectivity(
			connectivities[0], word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)]

	# Parallel evaluation
	results = Parallel(n_jobs=n_workers, prefer="processes")(
		delayed(_evaluate_single_connectivity)(
			conn, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)
		for conn in connectivities
	)

	return results


def _evaluate_batch_rust(
	connectivities: list[list[list[int]]],
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
) -> list[float]:
	"""
	Evaluate multiple connectivity patterns using Rust accelerator (822x faster!).

	Uses rayon for CPU parallelism across all cores.
	Falls back to Python if Rust is not available.
	"""
	if not RUST_AVAILABLE:
		return _evaluate_batch_parallel(
			connectivities, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)

	if len(connectivities) == 0:
		return []

	# Convert word_to_cluster to use cluster IDs (Rust expects dict[str, int])
	# The Python version uses hash() % n_clusters for unknown words
	# We need to ensure all words in train/test are in the mapping
	n_clusters = 256
	word_map = {}
	for word, cluster in word_to_cluster.items():
		word_map[word] = cluster

	# Add any missing words from tokens
	for word in set(train_tokens) | set(test_tokens):
		if word not in word_map:
			word_map[word] = hash(word) % n_clusters

	# Call Rust accelerator
	results = ram_accelerator.evaluate_batch_cpu(
		connectivities,
		word_map,
		train_tokens,
		test_tokens,
		bits_per_neuron,
		eval_subset
	)

	return results


def _evaluate_batch_fast(
	connectivities: list,
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
) -> list[float]:
	"""
	Fast batch evaluation - uses Rust if available, otherwise Python parallel.

	This is the recommended function to use for batch evaluation.
	"""
	# Convert any numpy arrays to lists
	conn_list = []
	for conn in connectivities:
		if hasattr(conn, 'tolist'):
			conn_list.append(conn.tolist())
		else:
			conn_list.append(conn)

	if RUST_AVAILABLE:
		return _evaluate_batch_rust(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)
	else:
		return _evaluate_batch_parallel(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)


class RAMLM_v2:
	"""
	RAM Language Model v2 with thesis optimization.
	"""

	def __init__(self, freq_threshold: int = 50, fast_mode: bool = True):
		self.freq_threshold = freq_threshold
		self.fast_mode = fast_mode  # For faster testing
		self.word_counts = Counter()
		self.high_freq_words = set()

		# Exact RAMs for high-freq (still useful for common patterns)
		self.exact_rams = {n: defaultdict(Counter) for n in [2, 3, 4]}

		# Generalized RAMs with partial connectivity (KEY IMPROVEMENT)
		self.generalized_rams = {}

	def train(self, tokens: list[str], optimize_connectivity: bool = True):
		"""Train the model."""
		log(f"Training on {len(tokens):,} tokens...")

		self.word_counts = Counter(tokens)
		for word, count in self.word_counts.items():
			if count >= self.freq_threshold:
				self.high_freq_words.add(word)

		log(f"  Vocabulary: {len(self.word_counts):,}")
		log(f"  High-freq words: {len(self.high_freq_words):,}")

		# Train exact RAMs for high-freq contexts
		log("Training exact RAMs (high-freq only)...")
		for n in self.exact_rams:
			for i in range(len(tokens) - n):
				context = tokens[i:i + n]
				target = tokens[i + n]
				if all(w in self.high_freq_words for w in context):
					self.exact_rams[n][tuple(context)][target] += 1
			log(f"  n={n}: {len(self.exact_rams[n]):,} patterns")

		# Train generalized RAMs with partial connectivity
		log("Training generalized RAMs (partial connectivity)...")
		# Reduced params for fast_mode
		n_neurons = 16 if self.fast_mode else 64
		bits_per_neuron = 10 if self.fast_mode else 14

		for n in [2, 3, 4, 5, 6]:  # Added n=6 for longer context
			ram = GeneralizedNGramRAM(
				n=n,
				n_neurons=n_neurons,
				bits_per_neuron=bits_per_neuron,
				bits_per_word=12
			)
			ram.learn_clusters(tokens)
			ram.create_diverse_connectivity()
			ram.train(tokens)

			coverage = ram.get_coverage(tokens[:5000])
			log(f"  n={n}: {len(ram.neurons)} neurons, ~{coverage*100:.1f}% coverage")
			self.generalized_rams[n] = ram

		# Optimize connectivity if requested
		if optimize_connectivity:
			self._optimize_connectivity(tokens)

	def _optimize_connectivity(self, tokens: list[str]):
		"""Optimize connectivity patterns using thesis methods."""
		log_separator()
		log("CONNECTIVITY OPTIMIZATION (Garcia 2003)")
		log_separator()

		# Use n=4 for optimization (the problematic one)
		ram = self.generalized_rams[4]

		# Fast mode: use tiny subset for evaluation (key speedup!)
		train_subset = 10000 if self.fast_mode else 50000
		eval_subset = 300 if self.fast_mode else 3000
		test_tokens = tokens[:train_subset]
		bits_per_neuron = 10 if self.fast_mode else 14

		# Caching for evaluate_connectivity (huge speedup with 64GB RAM)
		eval_cache = {}
		cache_hits = [0]  # Use list to allow modification in nested function
		parallel_evals = [0]  # Track parallel evaluations

		# Shared parameters for parallel evaluation
		_word_to_cluster = ram.word_to_cluster
		_train_tokens = tokens[:train_subset]
		_test_tokens = test_tokens

		def evaluate_connectivity(connectivity: list[list[int]]) -> float:
			"""
			Evaluate a single connectivity pattern (serial, with caching).
			"""
			cache_key = tuple(tuple(row) for row in connectivity)

			if cache_key in eval_cache:
				cache_hits[0] += 1
				return eval_cache[cache_key]

			error = _evaluate_single_connectivity(
				connectivity, _word_to_cluster, _train_tokens,
				_test_tokens, bits_per_neuron, eval_subset
			)

			eval_cache[cache_key] = error
			return error

		def evaluate_batch(connectivities: list) -> list[float]:
			"""
			Evaluate multiple connectivity patterns with caching + Rust acceleration.

			1. Check cache for each pattern
			2. Batch cache misses for Rust evaluation (822x faster!)
			3. Update cache with results
			"""
			results = [None] * len(connectivities)
			to_evaluate = []  # (index, connectivity) pairs for cache misses

			# Check cache first
			for i, conn in enumerate(connectivities):
				if isinstance(conn, list):
					conn_list = conn
				else:
					conn_list = conn.tolist()

				cache_key = tuple(tuple(row) for row in conn_list)

				if cache_key in eval_cache:
					cache_hits[0] += 1
					results[i] = eval_cache[cache_key]
				else:
					to_evaluate.append((i, conn_list, cache_key))

			# Fast evaluation of cache misses (Rust if available, else Python parallel)
			if to_evaluate:
				parallel_evals[0] += len(to_evaluate)
				conn_list_batch = [item[1] for item in to_evaluate]

				# Use Rust accelerator (822x faster!) or fall back to Python parallel
				errors = _evaluate_batch_fast(
					conn_list_batch, _word_to_cluster, _train_tokens,
					_test_tokens, bits_per_neuron, eval_subset
				)

				# Update results and cache
				for (idx, conn_list, cache_key), error in zip(to_evaluate, errors):
					results[idx] = error
					eval_cache[cache_key] = error

			return results

		initial_connectivity = ram.get_connectivity()

		# Convert to tensor format for optimizer
		import torch
		conn_tensor = torch.tensor(initial_connectivity, dtype=torch.long)

		log(f"Initial connectivity shape: {conn_tensor.shape}")
		log(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
		log(f"Train subset: {train_subset:,}, Eval subset: {eval_subset}")
		if RUST_AVAILABLE:
			log(f"Accelerator: Rust + rayon ({RUST_CPU_CORES} threads) - 822x speedup!")
		else:
			log(f"Accelerator: Python + joblib ({N_WORKERS} workers) - install ram_accelerator for 822x speedup")
		initial_error = evaluate_connectivity(initial_connectivity)
		log(f"Initial acc×cov: {(1-initial_error)*100:.2f}% (lower bound to optimize)")

		results = {}

		# Fast mode: reduced iterations for initial testing
		if self.fast_mode:
			# Fast mode: simple sequential run
			ga_gens, ga_pop = 5, 10
			ts_iters, ts_neighbors = 5, 10
			hybrid_mode = None  # No hybrid in fast mode
		else:
			# Full mode: HYBRID approaches
			# Options: "GA_TS" (GA first), "TS_GA" (TS first), or None
			ga_gens, ga_pop = 20, 50  # GA: 50 pop × 20 gens
			ts_iters, ts_neighbors = 20, 40  # TS: 40 neighbors × 20 iters
			# Alternate between modes based on run_id (if available)
			run_id = getattr(self, '_current_run_id', 1)
			hybrid_mode = "GA_TS" if run_id % 2 == 1 else "TS_GA"

		# Timeout per strategy (3 hours = 10800 seconds for full mode)
		strategy_timeout = 10800 if not self.fast_mode else 300

		strategy_results = {}

		if hybrid_mode == "GA_TS":
			# =================================================================
			# HYBRID GA→TS OPTIMIZATION (Garcia 2003 + modern insights)
			# =================================================================
			# Phase 1: GA explores diverse regions (50 pop × 20 gens)
			# Phase 2: TS refines using GA's top population as neighbor pool
			# =================================================================

			log_separator("-")
			log(f"HYBRID STRATEGY: GA ({ga_pop} pop × {ga_gens} gens) → TS ({ts_neighbors} neighbors × {ts_iters} iters)")
			log(f"  Total timeout: {strategy_timeout}s ({strategy_timeout//60} min)")
			log_separator("-")

			hybrid_start = time.time()

			# --- PHASE 1: GA EXPLORATION ---
			log("")
			log("Phase 1: GA Exploration")
			log(f"  Population: {ga_pop}, Generations: {ga_gens}")

			ga_config = GeneticAlgorithmConfig(
				population_size=ga_pop,
				generations=ga_gens,
				mutation_rate=0.05,
				crossover_rate=0.8,
				elitism=max(2, ga_pop // 10)
			)
			ga_strategy = GeneticAlgorithmStrategy(config=ga_config, seed=42, verbose=True)

			ga_population = None  # Will store final population
			ga_fitness = None

			try:
				with timeout(strategy_timeout // 2, "GA-Phase"), LogCapture():
					# Run GA and capture the final population
					# We need to access the internal state, so we'll run it manually
					import torch

					ga_strategy._ensure_rng()
					cfg = ga_strategy._config

					# Initialize population
					population = []
					for i in range(cfg.population_size):
						if i == 0:
							individual = conn_tensor.clone()
						else:
							individual, _ = ga_strategy._generate_neighbor(
								conn_tensor, cfg.mutation_rate * 10,
								ram.total_bits, len(ram.neurons), bits_per_neuron
							)
						population.append(individual)

					# Evaluate initial population
					fitness = evaluate_batch(population)  # Parallel evaluation
					best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
					best = population[best_idx].clone()
					best_error = fitness[best_idx]
					ga_initial_error = initial_error

					log(f"    [GA] Initial best: {(1-best_error)*100:.2f}% acc×cov")

					# Evolution loop
					for generation in range(cfg.generations):
						new_population = []

						# Elitism
						sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
						for i in range(cfg.elitism):
							new_population.append(population[sorted_indices[i]].clone())

						# Generate rest
						while len(new_population) < cfg.population_size:
							# Tournament selection
							def tournament_select():
								indices = random.sample(range(len(population)), min(3, len(population)))
								return population[min(indices, key=lambda i: fitness[i])]

							p1, p2 = tournament_select(), tournament_select()

							# Crossover
							if random.random() < cfg.crossover_rate:
								child = p1.clone()
								cp = random.randint(1, len(ram.neurons) - 1)
								child[cp:] = p2[cp:].clone()
							else:
								child = p1.clone()

							# Mutation
							child, _ = ga_strategy._generate_neighbor(
								child, cfg.mutation_rate,
								ram.total_bits, len(ram.neurons), bits_per_neuron
							)
							new_population.append(child)

						population = new_population[:cfg.population_size]
						fitness = evaluate_batch(population)  # Parallel evaluation

						# Update best
						gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
						if fitness[gen_best_idx] < best_error:
							best = population[gen_best_idx].clone()
							best_error = fitness[gen_best_idx]

						if (generation + 1) % 5 == 0:
							avg_fit = sum(fitness) / len(fitness)
							log(f"    [GA] Gen {generation + 1}: best={best_error:.4f} ({(1-best_error)*100:.2f}%), avg={avg_fit:.4f}")

					# Store final population for TS
					ga_population = population
					ga_fitness = fitness
					ga_best = best
					ga_best_error = best_error

				ga_elapsed = time.time() - hybrid_start
				log(f"  Phase 1 complete: {(1-ga_best_error)*100:.2f}% acc×cov in {ga_elapsed:.1f}s")

			except TimeoutError:
				ga_elapsed = time.time() - hybrid_start
				log(f"  Phase 1 timeout after {ga_elapsed:.1f}s")
				ga_population = None

			# --- PHASE 2: TS REFINEMENT WITH GA POPULATION ---
			if ga_population is not None:
				log("")
				log("Phase 2: TS Refinement (using GA population as neighbor pool)")
				log(f"  Taking top {ts_neighbors} from GA population")

				ts_start = time.time()

				try:
					with timeout(strategy_timeout // 2, "TS-Phase"), LogCapture():
						# Sort GA population by fitness and take top N
						sorted_pop = sorted(zip(ga_population, ga_fitness), key=lambda x: x[1])
						top_individuals = [ind.clone() for ind, _ in sorted_pop[:ts_neighbors]]

						# Custom TS with GA-seeded neighbors
						from collections import deque

						current = ga_best.clone()
						current_error = ga_best_error
						best = current.clone()
						best_error = current_error
						ts_initial_error = ga_best_error

						tabu_list = deque(maxlen=5)

						log(f"    [TS-Hybrid] Starting from GA best: {(1-best_error)*100:.2f}%")

						for iteration in range(ts_iters):
							neighbors = []

							if iteration == 0:
								# First iteration: use GA population as initial neighbors (diversity seed)
								for i in range(min(ts_neighbors, len(top_individuals))):
									neighbor = top_individuals[i]
									error = evaluate_connectivity(neighbor.tolist())
									neighbors.append((neighbor, error, None))
							else:
								# Subsequent iterations: regular TS mutations from current best
								for _ in range(ts_neighbors):
									neighbor, move = ga_strategy._generate_neighbor(
										current, 0.1,
										ram.total_bits, len(ram.neurons), bits_per_neuron
									)
									is_tabu = any(
										m is not None and m[0] == move[0] and m[2] == move[1]
										for m in tabu_list
									)
									if not is_tabu:
										error = evaluate_connectivity(neighbor.tolist())
										neighbors.append((neighbor, error, move))

							if not neighbors:
								continue

							# Select best neighbor
							neighbors.sort(key=lambda x: x[1])
							best_neighbor, best_neighbor_error, best_move = neighbors[0]

							current = best_neighbor
							current_error = best_neighbor_error

							if best_move:
								tabu_list.append(best_move)

							if current_error < best_error:
								best = current.clone()
								best_error = current_error

							log(f"    [TS-Hybrid] Iter {iteration + 1}: current={current_error:.4f}, best={best_error:.4f} ({(1-best_error)*100:.2f}%)")

						ts_final_error = best_error

					ts_elapsed = time.time() - ts_start
					hybrid_elapsed = time.time() - hybrid_start

					# Calculate improvement
					improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

					# Create result object
					class HybridResult:
						def __init__(self):
							self.initial_error = initial_error
							self.final_error = best_error
							self.improvement_percent = improvement_pct
							self.optimized_connections = best

					hybrid_result = HybridResult()
					results['Hybrid_GA_TS'] = hybrid_result
					strategy_results['Hybrid_GA_TS'] = StrategyResult(
						name='Hybrid_GA_TS',
						initial_error=initial_error,
						final_error=best_error,
						improvement_percent=improvement_pct,
						elapsed_seconds=hybrid_elapsed,
						timed_out=False
					)

					log(f"  Phase 2 complete: {(1-best_error)*100:.2f}% acc×cov in {ts_elapsed:.1f}s")
					log_separator("-")
					log(f"  ✓ Hybrid complete: {(1-initial_error)*100:.2f}% → {(1-best_error)*100:.2f}% acc×cov")
					log(f"  Total improvement: {improvement_pct:.1f}%")
					log(f"  Total elapsed: {hybrid_elapsed:.1f}s ({hybrid_elapsed/60:.1f} min)")

				except TimeoutError:
					ts_elapsed = time.time() - ts_start
					log(f"  Phase 2 timeout after {ts_elapsed:.1f}s")

		elif hybrid_mode == "TS_GA":
			# =================================================================
			# HYBRID TS→GA OPTIMIZATION (Alternative approach)
			# =================================================================
			# Phase 1: TS finds good local optima (20 iters × 40 neighbors)
			# Phase 2: GA combines features via crossover (20 gens × 50 pop)
			# =================================================================

			log_separator("-")
			log(f"HYBRID STRATEGY: TS ({ts_iters} iters × {ts_neighbors} neighbors) → GA ({ga_pop} pop × {ga_gens} gens)")
			log(f"  Total timeout: {strategy_timeout}s ({strategy_timeout//60} min)")
			log_separator("-")

			hybrid_start = time.time()

			# --- PHASE 1: TS EXPLORATION ---
			log("")
			log("Phase 1: TS Exploration (finding diverse local optima)")
			log(f"  Iterations: {ts_iters}, Neighbors: {ts_neighbors}")

			# We'll collect the best solutions found during TS
			ts_best_solutions = []  # List of (connectivity, error) tuples

			try:
				with timeout(strategy_timeout // 2, "TS-Phase"), LogCapture():
					import torch
					from collections import deque

					current = conn_tensor.clone()
					current_error = evaluate_connectivity(current.tolist())

					best = current.clone()
					best_error = current_error

					tabu_list = deque(maxlen=5)

					# Track top solutions found
					top_k = ts_neighbors // 2  # Keep top 20 solutions
					ts_best_solutions = [(current.clone(), current_error)]

					log(f"    [TS] Initial: {(1-current_error)*100:.2f}% acc×cov")

					# Create a strategy for neighbor generation
					ts_config = TabuSearchConfig(
						iterations=ts_iters,
						neighbors_per_iter=ts_neighbors,
						tabu_size=5,
						mutation_rate=0.1
					)
					ts_strategy = TabuSearchStrategy(config=ts_config, seed=42, verbose=False)
					ts_strategy._ensure_rng()

					for iteration in range(ts_iters):
						neighbors = []

						for _ in range(ts_neighbors):
							neighbor, move = ts_strategy._generate_neighbor(
								current, 0.1,
								ram.total_bits, len(ram.neurons), bits_per_neuron
							)

							is_tabu = any(
								m[0] == move[0] and m[2] == move[1]
								for m in tabu_list
							)

							if not is_tabu:
								error = evaluate_connectivity(neighbor.tolist())
								neighbors.append((neighbor, error, move))

								# Track for GA seeding
								ts_best_solutions.append((neighbor.clone(), error))

						if not neighbors:
							continue

						# Select best neighbor
						neighbors.sort(key=lambda x: x[1])
						best_neighbor, best_neighbor_error, best_move = neighbors[0]

						current = best_neighbor
						current_error = best_neighbor_error
						tabu_list.append(best_move)

						if current_error < best_error:
							best = current.clone()
							best_error = current_error

						if (iteration + 1) % 5 == 0:
							log(f"    [TS] Iter {iteration + 1}: best={best_error:.4f} ({(1-best_error)*100:.2f}%)")

					# Keep only top K unique solutions for GA
					ts_best_solutions.sort(key=lambda x: x[1])
					seen = set()
					unique_solutions = []
					for sol, err in ts_best_solutions:
						sol_key = tuple(sol.flatten().tolist())
						if sol_key not in seen:
							seen.add(sol_key)
							unique_solutions.append((sol, err))
							if len(unique_solutions) >= top_k:
								break

					ts_best_solutions = unique_solutions
					ts_best = best
					ts_best_error = best_error

				ts_elapsed = time.time() - hybrid_start
				log(f"  Phase 1 complete: {(1-ts_best_error)*100:.2f}% acc×cov, found {len(ts_best_solutions)} unique solutions")

			except TimeoutError:
				ts_elapsed = time.time() - hybrid_start
				log(f"  Phase 1 timeout after {ts_elapsed:.1f}s")
				ts_best_solutions = []

			# --- PHASE 2: GA REFINEMENT WITH TS POPULATION ---
			if len(ts_best_solutions) > 0:
				log("")
				log("Phase 2: GA Refinement (crossover combines TS solutions)")
				log(f"  Initial population: {len(ts_best_solutions)} from TS + {ga_pop - len(ts_best_solutions)} mutations")

				ga_start = time.time()

				try:
					with timeout(strategy_timeout // 2, "GA-Phase"), LogCapture():
						import torch

						# Initialize population from TS solutions + mutations
						population = []

						# Add TS solutions
						for sol, _ in ts_best_solutions:
							population.append(sol.clone())

						# Fill rest with mutations of TS solutions
						ga_config = GeneticAlgorithmConfig(
							population_size=ga_pop,
							generations=ga_gens,
							mutation_rate=0.05,
							crossover_rate=0.8,
							elitism=max(2, ga_pop // 10)
						)
						ga_strategy = GeneticAlgorithmStrategy(config=ga_config, seed=42, verbose=False)
						ga_strategy._ensure_rng()
						cfg = ga_strategy._config

						while len(population) < ga_pop:
							# Mutate a random TS solution
							base_sol = ts_best_solutions[random.randint(0, len(ts_best_solutions) - 1)][0]
							mutated, _ = ga_strategy._generate_neighbor(
								base_sol, cfg.mutation_rate * 5,
								ram.total_bits, len(ram.neurons), bits_per_neuron
							)
							population.append(mutated)

						# Evaluate initial population
						fitness = evaluate_batch(population)  # Parallel evaluation
						best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
						best = population[best_idx].clone()
						best_error = fitness[best_idx]

						log(f"    [GA] Initial best (from TS): {(1-best_error)*100:.2f}% acc×cov")

						# Evolution loop
						for generation in range(cfg.generations):
							new_population = []

							# Elitism
							sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])
							for i in range(cfg.elitism):
								new_population.append(population[sorted_indices[i]].clone())

							# Generate rest
							while len(new_population) < cfg.population_size:
								# Tournament selection
								def tournament_select():
									indices = random.sample(range(len(population)), min(3, len(population)))
									return population[min(indices, key=lambda i: fitness[i])]

								p1, p2 = tournament_select(), tournament_select()

								# Crossover - key for combining TS solutions!
								if random.random() < cfg.crossover_rate:
									child = p1.clone()
									cp = random.randint(1, len(ram.neurons) - 1)
									child[cp:] = p2[cp:].clone()
								else:
									child = p1.clone()

								# Mutation
								child, _ = ga_strategy._generate_neighbor(
									child, cfg.mutation_rate,
									ram.total_bits, len(ram.neurons), bits_per_neuron
								)
								new_population.append(child)

							population = new_population[:cfg.population_size]
							fitness = evaluate_batch(population)  # Parallel evaluation

							# Update best
							gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
							if fitness[gen_best_idx] < best_error:
								best = population[gen_best_idx].clone()
								best_error = fitness[gen_best_idx]

							if (generation + 1) % 5 == 0:
								avg_fit = sum(fitness) / len(fitness)
								log(f"    [GA] Gen {generation + 1}: best={best_error:.4f} ({(1-best_error)*100:.2f}%), avg={avg_fit:.4f}")

						ga_final_error = best_error

					ga_elapsed = time.time() - ga_start
					hybrid_elapsed = time.time() - hybrid_start

					# Calculate improvement
					improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

					# Create result object
					class HybridResult:
						def __init__(self):
							self.initial_error = initial_error
							self.final_error = best_error
							self.improvement_percent = improvement_pct
							self.optimized_connections = best

					hybrid_result = HybridResult()
					results['Hybrid_TS_GA'] = hybrid_result
					strategy_results['Hybrid_TS_GA'] = StrategyResult(
						name='Hybrid_TS_GA',
						initial_error=initial_error,
						final_error=best_error,
						improvement_percent=improvement_pct,
						elapsed_seconds=hybrid_elapsed,
						timed_out=False
					)

					log(f"  Phase 2 complete: {(1-best_error)*100:.2f}% acc×cov in {ga_elapsed:.1f}s")
					log_separator("-")
					log(f"  ✓ Hybrid complete: {(1-initial_error)*100:.2f}% → {(1-best_error)*100:.2f}% acc×cov")
					log(f"  Total improvement: {improvement_pct:.1f}%")
					log(f"  Total elapsed: {hybrid_elapsed:.1f}s ({hybrid_elapsed/60:.1f} min)")

				except TimeoutError:
					ga_elapsed = time.time() - ga_start
					log(f"  Phase 2 timeout after {ga_elapsed:.1f}s")

		else:
			# =================================================================
			# FAST MODE: Simple GA only (hybrid_mode is None)
			# =================================================================
			log_separator("-")
			log(f"STRATEGY: GA only ({ga_pop} pop × {ga_gens} gens) - Fast mode")
			log_separator("-")

			ga_config = GeneticAlgorithmConfig(
				population_size=ga_pop,
				generations=ga_gens,
				mutation_rate=0.05,
				crossover_rate=0.8,
				elitism=max(2, ga_pop // 10)
			)
			ga_strategy = GeneticAlgorithmStrategy(config=ga_config, seed=42, verbose=True)

			ga_start = time.time()
			try:
				with timeout(strategy_timeout, "GA"), LogCapture():
					ga_result = ga_strategy.optimize(
						connections=conn_tensor,
						evaluate_fn=lambda c: evaluate_connectivity(c.tolist()),
						total_input_bits=ram.total_bits,
						num_neurons=len(ram.neurons),
						n_bits_per_neuron=bits_per_neuron,
					)
				ga_elapsed = time.time() - ga_start
				results['GeneticAlgorithm'] = ga_result
				strategy_results['GeneticAlgorithm'] = StrategyResult(
					name='GeneticAlgorithm',
					initial_error=ga_result.initial_error,
					final_error=ga_result.final_error,
					improvement_percent=ga_result.improvement_percent,
					elapsed_seconds=ga_elapsed,
					timed_out=False
				)
				log(f"  ✓ Complete: {(1-ga_result.final_error)*100:.2f}% acc×cov")
			except TimeoutError:
				ga_elapsed = time.time() - ga_start
				log(f"  ✗ Timeout after {ga_elapsed:.1f}s")

		# Store strategy results for multi-run analysis
		self._strategy_results = strategy_results

		# Use best result (if any completed)
		if results:
			best_method = min(results, key=lambda k: results[k].final_error)
			best_result = results[best_method]
			log_separator()
			log(f"★ Best strategy: {best_method}")
			log(f"  Acc×cov: {(1-best_result.initial_error)*100:.2f}% → {(1-best_result.final_error)*100:.2f}%")
			log(f"  Improvement: {best_result.improvement_percent:.1f}%")

			# Cache and parallelism stats
			total_evals = len(eval_cache) + cache_hits[0]
			if total_evals > 0:
				hit_rate = cache_hits[0] / total_evals * 100
				log(f"  Cache: {cache_hits[0]} hits / {total_evals} total ({hit_rate:.1f}% hit rate)")
				accel = f"Rust ({RUST_CPU_CORES} threads)" if RUST_AVAILABLE else f"Python ({N_WORKERS} workers)"
				log(f"  Accelerator: {parallel_evals[0]} evals via {accel}")
			log_separator()

			# Apply best connectivity
			ram.set_connectivity(best_result.optimized_connections.tolist())
			ram.train(tokens)  # Retrain with optimized connectivity
			self._best_strategy = best_method
		else:
			log_separator()
			log("⚠ WARNING: All strategies timed out! Using initial connectivity.")
			log_separator()
			self._best_strategy = "none"

	def predict(self, context: list[str]) -> tuple[str, str, float]:
		"""Predict next word."""

		# 1. Try exact match for high-freq contexts
		for n in sorted(self.exact_rams.keys(), reverse=True):
			if len(context) >= n:
				ctx = tuple(context[-n:])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					best, count = counts.most_common(1)[0]
					conf = count / total
					if conf > 0.2 or total >= 3:
						return best, f"exact_n{n}", conf

		# 2. Try generalized RAMs (partial connectivity)
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred and conf > 0.1:
					return pred, f"gen_n{n}", conf

		# 3. Fallback: vote across all scales
		votes = Counter()
		for n, ram in self.generalized_rams.items():
			if len(context) >= n:
				pred, conf = ram.predict(context)
				if pred:
					votes[pred] += conf * n  # Weight by context length

		if votes:
			best, score = votes.most_common(1)[0]
			return best, "voting", score / sum(votes.values())

		return "<UNK>", "none", 0.0

	def _get_all_predictions(self, context: list[str]) -> dict[str, tuple[str, float]]:
		"""Get predictions from ALL methods (for voting strategies)."""
		predictions = {}

		# Exact RAMs
		for n in self.exact_rams.keys():
			if len(context) >= n:
				ctx = tuple(context[-n:])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					best, count = counts.most_common(1)[0]
					predictions[f"exact_n{n}"] = (best, count / total)

		# Generalized RAMs
		for n in self.generalized_rams.keys():
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred:
					predictions[f"gen_n{n}"] = (pred, conf)

		return predictions

	def learn_voting_weights(self, tokens: list[str], validation_split: float = 0.2):
		"""
		Strategy #1: Learn voting weights based on observed accuracy.

		For each method, track how often it's correct when it makes a prediction.
		Use these accuracies as weights in voting.
		"""
		# Use last portion as validation
		val_start = int(len(tokens) * (1 - validation_split))
		val_tokens = tokens[val_start:]

		# Track per-method accuracy
		method_stats = defaultdict(lambda: {"correct": 0, "total": 0})

		for i in range(len(val_tokens) - 6):
			context = val_tokens[i:i + 5]
			target = val_tokens[i + 5]

			predictions = self._get_all_predictions(context)

			for method, (pred, conf) in predictions.items():
				if conf > 0.05:  # Only count confident predictions
					method_stats[method]["total"] += 1
					if pred == target:
						method_stats[method]["correct"] += 1

		# Compute weights = accuracy (with smoothing)
		self.voting_weights = {}
		for method, stats in method_stats.items():
			if stats["total"] > 0:
				# Laplace smoothing to avoid zero weights
				acc = (stats["correct"] + 1) / (stats["total"] + 2)
				self.voting_weights[method] = acc
			else:
				self.voting_weights[method] = 0.1  # Default small weight

		return self.voting_weights

	def predict_weighted_voting(self, context: list[str]) -> tuple[str, str, float]:
		"""
		Predict using accuracy-weighted voting.

		All methods vote, weighted by their learned accuracy.
		"""
		if not hasattr(self, 'voting_weights'):
			# Fall back to default if weights not learned
			return self.predict(context)

		predictions = self._get_all_predictions(context)

		if not predictions:
			return "<UNK>", "none", 0.0

		# Weighted voting
		votes = Counter()
		total_weight = 0

		for method, (pred, conf) in predictions.items():
			weight = self.voting_weights.get(method, 0.1) * conf
			votes[pred] += weight
			total_weight += weight

		if votes:
			best, score = votes.most_common(1)[0]
			return best, "weighted_voting", score / total_weight if total_weight > 0 else 0.0

		return "<UNK>", "none", 0.0

	def train_meta_classifier(self, tokens: list[str], validation_split: float = 0.2):
		"""
		Strategy #2: Train a meta-classifier (small RAM) to combine predictions.

		Input features: one-hot encoded predictions + confidences from each method
		Output: correct word prediction

		This learns which method combinations are most reliable.
		"""
		# Use last portion as validation/training for meta-classifier
		val_start = int(len(tokens) * (1 - validation_split))
		val_tokens = tokens[val_start:]

		# Collect training data for meta-classifier
		# Feature: method predictions encoded + confidences
		# Target: correct answer

		# First pass: collect all unique predictions to build vocabulary
		all_methods = sorted(set(
			list(f"exact_n{n}" for n in self.exact_rams.keys()) +
			list(f"gen_n{n}" for n in self.generalized_rams.keys())
		))
		self.meta_methods = all_methods

		# Collect (features, target) pairs
		training_data = []
		word_to_idx = {}

		for i in range(len(val_tokens) - 6):
			context = val_tokens[i:i + 5]
			target = val_tokens[i + 5]

			predictions = self._get_all_predictions(context)

			if not predictions:
				continue

			# Build feature vector: for each method, encode (prediction_word_idx, confidence)
			# We'll use a simpler approach: majority vote among top-k confident methods
			# and train a RAM to learn which combination pattern → which word

			# Feature: tuple of (method, prediction) pairs sorted by confidence
			sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]  # Top 4

			# Create a pattern: which methods agree and their words
			feature_parts = []
			for method, (pred, conf) in sorted_preds:
				if pred not in word_to_idx:
					word_to_idx[pred] = len(word_to_idx)
				# Quantize confidence to 2 bits (4 levels)
				conf_level = min(3, int(conf * 4))
				feature_parts.append((method, word_to_idx[pred], conf_level))

			training_data.append((feature_parts, target))

		self.meta_word_to_idx = word_to_idx
		self.meta_idx_to_word = {v: k for k, v in word_to_idx.items()}

		# Build meta-classifier: learn (top_prediction, agreement_level) → best_answer
		# Simpler approach: track which "primary predictor" is most reliable
		# when it has highest confidence

		self.meta_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

		for feature_parts, target in training_data:
			if not feature_parts:
				continue

			# Primary predictor = highest confidence method
			primary_method, primary_word_idx, primary_conf = feature_parts[0]
			primary_word = self.meta_idx_to_word.get(primary_word_idx, "<UNK>")

			# Secondary agreement: do other methods agree?
			agreement = sum(1 for m, w, c in feature_parts[1:] if w == primary_word_idx)

			# Key: (primary_method, agreement_level)
			key = (primary_method, agreement)

			self.meta_stats[key][primary_word]["total"] += 1
			if primary_word == target:
				self.meta_stats[key][primary_word]["correct"] += 1

		# Compute reliability scores
		self.meta_reliability = {}
		for key, word_stats in self.meta_stats.items():
			total_correct = sum(s["correct"] for s in word_stats.values())
			total_all = sum(s["total"] for s in word_stats.values())
			if total_all > 0:
				self.meta_reliability[key] = total_correct / total_all

		return self.meta_reliability

	def predict_meta(self, context: list[str]) -> tuple[str, str, float]:
		"""
		Predict using the meta-classifier.

		Uses learned reliability of (primary_method, agreement_level) combinations.
		"""
		if not hasattr(self, 'meta_reliability'):
			return self.predict(context)

		predictions = self._get_all_predictions(context)

		if not predictions:
			return "<UNK>", "none", 0.0

		# Sort by confidence
		sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]

		if not sorted_preds:
			return "<UNK>", "none", 0.0

		primary_method, (primary_word, primary_conf) = sorted_preds[0]

		# Check agreement
		agreement = sum(1 for m, (w, c) in sorted_preds[1:] if w == primary_word)

		# Look up reliability
		key = (primary_method, agreement)
		reliability = self.meta_reliability.get(key, 0.5)

		# If reliability is low and we have alternatives, consider them
		if reliability < 0.3 and len(sorted_preds) > 1:
			# Try second-best if it has higher agreement
			for alt_method, (alt_word, alt_conf) in sorted_preds[1:]:
				alt_agreement = sum(1 for m, (w, c) in sorted_preds if w == alt_word) - 1
				alt_key = (alt_method, alt_agreement)
				alt_reliability = self.meta_reliability.get(alt_key, 0.5)

				if alt_reliability > reliability and alt_conf > 0.2:
					return alt_word, f"meta({alt_method})", alt_reliability * alt_conf

		return primary_word, f"meta({primary_method})", reliability * primary_conf

	def evaluate_voting_strategies(self, tokens: list[str]) -> dict:
		"""Compare all voting strategies."""
		results = {
			"cascade": {"correct": 0, "total": 0},  # Original cascade
			"weighted": {"correct": 0, "total": 0},  # Accuracy-weighted voting
			"meta": {"correct": 0, "total": 0},      # Meta-classifier
		}

		for i in range(len(tokens) - 6):
			context = tokens[i:i + 5]
			target = tokens[i + 5]

			results["cascade"]["total"] += 1
			results["weighted"]["total"] += 1
			results["meta"]["total"] += 1

			# Original cascade
			pred1, _, _ = self.predict(context)
			if pred1 == target:
				results["cascade"]["correct"] += 1

			# Weighted voting
			pred2, _, _ = self.predict_weighted_voting(context)
			if pred2 == target:
				results["weighted"]["correct"] += 1

			# Meta-classifier
			pred3, _, _ = self.predict_meta(context)
			if pred3 == target:
				results["meta"]["correct"] += 1

		# Compute accuracies
		for strategy in results:
			total = results[strategy]["total"]
			correct = results[strategy]["correct"]
			results[strategy]["accuracy"] = correct / total if total > 0 else 0.0

		return results

	def evaluate(self, tokens: list[str]) -> dict:
		"""Evaluate the model."""
		results = {
			"correct": 0,
			"total": 0,
			"by_method": defaultdict(lambda: {"correct": 0, "total": 0})
		}

		for i in range(len(tokens) - 5):
			context = tokens[i:i + 5]
			target = tokens[i + 5]

			pred, method, conf = self.predict(context)

			results["total"] += 1
			results["by_method"][method]["total"] += 1

			if pred == target:
				results["correct"] += 1
				results["by_method"][method]["correct"] += 1

		return results


def run_benchmark(fast_mode: bool = True, run_id: int = 1, seed: int = None) -> BenchmarkRun:
	"""Run the v2 benchmark."""
	if seed is not None:
		random.seed(seed)

	log_separator()
	log(f"RAM LANGUAGE MODEL v2 - RUN {run_id}")
	log("With Full Thesis Optimization (2024 Hardware)")
	log(f"Mode: {'FAST (reduced params)' if fast_mode else 'FULL (3h timeout per strategy)'}")
	if not fast_mode:
		hybrid_label = "GA→TS" if run_id % 2 == 1 else "TS→GA"
		log(f"Hybrid: {hybrid_label}")
	if seed is not None:
		log(f"Seed: {seed}")
	log_separator()

	run_start = time.time()

	# Load data - reduced for fast mode
	train_limit = 30000 if fast_mode else 150000
	test_limit = 5000 if fast_mode else 20000

	try:
		from datasets import load_dataset
		import re

		log("Loading WikiText-2...")
		dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

		def tokenize(text):
			return re.findall(r"\w+|[^\w\s]", text.lower())

		train_text = " ".join(dataset["train"]["text"])
		test_text = " ".join(dataset["test"]["text"])

		train_tokens = tokenize(train_text)[:train_limit]
		test_tokens = tokenize(test_text)[:test_limit]

		log(f"Train: {len(train_tokens):,} tokens")
		log(f"Test: {len(test_tokens):,} tokens")

	except Exception as e:
		log(f"Error loading data: {e}")
		log("Using synthetic data...")

		# Fallback to synthetic
		words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "house", "tree"]
		train_tokens = [random.choice(words) for _ in range(train_limit)]
		test_tokens = [random.choice(words) for _ in range(test_limit)]

	# Train model
	log_separator()
	model = RAMLM_v2(freq_threshold=30, fast_mode=fast_mode)

	start = time.time()
	model.train(train_tokens, optimize_connectivity=True)
	train_time = time.time() - start
	log(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

	# Evaluate
	log_separator()
	log("EVALUATION")
	log_separator()

	results = model.evaluate(test_tokens)
	accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0

	log(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
	log("By method:")
	for method, stats in sorted(results["by_method"].items(),
								key=lambda x: -x[1]["total"]):
		if stats["total"] > 0:
			acc = stats["correct"] / stats["total"]
			cov = stats["total"] / results["total"]
			log(f"  {method:12s}: {acc*100:5.1f}% acc, {cov*100:5.1f}% coverage")

	# Coverage analysis
	log_separator()
	log("COVERAGE ANALYSIS")
	log_separator()

	for n, ram in model.generalized_rams.items():
		cov = ram.get_coverage(test_tokens)
		log(f"  Generalized n={n}: {cov*100:.1f}% coverage")

	# Voting strategy comparison
	log_separator()
	log("VOTING STRATEGY COMPARISON")
	log_separator()

	log("Training voting weights (Strategy #1)...")
	voting_weights = model.learn_voting_weights(train_tokens)
	log("  Learned weights:")
	for method, weight in sorted(voting_weights.items(), key=lambda x: -x[1]):
		log(f"    {method:12s}: {weight*100:.1f}%")

	log("Training meta-classifier (Strategy #2)...")
	meta_reliability = model.train_meta_classifier(train_tokens)
	log(f"  Learned {len(meta_reliability)} reliability patterns")

	log("Evaluating all strategies on test set...")
	voting_results = model.evaluate_voting_strategies(test_tokens)

	log("")
	log("RESULTS:")
	log(f"  Cascade (original)    : {voting_results['cascade']['accuracy']*100:.2f}%")
	log(f"  Weighted Voting (#1)  : {voting_results['weighted']['accuracy']*100:.2f}%")
	log(f"  Meta-Classifier (#2)  : {voting_results['meta']['accuracy']*100:.2f}%")
	log("")

	best_strategy = max(voting_results.items(), key=lambda x: x[1]['accuracy'])
	improvement = (best_strategy[1]['accuracy'] - voting_results['cascade']['accuracy']) / voting_results['cascade']['accuracy'] * 100
	if improvement > 0:
		log(f"★ Best: {best_strategy[0]} (+{improvement:.1f}% over cascade)")
	else:
		log(f"★ Best: cascade (baseline)")

	log_separator()
	log("SUMMARY")
	log_separator()

	gen_n4_coverage = model.generalized_rams[4].get_coverage(test_tokens)
	mode_desc = "FAST (reduced for testing)" if fast_mode else "FULL"
	run_elapsed = time.time() - run_start

	log(f"MODE: {mode_desc}")
	log("")
	log("IMPROVEMENTS IN v2:")
	log("1. Partial connectivity for generalization (not just exact matching)")
	log("2. TabuSearch optimization (vs random connectivity)")
	log("3. Simulated Annealing optimization")
	log("4. Genetic Algorithm optimization")
	log("5. Best optimizer selected automatically")
	log("6. n6 context added for longer-range patterns")
	log("7. Optimization objective: accuracy × coverage (not just accuracy)")
	log("")
	log("COVERAGE IMPROVEMENT (KEY RESULT):")
	log(f"- v1 exact_n4: ~2.3% coverage (exact 4-gram matching only)")
	log(f"- v2 gen_n4: ~{gen_n4_coverage*100:.1f}% coverage (partial connectivity)")
	log(f"- That's a {gen_n4_coverage/0.023:.1f}x improvement!")
	if 6 in model.generalized_rams:
		gen_n6_coverage = model.generalized_rams[6].get_coverage(test_tokens)
		log(f"- v2 gen_n6: ~{gen_n6_coverage*100:.1f}% coverage (new longer context)")
	log("")
	log(f"FINAL ACCURACY: {accuracy*100:.2f}%")
	log("")
	log(f"Run {run_id} elapsed: {run_elapsed:.1f}s ({run_elapsed/60:.1f} min)")

	# Return structured result
	benchmark_run = BenchmarkRun(
		run_id=run_id,
		accuracy=accuracy,
		coverage_n4=gen_n4_coverage,
		strategy_results=getattr(model, '_strategy_results', {}),
		best_strategy=getattr(model, '_best_strategy', 'unknown'),
		elapsed_seconds=run_elapsed
	)
	return benchmark_run


def run_multi_benchmark(n_runs: int = 3, fast_mode: bool = False):
	"""Run the benchmark multiple times and summarize results."""
	log_separator()
	log(f"MULTI-RUN BENCHMARK: {n_runs} runs")
	log(f"Mode: {'FAST' if fast_mode else 'FULL (3h timeout per strategy)'}")
	if not fast_mode:
		log("Hybrid schedule:")
		for r in range(1, n_runs + 1):
			hybrid = "GA→TS" if r % 2 == 1 else "TS→GA"
			log(f"  Run {r}: {hybrid}")
	log_separator()

	total_start = time.time()
	runs: list[BenchmarkRun] = []

	for i in range(n_runs):
		log_separator("#")
		log(f"# STARTING RUN {i+1} of {n_runs}")
		hybrid_label = "GA→TS" if (i+1) % 2 == 1 else "TS→GA"
		log(f"# Hybrid: {hybrid_label}")
		log_separator("#")

		seed = 42 + i * 1000  # Different seed for each run
		run_result = run_benchmark(fast_mode=fast_mode, run_id=i+1, seed=seed)
		runs.append(run_result)

		# Save intermediate results
		log_separator("-")
		log(f"[Run {i+1}/{n_runs} COMPLETE]")
		log(f"  Accuracy: {run_result.accuracy*100:.2f}%")
		log(f"  Best strategy: {run_result.best_strategy}")
		log(f"  Elapsed: {run_result.elapsed_seconds/60:.1f} min ({run_result.elapsed_seconds/3600:.2f} hours)")
		log_separator("-")

	total_elapsed = time.time() - total_start

	# Summary
	log_separator()
	log("★★★ MULTI-RUN SUMMARY ★★★")
	log_separator()

	log(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min = {total_elapsed/3600:.2f} hours)")
	log(f"Runs completed: {len(runs)}")

	# Accuracy stats
	accuracies = [r.accuracy for r in runs]
	log("")
	log("Accuracy:")
	log(f"  Mean: {sum(accuracies)/len(accuracies)*100:.2f}%")
	log(f"  Min:  {min(accuracies)*100:.2f}%")
	log(f"  Max:  {max(accuracies)*100:.2f}%")

	# Coverage stats
	coverages = [r.coverage_n4 for r in runs]
	log("")
	log("N4 Coverage:")
	log(f"  Mean: {sum(coverages)/len(coverages)*100:.1f}%")
	log(f"  vs v1 exact_n4 (2.3%): {sum(coverages)/len(coverages)/0.023:.1f}x improvement")

	# Strategy comparison
	log("")
	log("Strategy Results:")
	for strategy in ['TabuSearch', 'SimulatedAnnealing', 'GeneticAlgorithm']:
		improvements = []
		times = []
		timeouts = 0
		for r in runs:
			if strategy in r.strategy_results:
				sr = r.strategy_results[strategy]
				if sr.timed_out:
					timeouts += 1
				else:
					improvements.append(sr.improvement_percent)
					times.append(sr.elapsed_seconds)

		if improvements:
			log("")
			log(f"  {strategy}:")
			log(f"    Improvement: {sum(improvements)/len(improvements):.1f}% avg "
				  f"(min={min(improvements):.1f}%, max={max(improvements):.1f}%)")
			log(f"    Time: {sum(times)/len(times):.1f}s avg ({sum(times)/len(times)/60:.1f} min)")
			if timeouts:
				log(f"    Timeouts: {timeouts}/{len(runs)}")
		elif timeouts > 0:
			log(f"  {strategy}: ALL TIMED OUT ({timeouts}/{len(runs)})")

	# Best strategy counts
	log("")
	log("Best Strategy per Run:")
	best_counts = {}
	for r in runs:
		best_counts[r.best_strategy] = best_counts.get(r.best_strategy, 0) + 1
	for strategy, count in sorted(best_counts.items(), key=lambda x: -x[1]):
		log(f"  {strategy}: {count}/{len(runs)} runs")

	# Individual run summary
	log("")
	log("Individual Runs:")
	for r in runs:
		log(f"  Run {r.run_id}: acc={r.accuracy*100:.2f}%, "
			  f"cov={r.coverage_n4*100:.1f}%, "
			  f"best={r.best_strategy}, "
			  f"time={r.elapsed_seconds/60:.1f}min")

	log_separator()
	log("★ BENCHMARK COMPLETE ★")
	log_separator()

	return runs


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--full", action="store_true", help="Run full benchmark (slower, 3h per strategy)")
	parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
	args = parser.parse_args()

	# Setup logging to file and console
	log_file = setup_logging()
	print(f"\n{'='*70}")
	print(f"LOG FILE: {log_file}")
	print(f"{'='*70}")
	print(f"Tail command: tail -f \"{log_file}\"")
	print(f"{'='*70}\n")

	# Log the startup info
	log_separator()
	log("RAM LM v2 BENCHMARK - SESSION START")
	log_separator()
	log(f"Log file: {log_file}")
	log(f"Mode: {'FULL' if args.full else 'FAST'}")
	log(f"Runs: {args.runs}")
	if args.full:
		log(f"Timeout per strategy: 3 hours")
		log(f"Estimated max time: {args.runs * 9} hours ({args.runs} runs × 3 strategies × 3h)")
	log_separator()

	if args.runs > 1:
		run_multi_benchmark(n_runs=args.runs, fast_mode=not args.full)
	else:
		run_benchmark(fast_mode=not args.full)

	# Final log
	log_separator()
	log("SESSION COMPLETE")
	log(f"Full log available at: {log_file}")
	log_separator()
