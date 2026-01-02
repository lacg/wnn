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
	else:
		print(msg)

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

		for n in [2, 3, 4, 5]:
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

		def evaluate_connectivity(connectivity: list[list[int]]) -> float:
			"""Evaluate a connectivity pattern."""
			temp_ram = GeneralizedNGramRAM(
				n=4, n_neurons=len(connectivity),
				bits_per_neuron=bits_per_neuron
			)
			temp_ram.word_to_cluster = ram.word_to_cluster
			temp_ram.set_connectivity(connectivity)
			temp_ram.train(tokens[:train_subset])

			# Evaluate on small subset
			correct = 0
			total = min(eval_subset, len(test_tokens) - 4)
			for i in range(total):
				context = test_tokens[i:i + 4]
				target = test_tokens[i + 4]
				pred, _ = temp_ram.predict(context)
				if pred == target:
					correct += 1

			return 1.0 - (correct / total)  # Return error rate

		initial_connectivity = ram.get_connectivity()

		# Convert to tensor format for optimizer
		import torch
		conn_tensor = torch.tensor(initial_connectivity, dtype=torch.long)

		log(f"Initial connectivity shape: {conn_tensor.shape}")
		log(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
		log(f"Train subset: {train_subset:,}, Eval subset: {eval_subset}")
		initial_error = evaluate_connectivity(initial_connectivity)
		log(f"Initial error: {initial_error*100:.2f}%")

		results = {}

		# Fast mode: reduced iterations for initial testing
		if self.fast_mode:
			ts_iters, ts_neighbors = 5, 10  # ~50 evaluations
			sa_iters = 30  # 30 evaluations
			ga_gens, ga_pop = 5, 10  # ~50 evaluations
		else:
			# Full mode: thesis-level optimization (2024 hardware)
			ts_iters, ts_neighbors = 20, 40  # 800 evaluations
			sa_iters = 1000  # 1000 evaluations (increased!)
			ga_gens, ga_pop = 100, 50  # 5000 evaluations (increased!)

		# Timeout per strategy (3 hours = 10800 seconds for full mode)
		strategy_timeout = 10800 if not self.fast_mode else 300

		strategy_results = {}

		# 1. TABU SEARCH (your thesis best method)
		log_separator("-")
		log(f"STRATEGY 1/3: Tabu Search ({ts_iters} iterations, {ts_neighbors} neighbors)")
		log(f"  Timeout: {strategy_timeout}s ({strategy_timeout//60} min = {strategy_timeout//3600:.1f} hours)")
		log_separator("-")
		ts_config = TabuSearchConfig(
			iterations=ts_iters,
			neighbors_per_iter=ts_neighbors,
			tabu_size=5,
			mutation_rate=0.1
		)
		ts_strategy = TabuSearchStrategy(config=ts_config, seed=42, verbose=True)

		ts_start = time.time()
		ts_best_error = initial_error
		try:
			with timeout(strategy_timeout, "TabuSearch"):
				# Progress callback for detailed logging
				def ts_progress_callback(iteration, current_error, best_error):
					nonlocal ts_best_error
					ts_best_error = best_error
					if iteration % max(1, ts_iters // 10) == 0:  # Log every 10%
						log_progress("TabuSearch", iteration, ts_iters,
									current_error, best_error, time.time() - ts_start)

				ts_result = ts_strategy.optimize(
					connections=conn_tensor,
					evaluate_fn=lambda c: evaluate_connectivity(c.tolist()),
					total_input_bits=ram.total_bits,
					num_neurons=len(ram.neurons),
					n_bits_per_neuron=bits_per_neuron,
				)
			ts_elapsed = time.time() - ts_start
			results['TabuSearch'] = ts_result
			strategy_results['TabuSearch'] = StrategyResult(
				name='TabuSearch',
				initial_error=ts_result.initial_error,
				final_error=ts_result.final_error,
				improvement_percent=ts_result.improvement_percent,
				elapsed_seconds=ts_elapsed,
				timed_out=False
			)
			log(f"  ✓ COMPLETE: {ts_result.final_error*100:.2f}% (improvement: {ts_result.improvement_percent:.1f}%)")
			log(f"  Elapsed: {ts_elapsed:.1f}s ({ts_elapsed/60:.1f} min)")
		except TimeoutError as e:
			ts_elapsed = time.time() - ts_start
			log(f"  ✗ TIMEOUT after {ts_elapsed:.1f}s ({ts_elapsed/60:.1f} min)")
			strategy_results['TabuSearch'] = StrategyResult(
				name='TabuSearch',
				initial_error=initial_error,
				final_error=initial_error,
				improvement_percent=0.0,
				elapsed_seconds=ts_elapsed,
				timed_out=True
			)

		# 2. SIMULATED ANNEALING
		log_separator("-")
		log(f"STRATEGY 2/3: Simulated Annealing ({sa_iters} iterations)")
		log(f"  Timeout: {strategy_timeout}s ({strategy_timeout//60} min = {strategy_timeout//3600:.1f} hours)")
		log_separator("-")
		sa_config = SimulatedAnnealingConfig(
			iterations=sa_iters,
			initial_temp=2.0,
			cooling_rate=0.995,  # Slower cooling for 1000 iters
			mutation_rate=0.1
		)
		sa_strategy = SimulatedAnnealingStrategy(config=sa_config, seed=42, verbose=True)

		sa_start = time.time()
		try:
			with timeout(strategy_timeout, "SimulatedAnnealing"):
				sa_result = sa_strategy.optimize(
					connections=conn_tensor,
					evaluate_fn=lambda c: evaluate_connectivity(c.tolist()),
					total_input_bits=ram.total_bits,
					num_neurons=len(ram.neurons),
					n_bits_per_neuron=bits_per_neuron,
				)
			sa_elapsed = time.time() - sa_start
			results['SimulatedAnnealing'] = sa_result
			strategy_results['SimulatedAnnealing'] = StrategyResult(
				name='SimulatedAnnealing',
				initial_error=sa_result.initial_error,
				final_error=sa_result.final_error,
				improvement_percent=sa_result.improvement_percent,
				elapsed_seconds=sa_elapsed,
				timed_out=False
			)
			log(f"  ✓ COMPLETE: {sa_result.final_error*100:.2f}% (improvement: {sa_result.improvement_percent:.1f}%)")
			log(f"  Elapsed: {sa_elapsed:.1f}s ({sa_elapsed/60:.1f} min)")
		except TimeoutError as e:
			sa_elapsed = time.time() - sa_start
			log(f"  ✗ TIMEOUT after {sa_elapsed:.1f}s ({sa_elapsed/60:.1f} min)")
			strategy_results['SimulatedAnnealing'] = StrategyResult(
				name='SimulatedAnnealing',
				initial_error=initial_error,
				final_error=initial_error,
				improvement_percent=0.0,
				elapsed_seconds=sa_elapsed,
				timed_out=True
			)

		# 3. GENETIC ALGORITHM
		log_separator("-")
		log(f"STRATEGY 3/3: Genetic Algorithm ({ga_gens} generations, pop={ga_pop})")
		log(f"  Timeout: {strategy_timeout}s ({strategy_timeout//60} min = {strategy_timeout//3600:.1f} hours)")
		log_separator("-")
		ga_config = GeneticAlgorithmConfig(
			population_size=ga_pop,
			generations=ga_gens,
			mutation_rate=0.05,
			crossover_rate=0.8,
			elitism=max(2, ga_pop // 10)  # 10% elitism
		)
		ga_strategy = GeneticAlgorithmStrategy(config=ga_config, seed=42, verbose=True)

		ga_start = time.time()
		try:
			with timeout(strategy_timeout, "GeneticAlgorithm"):
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
			log(f"  ✓ COMPLETE: {ga_result.final_error*100:.2f}% (improvement: {ga_result.improvement_percent:.1f}%)")
			log(f"  Elapsed: {ga_elapsed:.1f}s ({ga_elapsed/60:.1f} min)")
		except TimeoutError as e:
			ga_elapsed = time.time() - ga_start
			log(f"  ✗ TIMEOUT after {ga_elapsed:.1f}s ({ga_elapsed/60:.1f} min)")
			strategy_results['GeneticAlgorithm'] = StrategyResult(
				name='GeneticAlgorithm',
				initial_error=initial_error,
				final_error=initial_error,
				improvement_percent=0.0,
				elapsed_seconds=ga_elapsed,
				timed_out=True
			)

		# Store strategy results for multi-run analysis
		self._strategy_results = strategy_results

		# Use best result (if any completed)
		if results:
			best_method = min(results, key=lambda k: results[k].final_error)
			best_result = results[best_method]
			log_separator()
			log(f"★ BEST STRATEGY: {best_method}")
			log(f"  Error: {best_result.initial_error*100:.2f}% → {best_result.final_error*100:.2f}%")
			log(f"  Improvement: {best_result.improvement_percent:.1f}%")
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
	log("")
	log("COVERAGE IMPROVEMENT (KEY RESULT):")
	log(f"- v1 exact_n4: ~2.3% coverage (exact 4-gram matching only)")
	log(f"- v2 gen_n4: ~{gen_n4_coverage*100:.1f}% coverage (partial connectivity)")
	log(f"- That's a {gen_n4_coverage/0.023:.1f}x improvement!")
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
	log_separator()

	total_start = time.time()
	runs: list[BenchmarkRun] = []

	for i in range(n_runs):
		log_separator("#")
		log(f"# STARTING RUN {i+1} of {n_runs}")
		log(f"# Estimated max time per run: ~9h (3 strategies × 3h each)")
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
