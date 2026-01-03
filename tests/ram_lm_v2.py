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

from wnn.ram.enums import BenchmarkMode, TokenizerType

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
# TOKENIZER FACTORY - For fair comparison to published benchmarks
# ============================================================================
def get_tokenizer(tokenizer_type: TokenizerType):
	"""
	Get tokenizer function based on type.

	Published WikiText-2 perplexity benchmarks:
	- Word-level (WIKITEXT_WORD): LSTM ~65-100, AWD-LSTM ~57
	- GPT-2 BPE (GPT2_BPE): GPT-2 Small ~29, GPT-2 Large ~22

	Note: Word-level and BPE perplexities are NOT directly comparable!
	"""
	import re

	if tokenizer_type == TokenizerType.SIMPLE:
		# Original simple tokenizer (not standard, just for testing)
		def tokenize(text):
			return re.findall(r"\w+|[^\w\s]", text.lower())
		return tokenize, None  # No special vocab

	elif tokenizer_type == TokenizerType.WIKITEXT_WORD:
		# Standard WikiText-2 word-level preprocessing
		# - Lowercase (optional, some benchmarks don't)
		# - Keep punctuation as tokens
		# - Replace rare words with <unk> (we skip this, just use all words)
		def tokenize(text):
			# WikiText uses space-separated tokens with special handling
			# Raw WikiText-2 has tokens separated by spaces
			tokens = text.split()
			# Filter empty and normalize
			tokens = [t for t in tokens if t.strip()]
			return tokens
		return tokenize, None

	elif tokenizer_type == TokenizerType.GPT2_BPE:
		# GPT-2 BPE tokenization - requires tiktoken
		try:
			import tiktoken
			enc = tiktoken.get_encoding("gpt2")

			def tokenize(text):
				# Returns token IDs, we convert to strings for compatibility
				token_ids = enc.encode(text)
				# Convert IDs to string representation for our word-based model
				return [str(tid) for tid in token_ids]

			return tokenize, enc
		except ImportError:
			print("WARNING: tiktoken not installed. Install with: pip install tiktoken")
			print("Falling back to WIKITEXT_WORD tokenizer.")
			return get_tokenizer(TokenizerType.WIKITEXT_WORD)

	else:
		raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

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

# Import thesis optimization strategies (now using AcceleratedOptimizer internally)
from wnn.ram.strategies.connectivity import OptimizerResult


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
	perplexity: float
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

	# Precompute full 12-bit encoding for each word (7 cluster + 5 hash bits)
	# This MUST match Python's _word_to_bits() exactly for Rust to produce same results
	n_clusters = 256
	word_to_bits = {}

	def compute_word_bits(word: str, cluster: int) -> int:
		"""Compute 12-bit encoding: 7 cluster bits + 5 hash bits"""
		h = hash(word)
		bits = 0
		# First 7 bits from cluster
		for i in range(7):
			if (cluster >> i) & 1:
				bits |= 1 << i
		# Next 5 bits from hash
		for i in range(5):
			if (h >> i) & 1:
				bits |= 1 << (7 + i)
		return bits

	# Encode all words from word_to_cluster
	for word, cluster in word_to_cluster.items():
		word_to_bits[word] = compute_word_bits(word, cluster)

	# Add any missing words from tokens (use hash-based cluster)
	for word in set(train_tokens) | set(test_tokens):
		if word not in word_to_bits:
			cluster = hash(word) % n_clusters
			word_to_bits[word] = compute_word_bits(word, cluster)

	# Call Rust accelerator with precomputed bits
	results = ram_accelerator.evaluate_batch_cpu(
		connectivities,
		word_to_bits,  # Now passes full 12-bit encoding, not just cluster
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

	def __init__(self, freq_threshold: int = 50, mode: BenchmarkMode = BenchmarkMode.FAST,
				 n_neurons: int = None, cascade_threshold: float = 0.1,
				 strategy_sequence: list = None):
		self.freq_threshold = freq_threshold
		self.mode = mode
		self.word_counts = Counter()
		self.high_freq_words = set()

		# Configurable parameters (with mode-based defaults)
		if n_neurons is None:
			self.n_neurons = 16 if mode == BenchmarkMode.FAST else 64
		else:
			self.n_neurons = n_neurons
		self.cascade_threshold = cascade_threshold
		self.strategy_sequence = strategy_sequence if strategy_sequence else ['GA', 'TS']

		# Exact RAMs for high-freq (still useful for common patterns)
		# Now includes n=5 and n=6 for better coverage
		self.exact_rams = {n: defaultdict(Counter) for n in [2, 3, 4, 5, 6]}

		# Generalized RAMs with partial connectivity (KEY IMPROVEMENT)
		self.generalized_rams = {}

		# Track prediction method usage
		self.prediction_stats = Counter()

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
		log(f"Training generalized RAMs ({self.n_neurons} neurons)...")
		bits_per_neuron = 10 if self.mode == BenchmarkMode.FAST else 14

		for n in [2, 3, 4, 5, 6]:  # Added n=6 for longer context
			ram = GeneralizedNGramRAM(
				n=n,
				n_neurons=self.n_neurons,
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
		"""
		Optimize connectivity patterns for ALL generalized RAMs using CASCADE ACCURACY.

		This method optimizes each RAM's connectivity while evaluating the full
		cascade prediction accuracy (not individual RAM accuracy), ensuring
		optimization aligns with the actual goal.
		"""
		from wnn.ram.strategies.connectivity import (
			OptimizationStrategy,
			RUST_AVAILABLE as ACCEL_RUST_AVAILABLE,
			RUST_CPU_CORES as ACCEL_RUST_CORES,
		)
		from wnn.ram.strategies.connectivity.genetic_algorithm import (
			GeneticAlgorithmStrategy, GeneticAlgorithmConfig
		)
		from wnn.ram.strategies.connectivity.tabu_search import (
			TabuSearchStrategy, TabuSearchConfig
		)
		import torch

		log_separator()
		log("CONNECTIVITY OPTIMIZATION (Garcia 2003)")
		log("Using PERPLEXITY as optimization goal (lower is better)")
		log("(exact RAMs → generalized cascade → voting fallback)")
		log_separator()

		# Evaluation parameters (scale with mode)
		train_subset = 10000 if self.mode == BenchmarkMode.FAST else 50000
		eval_subset = 300 if self.mode == BenchmarkMode.FAST else 3000
		train_tokens = tokens[:train_subset]
		# Use DIFFERENT tokens for evaluation (after training portion)
		test_tokens = tokens[train_subset:train_subset + eval_subset * 2]

		# Strategy parameters (scale with mode)
		# Population sizes are multiples of 16 (CPU cores) for optimal parallel batching
		if self.mode == BenchmarkMode.FAST:
			ga_pop, ga_gens = 16, 10
			ts_neighbors, ts_iters = 16, 10
			ga_elitism_pct = 0.25  # 25% elitism
		elif self.mode == BenchmarkMode.OVERNIGHT:
			# Extended overnight: max 1000 iterations with early stopping
			ga_pop, ga_gens = 48, 1000
			ts_neighbors, ts_iters = 48, 1000
			ga_elitism_pct = 0.25
		else:  # BenchmarkMode.FULL
			ga_pop, ga_gens = 32, 100
			ts_neighbors, ts_iters = 32, 50
			ga_elitism_pct = 0.25

		# Configurable strategy sequence (default: GA then TS)
		strategy_sequence = getattr(self, 'strategy_sequence', ['GA', 'TS'])
		strategy_name = "→".join(strategy_sequence)

		bits_per_neuron = 10 if self.mode == BenchmarkMode.FAST else 14
		n_values = sorted(self.generalized_rams.keys())

		log(f"Optimizing {len(n_values)} RAMs: n={n_values}")
		log(f"Mode: {self.mode.name}")
		log(f"Train subset: {train_subset:,}, Eval subset: {eval_subset}")
		log(f"Strategy sequence: {strategy_name}")
		ga_elitism = max(2, int(ga_pop * ga_elitism_pct))
		ga_eval_per_gen = ga_pop - ga_elitism
		log(f"GA: {ga_pop} pop × {ga_gens} max gens (elite={ga_elitism} ({ga_elitism_pct*100:.0f}%), eval={ga_eval_per_gen}/gen)")
		log(f"TS: {ts_neighbors} neighbors × {ts_iters} max iters")
		log(f"Early stopping: <0.02% improvement over 5 iterations")
		if ACCEL_RUST_AVAILABLE:
			log(f"Accelerator: Rust PERPLEXITY eval ({ACCEL_RUST_CORES} threads)")
		else:
			log(f"Accelerator: Python (install ram_accelerator for speedup)")
		log("")

		# Pre-calculate exact RAM probabilities (these don't change during optimization)
		# exact_probs[i] = P(target|context) if covered by exact RAM, None if not covered
		# Also track exact_results for accuracy reporting
		exact_probs = []
		exact_results = []  # Keep for backward compatibility
		for i in range(min(eval_subset, len(test_tokens) - 6)):
			target = test_tokens[i + 6]

			# Try exact RAMs in priority order (n=4, n=3, n=2) - same as predict()
			exact_prob = None
			exact_pred = None
			for n in sorted(self.exact_rams.keys(), reverse=True):
				ctx = tuple(test_tokens[i + 6 - n:i + 6])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					best, count = counts.most_common(1)[0]
					conf = count / total
					if conf > 0.2 or total >= 3:
						# Get probability for target (not just best)
						target_count = counts.get(target, 0)
						exact_prob = target_count / total if total > 0 else 0.0
						exact_pred = (best == target)
						break

			exact_probs.append(exact_prob)
			exact_results.append(exact_pred)

		exact_covered = sum(1 for x in exact_probs if x is not None)
		exact_correct = sum(1 for x in exact_results if x is True)
		avg_exact_prob = sum(p for p in exact_probs if p is not None) / exact_covered if exact_covered > 0 else 0.0
		log(f"Pre-calculated exact RAM predictions:")
		log(f"  Covered: {exact_covered}/{len(exact_probs)} ({exact_covered/len(exact_probs)*100:.1f}%)")
		log(f"  Correct: {exact_correct}/{exact_covered} ({exact_correct/exact_covered*100:.1f}% of covered)")
		log(f"  Avg P(target): {avg_exact_prob:.3f}")
		log("")

		# Get initial connectivities for all RAMs
		all_connectivities = []
		for n in n_values:
			ram = self.generalized_rams[n]
			all_connectivities.append(ram.get_connectivity())

		# Build word_to_bits mapping (same for all RAMs)
		ram = self.generalized_rams[n_values[0]]
		word_to_bits = {}
		for word, cluster in ram.word_to_cluster.items():
			word_hash = hash(word) & 0x1F  # 5 bits
			encoding = (cluster << 5) | word_hash
			word_to_bits[word] = encoding

		# Check if Rust evaluators are available
		try:
			import ram_accelerator
			has_perplexity = hasattr(ram_accelerator, 'evaluate_fullnetwork_perplexity_batch_cpu')
			has_fullnetwork = hasattr(ram_accelerator, 'evaluate_fullnetwork_batch_cpu')
			has_cascade = hasattr(ram_accelerator, 'evaluate_cascade_batch_cpu')
		except ImportError:
			has_perplexity = False
			has_fullnetwork = False
			has_cascade = False

		# Get vocabulary size for perplexity smoothing
		vocab_size = len(self.word_counts)

		self._strategy_results = {}
		total_start = time.time()

		# Optimize each RAM using PERPLEXITY as the goal (lower is better)
		# IMPORTANT: Optimize from highest n to lowest (n=6 → n=2)
		# Higher n RAMs have priority in cascade, so optimizing them first has more impact
		for ram_idx, n in reversed(list(enumerate(n_values))):
			ram = self.generalized_rams[n]
			log(f"--- Optimizing n={n} RAM ({len(ram.neurons)} neurons) ---")

			conn_tensor = torch.tensor(all_connectivities[ram_idx], dtype=torch.long)
			num_neurons = len(ram.neurons)
			total_bits = ram.total_bits

			# Create batch evaluation function using PERPLEXITY (lower is better)
			def create_perplexity_batch_fn(ram_idx, target_n, exact_probs_ref, vocab_size_ref, cascade_threshold_ref, pop_size):
				eval_count = [0]
				global_best = [float('inf')]  # Track best perplexity (lower is better)
				def batch_eval(candidates):
					if has_perplexity:
						import time as time_module
						start = time_module.time()
						current_all_conns = all_connectivities
						candidate_lists = [c.tolist() if hasattr(c, 'tolist') else c for c in candidates]
						perplexities = ram_accelerator.evaluate_fullnetwork_perplexity_batch_cpu(
							current_all_conns, candidate_lists, ram_idx,
							word_to_bits, train_tokens, test_tokens,
							exact_probs_ref, eval_subset, vocab_size_ref,
							cascade_threshold_ref
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_ppl = min(perplexities)
						if batch_best_ppl < global_best[0]:
							global_best[0] = batch_best_ppl
						# Log every 5th batch
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {pop_size} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch PPL: {batch_best_ppl:.1f}, global PPL: {global_best[0]:.1f}"
							log(msg)
						return perplexities
					elif has_fullnetwork:
						# Fallback to accuracy-based (convert to perplexity-like score)
						import time as time_module
						start = time_module.time()
						current_all_conns = all_connectivities
						candidate_lists = [c.tolist() if hasattr(c, 'tolist') else c for c in candidates]
						# Use exact_results (bool) for accuracy-based fallback
						exact_results_fallback = [x is True if x is not None else None for x in exact_probs_ref]
						errors = ram_accelerator.evaluate_fullnetwork_batch_cpu(
							current_all_conns, candidate_lists, ram_idx,
							word_to_bits, train_tokens, test_tokens,
							exact_results_fallback, eval_subset
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_err = min(errors)
						batch_best_acc = (1 - batch_best_err) * 100
						if batch_best_err < global_best[0]:
							global_best[0] = batch_best_err
						global_best_acc = (1 - global_best[0]) * 100
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {pop_size} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch: {batch_best_acc:.2f}%, global: {global_best_acc:.2f}%"
							log(msg)
						return errors
					elif has_cascade:
						# Fallback to cascade-only (old behavior)
						import time as time_module
						start = time_module.time()
						current_all_conns = all_connectivities
						candidate_lists = [c.tolist() if hasattr(c, 'tolist') else c for c in candidates]
						errors = ram_accelerator.evaluate_cascade_batch_cpu(
							current_all_conns, candidate_lists, ram_idx,
							word_to_bits, train_tokens, test_tokens, eval_subset
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_err = min(errors)
						batch_best_acc = (1 - batch_best_err) * 100
						if batch_best_err < global_best[0]:
							global_best[0] = batch_best_err
						global_best_acc = (1 - global_best[0]) * 100
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {pop_size} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch: {batch_best_acc:.2f}%, global: {global_best_acc:.2f}%"
							log(msg)
						return errors
					else:
						# Python fallback (slower but correct perplexity)
						import time as time_module
						start = time_module.time()
						perplexities = []
						ram = self.generalized_rams[target_n]
						old_conn = ram.get_connectivity()
						for cand in candidates:
							# Temporarily apply candidate connectivity
							cand_list = cand.tolist() if hasattr(cand, 'tolist') else cand
							ram.set_connectivity(cand_list)
							# Evaluate perplexity (matches Rust logic)
							ppl = self._evaluate_perplexity_python(
								test_tokens, exact_probs_ref, vocab_size_ref,
								cascade_threshold_ref, eval_subset
							)
							perplexities.append(ppl)
						# Restore original connectivity
						ram.set_connectivity(old_conn)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_ppl = min(perplexities)
						if batch_best_ppl < global_best[0]:
							global_best[0] = batch_best_ppl
						if eval_count[0] % 5 == 1:
							msg = f"[Python n={target_n}] Batch {eval_count[0]}: {pop_size} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch PPL: {batch_best_ppl:.1f}, global PPL: {global_best[0]:.1f}"
							log(msg)
						return perplexities
				return batch_eval

			batch_fn = create_perplexity_batch_fn(ram_idx, n, exact_probs, vocab_size, self.cascade_threshold, ga_pop)

			# Initial perplexity (using same code path as optimization)
			initial_ppls = batch_fn([conn_tensor])
			initial_ppl = initial_ppls[0]

			# Log initial perplexity only for the first RAM
			if ram_idx == 0:
				log(f"Initial PERPLEXITY: {initial_ppl:.1f}")
				log("")
			else:
				# VERIFICATION: This should match previous RAM's final perplexity!
				log(f"  Initial PERPLEXITY: {initial_ppl:.1f} (should match prev final)")

			start_time = time.time()
			# Track best across all phases (lower perplexity is better)
			best_ppl = initial_ppl
			best_connectivity = conn_tensor.clone()
			current_connectivity = conn_tensor.clone()

			# Run strategy sequence (e.g., ['GA', 'TS'] or ['TS', 'GA', 'GA'])
			for step_idx, strategy_type in enumerate(strategy_sequence):
				step_num = step_idx + 1
				if strategy_type.upper() == 'GA':
					log(f"  [{step_num}/{len(strategy_sequence)} GA] Starting...")
					ga_config = GeneticAlgorithmConfig(
						population_size=ga_pop, generations=ga_gens,
						mutation_rate=0.01, crossover_rate=0.7, elitism=ga_elitism,
						early_stop_patience=5, early_stop_threshold_pct=0.02
					)
					ga = GeneticAlgorithmStrategy(config=ga_config, seed=42+n+step_idx, verbose=True)
					result = ga.optimize(current_connectivity, lambda x: batch_fn([x])[0],
						total_bits, num_neurons, bits_per_neuron, batch_fn)
					log(f"  [{step_num}/{len(strategy_sequence)} GA] Done: PPL {result.final_error:.1f}")
				elif strategy_type.upper() == 'TS':
					log(f"  [{step_num}/{len(strategy_sequence)} TS] Starting...")
					ts_config = TabuSearchConfig(
						iterations=ts_iters, neighbors_per_iter=ts_neighbors,
						early_stop_patience=5, early_stop_threshold_pct=0.02
					)
					ts = TabuSearchStrategy(config=ts_config, seed=42+n+step_idx, verbose=True)
					result = ts.optimize(current_connectivity, lambda x: batch_fn([x])[0],
						total_bits, num_neurons, bits_per_neuron, batch_fn)
					log(f"  [{step_num}/{len(strategy_sequence)} TS] Done: PPL {result.final_error:.1f}")
				else:
					log(f"  [WARNING] Unknown strategy: {strategy_type}, skipping")
					continue

				# Update current for next phase in sequence
				current_connectivity = result.optimized_connections.clone()

				# Track global best
				if result.final_error < best_ppl:
					best_ppl = result.final_error
					best_connectivity = result.optimized_connections.clone()
					log(f"  [BEST] {strategy_type} improved: PPL {best_ppl:.1f}")
				else:
					log(f"  [BEST] Keeping previous: PPL {best_ppl:.1f}")

			elapsed = time.time() - start_time
			final_ppl = best_ppl

			# Update connectivity with THE BEST from either phase
			all_connectivities[ram_idx] = best_connectivity.tolist()
			updated_sample = all_connectivities[ram_idx][0][:3]
			log(f"  [DEBUG] Updated n={n} connectivity sample: {updated_sample}")

			# VERIFICATION: Re-evaluate with updated connectivity
			verify_ppls = batch_fn([best_connectivity])
			verify_ppl = verify_ppls[0]
			if abs(verify_ppl - final_ppl) > 0.1:
				log(f"  [WARNING] Verification mismatch! Expected PPL {final_ppl:.1f}, got {verify_ppl:.1f}")

			# Apply optimized connectivity
			ram.set_connectivity(best_connectivity.tolist())
			ram.train(tokens)

			# Store results (using perplexity - lower is better)
			ppl_improvement = ((initial_ppl - final_ppl) / initial_ppl * 100) if initial_ppl > 0 else 0
			self._strategy_results[f"n{n}_{strategy_name}"] = StrategyResult(
				name=f"n{n}_{strategy_name}",
				initial_error=initial_ppl,  # Store perplexity in error field
				final_error=final_ppl,
				improvement_percent=ppl_improvement,
				elapsed_seconds=elapsed,
				timed_out=False,
			)

			log(f"  n={n}: PPL {initial_ppl:.1f} → {final_ppl:.1f} "
				f"({ppl_improvement:+.1f}%) in {elapsed:.1f}s")

		total_elapsed = time.time() - total_start
		log_separator()
		log(f"★ All RAMs optimized in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
		log_separator()
		self._best_strategy = strategy_name

	def _evaluate_cascade_python(self, train_tokens: list[str], test_tokens: list[str]) -> float:
		"""Python fallback for cascade evaluation."""
		correct = 0
		total = min(len(test_tokens) - 6, len(test_tokens))
		for i in range(total):
			context = test_tokens[max(0, i):i+6]
			target = test_tokens[i+6] if i+6 < len(test_tokens) else None
			if target is None:
				break
			# Try cascade prediction
			pred, _, _ = self.predict(context)
			if pred == target:
				correct += 1
		accuracy = correct / total if total > 0 else 0
		return 1 - accuracy  # Return error

	def _evaluate_perplexity_python(self, test_tokens: list[str], exact_probs: list,
									vocab_size: int, cascade_threshold: float, eval_subset: int) -> float:
		"""Python fallback for perplexity evaluation (matches Rust logic)."""
		import math

		min_prob = 1.0 / vocab_size
		total_log_prob = 0.0
		n_values = [2, 3, 4, 5, 6]

		total = min(eval_subset, len(test_tokens) - 6)
		for i in range(total):
			target = test_tokens[i + 6]

			# Use pre-computed exact prob if available
			if exact_probs[i] is not None:
				prob = max(exact_probs[i], min_prob)
			else:
				# Try cascade prediction (highest n first)
				found_prob = None
				for n in reversed(n_values):
					if n not in self.generalized_rams:
						continue
					context = test_tokens[max(0, i+6-n):i+6]
					if len(context) < n:
						continue
					pred, conf = self.generalized_rams[n].predict(context)
					if pred and conf > cascade_threshold:
						if pred == target:
							found_prob = max(float(conf), min_prob)
						else:
							found_prob = min_prob
						break

				# Voting fallback if no confident cascade prediction
				if found_prob is None:
					votes = {}
					total_weight = 0.0
					for n in n_values:
						if n not in self.generalized_rams:
							continue
						context = test_tokens[max(0, i+6-n):i+6]
						if len(context) < n:
							continue
						pred, conf = self.generalized_rams[n].predict(context)
						if pred:
							weight = conf * n
							votes[pred] = votes.get(pred, 0.0) + weight
							total_weight += weight

					if total_weight > 0:
						target_votes = votes.get(target, 0.0)
						found_prob = max(target_votes / total_weight, min_prob)
					else:
						found_prob = min_prob

				prob = found_prob

			total_log_prob += math.log(prob)

		# Perplexity = exp(-avg_log_prob)
		avg_log_prob = total_log_prob / total if total > 0 else 0
		return math.exp(-avg_log_prob)

	def predict(self, context: list[str], cascade_threshold: float = None) -> tuple[str, str, float]:
		"""Predict next word.

		Args:
			context: List of previous words
			cascade_threshold: Confidence threshold for generalized RAMs (default: self.cascade_threshold)
		"""
		if cascade_threshold is None:
			cascade_threshold = getattr(self, 'cascade_threshold', 0.1)

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
						method = f"exact_n{n}"
						self.prediction_stats[method] += 1
						return best, method, conf

		# 2. Try generalized RAMs (partial connectivity)
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred and conf > cascade_threshold:
					method = f"gen_n{n}"
					self.prediction_stats[method] += 1
					return pred, method, conf

		# 3. Fallback: vote across all scales
		votes = Counter()
		for n, ram in self.generalized_rams.items():
			if len(context) >= n:
				pred, conf = ram.predict(context)
				if pred:
					votes[pred] += conf * n  # Weight by context length

		if votes:
			best, score = votes.most_common(1)[0]
			self.prediction_stats["voting"] += 1
			return best, "voting", score / sum(votes.values())

		self.prediction_stats["none"] += 1
		return "<UNK>", "none", 0.0

	def log_prediction_stats(self):
		"""Log prediction method usage statistics."""
		total = sum(self.prediction_stats.values())
		if total == 0:
			log("No predictions made yet.")
			return

		log("Prediction method usage:")
		# Sort by n-level (exact first, then gen, then voting/none)
		for method in ["exact_n4", "exact_n3", "exact_n2",
					   "gen_n6", "gen_n5", "gen_n4", "gen_n3", "gen_n2",
					   "voting", "none"]:
			count = self.prediction_stats.get(method, 0)
			if count > 0:
				pct = 100 * count / total
				log(f"  {method}: {count:,} ({pct:.1f}%)")

	def reset_prediction_stats(self):
		"""Reset prediction statistics."""
		self.prediction_stats = Counter()

	def predict_threshold_voting(self, context: list[str], min_conf: float = None) -> tuple[str, str, float]:
		"""
		Strategy #3: Threshold-filtered voting.

		Only include votes from RAMs with confidence > min_conf.
		This filters out garbage predictions that pollute voting.
		"""
		if min_conf is None:
			min_conf = self.cascade_threshold

		# 1. Try exact match first (same as regular predict)
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

		# 2. Try generalized RAMs with confidence threshold
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred and conf > self.cascade_threshold:
					return pred, f"gen_n{n}", conf

		# 3. Threshold-filtered voting: only include votes with conf > min_conf
		votes = Counter()
		for n, ram in self.generalized_rams.items():
			if len(context) >= n:
				pred, conf = ram.predict(context)
				if pred and conf > min_conf:
					votes[pred] += conf * n

		if votes:
			best, score = votes.most_common(1)[0]
			return best, "threshold_voting", score / sum(votes.values())

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
				if conf > self.cascade_threshold:  # Only count confident predictions
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
		# Test multiple confidence thresholds
		thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

		results = {
			"cascade": {"correct": 0, "total": 0},  # Original cascade
			"weighted": {"correct": 0, "total": 0},  # Accuracy-weighted voting
			"meta": {"correct": 0, "total": 0},      # Meta-classifier
		}
		# Add threshold strategies
		for t in thresholds:
			results[f"thresh_{t:.2f}"] = {"correct": 0, "total": 0}

		for i in range(len(tokens) - 6):
			context = tokens[i:i + 5]
			target = tokens[i + 5]

			# Increment totals for all strategies
			for key in results:
				results[key]["total"] += 1

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

			# Threshold voting at various levels
			for t in thresholds:
				pred, _, _ = self.predict_threshold_voting(context, min_conf=t)
				if pred == target:
					results[f"thresh_{t:.2f}"]["correct"] += 1

		# Compute accuracies
		for strategy in results:
			total = results[strategy]["total"]
			correct = results[strategy]["correct"]
			results[strategy]["accuracy"] = correct / total if total > 0 else 0.0

		return results

	def evaluate(self, tokens: list[str]) -> dict:
		"""Evaluate the model."""
		self.reset_prediction_stats()  # Reset for fresh stats

		results = {
			"correct": 0,
			"total": 0,
			"by_method": defaultdict(lambda: {"correct": 0, "total": 0})
		}

		for i in range(len(tokens) - 6):
			context = tokens[i:i + 6]
			target = tokens[i + 6]

			pred, method, _ = self.predict(context)

			results["total"] += 1
			results["by_method"][method]["total"] += 1

			if pred == target:
				results["correct"] += 1
				results["by_method"][method]["correct"] += 1

		# Log usage and accuracy by method
		self.log_accuracy_by_method(results["by_method"])

		return results

	def log_accuracy_by_method(self, by_method: dict):
		"""Log accuracy breakdown by prediction method."""
		total_preds = sum(m["total"] for m in by_method.values())
		if total_preds == 0:
			log("No predictions made yet.")
			return

		log("Accuracy by prediction method:")
		log(f"  {'Method':<12} {'Usage':>8} {'Accuracy':>10} {'Correct':>10}")
		log(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10}")

		# Sort by cascade priority (exact first, then gen by n, then voting/none)
		method_order = ["exact_n4", "exact_n3", "exact_n2",
						"gen_n6", "gen_n5", "gen_n4", "gen_n3", "gen_n2",
						"voting", "none"]

		for method in method_order:
			if method in by_method:
				stats = by_method[method]
				total = stats["total"]
				correct = stats["correct"]
				usage_pct = 100 * total / total_preds
				acc = 100 * correct / total if total > 0 else 0
				log(f"  {method:<12} {usage_pct:>7.1f}% {acc:>9.1f}% {correct:>6}/{total:<6}")


def run_benchmark(
	mode: BenchmarkMode = BenchmarkMode.FAST,
	run_id: int = 1,
	seed: int = None,
	tokenizer_type: TokenizerType = TokenizerType.WIKITEXT_WORD,
	n_neurons: int = None,
	cascade_threshold: float = 0.1,
	strategy_sequence: list = None
) -> BenchmarkRun:
	"""Run the v2 benchmark."""
	if seed is not None:
		random.seed(seed)

	# Default strategy sequence
	if strategy_sequence is None:
		strategy_sequence = ['GA', 'TS']

	# Mode descriptions
	mode_descs = {
		BenchmarkMode.FAST: "FAST (GA 16×10, TS 16×10)",
		BenchmarkMode.FULL: "FULL (GA 32×100, TS 32×50)",
		BenchmarkMode.OVERNIGHT: "OVERNIGHT (GA 48×1000, TS 48×1000 + early stop)",
	}
	mode_desc = mode_descs[mode]

	log_separator()
	log(f"RAM LANGUAGE MODEL v2 - RUN {run_id}")
	log("With Full Thesis Optimization (2024 Hardware)")
	log(f"Mode: {mode_desc}")
	if mode != BenchmarkMode.FAST:
		hybrid_label = "GA→TS" if run_id % 2 == 1 else "TS→GA"
		log(f"Hybrid: {hybrid_label}")
	if seed is not None:
		log(f"Seed: {seed}")
	log_separator()

	run_start = time.time()

	# Load data - reduced for fast mode
	train_limit = 30000 if mode == BenchmarkMode.FAST else 150000
	test_limit = 5000 if mode == BenchmarkMode.FAST else 20000

	try:
		from datasets import load_dataset

		log("Loading WikiText-2...")
		log(f"Tokenizer: {tokenizer_type.name} (for comparable perplexity)")
		dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

		# Get tokenizer based on parameter
		tokenize, _ = get_tokenizer(tokenizer_type)

		train_text = " ".join(dataset["train"]["text"])
		test_text = " ".join(dataset["test"]["text"])

		train_tokens = tokenize(train_text)[:train_limit]
		test_tokens = tokenize(test_text)[:test_limit]

		# Log vocabulary size for reference
		train_vocab = set(train_tokens)
		test_vocab = set(test_tokens)
		total_vocab = train_vocab | test_vocab
		log(f"Train: {len(train_tokens):,} tokens, Vocab: {len(train_vocab):,}")
		log(f"Test: {len(test_tokens):,} tokens, Vocab: {len(test_vocab):,}")
		log(f"Total unique tokens: {len(total_vocab):,}")

	except Exception as e:
		log(f"Error loading data: {e}")
		log("Using synthetic data...")

		# Fallback to synthetic
		words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "house", "tree"]
		train_tokens = [random.choice(words) for _ in range(train_limit)]
		test_tokens = [random.choice(words) for _ in range(test_limit)]

	# Train model
	log_separator()
	model = RAMLM_v2(freq_threshold=30, mode=mode, n_neurons=n_neurons, cascade_threshold=cascade_threshold,
					 strategy_sequence=strategy_sequence)
	log(f"Model config: n_neurons={model.n_neurons}, cascade_threshold={model.cascade_threshold}")
	log(f"Strategy sequence: {' → '.join(strategy_sequence)}")

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

	# Calculate perplexity on test set
	# First compute exact_probs for the test set
	eval_subset = min(3000, len(test_tokens) - 6)
	exact_probs = []
	for i in range(eval_subset):
		target = test_tokens[i + 6]
		exact_prob = None
		for n in sorted(model.exact_rams.keys(), reverse=True):
			ctx = tuple(test_tokens[i + 6 - n:i + 6])
			if ctx in model.exact_rams[n]:
				counts = model.exact_rams[n][ctx]
				total = sum(counts.values())
				if counts.most_common(1)[0][1] / total > 0.2 or total >= 3:
					target_count = counts.get(target, 0)
					exact_prob = target_count / total if total > 0 else 0.0
					break
		exact_probs.append(exact_prob)

	vocab_size = len(model.word_counts)
	perplexity = model._evaluate_perplexity_python(
		test_tokens, exact_probs, vocab_size, model.cascade_threshold, eval_subset
	)

	log(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
	log(f"PERPLEXITY: {perplexity:.1f}")
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
	run_elapsed = time.time() - run_start

	log(f"MODE: {mode.name}")
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
	log(f"FINAL PERPLEXITY: {perplexity:.1f}")
	log("")
	log(f"Run {run_id} elapsed: {run_elapsed:.1f}s ({run_elapsed/60:.1f} min)")

	# Return structured result
	benchmark_run = BenchmarkRun(
		run_id=run_id,
		accuracy=accuracy,
		perplexity=perplexity,
		coverage_n4=gen_n4_coverage,
		strategy_results=getattr(model, '_strategy_results', {}),
		best_strategy=getattr(model, '_best_strategy', 'unknown'),
		elapsed_seconds=run_elapsed
	)
	return benchmark_run


def run_multi_benchmark(
	n_runs: int = 3,
	mode: BenchmarkMode = BenchmarkMode.FULL,
	tokenizer_type: TokenizerType = TokenizerType.WIKITEXT_WORD,
	n_neurons: int = None,
	cascade_threshold: float = 0.1,
	strategy_sequence: list = None
):
	"""Run the benchmark multiple times and summarize results."""
	if strategy_sequence is None:
		strategy_sequence = ['GA', 'TS']

	# Mode descriptions
	mode_descs = {
		BenchmarkMode.FAST: "FAST (GA 16×10, TS 16×10)",
		BenchmarkMode.FULL: "FULL (GA 32×100, TS 32×50)",
		BenchmarkMode.OVERNIGHT: "OVERNIGHT (GA 48×1000, TS 48×1000 + early stop)",
	}

	log_separator()
	log(f"MULTI-RUN BENCHMARK: {n_runs} runs")
	log(f"Mode: {mode_descs[mode]}")
	log(f"Tokenizer: {tokenizer_type.name}")
	log(f"Neurons: {n_neurons if n_neurons else 'default'}, Threshold: {cascade_threshold}")
	log(f"Strategy sequence: {' → '.join(strategy_sequence)}")
	log_separator()

	total_start = time.time()
	runs: list[BenchmarkRun] = []

	for i in range(n_runs):
		log_separator("#")
		log(f"# STARTING RUN {i+1} of {n_runs}")
		log(f"# Strategy: {' → '.join(strategy_sequence)}")
		log_separator("#")

		seed = 42 + i * 1000  # Different seed for each run
		run_result = run_benchmark(mode=mode, run_id=i+1, seed=seed, tokenizer_type=tokenizer_type,
								   n_neurons=n_neurons, cascade_threshold=cascade_threshold,
								   strategy_sequence=strategy_sequence)
		runs.append(run_result)

		# Save intermediate results
		log_separator("-")
		log(f"[Run {i+1}/{n_runs} COMPLETE]")
		log(f"  Accuracy: {run_result.accuracy*100:.2f}%")
		log(f"  Perplexity: {run_result.perplexity:.1f}")
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

	# Perplexity stats
	perplexities = [r.perplexity for r in runs]
	log("")
	log("Perplexity (lower is better):")
	log(f"  Mean: {sum(perplexities)/len(perplexities):.1f}")
	log(f"  Min:  {min(perplexities):.1f}")
	log(f"  Max:  {max(perplexities):.1f}")

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
	parser.add_argument("--full", action="store_true", help="Run full benchmark (GA 32×100, TS 32×50)")
	parser.add_argument("--overnight", action="store_true", help="Run extended overnight benchmark (GA 48×1000, TS 48×1000 with early stopping)")
	parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
	parser.add_argument("--tokenizer", type=str, default="word",
		choices=["simple", "word", "gpt2"],
		help="Tokenizer: simple (original), word (WikiText-2 standard), gpt2 (BPE)")
	parser.add_argument("--neurons", type=int, default=None,
		help="Number of neurons per RAM (default: 16 FAST, 64 FULL/OVERNIGHT)")
	parser.add_argument("--threshold", type=float, default=0.1,
		help="Cascade confidence threshold (default: 0.1)")
	parser.add_argument("--strategy", type=str, default="GA,TS",
		help="Optimization strategy sequence, comma-separated (default: GA,TS). Examples: GA, TS, GA,TS, TS,GA, GA,TS,GA")
	args = parser.parse_args()

	# Map tokenizer arg to enum
	tokenizer_map = {"simple": TokenizerType.SIMPLE, "word": TokenizerType.WIKITEXT_WORD, "gpt2": TokenizerType.GPT2_BPE}
	tokenizer_type = tokenizer_map[args.tokenizer]

	# Determine mode from flags
	if args.overnight:
		mode = BenchmarkMode.OVERNIGHT
	elif args.full:
		mode = BenchmarkMode.FULL
	else:
		mode = BenchmarkMode.FAST

	# Parse strategy sequence
	strategy_sequence = [s.strip().upper() for s in args.strategy.split(',')]

	# Mode descriptions for logging
	mode_descs = {
		BenchmarkMode.FAST: "FAST (GA 16×10, TS 16×10)",
		BenchmarkMode.FULL: "FULL (GA 32×100, TS 32×50)",
		BenchmarkMode.OVERNIGHT: "OVERNIGHT (GA 48×1000, TS 48×1000 + early stop)",
	}

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
	log(f"Mode: {mode_descs[mode]}")
	log(f"Tokenizer: {tokenizer_type.name}")
	log(f"Neurons: {args.neurons if args.neurons else 'default (16 FAST, 64 otherwise)'}")
	log(f"Cascade threshold: {args.threshold}")
	log(f"Strategy: {' → '.join(strategy_sequence)}")
	log(f"Runs: {args.runs}")
	if mode == BenchmarkMode.OVERNIGHT:
		# Estimate: ~30 min per RAM × 5 RAMs × runs
		est_hours = args.runs * 2.5
		log(f"Estimated time: ~{est_hours:.1f} hours ({args.runs} runs × ~2.5h each)")
	elif mode == BenchmarkMode.FULL:
		log(f"Timeout per strategy: 3 hours")
		log(f"Estimated max time: {args.runs * 9} hours ({args.runs} runs × 3 strategies × 3h)")
	log_separator()

	if args.runs > 1:
		run_multi_benchmark(n_runs=args.runs, mode=mode, tokenizer_type=tokenizer_type,
							n_neurons=args.neurons, cascade_threshold=args.threshold,
							strategy_sequence=strategy_sequence)
	else:
		run_benchmark(mode=mode, tokenizer_type=tokenizer_type,
					  n_neurons=args.neurons, cascade_threshold=args.threshold,
					  strategy_sequence=strategy_sequence)

	# Final log
	log_separator()
	log("SESSION COMPLETE")
	log(f"Full log available at: {log_file}")
	log_separator()
