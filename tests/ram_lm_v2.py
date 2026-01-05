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

from wnn.ram.core import (
	BenchmarkMode,
	AccelerationMode,
	get_effective_cores,
	get_batch_size_for_mode,
	set_detected_cores,
)
from wnn.ram.strategies import PerplexityCalculator
from wnn.tokenizers import TokenizerType, TokenizerFactory, Tokenizer
from wnn.lsh import RandomProjectionHasher, SimHasher, LSHType
from wnn.attention import AttentionType, create_attention
from wnn.representations import RepresentationType, create_encoder

# Number of parallel workers (leave some cores for system)
N_WORKERS = max(1, multiprocessing.cpu_count() - 4)  # 12 on 16-core M4 Max

# Maximum n-gram order (6 = use 6-grams, predict 7th token from 6-token context)
# This is used for indexing: context = tokens[i:i+MAX_NGRAM], target = tokens[i+MAX_NGRAM]
MAX_NGRAM = 6

# ============================================================================
# RUST ACCELERATOR (822x speedup over Python!)
# ============================================================================
try:
	import ram_accelerator
	RUST_AVAILABLE = True
	RUST_CPU_CORES = ram_accelerator.cpu_cores()
	METAL_AVAILABLE = ram_accelerator.metal_available()
	if METAL_AVAILABLE:
		METAL_INFO = ram_accelerator.metal_device_info()
		# M4 Max has 40 GPU cores - use as parallel workers
		METAL_GPU_CORES = 40  # Could parse from METAL_INFO if available
	else:
		METAL_INFO = None
		METAL_GPU_CORES = 0
	# Register detected cores with wnn.ram.core for centralized access
	set_detected_cores(RUST_CPU_CORES, METAL_GPU_CORES, METAL_AVAILABLE)
except ImportError:
	RUST_AVAILABLE = False
	RUST_CPU_CORES = 0
	METAL_AVAILABLE = False
	METAL_INFO = None
	METAL_GPU_CORES = 0

# Acceleration mode (set via command line, enum imported from wnn.ram.core)
ACCELERATION_MODE = AccelerationMode.CPU  # Default

# ============================================================================
# TOKENIZER FACTORY - For fair comparison to published benchmarks
# ============================================================================
def get_tokenizer(tokenizer_type: TokenizerType, bpe_vocab_size: int = 32000):
	"""
	Get tokenizer function based on type.

	Uses the new wnn.tokenizers module for consistent tokenization.

	Published WikiText-2 perplexity benchmarks:
	- Word-level (WORD): LSTM ~65-100, AWD-LSTM ~57
	- GPT-2 BPE (GPT2): GPT-2 Small ~29, GPT-2 Large ~22

	Note: Word-level and BPE perplexities are NOT directly comparable!

	Args:
		tokenizer_type: Type of tokenizer to use
		bpe_vocab_size: Vocabulary size for trainable BPE (default 32k)

	Returns:
		Tuple of (tokenize_function, tokenizer_instance)
	"""
	import re

	if tokenizer_type == TokenizerType.SIMPLE_WORD:
		# Original simple tokenizer (not standard, just for testing)
		def tokenize(text):
			return re.findall(r"\w+|[^\w\s]", text.lower())
		return tokenize, None

	elif tokenizer_type == TokenizerType.WORD:
		# Standard WikiText-2 word-level preprocessing
		# Raw WikiText-2 has tokens separated by spaces
		def tokenize(text):
			tokens = text.split()
			return [t for t in tokens if t.strip()]
		return tokenize, None

	elif tokenizer_type == TokenizerType.GPT2:
		# Pre-trained GPT-2 BPE tokenization
		try:
			tokenizer = TokenizerFactory.create(TokenizerType.GPT2)

			def tokenize(text):
				# Returns token IDs as strings for compatibility with our word-based model
				token_ids = tokenizer.encode(text)
				return [str(tid) for tid in token_ids]

			return tokenize, tokenizer
		except ImportError as e:
			print(f"WARNING: GPT-2 tokenizer unavailable: {e}")
			print("Falling back to WORD tokenizer.")
			return get_tokenizer(TokenizerType.WORD)

	elif tokenizer_type == TokenizerType.BPE:
		# Trainable BPE - needs to be trained on corpus before use
		# Returns the tokenizer instance; caller must train it
		tokenizer = TokenizerFactory.create(
			TokenizerType.BPE,
			vocab_size=bpe_vocab_size,
			show_progress=True
		)

		def tokenize(text):
			# After training, encode returns token IDs as strings
			token_ids = tokenizer.encode(text)
			return [str(tid) for tid in token_ids]

		return tokenize, tokenizer

	elif tokenizer_type == TokenizerType.CHARACTER:
		# Character-level tokenization
		tokenizer = TokenizerFactory.create(TokenizerType.CHARACTER)

		def tokenize(text):
			# Return characters as tokens
			return list(text)

		return tokenize, tokenizer

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
		# Use date-based folder structure: wnn/logs/YYYY/MM/DD/
		script_dir = os.path.dirname(os.path.abspath(__file__))
		project_root = os.path.dirname(script_dir)  # Go up from tests/ to wnn/
		now = datetime.now()
		log_dir = os.path.join(project_root, "logs", now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
		os.makedirs(log_dir, exist_ok=True)

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

	With LSH enabled, similar contexts map to SIMILAR addresses, enabling:
	- "the cat ate" ≈ "the dog ate" → can share learned patterns
	- Better generalization for semantically related contexts

	With learned encoder, similar words get similar binary codes:
	- "cat" ≈ "dog" → similar addresses → shared learned patterns
	- Enables semantic generalization (cat/dog both work in pet contexts)
	"""

	def __init__(self, n: int, n_neurons: int = 32, bits_per_neuron: int = 12,
				 bits_per_word: int = 12, use_lsh: bool = False, lsh_type: str = "simhash",
				 encoder=None):
		self.n = n
		self.n_neurons = n_neurons
		self.bits_per_neuron = bits_per_neuron
		self.bits_per_word = bits_per_word
		self.total_bits = n * bits_per_word

		self.neurons = []
		self.word_to_cluster = {}
		self.n_clusters = 256

		# LSH support for similarity-preserving hashing
		self.use_lsh = use_lsh
		self.lsh_type = lsh_type
		self._lsh_hasher = None
		self._lsh_trained = False

		# Learned encoder for semantic word representations
		# If None, uses simple hash-based encoding (fast but no generalization)
		self.encoder = encoder
		self._word_code_cache = {}  # Cache encoder results for speed

	def _word_to_bits(self, word: str) -> tuple:
		"""
		Convert word to bits using encoder (learned) or hash (fallback).

		With learned encoder: similar words → similar codes → generalization
		With hash fallback: random codes → no semantic generalization
		"""
		# Try learned encoder first (cached for speed)
		if self.encoder is not None:
			if word not in self._word_code_cache:
				code = self.encoder.encode(word)
				self._word_code_cache[word] = code
			code = self._word_code_cache[word]

			# Convert int code to bit tuple
			bits = []
			for i in range(self.bits_per_word):
				bits.append((code >> i) & 1)
			return tuple(bits)

		# Fallback: simple hash-based encoding (fast, no generalization)
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
		"""
		Convert context to bit vector.

		With LSH enabled, similar contexts map to similar bit patterns,
		enabling generalization between semantically related patterns.
		"""
		if self.use_lsh and self._lsh_trained:
			# LSH: project context to bits preserving similarity
			ctx = context[-self.n:]
			address = self._lsh_hasher.hash_context(ctx)
			# Convert integer to bit tuple
			bits = []
			for i in range(self.total_bits):
				bits.append((address >> i) & 1)
			return tuple(bits)
		else:
			# Standard: concatenate word bits
			bits = []
			for word in context[-self.n:]:
				bits.extend(self._word_to_bits(word))
			return tuple(bits)

	def train_lsh(self, tokens: list[str], seed: int = 42) -> None:
		"""
		Train LSH hasher on corpus.

		Must be called before train() if use_lsh=True.
		"""
		if not self.use_lsh:
			return

		# Create hasher with appropriate bit count
		if self.lsh_type == "simhash":
			self._lsh_hasher = SimHasher(n_bits=self.total_bits)
		else:
			self._lsh_hasher = RandomProjectionHasher(
				n_bits=self.total_bits,
				embedding_dim=64,
				min_count=2,
			)

		self._lsh_hasher.train(tokens, seed=seed)
		self._lsh_trained = True

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


def _evaluate_batch_metal(
	connectivities: list[list[list[int]]],
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
) -> list[float]:
	"""
	Evaluate multiple connectivity patterns using Metal GPU (40 cores on M4 Max).

	Falls back to CPU if Metal is not available.
	"""
	if not RUST_AVAILABLE or not METAL_AVAILABLE:
		return _evaluate_batch_rust(
			connectivities, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)

	if len(connectivities) == 0:
		return []

	# Precompute word bits (same as CPU version)
	n_clusters = 256
	word_to_bits = {}

	def compute_word_bits(word: str, cluster: int) -> int:
		h = hash(word)
		bits = 0
		for i in range(7):
			if (cluster >> i) & 1:
				bits |= 1 << i
		for i in range(5):
			if (h >> i) & 1:
				bits |= 1 << (7 + i)
		return bits

	for word, cluster in word_to_cluster.items():
		word_to_bits[word] = compute_word_bits(word, cluster)

	for word in set(train_tokens) | set(test_tokens):
		if word not in word_to_bits:
			cluster = hash(word) % n_clusters
			word_to_bits[word] = compute_word_bits(word, cluster)

	# Call Metal accelerator
	results = ram_accelerator.evaluate_batch_metal(
		connectivities,
		word_to_bits,
		train_tokens,
		test_tokens,
		bits_per_neuron,
		eval_subset
	)

	return results


def _evaluate_batch_hybrid(
	connectivities: list[list[list[int]]],
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
) -> list[float]:
	"""
	Evaluate using BOTH CPU (16 cores) and Metal GPU (40 cores) = 56 parallel workers.

	Splits the batch: 71% to GPU, 29% to CPU, runs in parallel, combines results.
	"""
	if not RUST_AVAILABLE or not METAL_AVAILABLE:
		return _evaluate_batch_rust(
			connectivities, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)

	if len(connectivities) == 0:
		return []

	# Split ratio based on core counts: GPU gets 40/(40+16) = 71%
	n_total = len(connectivities)
	gpu_ratio = METAL_GPU_CORES / (METAL_GPU_CORES + RUST_CPU_CORES)
	n_gpu = int(n_total * gpu_ratio)
	n_cpu = n_total - n_gpu

	# Split connectivities
	conn_gpu = connectivities[:n_gpu]
	conn_cpu = connectivities[n_gpu:]

	# Precompute word bits once (shared by both)
	n_clusters = 256
	word_to_bits = {}

	def compute_word_bits(word: str, cluster: int) -> int:
		h = hash(word)
		bits = 0
		for i in range(7):
			if (cluster >> i) & 1:
				bits |= 1 << i
		for i in range(5):
			if (h >> i) & 1:
				bits |= 1 << (7 + i)
		return bits

	for word, cluster in word_to_cluster.items():
		word_to_bits[word] = compute_word_bits(word, cluster)

	for word in set(train_tokens) | set(test_tokens):
		if word not in word_to_bits:
			cluster = hash(word) % n_clusters
			word_to_bits[word] = compute_word_bits(word, cluster)

	# Run GPU and CPU in parallel using threads
	import concurrent.futures

	results_gpu = []
	results_cpu = []

	def run_gpu():
		nonlocal results_gpu
		if conn_gpu:
			results_gpu = ram_accelerator.evaluate_batch_metal(
				conn_gpu, word_to_bits, train_tokens, test_tokens,
				bits_per_neuron, eval_subset
			)

	def run_cpu():
		nonlocal results_cpu
		if conn_cpu:
			results_cpu = ram_accelerator.evaluate_batch_cpu(
				conn_cpu, word_to_bits, train_tokens, test_tokens,
				bits_per_neuron, eval_subset
			)

	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		gpu_future = executor.submit(run_gpu)
		cpu_future = executor.submit(run_cpu)
		gpu_future.result()
		cpu_future.result()

	# Combine results in original order
	return list(results_gpu) + list(results_cpu)


def _evaluate_batch_fast(
	connectivities: list,
	word_to_cluster: dict,
	train_tokens: list[str],
	test_tokens: list[str],
	bits_per_neuron: int,
	eval_subset: int,
) -> list[float]:
	"""
	Fast batch evaluation - uses selected acceleration mode.

	Modes:
	- CPU: Rust + rayon parallelism (16 cores)
	- METAL: Metal GPU (40 cores)
	- HYBRID: Both CPU + GPU (56 cores)

	This is the recommended function to use for batch evaluation.
	"""
	global ACCELERATION_MODE

	# Convert any numpy arrays to lists
	conn_list = []
	for conn in connectivities:
		if hasattr(conn, 'tolist'):
			conn_list.append(conn.tolist())
		else:
			conn_list.append(conn)

	if not RUST_AVAILABLE:
		return _evaluate_batch_parallel(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)

	# Use selected acceleration mode
	if ACCELERATION_MODE == AccelerationMode.HYBRID and METAL_AVAILABLE:
		return _evaluate_batch_hybrid(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)
	elif ACCELERATION_MODE == AccelerationMode.METAL and METAL_AVAILABLE:
		return _evaluate_batch_metal(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)
	else:
		# Default to CPU (Rust + rayon)
		return _evaluate_batch_rust(
			conn_list, word_to_cluster, train_tokens,
			test_tokens, bits_per_neuron, eval_subset
		)


class RAMMetaClassifier:
	"""
	RAM-based meta-classifier for selecting which prediction method to trust.

	Architecture:
	- Input layer: prediction features (24 bits) + agreement (2 bits) = 26 bits
	- Output layer: RAM neurons vote on which method to trust

	Uses agreement as additional signal - when multiple methods agree,
	that prediction is more likely correct.
	"""

	def __init__(self, methods: list[str], n_neurons: int = 32, bits_per_neuron: int = 10,
				 state_history: int = 0):  # state_history kept for API compatibility
		self.methods = methods  # e.g., ['exact_n2', 'gen_n2', 'exact_n3', ...]
		self.n_methods = len(methods)
		self.method_to_idx = {m: i for i, m in enumerate(methods)}
		self.n_neurons = n_neurons
		self.bits_per_neuron = bits_per_neuron

		# Create RAM neurons for each method (voting ensemble)
		self.neurons_per_method = {m: [] for m in methods}

		# Input encoding:
		# - Prediction features: top-4 methods × (method_id: 4 bits + conf: 2 bits) = 24 bits
		# - Agreement pattern: 2 bits (how many of top-4 agree)
		self.pred_bits = 24
		self.agree_bits = 2
		self.input_bits = self.pred_bits + self.agree_bits

		# Create neurons with random connectivity
		for method in methods:
			neurons = []
			for _ in range(n_neurons):
				connected = random.sample(range(self.input_bits), min(bits_per_neuron, self.input_bits))
				neurons.append({
					'connected': connected,
					'ram': defaultdict(lambda: {'correct': 0, 'total': 0})
				})
			self.neurons_per_method[method] = neurons

	def reset_state(self):
		"""Reset state (no-op - kept for API compatibility)."""
		pass

	def _encode_agreement(self, sorted_preds: list) -> list:
		"""Encode agreement level (how many methods agree with top prediction)."""
		if not sorted_preds:
			return [0, 0]

		top_word = sorted_preds[0][1][0]  # Word predicted by top method
		agree_count = sum(1 for _, (w, _) in sorted_preds[1:4] if w == top_word)
		# Encode as 2 bits: 0=none agree, 1=one agrees, 2=two agree, 3=all agree
		level = min(3, agree_count)
		return [(level >> 1) & 1, level & 1]

	def _encode_input(self, sorted_preds: list) -> tuple:
		"""Encode predictions + agreement into input bits (26 bits total)."""
		bits = []

		# Part 1: Prediction features (24 bits)
		for i in range(4):
			if i < len(sorted_preds):
				method, (word, conf) = sorted_preds[i]
				method_id = self.method_to_idx.get(method, 0)
				conf_level = min(3, int(conf * 4))

				# 4 bits for method_id
				for b in range(4):
					bits.append((method_id >> (3 - b)) & 1)
				# 2 bits for confidence
				for b in range(2):
					bits.append((conf_level >> (1 - b)) & 1)
			else:
				bits.extend([0] * 6)

		# Part 2: Agreement pattern
		bits.extend(self._encode_agreement(sorted_preds))

		return tuple(bits)

	def update_state(self, method: str, was_correct: bool):
		"""Update state (no-op - kept for API compatibility)."""
		pass

	def _get_address(self, full_bits: tuple, connected: list) -> tuple:
		"""Extract address from connected bits."""
		return tuple(full_bits[b] for b in connected if b < len(full_bits))

	def train(self, sorted_preds: list, correct_method: str):
		"""Train on one example."""
		input_bits = self._encode_input(sorted_preds)

		# Train each method's neurons
		for method, neurons in self.neurons_per_method.items():
			is_correct = (method == correct_method)
			for neuron in neurons:
				addr = self._get_address(input_bits, neuron['connected'])
				if addr:
					neuron['ram'][addr]['total'] += 1
					if is_correct:
						neuron['ram'][addr]['correct'] += 1

	def predict(self, sorted_preds: list) -> tuple[str, float]:
		"""Predict which method to trust."""
		input_bits = self._encode_input(sorted_preds)

		# Collect votes from all neurons for each method
		method_scores = {}

		for method, neurons in self.neurons_per_method.items():
			votes = []
			for neuron in neurons:
				addr = self._get_address(input_bits, neuron['connected'])
				if addr and addr in neuron['ram']:
					stats = neuron['ram'][addr]
					if stats['total'] > 0:
						reliability = stats['correct'] / stats['total']
						votes.append(reliability)

			if votes:
				method_scores[method] = sum(votes) / len(votes)
			else:
				method_scores[method] = 0.5

		if not method_scores:
			return sorted_preds[0][0] if sorted_preds else None, 0.5

		best_method = max(method_scores, key=method_scores.get)
		return best_method, method_scores[best_method]


class RAMLM_v2:
	"""
	RAM Language Model v2 with thesis optimization.
	"""

	def __init__(self, freq_threshold: int = 50, mode: BenchmarkMode = BenchmarkMode.FAST,
				 n_neurons: int = None, bits_per_neuron: int = None, cascade_threshold: float = 0.1,
				 strategy_sequence: list = None, smoothing_type: str = "none",
				 use_lsh: bool = False, lsh_type: str = "simhash",
				 attention_type: AttentionType = AttentionType.NONE,
				 representation_type: RepresentationType = RepresentationType.COOCCURRENCE):
		self.freq_threshold = freq_threshold
		self.mode = mode
		self.word_counts = Counter()
		self.high_freq_words = set()
		self.smoothing_type = smoothing_type
		self.smoothing = None  # Trained during train()
		self.use_lsh = use_lsh
		self.lsh_type = lsh_type
		self.attention_type = attention_type
		self.attention = None  # Trained during train()
		self.representation_type = representation_type
		self.binary_encoder = None  # Trained during train()

		# Configurable parameters (with mode-based defaults)
		if n_neurons is None:
			self.n_neurons = 16 if mode == BenchmarkMode.FAST else 64
		else:
			self.n_neurons = n_neurons
		if bits_per_neuron is None:
			self.bits_per_neuron = 10 if mode == BenchmarkMode.FAST else 14
		else:
			self.bits_per_neuron = bits_per_neuron
		self.cascade_threshold = cascade_threshold
		self.strategy_sequence = strategy_sequence if strategy_sequence else ['GA', 'TS']

		# Exact RAMs for high-freq (still useful for common patterns)
		# Now includes n=5 and n=6 for better coverage
		self.exact_rams = {n: defaultdict(Counter) for n in [2, 3, 4, 5, 6]}

		# Generalized RAMs with partial connectivity (KEY IMPROVEMENT)
		self.generalized_rams = {}

		# Track prediction method usage
		self.prediction_stats = Counter()

	def train(self, tokens: list[str], optimize_connectivity: bool = True, final_test_tokens: list[str] = None, validation_tokens: list[str] = None):
		"""Train the model.

		Args:
			tokens: Training tokens
			optimize_connectivity: Whether to run connectivity optimization
			final_test_tokens: Optional test tokens for pre/post optimization comparison
			validation_tokens: Optional validation tokens for overfitting detection during optimization
		"""
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

		# Train smoothing model if enabled
		if self.smoothing_type != "none":
			log(f"Training {self.smoothing_type} smoothing model...")
			from wnn.smoothing import KneserNeySmoothing, SimpleBackoffSmoothing, AddKSmoothing
			if self.smoothing_type == "kneser_ney":
				self.smoothing = KneserNeySmoothing(max_order=6)
			elif self.smoothing_type == "backoff":
				self.smoothing = SimpleBackoffSmoothing(max_order=6)
			elif self.smoothing_type == "add_k":
				self.smoothing = AddKSmoothing(max_order=6, k=0.5)
			self.smoothing.train(tokens)
			log(f"  Vocab size: {self.smoothing.vocab_size:,}")

		# Train attention mechanism if enabled
		if self.attention_type != AttentionType.NONE:
			log(f"Training {self.attention_type.name.lower()} attention...")
			self.attention = create_attention(self.attention_type, max_context=6)
			self.attention.train(tokens)
			log(f"  Attention trained on {len(tokens):,} tokens")

		# Train binary encoder for learned representations (if not HASH type)
		self.binary_encoder = create_encoder(
			self.representation_type,
			n_bits=12,
			context_window=2,
			min_freq=3,
		)
		if self.binary_encoder is not None:
			log(f"Training {self.representation_type.name.lower()} binary encoder...")
			self.binary_encoder.train(tokens)
			stats = self.binary_encoder.get_stats()
			log(f"  Vocab: {stats['vocab_size']:,}, Unique codes: {stats.get('unique_codes', 'N/A')}")
		else:
			log(f"Using {self.representation_type.name.lower()} encoding (no training needed)")

		# Train generalized RAMs with partial connectivity
		lsh_label = f" + {self.lsh_type.upper()} LSH" if self.use_lsh else ""
		encoder_label = f" + {self.representation_type.name}" if self.binary_encoder else ""
		log(f"Training generalized RAMs ({self.n_neurons} neurons{lsh_label}{encoder_label})...")
		bits_per_neuron = 10 if self.mode == BenchmarkMode.FAST else 14

		for n in [2, 3, 4, 5, 6]:  # Added n=6 for longer context
			ram = GeneralizedNGramRAM(
				n=n,
				n_neurons=self.n_neurons,
				bits_per_neuron=bits_per_neuron,
				bits_per_word=12,
				use_lsh=self.use_lsh,
				lsh_type=self.lsh_type,
				encoder=self.binary_encoder,  # Pass learned encoder for semantic generalization
			)
			ram.learn_clusters(tokens)
			# Train LSH hasher if enabled (before train())
			if self.use_lsh:
				ram.train_lsh(tokens)
			ram.create_diverse_connectivity()
			ram.train(tokens)

			coverage = ram.get_coverage(tokens[:5000])
			log(f"  n={n}: {len(ram.neurons)} neurons, ~{coverage*100:.1f}% coverage")
			self.generalized_rams[n] = ram

		# Optimize connectivity if requested
		if optimize_connectivity:
			self._optimize_connectivity(tokens, final_test_tokens, validation_tokens)

	def _optimize_connectivity(self, tokens: list[str], final_test_tokens: list[str] = None, validation_tokens: list[str] = None):
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
			OverfittingMonitor,
			HEALTHY_THRESHOLD,
			WARNING_THRESHOLD,
			SEVERE_THRESHOLD,
			CRITICAL_THRESHOLD,
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
		train_tokens = tokens[:train_subset]

		# Multi-window sampling: sample N windows from across the 2M token corpus
		# Each window is continuous (for valid n-gram contexts), but windows come from
		# different Wikipedia articles - reduces topical bias in optimization evaluation
		# Standard: 100 windows × 200 tokens = 20k eval tokens (10x more than before)
		num_windows = 20 if self.mode == BenchmarkMode.FAST else 100
		window_size = 200  # Large enough for 6-gram contexts with margin

		# Sample random windows from across the FULL training corpus
		# Not just after train_subset - we want diversity from all 2M tokens
		corpus_end = len(tokens) - window_size - 10  # Safety margin
		if corpus_end > window_size * num_windows:
			import random as rand_module
			# Consistent seed for reproducibility (different from model seed)
			sampling_rng = rand_module.Random(42)
			window_starts = sampling_rng.sample(range(0, corpus_end), num_windows)
			window_starts.sort()  # Cache-friendly access
			# Concatenate windows into test_tokens for optimization evaluation
			test_tokens = []
			for start in window_starts:
				test_tokens.extend(tokens[start:start + window_size])
			log(f"Multi-window sampling: {num_windows} windows × {window_size} tokens = {len(test_tokens):,} eval tokens")
		else:
			# Fallback if corpus too small
			test_tokens = tokens[train_subset:train_subset + num_windows * window_size]
			log(f"Single-window fallback: {len(test_tokens):,} eval tokens")

		eval_total = len(test_tokens) - MAX_NGRAM  # Usable positions (need MAX_NGRAM for context)
		# Split eval into train (for optimization) and validation (for overfitting detection)
		eval_train = int(eval_total * 0.67)  # ~13k for optimization
		eval_val = eval_total - eval_train    # ~7k for validation

		# Strategy parameters (scale with mode and available cores)
		# Population sizes scale with effective cores for optimal parallel batching
		# Elitism 25% across all modes
		ga_elitism_pct = 0.25

		# Get batch sizes based on acceleration mode and benchmark mode
		ga_pop, ga_gens = get_batch_size_for_mode(ACCELERATION_MODE, self.mode)
		ts_neighbors, ts_iters = ga_pop, ga_gens  # Same sizing for TS

		# Adjust iterations for different modes (get_batch_size_for_mode returns base values)
		match self.mode:
			case BenchmarkMode.FULL:
				ts_iters = 50  # TS uses fewer iterations than GA in FULL mode
			case BenchmarkMode.OVERNIGHT:
				pass  # Keep 1000 for both GA and TS
			case _:
				pass  # FAST uses same for both

		# Configurable strategy sequence (default: GA then TS)
		strategy_sequence = getattr(self, 'strategy_sequence', ['GA', 'TS'])
		strategy_name = "→".join(strategy_sequence)

		bits_per_neuron = self.bits_per_neuron
		n_values = sorted(self.generalized_rams.keys())

		log(f"Optimizing {len(n_values)} RAMs: n={n_values}")
		log(f"Mode: {self.mode.name}")
		log(f"Train: {train_subset:,} tokens | Eval: {num_windows} windows × {window_size} = {eval_train:,} opt + {eval_val:,} val")
		log(f"Strategy sequence: {strategy_name}")
		ga_elitism = max(2, int(ga_pop * ga_elitism_pct))
		ga_eval_per_gen = ga_pop - ga_elitism
		early_stop_patience = 1  # Must match GA/TS config defaults
		early_stop_threshold_pct = 0.1  # Require 0.1% improvement to continue (was 5.0% - stopped too early)
		log(f"GA: {ga_pop} pop × {ga_gens} max gens (elite={ga_elitism} ({ga_elitism_pct*100:.0f}%), eval={ga_eval_per_gen}/gen)")
		log(f"TS: {ts_neighbors} neighbors × {ts_iters} max iters")
		log(f"Early stopping: <{early_stop_threshold_pct}% improvement over {(early_stop_patience+1)*5} gens/iters ({early_stop_patience+1} checks)")
		match (ACCEL_RUST_AVAILABLE, ACCELERATION_MODE, METAL_AVAILABLE):
			case (True, AccelerationMode.HYBRID, True):
				log(f"Accelerator: Rust PERPLEXITY eval ({ACCELERATION_MODE.name}: {ACCEL_RUST_CORES} CPU + {METAL_GPU_CORES} GPU = {ACCEL_RUST_CORES + METAL_GPU_CORES} cores)")
			case (True, AccelerationMode.METAL, True):
				log(f"Accelerator: Rust PERPLEXITY eval ({ACCELERATION_MODE.name}: {METAL_GPU_CORES} GPU cores)")
			case (True, _, _):
				log(f"Accelerator: Rust PERPLEXITY eval ({ACCELERATION_MODE.name}: {ACCEL_RUST_CORES} threads)")
			case _:
				log(f"Accelerator: Python (install ram_accelerator for speedup)")
		log("")

		# Pre-calculate exact RAM probabilities for OPTIMIZATION (from multi-window sampled training data)
		# exact_probs[i] = P(target|context) if covered by exact RAM, None if not covered
		exact_probs_train = []
		exact_results = []
		for i in range(min(eval_total, len(test_tokens) - MAX_NGRAM)):
			target = test_tokens[i + MAX_NGRAM]
			exact_prob = None
			exact_pred = None
			for n in sorted(self.exact_rams.keys(), reverse=True):
				ctx = tuple(test_tokens[i + MAX_NGRAM - n:i + MAX_NGRAM])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					best, count = counts.most_common(1)[0]
					conf = count / total
					if conf > 0.2 or total >= 3:
						target_count = counts.get(target, 0)
						exact_prob = target_count / total if total > 0 else 0.0
						exact_pred = (best == target)
						break
			exact_probs_train.append(exact_prob)
			exact_results.append(exact_pred)

		exact_covered_train = sum(1 for x in exact_probs_train if x is not None)
		exact_correct_train = sum(1 for x in exact_results if x is True)
		avg_exact_prob_train = sum(p for p in exact_probs_train if p is not None) / exact_covered_train if exact_covered_train > 0 else 0.0
		log(f"Exact RAM predictions (optimization eval - from train corpus):")
		log(f"  Covered: {exact_covered_train}/{len(exact_probs_train)} ({exact_covered_train/len(exact_probs_train)*100:.1f}%)")
		log(f"  Correct: {exact_correct_train}/{exact_covered_train} ({exact_correct_train/exact_covered_train*100:.1f}% of covered)")
		log(f"  Avg P(target): {avg_exact_prob_train:.3f}")

		# Pre-calculate exact RAM probabilities for VALIDATION (WikiText-2 validation split)
		# This is DIFFERENT articles from train - detects true generalization issues
		val_tokens_for_eval = validation_tokens if validation_tokens is not None else test_tokens
		val_eval_size = min(5000, len(val_tokens_for_eval) - MAX_NGRAM)  # Use 5k from validation split
		exact_probs_val = []
		for i in range(val_eval_size):
			target = val_tokens_for_eval[i + MAX_NGRAM]
			exact_prob = None
			for n in sorted(self.exact_rams.keys(), reverse=True):
				ctx = tuple(val_tokens_for_eval[i + MAX_NGRAM - n:i + MAX_NGRAM])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					best, count = counts.most_common(1)[0]
					conf = count / total
					if conf > 0.2 or total >= 3:
						target_count = counts.get(target, 0)
						exact_prob = target_count / total if total > 0 else 0.0
						break
			exact_probs_val.append(exact_prob)

		exact_covered_val = sum(1 for x in exact_probs_val if x is not None)
		avg_exact_prob_val = sum(p for p in exact_probs_val if p is not None) / exact_covered_val if exact_covered_val > 0 else 0.0
		log(f"Exact RAM predictions (validation - WikiText-2 validation split):")
		log(f"  Covered: {exact_covered_val}/{len(exact_probs_val)} ({exact_covered_val/len(exact_probs_val)*100:.1f}%)")
		log(f"  Avg P(target): {avg_exact_prob_val:.3f}")
		log("")

		# Get initial connectivities for all RAMs
		all_connectivities = []
		for n in n_values:
			ram = self.generalized_rams[n]
			all_connectivities.append(ram.get_connectivity())

		# Build word_to_bits mapping (same for all RAMs)
		# Include both training AND validation vocabulary
		ram = self.generalized_rams[n_values[0]]
		word_to_bits = {}
		for word, cluster in ram.word_to_cluster.items():
			word_hash = hash(word) & 0x1F  # 5 bits
			encoding = (cluster << 5) | word_hash
			word_to_bits[word] = encoding

		# Add validation vocabulary words not in training (use hash-only encoding)
		# These words won't have learned clusters, so use cluster 0 + hash
		if validation_tokens is not None:
			for word in set(validation_tokens):
				if word not in word_to_bits:
					word_hash = hash(word) & 0x1F  # 5 bits
					encoding = word_hash  # cluster 0 + hash
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

		# PRE-OPTIMIZATION: Evaluate on test and validation sets if provided
		# This gives us baselines to compare with post-optimization
		self._pre_optimization_test_ppl = None
		self._pre_optimization_val_ppl = None
		self._cached_test_exact_probs = None  # Cache for reuse in final evaluation
		if final_test_tokens is not None and len(final_test_tokens) > MAX_NGRAM:
			log("")
			log("╔══════════════════════════════════════════════════════════════════╗")
			log("║  PRE-OPTIMIZATION BASELINE EVALUATION                            ║")
			log("╚══════════════════════════════════════════════════════════════════╝")

			# Helper to compute exact_probs for a token set (needed for PPL calculation)
			def compute_exact_probs(tokens_to_eval):
				"""Compute exact_probs (from exact RAMs) - P(target|context) or None."""
				eval_size = len(tokens_to_eval) - MAX_NGRAM
				exact_probs = []
				for i in range(eval_size):
					target = tokens_to_eval[i + MAX_NGRAM]
					exact_prob = None
					for n in sorted(self.exact_rams.keys(), reverse=True):
						ctx = tuple(tokens_to_eval[i + MAX_NGRAM - n:i + MAX_NGRAM])
						if ctx in self.exact_rams[n]:
							counts = self.exact_rams[n][ctx]
							total = sum(counts.values())
							_, count = counts.most_common(1)[0]
							conf = count / total
							if conf > 0.2 or total >= 3:
								target_count = counts.get(target, 0)
								exact_prob = target_count / total if total > 0 else 0.0
								break
					exact_probs.append(exact_prob)
				return exact_probs

			# Helper to compute PPL using Rust accelerator (batch of 1 = current connectivity)
			def compute_ppl_rust(tokens_to_eval, exact_probs_list):
				"""Compute PPL using Rust accelerator with current connectivity."""
				eval_size = len(tokens_to_eval) - MAX_NGRAM
				# Get current connectivities from all RAMs
				n_values = sorted(self.generalized_rams.keys())
				all_conns = [
					[list(neuron.connected_bits) for neuron in self.generalized_rams[n].neurons]
					for n in n_values
				]
				# Use first RAM's connectivity as dummy candidate (we evaluate current state)
				dummy_candidate = all_conns[0][0] if all_conns and all_conns[0] else [0, 1, 2]
				# Export word_to_bits
				_, _, w2b = self._export_rams_for_rust()
				# Call Rust with batch of 1 (current connectivity, ram_idx=0 is ignored for full eval)
				perplexities = ram_accelerator.evaluate_fullnetwork_perplexity_batch_cpu(
					all_conns, [dummy_candidate], 0,  # ram_idx=0, single candidate
					w2b, tokens[:train_subset], tokens_to_eval,
					exact_probs_list, eval_size, vocab_size, self.cascade_threshold
				)
				return perplexities[0] if perplexities else float('inf')

			# Check if Rust accelerator with perplexity is available
			has_rust_ppl = ACCEL_RUST_AVAILABLE and hasattr(ram_accelerator, 'evaluate_fullnetwork_perplexity_batch_cpu')

			# Test set baseline (cache exact_probs - they don't change during optimization)
			test_exact_probs = compute_exact_probs(final_test_tokens)
			self._cached_test_exact_probs = test_exact_probs
			if has_rust_ppl:
				pre_test_ppl = compute_ppl_rust(final_test_tokens, test_exact_probs)
			else:
				pre_test_ppl = self._evaluate_perplexity_python(
					final_test_tokens, test_exact_probs, vocab_size, self.cascade_threshold,
					len(final_test_tokens) - MAX_NGRAM
				)
			self._pre_optimization_test_ppl = pre_test_ppl
			log(f"★ TEST SET PPL (before optimization): {pre_test_ppl:.1f}")

			# Validation set baseline
			if validation_tokens is not None and len(validation_tokens) > MAX_NGRAM:
				val_exact_probs = compute_exact_probs(validation_tokens)
				if has_rust_ppl:
					pre_val_ppl = compute_ppl_rust(validation_tokens, val_exact_probs)
				else:
					pre_val_ppl = self._evaluate_perplexity_python(
						validation_tokens, val_exact_probs, vocab_size, self.cascade_threshold,
						len(validation_tokens) - MAX_NGRAM
					)
				self._pre_optimization_val_ppl = pre_val_ppl
				log(f"★ VALIDATION SET PPL (before optimization): {pre_val_ppl:.1f}")
			log("")

		# Optimize each RAM using PERPLEXITY as the goal (lower is better)
		# IMPORTANT: Optimize from highest n to lowest (n=6 → n=2)
		# Higher n RAMs have priority in cascade, so optimizing them first has more impact
		# Track global baseline for overfitting detection (set from first RAM's initial values)
		global_baseline_ratio = None  # Will be set from first RAM's initial train/val PPL

		for ram_idx, n in reversed(list(enumerate(n_values))):
			ram = self.generalized_rams[n]
			log(f"--- Optimizing n={n} RAM ({len(ram.neurons)} neurons × {bits_per_neuron} bits) ---")

			conn_tensor = torch.tensor(all_connectivities[ram_idx], dtype=torch.long)
			num_neurons = len(ram.neurons)
			total_bits = ram.total_bits

			# Create batch evaluation function using PERPLEXITY (lower is better)
			# eval_tokens: tokens to evaluate on (test_tokens for optimization, val_tokens_for_eval for validation)
			def create_perplexity_batch_fn(ram_idx, target_n, exact_probs_ref, eval_size, vocab_size_ref, cascade_threshold_ref, pop_size, eval_tokens):
				eval_count = [0]
				global_best = [float('inf')]  # Track best perplexity (lower is better)
				def batch_eval(candidates, total_pop=None):
					# total_pop: actual population size (including cached elites)
					# len(candidates): new candidates being evaluated this batch
					actual_total_pop = total_pop if total_pop is not None else len(candidates)
					if has_perplexity:
						import time as time_module
						start = time_module.time()
						current_all_conns = all_connectivities
						candidate_lists = [c.tolist() if hasattr(c, 'tolist') else c for c in candidates]
						perplexities = ram_accelerator.evaluate_fullnetwork_perplexity_batch_cpu(
							current_all_conns, candidate_lists, ram_idx,
							word_to_bits, train_tokens, eval_tokens,
							exact_probs_ref, eval_size, vocab_size_ref,
							cascade_threshold_ref
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_ppl = min(perplexities)
						if batch_best_ppl < global_best[0]:
							global_best[0] = batch_best_ppl
						# Log every 5th batch
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {actual_total_pop} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch PPL: {batch_best_ppl:.1f}, global PPL: {global_best[0]:.1f}"
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
							word_to_bits, train_tokens, eval_tokens,
							exact_results_fallback, eval_size
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_err = min(errors)
						batch_best_acc = (1 - batch_best_err) * 100
						if batch_best_err < global_best[0]:
							global_best[0] = batch_best_err
						global_best_acc = (1 - global_best[0]) * 100
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {actual_total_pop} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch: {batch_best_acc:.2f}%, global: {global_best_acc:.2f}%"
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
							word_to_bits, train_tokens, eval_tokens, eval_size
						)
						elapsed = time_module.time() - start
						eval_count[0] += 1
						batch_best_err = min(errors)
						batch_best_acc = (1 - batch_best_err) * 100
						if batch_best_err < global_best[0]:
							global_best[0] = batch_best_err
						global_best_acc = (1 - global_best[0]) * 100
						if eval_count[0] % 5 == 1:
							msg = f"[Rust n={target_n}] Batch {eval_count[0]}: {actual_total_pop} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch: {batch_best_acc:.2f}%, global: {global_best_acc:.2f}%"
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
								eval_tokens, exact_probs_ref, vocab_size_ref,
								cascade_threshold_ref, eval_size
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
							msg = f"[Python n={target_n}] Batch {eval_count[0]}: {actual_total_pop} total population ({len(candidates)} new) | {elapsed*1000:.0f}ms | batch PPL: {batch_best_ppl:.1f}, global PPL: {global_best[0]:.1f}"
							log(msg)
						return perplexities
				return batch_eval

			batch_fn = create_perplexity_batch_fn(ram_idx, n, exact_probs_train, eval_total, vocab_size, self.cascade_threshold, ga_pop, test_tokens)

			# Create validation batch function for overfitting detection DURING optimization
			# Uses WikiText-2 VALIDATION split (different articles from train) for true generalization check
			val_batch_fn = create_perplexity_batch_fn(ram_idx, n, exact_probs_val, val_eval_size, vocab_size, self.cascade_threshold, 1, val_tokens_for_eval)

			# Initial perplexity (using same code path as optimization)
			initial_ppls = batch_fn([conn_tensor])
			initial_ppl = initial_ppls[0]

			# Calculate initial validation PPL for baseline
			initial_val_ppls = val_batch_fn([conn_tensor])
			initial_val_ppl = initial_val_ppls[0]

			# For first RAM (n=6), set global baseline ratio
			is_first_ram = (n == max(n_values))
			if is_first_ram and global_baseline_ratio is None:
				global_baseline_ratio = initial_val_ppl / initial_ppl if initial_ppl > 0 else 1.0
				log(f"  Initial PERPLEXITY: train={initial_ppl:.1f}, val={initial_val_ppl:.1f}")
				log(f"  Global baseline ratio: {global_baseline_ratio:.2f}x (val/train)")
				log(f"  Overfitting thresholds: <{HEALTHY_THRESHOLD}% healthy, >{WARNING_THRESHOLD}% warn, >{CRITICAL_THRESHOLD}% stop")
				log("")
			else:
				# VERIFICATION: This should match previous RAM's final perplexity!
				log(f"  Initial PERPLEXITY: {initial_ppl:.1f} (should match prev final)")

			# Overfitting monitor: BASELINE-RELATIVE gap detection with hysteresis
			# Monitors val/train RATIO CHANGE (delta) from GLOBAL baseline (not per-RAM)
			# This works when train/val have different distributions (1000%+ absolute gaps)
			# Uses thresholds from base.py (single source of truth):
			# - delta < HEALTHY_THRESHOLD (-5%):  Improving → exit diversity mode
			# - delta > WARNING_THRESHOLD (0%):   Any worsening → activate diversity mode
			# - delta > CRITICAL_THRESHOLD (20%): Severe → early stop (after grace period)
			overfitting_monitor = OverfittingMonitor(
				validation_fn=lambda conn: val_batch_fn([conn])[0],
				# Use default thresholds from base.py constants
				grace_checks=1,  # 1 check = 5 generations grace period
				logger=log,
				global_baseline_ratio=global_baseline_ratio,  # Use global baseline from first RAM
			)

			start_time = time.time()
			# Track best across all phases (lower perplexity is better)
			best_ppl = initial_ppl
			best_connectivity = conn_tensor.clone()
			current_connectivity = conn_tensor.clone()

			# Track if any optimizer early-stopped due to overfitting
			any_overfitting_stop = False

			# Run strategy sequence (e.g., ['GA', 'TS'] or ['TS', 'GA', 'GA'])
			for step_idx, strategy_type in enumerate(strategy_sequence):
				step_num = step_idx + 1
				# Reset monitor for each strategy so each gets its own grace period
				overfitting_monitor.reset()

				if strategy_type.upper() == 'GA':
					log(f"  [{step_num}/{len(strategy_sequence)} GA] Starting...")
					ga_config = GeneticAlgorithmConfig(
						population_size=ga_pop, generations=ga_gens,
						mutation_rate=0.01, crossover_rate=0.7, elitism=ga_elitism,
						early_stop_patience=early_stop_patience, early_stop_threshold_pct=early_stop_threshold_pct
					)
					ga = GeneticAlgorithmStrategy(config=ga_config, seed=42+n+step_idx, verbose=True, logger=log)
					result = ga.optimize(current_connectivity, lambda x: batch_fn([x])[0],
						total_bits, num_neurons, bits_per_neuron, batch_fn,
						overfitting_callback=overfitting_monitor)
					if result.early_stopped_overfitting:
						any_overfitting_stop = True
						log(f"  [{step_num}/{len(strategy_sequence)} GA] Done: PPL {result.final_error:.1f} (stopped: overfitting)")
					else:
						log(f"  [{step_num}/{len(strategy_sequence)} GA] Done: PPL {result.final_error:.1f}")
				elif strategy_type.upper() == 'TS':
					log(f"  [{step_num}/{len(strategy_sequence)} TS] Starting...")
					ts_config = TabuSearchConfig(
						iterations=ts_iters, neighbors_per_iter=ts_neighbors,
						early_stop_patience=early_stop_patience, early_stop_threshold_pct=early_stop_threshold_pct
					)
					ts = TabuSearchStrategy(config=ts_config, seed=42+n+step_idx, verbose=True, logger=log)
					result = ts.optimize(current_connectivity, lambda x: batch_fn([x])[0],
						total_bits, num_neurons, bits_per_neuron, batch_fn,
						overfitting_callback=overfitting_monitor)
					if result.early_stopped_overfitting:
						any_overfitting_stop = True
						log(f"  [{step_num}/{len(strategy_sequence)} TS] Done: PPL {result.final_error:.1f} (stopped: overfitting)")
					else:
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

			# VALIDATION: Final evaluation on held-out validation set
			# (val_batch_fn already created earlier for callback)
			val_ppls = val_batch_fn([best_connectivity])
			val_ppl = val_ppls[0]
			current_ratio = val_ppl / final_ppl if final_ppl > 0 else 1.0
			ratio_delta_pct = ((current_ratio - global_baseline_ratio) / global_baseline_ratio * 100) if global_baseline_ratio and global_baseline_ratio > 0 else 0
			overfit_note = " (early stopped)" if any_overfitting_stop else ""

			# Status indicator based on ratio delta
			if ratio_delta_pct > 20:
				status = "🔴"
			elif ratio_delta_pct > 10:
				status = "🟡"
			elif ratio_delta_pct > 5:
				status = "🟠"
			else:
				status = "🟢"
			log(f"  [VALIDATION] train={final_ppl:.1f}, val={val_ppl:.1f}, ratio={current_ratio:.2f}x (Δ={ratio_delta_pct:+.1f}% vs baseline {global_baseline_ratio:.2f}x) {status}{overfit_note}")

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

			log(f"  n={n}: Train PPL {initial_ppl:.1f} → {final_ppl:.1f} "
				f"({ppl_improvement:+.1f}%), Val PPL: {val_ppl:.1f}, in {elapsed:.1f}s")

		total_elapsed = time.time() - total_start
		log_separator()
		log(f"★ All RAMs optimized in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
		log(f"★ Final Train PPL: {final_ppl:.1f}, Final Val PPL: {val_ppl:.1f}")
		log_separator()
		self._best_strategy = strategy_name

		# Store final optimization PPL for reporting (this is on held-out training data)
		self._optimization_final_ppl = final_ppl
		self._optimization_val_ppl = val_ppl  # Store validation PPL for overfitting analysis

	def _evaluate_perplexity_python(self, test_tokens: list[str], exact_probs: list,
									vocab_size: int, cascade_threshold: float, eval_subset: int) -> float:
		"""Python fallback for perplexity evaluation (matches Rust logic)."""
		import math

		min_prob = 1.0 / vocab_size
		total_log_prob = 0.0
		n_values = list(range(2, MAX_NGRAM + 1))  # [2, 3, 4, 5, 6] for MAX_NGRAM=6

		total = min(eval_subset, len(test_tokens) - MAX_NGRAM)
		for i in range(total):
			target = test_tokens[i + MAX_NGRAM]

			# Use pre-computed exact prob if available
			if exact_probs[i] is not None:
				prob = max(exact_probs[i], min_prob)
			else:
				# Try cascade prediction (highest n first)
				found_prob = None
				for n in reversed(n_values):
					if n not in self.generalized_rams:
						continue
					context = test_tokens[max(0, i + MAX_NGRAM - n):i + MAX_NGRAM]
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
						context = test_tokens[max(0, i + MAX_NGRAM - n):i + MAX_NGRAM]
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

		# 3. Fallback: vote across all scales (with optional attention weighting)
		votes = Counter()

		# Get attention weights if enabled
		if self.attention is not None:
			attn_weights = self.attention.get_weights(context[-6:])
			# Average attention weight for context gives importance multiplier
			attn_multiplier = float(attn_weights.mean()) * len(attn_weights)
		else:
			attn_multiplier = 1.0

		for n, ram in self.generalized_rams.items():
			if len(context) >= n:
				pred, conf = ram.predict(context)
				if pred:
					# Weight by context length and attention importance
					weight = conf * n * attn_multiplier
					votes[pred] += weight

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

	def _export_rams_for_rust(self) -> tuple:
		"""
		Export RAM data in Rust-compatible format for batch prediction.

		Returns:
			(generalized_rams, exact_rams, word_to_bits) tuple where:
			- generalized_rams: List of (n, connectivity, memory_per_neuron)
			  - connectivity: List of bit indices per neuron
			  - memory: List of {address_u64: {word: count}} per neuron
			- exact_rams: List of (n, contexts)
			  - contexts: {context_bits_as_tuple: {word: count}}
			- word_to_bits: {word: bits_as_u64}
		"""
		def bits_tuple_to_u64(bits: tuple) -> int:
			"""Convert tuple of bits to packed u64."""
			result = 0
			for i, bit in enumerate(bits):
				if bit:
					result |= (1 << i)
			return result

		# Export generalized RAMs
		gen_rams_export = []
		word_to_bits = {}

		# Helper to encode a word to bits (uses learned encoder or hash fallback)
		def encode_word(word: str) -> int:
			"""Convert word to 12-bit code using encoder or hash fallback."""
			if self.binary_encoder is not None:
				# Use learned encoder (returns 12-bit code directly)
				return self.binary_encoder.encode(word)
			else:
				# Hash fallback: 7 cluster bits + 5 hash bits = 12 bits
				h = hash(word)
				cluster = h % 128  # 7 bits for cluster
				bits = cluster  # bits 0-6
				for i in range(5):  # bits 7-11
					if (h >> (i + 7)) & 1:
						bits |= (1 << (7 + i))
				return bits

		for n in sorted(self.generalized_rams.keys()):
			ram = self.generalized_rams[n]

			# Build word_to_bits from the RAM's word_to_cluster mapping
			for word in ram.word_to_cluster.keys():
				if word not in word_to_bits:
					word_to_bits[word] = encode_word(word)

			# Export connectivity and memory for each neuron
			connectivity = []
			memory_per_neuron = []

			for neuron in ram.neurons:
				# Connectivity: list of bit indices
				connectivity.append(neuron.connected_bits)

				# Memory: convert tuple addresses to u64
				neuron_memory = {}
				for addr_tuple, word_counts in neuron.ram.items():
					addr_u64 = bits_tuple_to_u64(addr_tuple)
					# Convert Counter to regular dict
					neuron_memory[addr_u64] = dict(word_counts)

				memory_per_neuron.append(neuron_memory)

			gen_rams_export.append((n, connectivity, memory_per_neuron))

		# Export exact RAMs
		exact_rams_export = []
		for n in sorted(self.exact_rams.keys()):
			contexts = {}
			for ctx_words, word_counts in self.exact_rams[n].items():
				# Convert context words to bits and pack
				ctx_bits = []
				for word in ctx_words:
					if word not in word_to_bits:
						word_to_bits[word] = encode_word(word)
					ctx_bits.append(word_to_bits[word])
				contexts[tuple(ctx_bits)] = dict(word_counts)
			exact_rams_export.append((n, contexts))

		return gen_rams_export, exact_rams_export, word_to_bits

	def _precompute_all_predictions(self, tokens: list[str], n_workers: int = None) -> list[dict]:
		"""
		Pre-compute predictions for ALL positions in tokens using parallel workers.

		This is a key acceleration: instead of calling _get_all_predictions()
		245k times sequentially, we batch them and process in parallel.

		Uses Rust accelerator when available for maximum performance (rayon CPU parallelism).

		Returns: List of dicts, one per position (len = len(tokens) - MAX_NGRAM)
		"""
		if n_workers is None:
			n_workers = get_effective_cores(ACCELERATION_MODE)

		n_positions = len(tokens) - MAX_NGRAM
		if n_positions <= 0:
			return []

		# Try Rust acceleration first (much faster for batch prediction)
		if RUST_AVAILABLE and n_positions > 1000:
			try:
				return self._precompute_with_rust(tokens)
			except Exception as e:
				# Fall back to Python if Rust fails
				import sys
				print(f"  [Rust prediction failed: {e}, using Python]", file=sys.stderr)

		# Python fallback with joblib parallelism
		# Use MAX_NGRAM-token contexts to predict token at i+MAX_NGRAM
		contexts = [tokens[i:i + MAX_NGRAM] for i in range(n_positions)]
		chunk_size = max(1000, n_positions // n_workers)

		def process_chunk(start_idx: int, end_idx: int) -> list[dict]:
			"""Process a chunk of contexts."""
			results = []
			for i in range(start_idx, end_idx):
				results.append(self._get_all_predictions(contexts[i]))
			return results

		# Split into chunks
		chunks = []
		for start in range(0, n_positions, chunk_size):
			end = min(start + chunk_size, n_positions)
			chunks.append((start, end))

		# Use joblib for parallel processing (if worthwhile)
		if n_workers > 1 and n_positions > 10000:
			# Parallel processing for large datasets
			results_by_chunk = Parallel(n_jobs=n_workers, prefer="threads")(
				delayed(process_chunk)(start, end) for start, end in chunks
			)
			# Flatten results
			all_predictions = []
			for chunk_results in results_by_chunk:
				all_predictions.extend(chunk_results)
		else:
			# Sequential for small datasets (avoid overhead)
			all_predictions = []
			for start, end in chunks:
				all_predictions.extend(process_chunk(start, end))

		return all_predictions

	def _precompute_with_rust(self, tokens: list[str]) -> list[dict]:
		"""
		Use Rust accelerator for batch prediction (rayon parallel).

		This is significantly faster than Python/joblib for large datasets
		because Rust uses efficient FxHashMap and parallel iteration.
		"""
		# Export RAM data to Rust format
		gen_rams, exact_rams, word_to_bits = self._export_rams_for_rust()

		# Call appropriate Rust function based on acceleration mode
		if ACCELERATION_MODE == AccelerationMode.METAL and METAL_AVAILABLE:
			results = ram_accelerator.predict_all_batch_metal(
				gen_rams, exact_rams, word_to_bits, tokens
			)
		elif ACCELERATION_MODE == AccelerationMode.HYBRID and METAL_AVAILABLE:
			results = ram_accelerator.predict_all_batch_hybrid(
				gen_rams, exact_rams, word_to_bits, tokens
			)
		else:
			results = ram_accelerator.predict_all_batch_cpu(
				gen_rams, exact_rams, word_to_bits, tokens
			)

		return results

	def _compute_target_probability(self, context: list[str], target: str) -> float:
		"""
		Compute P(target) using the cascade prediction logic.

		This matches the logic in _evaluate_perplexity_python for consistent PPL.
		Uses:
		1. Exact RAM count-based probability if available
		2. Generalized RAM confidence if cascade match (with optional smoothing interpolation)
		3. Vote-weighted probability if voting fallback
		4. Smoothing fallback for unseen contexts (if enabled)

		IMPORTANT: This gives partial credit to targets that received some
		probability mass, even if they didn't win the argmax prediction.
		"""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4
		n_values = [2, 3, 4, 5, 6]

		# 1. Try exact match first (count-based probability)
		for n in sorted(self.exact_rams.keys(), reverse=True):
			if len(context) >= n:
				ctx = tuple(context[-n:])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					# Check if prediction is confident enough
					if counts.most_common(1)[0][1] / total > 0.2 or total >= 3:
						# Return P(target) from count distribution
						target_count = counts.get(target, 0)
						if target_count > 0:
							return max(target_count / total, min_prob)
						# Target not in exact counts - fall through to other methods

		# 2. Try generalized RAMs with cascade
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			if n not in self.generalized_rams:
				continue
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred and conf > self.cascade_threshold:
					# Cascade match - if correct, use confidence
					if pred == target:
						return max(float(conf), min_prob)
					else:
						# Wrong prediction - use smoothing if available
						if self.smoothing is not None:
							ctx_tuple = tuple(context[-5:]) if len(context) >= 5 else tuple(context)
							return self.smoothing.probability(target, ctx_tuple)
						return min_prob

		# 3. Voting fallback - aggregate votes and compute P(target)
		votes = {}
		total_weight = 0.0
		for n in n_values:
			if n not in self.generalized_rams:
				continue
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred:
					weight = conf * n
					votes[pred] = votes.get(pred, 0.0) + weight
					total_weight += weight

		if total_weight > 0:
			# Return P(target) from vote distribution (partial credit!)
			target_votes = votes.get(target, 0.0)
			if target_votes > 0:
				return max(target_votes / total_weight, min_prob)

		# 4. Smoothing fallback for unseen contexts
		if self.smoothing is not None:
			ctx_tuple = tuple(context[-5:]) if len(context) >= 5 else tuple(context)
			return self.smoothing.probability(target, ctx_tuple)

		return min_prob

	def _predict_cascade_from_cached(self, predictions: dict) -> tuple[str, str, float]:
		"""
		Cascade prediction using CACHED predictions (from _precompute_all_predictions).

		Replicates the logic of predict() but uses cached data instead of fresh lookups.
		This is 100x+ faster when evaluating 287k positions.

		Returns: (prediction, method, confidence)
		"""
		# 1. Try exact match first (highest n to lowest)
		for n in sorted(self.exact_rams.keys(), reverse=True):
			key = f"exact_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				# Exact match uses conf > 0.2 or total >= 3 (encoded in caching)
				if conf > 0.2:
					return pred, key, conf

		# 2. Try generalized RAMs with cascade threshold
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			key = f"gen_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if pred and conf > self.cascade_threshold:
					return pred, key, conf

		# 3. Voting fallback
		votes = Counter()
		for method, (pred, conf) in predictions.items():
			if pred:
				n = int(method.split("_n")[1]) if "_n" in method else 1
				weight = conf * n
				votes[pred] += weight

		if votes:
			best, score = votes.most_common(1)[0]
			total_weight = sum(votes.values())
			return best, "vote", score / total_weight if total_weight > 0 else 0.0

		return "<UNK>", "none", 0.0

	def _compute_target_probability_from_cached(self, predictions: dict, target: str) -> float:
		"""
		Compute P(target) using CACHED predictions (from _precompute_all_predictions).

		Replicates the logic of _compute_target_probability() but uses cached data.
		This is 100x+ faster when evaluating 287k positions.

		NOTE: This is an approximation for exact RAMs - we only cache the winner's
		confidence, not the full count distribution. For cascade evaluation this
		is acceptable since the probability model is:
		- If winner == target: P(target) = confidence
		- If winner != target: P(target) = min_prob (or smoothing)
		"""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4

		# 1. Try exact match first (highest n to lowest)
		for n in sorted(self.exact_rams.keys(), reverse=True):
			key = f"exact_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if conf > 0.2:  # Confident exact match
					if pred == target:
						return max(conf, min_prob)
					# Target not the winner - try smoothing or continue cascade
					break

		# 2. Try generalized RAMs with cascade
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			key = f"gen_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if pred and conf > self.cascade_threshold:
					if pred == target:
						return max(float(conf), min_prob)
					# Wrong prediction - return min_prob (matches cascade behavior)
					return min_prob

		# 3. Voting fallback - aggregate votes and compute P(target)
		votes = {}
		total_weight = 0.0
		for method, (pred, conf) in predictions.items():
			if pred:
				n = int(method.split("_n")[1]) if "_n" in method else 1
				weight = conf * n
				votes[pred] = votes.get(pred, 0.0) + weight
				total_weight += weight

		if total_weight > 0:
			target_votes = votes.get(target, 0.0)
			if target_votes > 0:
				return max(target_votes / total_weight, min_prob)

		return min_prob

	def learn_voting_weights(self, tokens: list[str], validation_split: float = 0.2):
		"""
		Strategy #1: Learn voting weights based on observed accuracy.

		For each method, track how often it's correct when it makes a prediction.
		Use these accuracies as weights in voting.

		ACCELERATED: Uses parallel pre-computation of predictions.
		"""
		# Use last portion as validation
		val_start = int(len(tokens) * (1 - validation_split))
		val_tokens = tokens[val_start:]

		# Pre-compute all predictions in parallel (key acceleration!)
		all_predictions = self._precompute_all_predictions(val_tokens)

		# Track per-method accuracy (fast aggregation over cached predictions)
		method_stats = defaultdict(lambda: {"correct": 0, "total": 0})

		for i, predictions in enumerate(all_predictions):
			target = val_tokens[i + 5]

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

		ACCELERATED: Uses parallel pre-computation of predictions.
		"""
		# Use last portion as validation/training for meta-classifier
		val_start = int(len(tokens) * (1 - validation_split))
		val_tokens = tokens[val_start:]

		# Pre-compute all predictions in parallel (key acceleration!)
		all_predictions = self._precompute_all_predictions(val_tokens)

		# Collect training data for meta-classifier
		# Feature: method predictions encoded + confidences
		# Target: correct answer

		# First pass: collect all unique predictions to build vocabulary
		all_methods = sorted(set(
			list(f"exact_n{n}" for n in self.exact_rams.keys()) +
			list(f"gen_n{n}" for n in self.generalized_rams.keys())
		))
		self.meta_methods = all_methods

		# Collect (features, target) pairs using cached predictions
		training_data = []
		word_to_idx = {}

		for i, predictions in enumerate(all_predictions):
			target = val_tokens[i + 5]

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
			primary_method, primary_word_idx, _ = feature_parts[0]
			primary_word = self.meta_idx_to_word.get(primary_word_idx, "<UNK>")

			# Secondary agreement: do other methods agree?
			agreement = sum(1 for _, w, _ in feature_parts[1:] if w == primary_word_idx)

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
		agreement = sum(1 for _, (w, _) in sorted_preds[1:] if w == primary_word)

		# Look up reliability
		key = (primary_method, agreement)
		reliability = self.meta_reliability.get(key, 0.5)

		# If reliability is low and we have alternatives, consider them
		if reliability < 0.3 and len(sorted_preds) > 1:
			# Try second-best if it has higher agreement
			for alt_method, (alt_word, alt_conf) in sorted_preds[1:]:
				alt_agreement = sum(1 for _, (w, _) in sorted_preds if w == alt_word) - 1
				alt_key = (alt_method, alt_agreement)
				alt_reliability = self.meta_reliability.get(alt_key, 0.5)

				if alt_reliability > reliability and alt_conf > 0.2:
					return alt_word, f"meta({alt_method})", alt_reliability * alt_conf

		return primary_word, f"meta({primary_method})", reliability * primary_conf

	def train_ram_meta_classifier(self, tokens: list[str], n_neurons: int = 32,
								   bits_per_neuron: int = 12, state_history: int = 4,
								   validation_split: float = 0.2):
		"""
		Strategy #3: Train a RAM-based meta-classifier with STATE LAYER.

		Unlike the statistics-based meta-classifier, this uses actual RAM neurons
		with temporal context to learn complex patterns about which method to trust.

		The state layer tracks recent prediction outcomes, enabling patterns like:
		- "After gen_n3 was wrong twice, trust exact RAMs more"
		- "When in a run of correct predictions, keep trusting current method"

		ACCELERATED: Pre-computes predictions in parallel, then trains sequentially
		(state updates require sequential ordering).
		"""
		# Build list of all methods
		all_methods = sorted(set(
			list(f"exact_n{n}" for n in self.exact_rams.keys()) +
			list(f"gen_n{n}" for n in self.generalized_rams.keys())
		))

		# Create RAM meta-classifier with state layer
		self.ram_meta = RAMMetaClassifier(all_methods, n_neurons, bits_per_neuron, state_history)

		# Use last portion as validation/training
		val_start = int(len(tokens) * (1 - validation_split))
		val_tokens = tokens[val_start:]

		# Pre-compute all predictions in parallel (key acceleration!)
		all_predictions = self._precompute_all_predictions(val_tokens)

		# Reset state before training
		self.ram_meta.reset_state()

		# Training loop is sequential (state updates depend on order)
		trained_count = 0
		for i, predictions in enumerate(all_predictions):
			target = val_tokens[i + 5]

			if not predictions:
				continue

			# Sort predictions by confidence
			sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]

			# Find which method(s) predicted correctly
			correct_methods = [m for m, (w, _) in predictions.items() if w == target]

			if correct_methods:
				# Train: prefer exact over generalized, higher n over lower
				# Priority: exact_n6 > exact_n5 > ... > gen_n6 > gen_n5 > ...
				def method_priority(m):
					is_exact = m.startswith('exact')
					n = int(m.split('_n')[1])
					return (is_exact, n)

				best_correct = max(correct_methods, key=method_priority)
				self.ram_meta.train(sorted_preds, best_correct)
				trained_count += 1

				# Update state: best_correct method was correct
				self.ram_meta.update_state(best_correct, was_correct=True)
			else:
				# No method was correct - still update state with top method being wrong
				if sorted_preds:
					top_method = sorted_preds[0][0]
					self.ram_meta.update_state(top_method, was_correct=False)

		return trained_count

	def predict_ram_meta(self, context: list[str]) -> tuple[str, str, float]:
		"""
		Predict using the RAM-based meta-classifier.
		"""
		if not hasattr(self, 'ram_meta'):
			return self.predict(context)

		predictions = self._get_all_predictions(context)

		if not predictions:
			return "<UNK>", "none", 0.0

		# Sort by confidence
		sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]

		if not sorted_preds:
			return "<UNK>", "none", 0.0

		# Ask RAM meta-classifier which method to trust
		best_method, ram_confidence = self.ram_meta.predict(sorted_preds)

		# Find that method's prediction
		if best_method in predictions:
			word, conf = predictions[best_method]
			return word, f"ram_meta({best_method})", ram_confidence * conf

		# Fallback to highest confidence
		method, (word, conf) = sorted_preds[0]
		return word, f"ram_meta_fallback({method})", conf

	def _compute_exact_probs(self, tokens: list[str]) -> list:
		"""
		Pre-compute exact RAM probabilities for P(target) calculation.

		This gives the ACTUAL probability of the target word from exact RAM
		count distributions, not just winner confidence. Essential for
		accurate perplexity calculation.

		Returns: List of (exact_prob or None) for each position
		"""
		eval_size = len(tokens) - MAX_NGRAM
		exact_probs = []
		for i in range(eval_size):
			target = tokens[i + MAX_NGRAM]
			exact_prob = None
			for n in sorted(self.exact_rams.keys(), reverse=True):
				ctx = tuple(tokens[i + MAX_NGRAM - n:i + MAX_NGRAM])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					_, count = counts.most_common(1)[0]
					conf = count / total
					if conf > 0.2 or total >= 3:
						target_count = counts.get(target, 0)
						exact_prob = target_count / total if total > 0 else 0.0
						break
			exact_probs.append(exact_prob)
		return exact_probs

	def evaluate_voting_strategies(self, tokens: list[str], max_samples: int = None) -> dict:
		"""
		Compare all voting strategies (accuracy and perplexity).

		Args:
			tokens: List of tokens to evaluate on.
			max_samples: Maximum number of samples to evaluate. If None, uses all.
				Pass the same value as main evaluation's eval_subset for consistency.

		IMPORTANT: Uses PerplexityCalculator for consistent PPL calculation.
		For "cascade" strategy, uses exact_probs (actual P(target) from count
		distributions) to match the main evaluation's perplexity calculation.

		ACCELERATED: Pre-computes all predictions in parallel, then evaluates
		strategies using cached predictions. RAM_META still runs sequentially
		due to state updates, but predictions are cached.
		"""
		# Test multiple confidence thresholds
		thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

		# Get vocab size for perplexity calculation
		vocab_size = len(self.word_counts) if hasattr(self, 'word_counts') else 10000

		# Create PerplexityCalculators for each strategy
		calculators = {
			"cascade": PerplexityCalculator(vocab_size),
			"weighted": PerplexityCalculator(vocab_size),
			"meta": PerplexityCalculator(vocab_size),
			"ram_meta": PerplexityCalculator(vocab_size),
		}
		for t in thresholds:
			calculators[f"thresh_{t:.2f}"] = PerplexityCalculator(vocab_size)

		# Limit tokens if max_samples specified (for consistency with main eval)
		n_available = len(tokens) - MAX_NGRAM
		if max_samples is not None and max_samples < n_available:
			eval_tokens = tokens[:max_samples + MAX_NGRAM]  # Need MAX_NGRAM extra for context
		else:
			eval_tokens = tokens

		# Pre-compute exact_probs for accurate P(target) from exact RAMs
		# This is CRITICAL: cached predictions only have winner, not full distribution
		exact_probs = self._compute_exact_probs(eval_tokens)

		# Pre-compute all predictions in parallel (KEY ACCELERATION!)
		all_predictions = self._precompute_all_predictions(eval_tokens)

		# Reset RAM meta state before evaluation (fresh temporal context)
		if hasattr(self, 'ram_meta'):
			self.ram_meta.reset_state()

		min_prob = 1.0 / vocab_size

		# Evaluate each position using cached predictions
		for i, predictions in enumerate(all_predictions):
			context = eval_tokens[i:i + MAX_NGRAM]  # MAX_NGRAM-token context to predict i+MAX_NGRAM
			target = eval_tokens[i + MAX_NGRAM]

			# CASCADE: Use exact_probs when available (matches main eval exactly)
			if exact_probs[i] is not None:
				cascade_prob = max(exact_probs[i], min_prob)
			else:
				# Fall back to cached predictions for generalized RAM cascade
				cascade_prob = self._compute_target_probability_from_cached(predictions, target)
			pred1, _, _ = self._predict_cascade_from_cached(predictions)
			calculators["cascade"].add_from_probability(cascade_prob, is_correct=(pred1 == target))

			# WEIGHTED: Use weighted vote distribution for P(target)
			# Use cached predictions for weighted voting
			pred2 = self._predict_weighted_from_cached(predictions)
			weighted_prob = self._compute_weighted_prob_from_cached(predictions, target)
			calculators["weighted"].add_from_probability(weighted_prob, is_correct=(pred2 == target))

			# META: Meta-classifier - use cached cascade probability
			pred3 = self._predict_meta_from_cached(predictions)
			# Use cached cascade probability for partial credit (meta just picks a method)
			meta_prob = self._compute_target_probability_from_cached(predictions, target)
			calculators["meta"].add_from_probability(meta_prob, is_correct=(pred3 == target))

			# RAM_META: RAM-based meta-classifier with state tracking
			# Uses cached predictions + sequential state updates
			if hasattr(self, 'ram_meta'):
				pred4, method4, _ = self._predict_ram_meta_from_cached(predictions)
				is_correct = (pred4 == target)
				# Use cached cascade probability for partial credit
				ram_meta_prob = self._compute_target_probability_from_cached(predictions, target)
				calculators["ram_meta"].add_from_probability(ram_meta_prob, is_correct=is_correct)

				# Update state with actual outcome (enables temporal learning)
				if '(' in method4:
					used_method = method4.split('(')[1].rstrip(')')
				else:
					used_method = method4
				self.ram_meta.update_state(used_method, is_correct)

			# THRESHOLD: Use threshold-filtered vote distribution
			for t in thresholds:
				pred = self._predict_threshold_from_cached(predictions, context, t)
				thresh_prob = self._compute_threshold_prob_from_cached(predictions, context, target, t)
				calculators[f"thresh_{t:.2f}"].add_from_probability(thresh_prob, is_correct=(pred == target))

		# Collect results from all calculators
		results = {}
		for strategy, calc in calculators.items():
			stats = calc.get_stats()
			results[strategy] = {
				"correct": stats["correct"],
				"total": stats["total"],
				"accuracy": stats["accuracy"],
				"perplexity": stats["perplexity"],
			}

		return results

	def _predict_weighted_from_cached(self, predictions: dict) -> str:
		"""Predict using weighted voting from cached predictions."""
		if not predictions or not hasattr(self, 'voting_weights'):
			return "<UNK>"

		votes = Counter()
		for method, (pred, conf) in predictions.items():
			weight = self.voting_weights.get(method, 0.1) * conf
			votes[pred] += weight

		if votes:
			return votes.most_common(1)[0][0]
		return "<UNK>"

	def _compute_weighted_prob_from_cached(self, predictions: dict, target: str) -> float:
		"""Compute P(target) using weighted voting from cached predictions."""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4

		if not predictions or not hasattr(self, 'voting_weights'):
			return min_prob

		votes = {}
		total_weight = 0.0
		for method, (pred, conf) in predictions.items():
			weight = self.voting_weights.get(method, 0.1) * conf
			votes[pred] = votes.get(pred, 0.0) + weight
			total_weight += weight

		if total_weight > 0:
			return max(votes.get(target, 0.0) / total_weight, min_prob)
		return min_prob

	def _predict_meta_from_cached(self, predictions: dict) -> str:
		"""Predict using meta-classifier from cached predictions."""
		if not predictions or not hasattr(self, 'meta_reliability'):
			return "<UNK>"

		# Sort by confidence
		sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]
		if not sorted_preds:
			return "<UNK>"

		primary_method, (primary_word, primary_conf) = sorted_preds[0]

		# Count agreement
		primary_word_idx = self.meta_word_to_idx.get(primary_word, -1)
		agreement = 0
		if primary_word_idx >= 0:
			for method, (pred, _) in sorted_preds[1:]:
				if self.meta_word_to_idx.get(pred, -1) == primary_word_idx:
					agreement += 1

		return primary_word

	def _predict_ram_meta_from_cached(self, predictions: dict) -> tuple[str, str, float]:
		"""
		Predict using RAM-based meta-classifier from CACHED predictions.

		This is the accelerated version of predict_ram_meta() that uses
		pre-computed predictions instead of doing fresh lookups.
		"""
		if not predictions or not hasattr(self, 'ram_meta'):
			# Fallback to cascade from cached
			return self._predict_cascade_from_cached(predictions)

		# Sort by confidence (same as predict_ram_meta)
		sorted_preds = sorted(predictions.items(), key=lambda x: -x[1][1])[:4]
		if not sorted_preds:
			return "<UNK>", "none", 0.0

		# Ask RAM meta-classifier which method to trust
		best_method, ram_confidence = self.ram_meta.predict(sorted_preds)

		# Find that method's prediction
		if best_method in predictions:
			word, conf = predictions[best_method]
			return word, f"ram_meta({best_method})", ram_confidence * conf

		# Fallback to highest confidence
		method, (word, conf) = sorted_preds[0]
		return word, f"ram_meta_fallback({method})", conf

	def _predict_threshold_from_cached(self, predictions: dict, context: list[str], min_conf: float) -> str:
		"""Predict using threshold voting from cached predictions."""
		# 1. Try exact match first (from predictions dict)
		for n in sorted(self.exact_rams.keys(), reverse=True):
			key = f"exact_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if conf > 0.2:
					return pred

		# 2. Try generalized cascade
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			key = f"gen_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if conf > self.cascade_threshold:
					return pred

		# 3. Threshold-filtered voting
		votes = Counter()
		for method, (pred, conf) in predictions.items():
			if conf > min_conf:
				n = int(method.split('_n')[1]) if '_n' in method else 1
				weight = conf * n
				votes[pred] += weight

		if votes:
			return votes.most_common(1)[0][0]
		return "<UNK>"

	def _compute_threshold_prob_from_cached(self, predictions: dict, context: list[str], target: str, min_conf: float) -> float:
		"""Compute P(target) using threshold voting from cached predictions."""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4

		# 1. Try exact match first
		for n in sorted(self.exact_rams.keys(), reverse=True):
			if len(context) >= n:
				ctx = tuple(context[-n:])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					if counts.most_common(1)[0][1] / total > 0.2 or total >= 3:
						target_count = counts.get(target, 0)
						return max(target_count / total, min_prob)

		# 2. Try generalized cascade
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			key = f"gen_n{n}"
			if key in predictions:
				pred, conf = predictions[key]
				if conf > self.cascade_threshold:
					return max(float(conf), min_prob) if pred == target else min_prob

		# 3. Threshold-filtered voting
		votes = {}
		total_weight = 0.0
		for method, (pred, conf) in predictions.items():
			if conf > min_conf:
				n = int(method.split('_n')[1]) if '_n' in method else 1
				weight = conf * n
				votes[pred] = votes.get(pred, 0.0) + weight
				total_weight += weight

		if total_weight > 0:
			return max(votes.get(target, 0.0) / total_weight, min_prob)
		return min_prob

	def _compute_weighted_target_probability(self, context: list[str], target: str) -> float:
		"""Compute P(target) using weighted voting (with learned weights)."""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4

		if not hasattr(self, 'voting_weights') or not self.voting_weights:
			return self._compute_target_probability(context, target)

		predictions = self._get_all_predictions(context)
		if not predictions:
			return min_prob

		# Build weighted vote distribution
		votes = {}
		total_weight = 0.0
		for method, (pred, conf) in predictions.items():
			weight = self.voting_weights.get(method, 0.1) * conf
			votes[pred] = votes.get(pred, 0.0) + weight
			total_weight += weight

		if total_weight > 0:
			target_votes = votes.get(target, 0.0)
			return max(target_votes / total_weight, min_prob)

		return min_prob

	def _compute_threshold_target_probability(self, context: list[str], target: str, min_conf: float) -> float:
		"""Compute P(target) using threshold-filtered voting."""
		min_prob = 1.0 / len(self.word_counts) if hasattr(self, 'word_counts') else 1e-4

		# 1. Try exact match first
		for n in sorted(self.exact_rams.keys(), reverse=True):
			if len(context) >= n:
				ctx = tuple(context[-n:])
				if ctx in self.exact_rams[n]:
					counts = self.exact_rams[n][ctx]
					total = sum(counts.values())
					if counts.most_common(1)[0][1] / total > 0.2 or total >= 3:
						target_count = counts.get(target, 0)
						return max(target_count / total, min_prob)

		# 2. Try generalized cascade
		for n in sorted(self.generalized_rams.keys(), reverse=True):
			if len(context) >= n:
				pred, conf = self.generalized_rams[n].predict(context)
				if pred and conf > self.cascade_threshold:
					if pred == target:
						return max(float(conf), min_prob)
					else:
						return min_prob

		# 3. Threshold-filtered voting
		votes = {}
		total_weight = 0.0
		for n, ram in self.generalized_rams.items():
			if len(context) >= n:
				pred, conf = ram.predict(context)
				if pred and conf > min_conf:
					weight = conf * n
					votes[pred] = votes.get(pred, 0.0) + weight
					total_weight += weight

		if total_weight > 0:
			target_votes = votes.get(target, 0.0)
			return max(target_votes / total_weight, min_prob)

		return min_prob

	def evaluate(self, tokens: list[str]) -> dict:
		"""
		Evaluate the model using CACHED predictions for acceleration.

		This method now uses _precompute_all_predictions() for 100x+ speedup
		on large datasets (214k tokens evaluates in seconds, not hours).
		"""
		self.reset_prediction_stats()  # Reset for fresh stats

		results = {
			"correct": 0,
			"total": 0,
			"by_method": defaultdict(lambda: {"correct": 0, "total": 0})
		}

		# Pre-compute all predictions in parallel (KEY ACCELERATION!)
		# Uses Rust batch prediction for massive speedup
		all_predictions = self._precompute_all_predictions(tokens)

		for i, predictions in enumerate(all_predictions):
			target = tokens[i + MAX_NGRAM]  # Context is tokens[i:i+MAX_NGRAM], target is i+MAX_NGRAM

			# Use cached prediction instead of fresh lookup
			pred, method, _ = self._predict_cascade_from_cached(predictions)

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
	tokenizer_type: TokenizerType = TokenizerType.WORD,
	n_neurons: int = None,
	bits_per_neuron: int = None,
	cascade_threshold: float = 0.1,
	strategy_sequence: list = None,
	full_data: bool = False,
	smoothing_type: str = "none",
	use_lsh: bool = False,
	lsh_type: str = "simhash",
	attention_type: AttentionType = AttentionType.NONE,
	representation_type: RepresentationType = RepresentationType.COOCCURRENCE,
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

	# Load data - reduced for fast mode, full for --full-data
	if full_data:
		train_limit = None  # Use all ~2M tokens
		test_limit = None   # Use all ~245k test, ~214k validation
		log("Using FULL WikiText-2 dataset (~2M train, ~245k test, ~214k validation)")
	else:
		train_limit = 30000 if mode == BenchmarkMode.FAST else 150000
		test_limit = 5000 if mode == BenchmarkMode.FAST else 20000

	try:
		from datasets import load_dataset

		log("Loading WikiText-2...")
		log(f"Tokenizer: {tokenizer_type.name} (for comparable perplexity)")
		dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

		# Get tokenizer based on parameter
		tokenize, tokenizer_instance = get_tokenizer(tokenizer_type)

		train_text = " ".join(dataset["train"]["text"])
		test_text = " ".join(dataset["test"]["text"])
		validation_text = " ".join(dataset["validation"]["text"])

		# For trainable tokenizers (BPE, CHARACTER), train on the corpus first
		if tokenizer_instance is not None and hasattr(tokenizer_instance, 'is_trainable') and tokenizer_instance.is_trainable:
			if not tokenizer_instance.is_trained:
				log(f"Training {tokenizer_type.name} tokenizer on corpus...")
				# Train on training text only (not test/val to avoid data leakage)
				tokenizer_instance.train([train_text])
				log(f"  Vocabulary size: {tokenizer_instance.vocab_size:,}")

		# Tokenize full corpora
		all_train_tokens = tokenize(train_text)
		all_test_tokens = tokenize(test_text)
		all_validation_tokens = tokenize(validation_text)

		# Helper function for multi-window sampling
		def sample_windows(all_tokens, total_tokens, window_size, seed, name):
			"""Sample N windows from corpus to reduce topical bias."""
			import random as rand_module
			num_windows = total_tokens // window_size
			corpus_end = len(all_tokens) - window_size - 10
			if corpus_end > window_size * num_windows:
				rng = rand_module.Random(seed)
				starts = rng.sample(range(0, corpus_end), num_windows)
				starts.sort()
				tokens = []
				for start in starts:
					tokens.extend(all_tokens[start:start + window_size])
				log(f"{name}: {num_windows} windows × {window_size:,} = {len(tokens):,} tokens (multi-window)")
				return tokens
			else:
				tokens = all_tokens[:total_tokens]
				log(f"{name}: {len(tokens):,} tokens (single-window fallback)")
				return tokens

		# Data sampling strategy
		if full_data:
			# Use all tokens directly (no sampling)
			train_tokens = all_train_tokens
			test_tokens = all_test_tokens
			validation_tokens = all_validation_tokens
			log(f"Train: {len(train_tokens):,} tokens (full dataset)")
			log(f"Test: {len(test_tokens):,} tokens (full dataset)")
			log(f"Validation: {len(validation_tokens):,} tokens (full dataset)")
		else:
			# Multi-window sampling for all splits - reduces topical bias
			# Each split uses different seed for independence
			train_tokens = sample_windows(all_train_tokens, train_limit, 10000, seed=123, name="Train")
			test_tokens = sample_windows(all_test_tokens, test_limit, 2000, seed=456, name="Test")
			validation_tokens = sample_windows(all_validation_tokens, test_limit, 2000, seed=789, name="Validation")

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
		fallback_train = train_limit if train_limit else 100000
		fallback_test = test_limit if test_limit else 10000
		train_tokens = [random.choice(words) for _ in range(fallback_train)]
		test_tokens = [random.choice(words) for _ in range(fallback_test)]
		validation_tokens = [random.choice(words) for _ in range(fallback_test)]

	# Train model
	log_separator()
	model = RAMLM_v2(freq_threshold=30, mode=mode, n_neurons=n_neurons, bits_per_neuron=bits_per_neuron,
					 cascade_threshold=cascade_threshold, strategy_sequence=strategy_sequence,
					 smoothing_type=smoothing_type, use_lsh=use_lsh, lsh_type=lsh_type,
					 attention_type=attention_type, representation_type=representation_type)
	log(f"Model config: n_neurons={model.n_neurons}, bits_per_neuron={model.bits_per_neuron}, cascade_threshold={model.cascade_threshold}")
	if smoothing_type != "none":
		log(f"Smoothing: {smoothing_type}")
	if attention_type != AttentionType.NONE:
		log(f"Attention: {attention_type.name.lower()}")
	log(f"Representation: {representation_type.name.lower()}")
	if use_lsh:
		log(f"LSH: {lsh_type} (similarity-preserving context hashing)")
	log(f"Strategy sequence: {' → '.join(strategy_sequence)}")

	start = time.time()
	model.train(train_tokens, optimize_connectivity=True, final_test_tokens=test_tokens, validation_tokens=validation_tokens)
	train_time = time.time() - start
	log(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

	# Evaluate
	log_separator()
	log("EVALUATION")
	log_separator()

	results = model.evaluate(test_tokens)
	accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0

	# Calculate perplexity on FULL test set (no sampling - use cached data)
	# exact_probs don't change during optimization (exact RAMs are dict-based)
	# so we reuse the cached version from pre-optimization baseline
	eval_subset = len(test_tokens) - MAX_NGRAM  # Use all positions
	exact_probs = getattr(model, '_cached_test_exact_probs', None)
	if exact_probs is None:
		# Fallback: compute if not cached (shouldn't happen in normal flow)
		log("Computing exact_probs (not cached)...")
		exact_probs = []
		for i in range(eval_subset):
			target = test_tokens[i + MAX_NGRAM]
			exact_prob = None
			for n in sorted(model.exact_rams.keys(), reverse=True):
				ctx = tuple(test_tokens[i + MAX_NGRAM - n:i + MAX_NGRAM])
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
	# Report all PPLs for clarity
	optimization_ppl = getattr(model, '_optimization_final_ppl', None)
	optimization_val_ppl = getattr(model, '_optimization_val_ppl', None)
	pre_test_ppl = getattr(model, '_pre_optimization_test_ppl', None)
	pre_val_ppl = getattr(model, '_pre_optimization_val_ppl', None)
	log("Perplexity Summary:")
	if pre_test_ppl:
		log(f"  Test (before opt):   {pre_test_ppl:.1f}")
	if pre_val_ppl:
		log(f"  Val (before opt):    {pre_val_ppl:.1f}")
	if optimization_ppl:
		log(f"  Train (after opt):   {optimization_ppl:.1f}  ← optimized for this")
	if optimization_val_ppl:
		log(f"  Val (after opt):     {optimization_val_ppl:.1f}  ← overfitting check")
	log(f"  Test (after opt):    {perplexity:.1f}  ← generalization")
	# Show improvement/degradation
	if pre_test_ppl:
		test_change = perplexity - pre_test_ppl
		test_change_pct = (test_change / pre_test_ppl * 100) if pre_test_ppl > 0 else 0
		if test_change < 0:
			log(f"★ Test PPL IMPROVED by {-test_change:.1f} ({-test_change_pct:.1f}%) - optimization helps generalization!")
		else:
			log(f"⚠️ Test PPL DEGRADED by {test_change:.1f} (+{test_change_pct:.1f}%) - optimization hurts generalization")
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

	log("Training RAM meta-classifier (Strategy #3)...")
	# 32 neurons × 10 bits for 26-bit input space (pred features + agreement)
	ram_meta_trained = model.train_ram_meta_classifier(train_tokens, n_neurons=32, bits_per_neuron=10)
	log(f"  Trained on {ram_meta_trained} examples (32 neurons × 10 bits)")

	# Evaluate voting strategies on VALIDATION set (for overfitting comparison)
	log("Evaluating all strategies on validation set...")
	val_voting_results = model.evaluate_voting_strategies(validation_tokens)

	# Get validation PPLs
	val_ppl_cascade = val_voting_results['cascade']['perplexity']
	val_ppl_weighted = val_voting_results['weighted']['perplexity']
	val_ppl_meta = val_voting_results['meta']['perplexity']
	val_ppl_ram_meta = val_voting_results['ram_meta']['perplexity']

	# Evaluate voting strategies on TEST set (for generalization)
	log("Evaluating all strategies on test set...")
	voting_results = model.evaluate_voting_strategies(test_tokens)

	# Get test PPLs
	ppl_cascade = voting_results['cascade']['perplexity']
	ppl_weighted = voting_results['weighted']['perplexity']
	ppl_meta = voting_results['meta']['perplexity']
	ppl_ram_meta = voting_results['ram_meta']['perplexity']

	# Get accuracies for table (from test set)
	acc_cascade = voting_results['cascade']['accuracy'] * 100
	acc_weighted = voting_results['weighted']['accuracy'] * 100
	acc_meta = voting_results['meta']['accuracy'] * 100
	acc_ram_meta = voting_results['ram_meta']['accuracy'] * 100

	log("")
	log("RESULTS SUMMARY (table format):")
	log("─" * 85)
	log(f"{'Metric':<35} {'Cascade':>10} {'Vote #1':>10} {'Vote #2':>10} {'Vote #3':>10}")
	log("─" * 85)
	log(f"{'Accuracy (%):':<35} {acc_cascade:>10.2f} {acc_weighted:>10.2f} {acc_meta:>10.2f} {acc_ram_meta:>10.2f}")
	if optimization_ppl:
		log(f"{'Optimization PPL (train):':<35} {optimization_ppl:>10.1f} {'-':>10} {'-':>10} {'-':>10}")
	if pre_val_ppl:
		log(f"{'Val PPL (before optimization):':<35} {pre_val_ppl:>10.1f} {'-':>10} {'-':>10} {'-':>10}")
	log(f"{'Val PPL (after optimization):':<35} {val_ppl_cascade:>10.1f} {val_ppl_weighted:>10.1f} {val_ppl_meta:>10.1f} {val_ppl_ram_meta:>10.1f}")
	if pre_test_ppl:
		log(f"{'Test PPL (before optimization):':<35} {pre_test_ppl:>10.1f} {'-':>10} {'-':>10} {'-':>10}")
	log(f"{'Test PPL (after optimization):':<35} {ppl_cascade:>10.1f} {ppl_weighted:>10.1f} {ppl_meta:>10.1f} {ppl_ram_meta:>10.1f}")
	log("─" * 85)

	# Find best strategy by perplexity (lower is better)
	main_strategies = ['cascade', 'weighted', 'meta', 'ram_meta']
	strategy_names = {'cascade': 'Cascade', 'weighted': 'Vote #1', 'meta': 'Vote #2', 'ram_meta': 'Vote #3'}

	best_ppl_strategy = min(main_strategies, key=lambda x: voting_results[x]['perplexity'])
	best_ppl = voting_results[best_ppl_strategy]['perplexity']
	ppl_improvement = (ppl_cascade - best_ppl) / ppl_cascade * 100 if ppl_cascade > 0 else 0

	best_acc_strategy = max(main_strategies, key=lambda x: voting_results[x]['accuracy'])
	best_acc = voting_results[best_acc_strategy]['accuracy'] * 100
	acc_improvement = (best_acc - acc_cascade) / acc_cascade * 100 if acc_cascade > 0 else 0

	log("")
	if ppl_improvement > 0.1:
		log(f"★ Best PPL: {strategy_names[best_ppl_strategy]} ({best_ppl:.1f}, -{ppl_improvement:.1f}% vs Cascade)")
	else:
		log(f"★ Best PPL: Cascade ({ppl_cascade:.1f}, baseline)")

	if acc_improvement > 0.1:
		log(f"★ Best ACC: {strategy_names[best_acc_strategy]} ({best_acc:.2f}%, +{acc_improvement:.1f}% vs Cascade)")
	else:
		log(f"★ Best ACC: Cascade ({acc_cascade:.2f}%, baseline)")

	# Show test PPL improvement if we have before/after
	if pre_test_ppl:
		test_change = ppl_cascade - pre_test_ppl
		test_change_pct = (test_change / pre_test_ppl * 100) if pre_test_ppl > 0 else 0
		if test_change < 0:
			log(f"★ Optimization IMPROVED test PPL by {-test_change:.1f} ({-test_change_pct:.1f}%)")
		else:
			log(f"⚠️ Optimization DEGRADED test PPL by {test_change:.1f} (+{test_change_pct:.1f}%)")

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
	if optimization_ppl:
		log(f"FINAL PPL (held-out train): {optimization_ppl:.1f}")
	log(f"FINAL PPL (test set): {perplexity:.1f}")
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
	tokenizer_type: TokenizerType = TokenizerType.WORD,
	n_neurons: int = None,
	bits_per_neuron: int = None,
	cascade_threshold: float = 0.1,
	strategy_sequence: list = None,
	full_data: bool = False,
	smoothing_type: str = "none",
	use_lsh: bool = False,
	lsh_type: str = "simhash",
	attention_type: AttentionType = AttentionType.NONE,
	representation_type: RepresentationType = RepresentationType.COOCCURRENCE,
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
								   n_neurons=n_neurons, bits_per_neuron=bits_per_neuron,
								   cascade_threshold=cascade_threshold, strategy_sequence=strategy_sequence,
								   full_data=full_data, smoothing_type=smoothing_type,
								   use_lsh=use_lsh, lsh_type=lsh_type,
								   attention_type=attention_type,
								   representation_type=representation_type)
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
	parser.add_argument("--tokenizer", type=str, default="gpt2",
		choices=["simple", "word", "bpe", "gpt2", "char"],
		help="Tokenizer: simple, word (WikiText-2), bpe (trainable 32k), gpt2 (50k, default), char")
	parser.add_argument("--neurons", type=int, default=None,
		help="Number of neurons per RAM (default: 16 FAST, 64 FULL/OVERNIGHT)")
	parser.add_argument("--bits", type=int, default=None,
		help="Bits per neuron (connectivity, default: 10 FAST, 14 FULL/OVERNIGHT)")
	parser.add_argument("--threshold", type=float, default=0.1,
		help="Cascade confidence threshold (default: 0.1)")
	parser.add_argument("--strategy", type=str, default="GA,TS",
		help="Optimization strategy sequence, comma-separated (default: GA,TS). Examples: GA, TS, GA,TS, TS,GA, GA,TS,GA")
	parser.add_argument("--full-data", action="store_true",
		help="Use full WikiText-2 dataset (2M train, 245k test, 214k val) instead of subsampled data")
	parser.add_argument("--smoothing", type=str, default="kneser_ney",
		choices=["none", "kneser_ney", "backoff", "add_k"],
		help="Smoothing for cascade fallback: none (1/vocab), kneser_ney (default, recommended), backoff, add_k")
	parser.add_argument("--lsh", action="store_true", default=True,
		help="Enable LSH context hashing (default: enabled)")
	parser.add_argument("--no-lsh", action="store_false", dest="lsh",
		help="Disable LSH context hashing")
	parser.add_argument("--lsh-type", type=str, default="simhash",
		choices=["simhash", "random_projection"],
		help="LSH algorithm: simhash (fast) or random_projection (learned embeddings)")
	parser.add_argument("--attention", type=str, default="hybrid",
		choices=["none", "position", "content", "hybrid", "sparse"],
		help="Attention mechanism: none, position, content, hybrid (default), sparse")
	parser.add_argument("--representation", type=str, default="ram_learned",
		choices=["hash", "cooccurrence", "mutual_info", "ram_learned"],
		help="Binary representation: hash (simple), ram_learned (default, fast, 4x accuracy), mutual_info, cooccurrence (SLOW)")
	parser.add_argument("--accel", type=str, default="hybrid",
		choices=["cpu", "metal", "hybrid"],
		help="Acceleration: cpu (16 cores), metal (40 GPU cores), hybrid (default, 56 cores = CPU+GPU)")
	args = parser.parse_args()

	# Map tokenizer arg to enum
	tokenizer_map = {
		"simple": TokenizerType.SIMPLE_WORD,
		"word": TokenizerType.WORD,
		"bpe": TokenizerType.BPE,
		"gpt2": TokenizerType.GPT2,
		"char": TokenizerType.CHARACTER,
	}
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

	# Map attention arg to enum
	attention_map = {
		"none": AttentionType.NONE,
		"position": AttentionType.POSITION,
		"content": AttentionType.CONTENT,
		"hybrid": AttentionType.HYBRID,
		"sparse": AttentionType.SPARSE,
	}
	attention_type = attention_map[args.attention]

	# Map representation arg to enum
	representation_map = {
		"hash": RepresentationType.HASH,
		"cooccurrence": RepresentationType.COOCCURRENCE,
		"mutual_info": RepresentationType.MUTUAL_INFO,
		"ram_learned": RepresentationType.RAM_LEARNED,
	}
	representation_type = representation_map[args.representation]

	# Map acceleration arg to enum and set global
	accel_map = {
		"cpu": AccelerationMode.CPU,
		"metal": AccelerationMode.METAL,
		"hybrid": AccelerationMode.HYBRID,
	}
	ACCELERATION_MODE = accel_map[args.accel]

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
	log(f"Bits per neuron: {args.bits if args.bits else 'default (10 FAST, 14 otherwise)'}")
	log(f"Cascade threshold: {args.threshold}")
	log(f"Strategy: {' → '.join(strategy_sequence)}")
	log(f"Data: {'FULL WikiText-2 (~2M train, ~245k test, ~214k val)' if args.full_data else 'sampled'}")
	log(f"Smoothing: {args.smoothing}")
	log(f"LSH: {'enabled (' + args.lsh_type + ')' if args.lsh else 'disabled'}")
	log(f"Attention: {attention_type.name.lower()}")
	log(f"Representation: {representation_type.name.lower()}")
	accel_desc = {
		AccelerationMode.CPU: f"CPU ({RUST_CPU_CORES} cores)",
		AccelerationMode.METAL: f"Metal GPU ({METAL_GPU_CORES} cores)" if METAL_AVAILABLE else "Metal (unavailable, fallback to CPU)",
		AccelerationMode.HYBRID: f"Hybrid CPU+GPU ({RUST_CPU_CORES}+{METAL_GPU_CORES}={RUST_CPU_CORES+METAL_GPU_CORES} cores)" if METAL_AVAILABLE else "Hybrid (unavailable, fallback to CPU)",
	}
	log(f"Acceleration: {accel_desc[ACCELERATION_MODE]}")
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
							n_neurons=args.neurons, bits_per_neuron=args.bits,
							cascade_threshold=args.threshold, strategy_sequence=strategy_sequence,
							full_data=args.full_data, smoothing_type=args.smoothing,
							use_lsh=args.lsh, lsh_type=args.lsh_type,
							attention_type=attention_type,
							representation_type=representation_type)
	else:
		run_benchmark(mode=mode, tokenizer_type=tokenizer_type,
					  n_neurons=args.neurons, bits_per_neuron=args.bits,
					  cascade_threshold=args.threshold, strategy_sequence=strategy_sequence,
					  full_data=args.full_data, smoothing_type=args.smoothing,
					  use_lsh=args.lsh, lsh_type=args.lsh_type,
					  attention_type=attention_type,
					  representation_type=representation_type)

	# Final log
	log_separator()
	log("SESSION COMPLETE")
	log(f"Full log available at: {log_file}")
	log_separator()
