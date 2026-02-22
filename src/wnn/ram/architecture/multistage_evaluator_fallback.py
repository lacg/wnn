"""
Python-only fallback for multi-stage tiered evaluation.

Used when the Rust accelerator is not available. Implements a simple dict-based
evaluator: train = dict[context_bits → class_counts], eval = lookup + normalize + CE.

Slow but correct reference implementation.
"""

import math
import random
from collections import defaultdict


class TieredFallbackEvaluator:
	"""Dict-based tiered evaluator for testing without Rust.

	For each example:
	  - Training: increment count[address][true_class] and decrement for negatives
	  - Evaluation: lookup counts, softmax, compute CE
	"""

	def __init__(
		self,
		train_input_bits: list[bool],
		train_targets: list[int],
		eval_input_bits: list[bool],
		eval_targets: list[int],
		total_input_bits: int,
		num_clusters: int,
		num_train: int,
		num_eval: int,
		num_negatives: int = 10,
	):
		self.train_input_bits = train_input_bits
		self.train_targets = train_targets
		self.eval_input_bits = eval_input_bits
		self.eval_targets = eval_targets
		self.total_input_bits = total_input_bits
		self.num_clusters = num_clusters
		self.num_train = num_train
		self.num_eval = num_eval
		self.num_negatives = num_negatives

	def evaluate(
		self,
		bits_per_neuron: list[int],
		neurons_per_cluster: list[int],
		connections: list[int],
	) -> tuple[float, float]:
		"""Train and evaluate a single genome.

		Returns (cross_entropy, accuracy).
		"""
		# Build neuron layout
		cluster_neuron_starts = []
		cumul = 0
		for nc in neurons_per_cluster:
			cluster_neuron_starts.append(cumul)
			cumul += nc

		neuron_conn_offsets = []
		conn_off = 0
		for b in bits_per_neuron:
			neuron_conn_offsets.append(conn_off)
			conn_off += b

		# Memory: dict[cluster_id][neuron_local][address] → value (True/False)
		# Using simple dict for clarity
		memory: dict[int, dict[int, dict[int, bool]]] = defaultdict(
			lambda: defaultdict(dict)
		)

		# Train
		rng = random.Random(42)
		for ex_idx in range(self.num_train):
			input_start = ex_idx * self.total_input_bits
			input_bits = self.train_input_bits[input_start:input_start + self.total_input_bits]
			true_cluster = self.train_targets[ex_idx]

			# Write TRUE for true_cluster neurons
			neuron_base = cluster_neuron_starts[true_cluster]
			nc = neurons_per_cluster[true_cluster]
			for n in range(nc):
				global_n = neuron_base + n
				n_bits = bits_per_neuron[global_n]
				conn_start = neuron_conn_offsets[global_n]
				address = self._compute_address(input_bits, connections[conn_start:], n_bits)
				mem = memory[true_cluster][n]
				if address not in mem:
					mem[address] = True  # TRUE wins over FALSE
				# If already TRUE, keep TRUE

			# Write FALSE for negative clusters
			for _ in range(self.num_negatives):
				false_cluster = rng.randint(0, self.num_clusters - 1)
				if false_cluster == true_cluster:
					continue
				neuron_base = cluster_neuron_starts[false_cluster]
				nc = neurons_per_cluster[false_cluster]
				for n in range(nc):
					global_n = neuron_base + n
					n_bits = bits_per_neuron[global_n]
					conn_start = neuron_conn_offsets[global_n]
					address = self._compute_address(input_bits, connections[conn_start:], n_bits)
					mem = memory[false_cluster][n]
					if address not in mem:
						mem[address] = False

		# Evaluate
		total_ce = 0.0
		correct = 0
		epsilon = 1e-10

		for ex_idx in range(self.num_eval):
			input_start = ex_idx * self.total_input_bits
			input_bits = self.eval_input_bits[input_start:input_start + self.total_input_bits]
			true_cluster = self.eval_targets[ex_idx]

			# Compute scores for all clusters
			scores = []
			for c in range(self.num_clusters):
				neuron_base = cluster_neuron_starts[c]
				nc = neurons_per_cluster[c]
				if nc == 0:
					scores.append(0.0)
					continue
				total = 0.0
				for n in range(nc):
					global_n = neuron_base + n
					n_bits = bits_per_neuron[global_n]
					conn_start = neuron_conn_offsets[global_n]
					address = self._compute_address(input_bits, connections[conn_start:], n_bits)
					mem = memory[c][n]
					if address in mem:
						total += 1.0 if mem[address] else 0.0
					else:
						total += 0.0  # EMPTY = 0.0
				scores.append(total / nc)

			# Softmax CE
			max_score = max(scores)
			exp_scores = [math.exp(s - max_score) for s in scores]
			sum_exp = sum(exp_scores)
			target_prob = exp_scores[true_cluster] / sum_exp
			ce = -math.log(target_prob + epsilon)
			total_ce += ce

			predicted = scores.index(max(scores))
			if predicted == true_cluster:
				correct += 1

		avg_ce = total_ce / max(self.num_eval, 1)
		accuracy = correct / max(self.num_eval, 1)
		return avg_ce, accuracy

	@staticmethod
	def _compute_address(input_bits: list[bool], connections: list[int], bits: int) -> int:
		"""Compute RAM address from input bits and connections."""
		address = 0
		for b in range(bits):
			conn_idx = connections[b]
			if 0 <= conn_idx < len(input_bits) and input_bits[conn_idx]:
				address |= 1 << (bits - 1 - b)
		return address
