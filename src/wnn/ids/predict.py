"""Lightweight Python WNN for per-example prediction.

Replicates the Rust RAM WNN training/inference logic in pure Python+numpy.
Used for the bitwise ECOC approach where we need per-example bit predictions
to combine into codewords for nearest-codeword decoding.

This is intentionally simple and slow — it runs once after optimization
finds the best genome. For training/optimization we use the Rust accelerator.
"""

import numpy as np
from .dataset import IDSDataset
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


def predict_ids(
	genome: ClusterGenome,
	dataset: IDSDataset,
	classification: str = "binary",
	empty_value: float = 0.0,
) -> np.ndarray:
	"""Train a WNN on training data and return per-example test predictions.

	Replicates the Rust ternary RAM neuron logic:
	- 3-state memory: FALSE (0.0), TRUE (1.0), EMPTY (empty_value)
	- Training: write TRUE for target class, FALSE for all negatives
	- Inference: for each class, average neuron outputs → argmax

	Args:
		genome: Optimized genome with bits_per_neuron, neurons_per_cluster, connections
		dataset: IDSDataset with X_train, y_train, X_test
		classification: "binary" or "multi"
		empty_value: Value for EMPTY cells (default 0.0 = abstain)

	Returns:
		np.ndarray of shape (n_test,) with predicted class indices
	"""
	if classification == "binary":
		y_train = dataset.y_train_binary
		num_classes = 2
	else:
		y_train = dataset.y_train_multi
		num_classes = len(dataset.category_names)

	X_train = dataset.X_train
	X_test = dataset.X_test
	total_features = X_train.shape[1]

	neurons_per_cluster = genome.neurons_per_cluster
	bits_per_neuron = genome.bits_per_neuron
	connections = genome.connections

	assert len(neurons_per_cluster) == num_classes
	total_neurons = sum(neurons_per_cluster)
	assert len(bits_per_neuron) == total_neurons

	# Parse connections into per-neuron connection lists
	neuron_connections = []
	conn_idx = 0
	for n_bits in bits_per_neuron:
		neuron_connections.append(connections[conn_idx:conn_idx + n_bits])
		conn_idx += n_bits

	# Build neuron-to-cluster mapping
	neuron_cluster = []
	for cluster_id, n_neurons in enumerate(neurons_per_cluster):
		neuron_cluster.extend([cluster_id] * n_neurons)

	# Allocate memory: each neuron has 2^bits cells, initialized to EMPTY (2)
	# Cell values: 0=FALSE, 1=TRUE, 2=EMPTY
	memories = []
	for n_bits in bits_per_neuron:
		mem_size = 1 << n_bits
		# Initialize all cells to EMPTY (2)
		memories.append(np.full(mem_size, 2, dtype=np.int8))

	# ── Training ──────────────────────────────────────────────────────
	n_train = X_train.shape[0]
	for ex_idx in range(n_train):
		example = X_train[ex_idx]
		target = int(y_train[ex_idx])

		# For each neuron: compute address, write TRUE/FALSE
		neuron_idx = 0
		for cluster_id in range(num_classes):
			for _ in range(neurons_per_cluster[cluster_id]):
				# Compute address from connections
				conns = neuron_connections[neuron_idx]
				addr = 0
				for bit_pos, conn in enumerate(conns):
					if conn >= 0 and conn < total_features and example[conn]:
						addr |= (1 << bit_pos)

				# Write: TRUE for target class, FALSE for others
				if cluster_id == target:
					memories[neuron_idx][addr] = 1  # TRUE
				else:
					memories[neuron_idx][addr] = 0  # FALSE

				neuron_idx += 1

	# ── Inference on test set ─────────────────────────────────────────
	n_test = X_test.shape[0]
	predictions = np.zeros(n_test, dtype=np.int64)

	# Cell value to float: FALSE=0.0, TRUE=1.0, EMPTY=empty_value
	cell_weights = np.array([0.0, 1.0, empty_value], dtype=np.float64)

	for ex_idx in range(n_test):
		example = X_test[ex_idx]
		scores = np.zeros(num_classes, dtype=np.float64)

		neuron_idx = 0
		for cluster_id in range(num_classes):
			n_neurons = neurons_per_cluster[cluster_id]
			cluster_sum = 0.0
			for _ in range(n_neurons):
				conns = neuron_connections[neuron_idx]
				addr = 0
				for bit_pos, conn in enumerate(conns):
					if conn >= 0 and conn < total_features and example[conn]:
						addr |= (1 << bit_pos)

				cell_val = memories[neuron_idx][addr]
				cluster_sum += cell_weights[cell_val]
				neuron_idx += 1

			scores[cluster_id] = cluster_sum / n_neurons if n_neurons > 0 else 0.0

		predictions[ex_idx] = np.argmax(scores)

	return predictions
