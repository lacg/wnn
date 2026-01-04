"""
Base classes for connectivity optimization strategies.

Based on Garcia (2003) thesis on global optimization methods for
choosing connectivity patterns of weightless neural networks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from torch import Tensor, manual_seed


@dataclass
class OptimizerResult:
	"""Result of connectivity optimization."""

	initial_connections: Tensor
	optimized_connections: Tensor
	initial_error: float
	final_error: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: list = field(default_factory=list)

	def __repr__(self) -> str:
		return (
			f"OptimizerResult("
			f"method={self.method_name}, "
			f"initial_error={self.initial_error:.4f}, "
			f"final_error={self.final_error:.4f}, "
			f"improvement={self.improvement_percent:.2f}%)"
		)


class OptimizerStrategyBase(ABC):
	"""
	Abstract base class for connectivity optimization strategies.

	Implements the Strategy pattern for global optimization of RAM neuron
	connectivity patterns. Based on Garcia (2003) thesis.

	Subclasses must implement:
	- optimize(): The main optimization loop
	- name property

	Usage:
		strategy = TabuSearchStrategy(config)
		result = strategy.optimize(
			connections=initial_connections,
			evaluate_fn=lambda conn: compute_error(conn),
			total_input_bits=16,
			num_neurons=8,
			n_bits_per_neuron=4,
		)
	"""

	def __init__(
		self,
		seed: Optional[int] = None,
		verbose: bool = False,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._seed = seed
		self._verbose = verbose
		self._logger = logger or print
		self._rng_initialized = False

	def _log(self, msg: str) -> None:
		"""Log a message using the configured logger."""
		if self._verbose:
			self._logger(msg)

	@property
	def seed(self) -> Optional[int]:
		return self._seed

	@property
	def verbose(self) -> bool:
		return self._verbose

	@property
	@abstractmethod
	def name(self) -> str:
		"""Return the strategy name."""
		...

	def _ensure_rng(self) -> None:
		"""Initialize RNG if needed."""
		if not self._rng_initialized and self._seed is not None:
			manual_seed(self._seed)
			import random
			random.seed(self._seed)
			self._rng_initialized = True

	@abstractmethod
	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""
		Optimize connectivity pattern.

		Args:
			connections: Initial connectivity [num_neurons, n_bits_per_neuron]
			evaluate_fn: Function that evaluates a connectivity pattern,
				returns error rate (lower is better)
			total_input_bits: Size of input vector
			num_neurons: Number of RAM neurons
			n_bits_per_neuron: Fan-in per neuron

		Returns:
			OptimizerResult with optimized connections and statistics
		"""
		...

	def _generate_neighbor(
		self,
		connections: Tensor,
		mutation_rate: float,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> tuple[Tensor, tuple]:
		"""
		Generate neighbor by mutating connections.

		From Garcia (2003) thesis:
		- For each connection, with probability mutation_rate, modify it
		- Generate random value 0..total_input_bits-1
		- Allows changing which input bit a connection observes

		Returns:
			(new_connections, move) where move = (neuron_idx, old_value, new_value)
		"""
		import random

		neighbor = connections.clone()
		last_move = (0, 0, 0)

		for neuron_idx in range(num_neurons):
			for conn_idx in range(n_bits_per_neuron):
				if random.random() < mutation_rate:
					old_value = int(neighbor[neuron_idx, conn_idx].item())
					new_value = random.randint(0, total_input_bits - 1)
					neighbor[neuron_idx, conn_idx] = new_value
					last_move = (neuron_idx, old_value, new_value)

		return neighbor, last_move

	def compute_orthogonality(
		self,
		connections: Tensor,
		sample_inputs: Tensor,
		n_bits_per_neuron: int,
	) -> float:
		"""
		Compute orthogonality score for a connectivity pattern.

		Higher orthogonality means neurons observe different aspects of input,
		leading to better discrimination. From Garcia (2003), SA improved
		orthogonality by 50%, leading to 10% classification improvement.

		Args:
			connections: [num_neurons, n_bits_per_neuron]
			sample_inputs: [batch_size, total_input_bits]
			n_bits_per_neuron: Fan-in per neuron

		Returns:
			Orthogonality score (higher is better, range 0-2)
		"""
		from torch import arange, zeros, int64

		num_neurons = connections.shape[0]
		binary_bases = 2 ** arange(n_bits_per_neuron - 1, -1, -1, dtype=int64)

		# For each neuron, compute its address distribution over samples
		address_distributions = []
		for neuron_idx in range(num_neurons):
			neuron_conn = connections[neuron_idx]
			gathered = sample_inputs[:, neuron_conn].to(int64)
			addresses = (gathered * binary_bases).sum(dim=1)

			# Compute histogram distribution
			hist = zeros(2 ** n_bits_per_neuron)
			for addr in addresses:
				hist[addr] += 1
			hist = hist / hist.sum()
			address_distributions.append(hist)

		# Compute pairwise L1 distances (total variation) between distributions
		total_distance = 0.0
		count = 0
		for i in range(num_neurons):
			for j in range(i + 1, num_neurons):
				distance = (address_distributions[i] - address_distributions[j]).abs().sum().item()
				total_distance += distance
				count += 1

		return total_distance / count if count > 0 else 0.0

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(seed={self._seed}, verbose={self._verbose})"
