"""
Tabu Search strategy for connectivity optimization.

From Garcia (2003) thesis, Tabu Search achieved the best results:
- 17.27% error reduction on T2 topology
- Only 5 iterations needed for convergence
- 30 neighbors per iteration
- Tabu list size of 5 to avoid cycling
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from torch import Tensor

from wnn.ram.strategies.connectivity.base import (
	OptimizerResult,
	OptimizerStrategyBase,
)


@dataclass
class TabuSearchConfig:
	"""
	Configuration for Tabu Search optimization.

	Parameters tuned from Garcia (2003) experiments:
	- iterations: 5 was sufficient for convergence
	- neighbors_per_iter: 30 neighbors tested per iteration
	- tabu_size: 5 recent moves stored to avoid cycling
	- mutation_rate: 0.001 per-connection probability
	"""
	iterations: int = 5
	neighbors_per_iter: int = 30
	tabu_size: int = 5
	mutation_rate: float = 0.001


class TabuSearchStrategy(OptimizerStrategyBase):
	"""
	Tabu Search optimization for RAM neuron connectivity patterns.

	Key features from Garcia (2003):
	- Generate multiple neighbors per iteration (unlike SA which uses 1)
	- Keep tabu list to avoid cycling back to recent solutions
	- Always move to best neighbor (even if worse - characteristic of TS)
	- Fewer iterations needed than SA due to parallel exploration

	This was the best-performing method in the thesis, achieving:
	- 17.27% error reduction (24.95% -> 20.64% on Buffalo T2)
	- Consistent results with low variance
	- Fast convergence in just 5 iterations
	"""

	def __init__(
		self,
		config: Optional[TabuSearchConfig] = None,
		seed: Optional[int] = None,
		verbose: bool = False,
	):
		super().__init__(seed=seed, verbose=verbose)
		self._config = config or TabuSearchConfig()

	@property
	def config(self) -> TabuSearchConfig:
		return self._config

	@property
	def name(self) -> str:
		return "TabuSearch"

	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""
		Run Tabu Search optimization.

		The algorithm:
		1. Start with initial connections
		2. Generate N neighbors by mutation
		3. Select best non-tabu neighbor
		4. Add move to tabu list (avoid reverting)
		5. Repeat for configured iterations
		"""
		self._ensure_rng()
		cfg = self._config

		current = connections.clone()
		current_error = evaluate_fn(current)

		best = current.clone()
		best_error = current_error
		initial_error = current_error

		# Tabu list stores recent moves to avoid cycling
		# Each entry is (neuron_idx, old_connection, new_connection)
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		history = [(0, current_error)]

		if self._verbose:
			print(f"[TS] Initial error: {current_error:.4f}")

		for iteration in range(cfg.iterations):
			# Generate neighbors
			neighbors = []
			for _ in range(cfg.neighbors_per_iter):
				neighbor, move = self._generate_neighbor(
					current, cfg.mutation_rate,
					total_input_bits, num_neurons, n_bits_per_neuron
				)

				# Check if move is tabu (reverting a recent change)
				is_tabu = any(
					m[0] == move[0] and m[2] == move[1]
					for m in tabu_list
				)

				if not is_tabu:
					error = evaluate_fn(neighbor)
					neighbors.append((neighbor, error, move))

			if not neighbors:
				continue

			# Select best non-tabu neighbor
			neighbors.sort(key=lambda x: x[1])
			best_neighbor, best_neighbor_error, best_move = neighbors[0]

			# Update current (always move to best neighbor - TS characteristic)
			current = best_neighbor
			current_error = best_neighbor_error

			# Add move to tabu list
			tabu_list.append(best_move)

			# Update global best
			if current_error < best_error:
				best = current.clone()
				best_error = current_error

			history.append((iteration + 1, best_error))

			if self._verbose:
				print(f"[TS] Iter {iteration + 1}: current={current_error:.4f}, best={best_error:.4f}")

		improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

		return OptimizerResult(
			initial_connections=connections,
			optimized_connections=best,
			initial_error=initial_error,
			final_error=best_error,
			improvement_percent=improvement_pct,
			iterations_run=cfg.iterations,
			method_name=self.name,
			history=history,
		)

	def __repr__(self) -> str:
		return f"TabuSearchStrategy(config={self._config}, seed={self._seed}, verbose={self._verbose})"
