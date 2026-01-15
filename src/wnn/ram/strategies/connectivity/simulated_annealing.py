"""
Simulated Annealing strategy for connectivity optimization.

From Garcia (2003) thesis:
- Good for escaping local minima via probabilistic acceptance
- 600 iterations with temperature decay
- Best temperature schedule: T(i+1) = 0.95 * T(i)
- Initial temperature: 1.0
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from torch import Tensor

from wnn.ram.strategies.connectivity.base import OptimizerStrategyBase
from wnn.ram.strategies.connectivity.generic_strategies import OptimizerResult


@dataclass
class SimulatedAnnealingConfig:
	"""
	Configuration for Simulated Annealing optimization.

	Parameters tuned from Garcia (2003) experiments:
	- iterations: 600 was used for convergence
	- initial_temp: 1.0 (from best set {1, 0.5, 0.1})
	- cooling_rate: 0.95 (from best set {0.99, 0.95, 0.9, 0.85})
	- mutation_rate: Modified from GA to avoid excessive deactivation
	"""
	iterations: int = 600
	initial_temp: float = 1.0
	cooling_rate: float = 0.95
	mutation_rate: float = 0.01


class SimulatedAnnealingStrategy(OptimizerStrategyBase):
	"""
	Simulated Annealing optimization for RAM neuron connectivity patterns.

	Key features from Garcia (2003):
	- Single neighbor per iteration (unlike TS which uses 30)
	- Probabilistic acceptance: worse solutions accepted with P = exp(-delta/T)
	- Temperature decay prevents premature convergence
	- Special neighbor generation to avoid excessive connection deactivation

	SA improved "orthogonality" of discriminators by 50%, leading to
	10% improvement in discrimination accuracy.
	"""

	def __init__(
		self,
		config: Optional[SimulatedAnnealingConfig] = None,
		seed: Optional[int] = None,
		verbose: bool = False,
	):
		super().__init__(seed=seed, verbose=verbose)
		self._config = config or SimulatedAnnealingConfig()

	@property
	def config(self) -> SimulatedAnnealingConfig:
		return self._config

	@property
	def name(self) -> str:
		return "SimulatedAnnealing"

	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult[Tensor]:
		"""
		Run Simulated Annealing optimization.

		The algorithm:
		1. Start with initial connections and high temperature
		2. Generate single neighbor by mutation
		3. Accept if better, or with probability exp(-delta/T) if worse
		4. Cool down: T = T * cooling_rate
		5. Repeat until temperature is very low
		"""
		self._ensure_rng()
		cfg = self._config

		current = connections.clone()
		current_error = evaluate_fn(current)

		best = current.clone()
		best_error = current_error
		initial_error = current_error

		temperature = cfg.initial_temp
		history = [(0, current_error)]

		if self._verbose:
			print(f"[SA] Initial error: {current_error:.4f}, T={temperature:.4f}")

		for iteration in range(cfg.iterations):
			# Generate neighbor
			neighbor, _ = self._generate_neighbor(
				current, cfg.mutation_rate,
				total_input_bits, num_neurons, n_bits_per_neuron
			)
			neighbor_error = evaluate_fn(neighbor)

			# Metropolis criterion
			delta = neighbor_error - current_error

			if delta < 0:
				# Accept improvement
				accept = True
			else:
				# Accept worse solution with probability exp(-delta/T)
				accept = random.random() < math.exp(-delta / temperature) if temperature > 0 else False

			if accept:
				current = neighbor
				current_error = neighbor_error

				if current_error < best_error:
					best = current.clone()
					best_error = current_error

			# Cool down
			temperature *= cfg.cooling_rate

			if (iteration + 1) % 100 == 0:
				history.append((iteration + 1, best_error))
				if self._verbose:
					print(f"[SA] Iter {iteration + 1}: current={current_error:.4f}, "
						f"best={best_error:.4f}, T={temperature:.6f}")

		improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

		return OptimizerResult(
			initial_genome=connections,
			best_genome=best,
			initial_fitness=initial_error,
			final_fitness=best_error,
			improvement_percent=improvement_pct,
			iterations_run=cfg.iterations,
			method_name=self.name,
			history=history,
		)

	def __repr__(self) -> str:
		return f"SimulatedAnnealingStrategy(config={self._config}, seed={self._seed}, verbose={self._verbose})"
