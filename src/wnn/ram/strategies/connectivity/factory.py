"""
Factory for creating connectivity optimization strategies.
"""

from typing import Optional, Union

from wnn.ram.core import OptimizationMethod
from wnn.ram.strategies.connectivity.base import OptimizerStrategyBase
from wnn.ram.strategies.connectivity.tabu_search import TabuSearchStrategy, TabuSearchConfig
from wnn.ram.strategies.connectivity.simulated_annealing import SimulatedAnnealingStrategy, SimulatedAnnealingConfig
from wnn.ram.strategies.connectivity.genetic_algorithm import GeneticAlgorithmStrategy, GeneticAlgorithmConfig


ConfigType = Union[TabuSearchConfig, SimulatedAnnealingConfig, GeneticAlgorithmConfig, None]


class OptimizerStrategyFactory:
	"""
	Factory for creating connectivity optimization strategies.

	Usage:
		# Default Tabu Search (recommended - best results from thesis)
		strategy = OptimizerStrategyFactory.create(OptimizationMethod.TABU_SEARCH)

		# With custom config
		config = TabuSearchConfig(iterations=10, neighbors_per_iter=50)
		strategy = OptimizerStrategyFactory.create(
			OptimizationMethod.TABU_SEARCH,
			config=config,
			verbose=True
		)

		# Use the strategy
		result = strategy.optimize(connections, evaluate_fn, ...)
	"""

	@staticmethod
	def create(
		method: OptimizationMethod,
		config: ConfigType = None,
		seed: Optional[int] = None,
		verbose: bool = False,
	) -> OptimizerStrategyBase:
		"""
		Create an optimization strategy.

		Args:
			method: Which optimization method to use
			config: Strategy-specific configuration (optional)
			seed: Random seed for reproducibility
			verbose: Print progress during optimization

		Returns:
			OptimizerStrategyBase subclass instance
		"""
		if method == OptimizationMethod.TABU_SEARCH:
			ts_config = config if isinstance(config, TabuSearchConfig) else None
			return TabuSearchStrategy(config=ts_config, seed=seed, verbose=verbose)

		elif method == OptimizationMethod.SIMULATED_ANNEALING:
			sa_config = config if isinstance(config, SimulatedAnnealingConfig) else None
			return SimulatedAnnealingStrategy(config=sa_config, seed=seed, verbose=verbose)

		elif method == OptimizationMethod.GENETIC_ALGORITHM:
			ga_config = config if isinstance(config, GeneticAlgorithmConfig) else None
			return GeneticAlgorithmStrategy(config=ga_config, seed=seed, verbose=verbose)

		else:
			raise ValueError(f"Unknown optimization method: {method}")

	@staticmethod
	def create_default() -> OptimizerStrategyBase:
		"""
		Create the default (recommended) strategy.

		Returns TabuSearchStrategy, which achieved best results in Garcia (2003):
		- 17.27% error reduction
		- Only 5 iterations needed
		- Consistent low-variance results
		"""
		return TabuSearchStrategy()
