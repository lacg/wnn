"""
CE-only fitness calculator.

Ranks genomes purely by cross-entropy loss, ignoring accuracy.
This is the current/default behavior.
"""

from typing import TypeVar

from .FitnessCalculator import FitnessCalculator

G = TypeVar('G')


class FitnessCalculatorCE(FitnessCalculator[G]):
	"""
	Fitness calculator using only cross-entropy.

	fitness = CE (lower is better)

	Accuracy is ignored in fitness calculation but can still be used
	for threshold filtering before ranking.
	"""

	def fitness(self, population: list[tuple[G, float, float]]) -> list[float]:
		"""
		Compute fitness as pure CE.

		Args:
			population: List of (genome, ce, accuracy) tuples

		Returns:
			List of CE values (fitness = CE, lower is better)
		"""
		return [t[1] for t in population]

	@property
	def name(self) -> str:
		return "CE"
