"""Controller fitness calculator — ranks by closed-loop reward.

For closed-loop control (drone attitude), genome quality is the closed-loop
reward of the trained controller (mean -attitude_error², penalises tumbling,
rewards reach+hold). The evaluator stores that raw reward in the dedicated
`Metrics.fitness` field (and mirrors -reward into `ce` so the GA loop's
"lower CE = better" plumbing still works). This calculator reads the reward
explicitly and returns -reward, honouring the framework's lower-is-better
convention — so we never lean on the ce-slot meaning for control tasks.
"""

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorController(FitnessCalculator):
	"""Fitness = -closed_loop_reward (lower is better → maximises reward)."""

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		out = []
		for m in metrics_list:
			# Prefer the dedicated reward field; fall back to the ce mirror.
			reward = m.fitness if m.fitness is not None else -m.ce
			out.append(-float(reward))
		return out

	@property
	def name(self) -> str:
		return "Controller"
