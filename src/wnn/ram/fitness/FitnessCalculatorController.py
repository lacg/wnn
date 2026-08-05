"""Controller fitness calculator — ranks by closed-loop reward.

For closed-loop control (drone attitude), genome quality is the closed-loop
reward of the trained controller (mean -attitude_error², penalises tumbling,
rewards reach+hold). Since 05/08/2026 the reward has its OWN field
(`ControllerMetrics.reward`) — the old `-ce` mirror is gone, and this
calculator refuses metrics that lack the field rather than guessing (a legacy
pre-05/08 checkpoint's cached metrics load as IDSMetrics; drop them and
re-evaluate the population).
"""

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorController(FitnessCalculator):
	"""Fitness = -closed_loop_reward (lower is better → maximises reward)."""

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		out = []
		for m in metrics_list:
			reward = getattr(m, "reward", None)
			if reward is None:
				raise TypeError(
					"FitnessCalculatorController needs ControllerMetrics with a "
					"reward field; got legacy/IDS metrics (pre-05/08/2026 cached "
					"checkpoint?) — drop the cached metrics and re-evaluate.")
			out.append(-float(reward))
		return out

	@property
	def name(self) -> str:
		return "Controller"
