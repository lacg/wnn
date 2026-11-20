"""The output of SupervisedCutFitter: per-feature thresholds (+ rank maps for re-coded categoricals)."""

from dataclasses import dataclass, field

import numpy as np

from wnn.representations.category_rank_map import CategoryRankMap


@dataclass
class SupervisedCuts:
	thresholds: dict[str, np.ndarray]
	# Categoricals re-coded as a thermometer over their train attack-rate rank.
	rank_maps: dict[str, CategoryRankMap] = field(default_factory=dict)

	def bits(self) -> dict[str, int]:
		return {name: len(cuts) for name, cuts in self.thresholds.items()}
