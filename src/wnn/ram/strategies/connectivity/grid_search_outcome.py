"""
GridSearchOutcome — neutral result of a genome-agnostic grid search.

Produced by `GenericGridSearch.run()` and adapted to each strand's own return
type by the subclass (IDS → `OptimizerResult`; controller → the stage0 tuple).
Kept genome-/metric-agnostic: `Any` throughout so the shared core never depends
on `ClusterGenome`/`RecurrentArchGenome` or a specific `Metrics` type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridSearchOutcome:
	"""What `GenericGridSearch.run()` hands back.

	`seed_population`/`seed_metrics`/`seed_points` are aligned, fitness-sorted
	(best first) and trimmed to `population_size`. `best_*` is index 0 of those
	(the fitness-best genome), surfaced explicitly for callers that need the
	winner's point/spec (the controller derives its next-stage spec from it).
	`ranked_points` is the top-of-search config ranking (before expansion).
	"""

	seed_population: list = field(default_factory=list)
	seed_metrics: list = field(default_factory=list)
	seed_points: list = field(default_factory=list)
	best_genome: Any = None
	best_point: Any = None
	best_metrics: Any = None
	ranked_points: list = field(default_factory=list)
	extra: dict = field(default_factory=dict)
