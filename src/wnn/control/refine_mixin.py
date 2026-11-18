"""CTRL-58: the controller GA strategies' side of the staged-refinement switch.

GenericGAStrategy exposes two no-op hooks (`_refine_switch`, `_refine_steady_floor`);
this mixin implements them against the run's RefineLatch, and overrides the fitness
setup so a stage that starts AFTER the latch fired (or that IS the trigger stage) ranks
under the refinement fitness from generation 0. `_wire_refine` (phased_ga) sets
`_refine_latch`; absent it (every run without --refine-fitness) all three methods fall
through to the base behaviour, bit-identical.
"""

from __future__ import annotations

from typing import Any, Optional


class RefineFitnessMixin:
	"""Mixed into ControllerArchGAStrategy (and so into its MEMORY / mixed subclasses)."""

	def _setup_fitness_calculator(self) -> Any:
		latch = getattr(self, "_refine_latch", None)
		if latch is None or not latch.stage_start(getattr(self, "_stage_name", "")):
			return super()._setup_fitness_calculator()
		return self._adopt_refine_calculator(latch, "stage start")

	def _refine_switch(self, generation: int, population: list) -> Optional[Any]:
		"""After selection: the refinement calculator on the generation the latch fires
		on this stage's TRAIN pool, else None (already refining, or off)."""
		latch = getattr(self, "_refine_latch", None)
		if latch is None or getattr(self, "_refine_active", False):
			return None
		if not latch.observe(getattr(self, "_stage_name", ""), generation, [t[1] for t in population]):
			return None
		return self._adopt_refine_calculator(latch, f"gen {generation + 1}")

	def _refine_steady_floor(self) -> Optional[float]:
		"""Magnitude patience watches steady (with this floor, degrees) once refining."""
		if not getattr(self, "_refine_active", False):
			return None
		return self._refine_latch.config.patience_steady_floor

	def _adopt_refine_calculator(self, latch, when: str) -> Any:
		calc = latch.calculator()
		self._fitness_calculator = calc
		self._refine_active = True
		self._log.info(f"[{self.name}] Fitness calculator ({when}, refinement latched): {calc.name}")
		return calc
