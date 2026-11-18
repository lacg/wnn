"""CTRL-58: the one-way latch that switches the GA to the refinement fitness.

ONE latch per run (built by ControllerOrchestrator, handed to every GA stage), so once
it fires every later stage starts under the refinement fitness. It fires on whichever
comes first:

  * POPULATION — after a generation's selection, >= trigger_frac of the TRAIN pool has
    stable >= S, err <= E, steady <= D and pos <= P, for trigger_gens consecutive
    generations (Luiz 18/11: the streak CARRIES ACROSS stages — consecutive GA
    generations of the run, not of one stage);
  * STAGE — trigger_stage starts (default MEMORY).

It reads ONLY the metrics the GA ranks (the train folds). It never sees a val or test
number, and it never reverts.
"""

import math
from typing import Callable, Optional

from .refine_config import RefineConfig


def refine_pos_value(m, pos_metric: str) -> float:
	"""The pos value the trigger compares (the column the fitness ranks). A missing hold
	value is NaN, which never satisfies `<= P`; a missing pos column is refused."""
	attr = "mean_position_hold_m" if pos_metric == "hold" else "mean_position_error_m"
	v = getattr(m, attr, None)
	if v is None:
		if pos_metric == "hold":
			return float("nan")
		raise ValueError(f"--refine-fitness trigger needs Metrics.{attr}; the scorer does not report it.")
	return float(v)


class RefineLatch:
	"""One-way switch: base fitness → refinement fitness (see module note)."""

	def __init__(self, cfg: RefineConfig, make_calculator: Callable[[], object],
	             pos_metric: str, log: Callable[[str], None] = print):
		self.config = cfg
		self._make_calculator = make_calculator
		self._pos_metric = pos_metric
		self._log = log
		self._streak = 0
		self.fired = False
		self.fired_stage: Optional[str] = None
		self.fired_gen: Optional[int] = None
		self.reason: Optional[str] = None

	def calculator(self):
		"""A fresh refinement calculator (each stage owns its own instance)."""
		return self._make_calculator()

	def stage_start(self, stage: str) -> bool:
		"""Called as a GA stage sets up its fitness. True ⇒ the stage runs under the
		refinement fitness from generation 0 (already fired, or this IS trigger_stage).
		The population streak is NOT reset here (it carries across stages)."""
		if self.fired:
			return True
		if stage == self.config.trigger_stage:
			self._fire(stage, 0, f"stage start ({stage.upper()})")
			return True
		return False

	def observe(self, stage: str, generation: int, metrics_list: list) -> bool:
		"""After a generation's selection. True exactly on the generation the latch fires
		(`generation` is the loop's 0-based index; the log prints it 1-based, as the gen
		line does)."""
		if self.fired:
			return False
		frac = self.qualifying_fraction(metrics_list)
		self._streak = self._streak + 1 if frac >= self.config.trigger_frac else 0
		if self._streak < self.config.trigger_gens:
			return False
		self._fire(stage, generation + 1, self._population_reason(frac))
		return True

	def qualifying_fraction(self, metrics_list: list) -> float:
		if not metrics_list:
			return 0.0
		return sum(1 for m in metrics_list if self._qualifies(m)) / len(metrics_list)

	def _qualifies(self, m) -> bool:
		c = self.config
		err = getattr(m, "mean_attitude_error_deg", None)
		steady = getattr(m, "mean_steady_error_deg", None)
		if err is None or steady is None:
			raise ValueError("--refine-fitness trigger needs err and steady on every TRAIN metric.")
		pos = refine_pos_value(m, self._pos_metric)
		return (float(m.stable_rate) >= c.trigger_stable and float(err) <= c.trigger_err
		        and float(steady) <= c.trigger_steady and not math.isnan(pos) and pos <= c.trigger_pos)

	def _population_reason(self, frac: float) -> str:
		c = self.config
		return (f"{frac:.0%} of TRAIN pool >= {c.trigger_frac:.0%} with stable>={c.trigger_stable:g} "
		        f"err<={c.trigger_err:g}° steady<={c.trigger_steady:g}° pos<={c.trigger_pos:g}m "
		        f"for {c.trigger_gens} consecutive gen(s)")

	def state(self) -> dict:
		"""The latch as plain data — checkpoint meta / winner marker (fired, stage, generation,
		reason) plus the running streak, so an emergency resume continues it exactly."""
		return {"fired": self.fired, "stage": self.fired_stage, "generation": self.fired_gen,
		        "reason": self.reason, "streak": self._streak}

	def restore(self, state: dict) -> None:
		"""Inverse of state() (emergency resume)."""
		self.fired = bool(state.get("fired", False))
		self.fired_stage = state.get("stage")
		self.fired_gen = state.get("generation")
		self.reason = state.get("reason")
		self._streak = int(state.get("streak", 0) or 0)
		self._log(f"[refine] latch restored from the emergency dump: {self.state()}")

	def _fire(self, stage: str, gen: int, reason: str) -> None:
		self.fired, self.fired_stage, self.fired_gen, self.reason = True, stage, gen, reason
		self._log(f"[refine] trigger fired gen {gen} stage {stage.upper()} reason {reason}")
