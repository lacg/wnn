"""SaveCadence — the shared "should I checkpoint this generation?" decision.

Extracted (13/06/2026) from the IDS ``CheckpointManager`` so the controller
phased-GA can reuse the SAME adaptive-cadence brain instead of copy-pasting it.

The rule (unchanged from CheckpointManager's original ``should_save_now``):
  * If ``target_loss_seconds`` is set, save once that many seconds of wall-clock
    have elapsed since the last save — but never let more than ``max_interval``
    generations pass without one. This SELF-ADJUSTS to per-gen cost:
      - a fast gen (~30 s) accumulates until the budget is hit → I/O throttled
        to ~every ``max_interval`` gens;
      - a single slow gen (e.g. 46M IDS or the 40-min/gen controller NEURONS
        stage) trips ``elapsed >= budget`` immediately → saves EVERY gen.
    Work lost on a hard crash is bounded to ~``target_loss_seconds``.
  * If ``target_loss_seconds`` is None, save every generation (legacy behaviour).

This object owns ONLY the timing decision — it never touches disk. The caller
does the actual write (IDS json, controller yaml.gz) and then calls
``mark_saved`` so the next decision measures from the right baseline.

``monotonic`` is injected so it stays unit-testable without real sleeps and so
the controller (whose scripts ban ``time``-free determinism elsewhere) can pass
its own clock. Defaults to ``time.monotonic``.
"""

from typing import Callable, Optional


class SaveCadence:
	"""Decide when to checkpoint, adaptively by wall-clock. Timing only."""

	def __init__(self, target_loss_seconds: Optional[float], max_interval: int,
	             monotonic: Optional[Callable[[], float]] = None):
		self._budget = target_loss_seconds
		self._max_interval = max(1, int(max_interval))
		if monotonic is None:
			import time
			monotonic = time.monotonic
		self._now = monotonic
		self._last_save_monotonic: Optional[float] = None
		self._last_save_gen: int = -1

	def should_save_now(self, generation: int) -> bool:
		"""True iff this generation should be checkpointed. The FIRST generation
		seen only establishes the time baseline (returns False — nothing to lose
		yet). With no budget, always True (caller saves every gen)."""
		if self._budget is None:
			return True
		now = self._now()
		if self._last_save_monotonic is None:
			self._last_save_monotonic = now
			self._last_save_gen = generation
			return False
		elapsed = now - self._last_save_monotonic
		gens_since = generation - self._last_save_gen
		return elapsed >= self._budget or gens_since >= self._max_interval

	def mark_saved(self, generation: int) -> None:
		"""Record that a save just happened at ``generation`` — resets both the
		wall-clock and the gen-count baselines."""
		self._last_save_monotonic = self._now()
		self._last_save_gen = generation
