"""CarryState — full-population carry between optimization phases."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CarryState:
	"""What flows from one phase to the next.

	HARD RULE (06/06/2026, reaffirmed by the controller 30/05 fix): each phase
	continues the previous phase's WHOLE population — never a rebuilt one.
	`update()` only overwrites a field when the new value is non-empty, so a
	cancelled/failed/skipped phase falls back to the last good carry instead
	of seeding the next phase with nothing.
	"""
	genome: Optional[Any] = None
	population: Optional[list] = None
	threshold: Optional[float] = None
	# Strand-specific extras that ride along (e.g. controller spec)
	extra: dict = field(default_factory=dict)

	def update(
		self,
		genome: Optional[Any] = None,
		population: Optional[list] = None,
		threshold: Optional[float] = None,
	) -> None:
		"""Overwrite each field only if the new value is non-empty."""
		if genome is not None:
			self.genome = genome
		if population:
			self.population = population
		if threshold is not None:
			self.threshold = threshold
