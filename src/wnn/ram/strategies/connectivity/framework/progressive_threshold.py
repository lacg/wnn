"""Progressive accuracy-threshold schedule across iterations."""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Progressive Accuracy Threshold
# =============================================================================

@dataclass
class ProgressiveThresholdConfig:
	"""Configuration for progressive accuracy threshold.

	The threshold increases both within a phase (as progress goes 0→1) and
	across phases (each phase starts where the previous ended).

	Formula: threshold = base + (phase_index + progress) * delta

	Example with base=0.0001 (0.01%), delta=0.0002 (0.02%):
		Phase 1a (idx=0): 0.01% → 0.03%
		Phase 1b (idx=1): 0.03% → 0.05%
		Phase 2a (idx=2): 0.05% → 0.07%
		...
	"""
	base: float = 0.0001      # Starting threshold (0.01%)
	delta: float = 0.0002     # Increase per phase (0.02%)


class ProgressiveThreshold:
	"""
	Computes accuracy threshold that increases with optimization progress.

	The threshold gets stricter as optimization progresses, both within
	a single phase and across phases (curriculum learning).

	Usage:
		threshold = ProgressiveThreshold(phase_index=0)

		# In optimization loop:
		progress = iteration / total_iterations  # 0.0 to 1.0
		min_accuracy = threshold.get(progress)

		# For next phase:
		threshold = ProgressiveThreshold(phase_index=1)
	"""

	def __init__(
		self,
		phase_index: int = 0,
		config: Optional[ProgressiveThresholdConfig] = None,
	):
		"""
		Initialize progressive threshold.

		Args:
			phase_index: Current phase (0=1a, 1=1b, 2=2a, 3=2b, 4=3a, 5=3b)
			config: Optional configuration for base and delta values
		"""
		self._config = config or ProgressiveThresholdConfig()
		self._phase_index = phase_index

	@property
	def phase_index(self) -> int:
		"""Current phase index."""
		return self._phase_index

	@property
	def start_threshold(self) -> float:
		"""Threshold at start of this phase (progress=0)."""
		return self._config.base + self._phase_index * self._config.delta

	@property
	def end_threshold(self) -> float:
		"""Threshold at end of this phase (progress=1)."""
		return self._config.base + (self._phase_index + 1) * self._config.delta

	def get(self, progress: float) -> float:
		"""
		Get threshold for current progress within the phase.

		Args:
			progress: Progress through current phase (0.0 to 1.0)

		Returns:
			Accuracy threshold for filtering candidates
		"""
		progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
		return self._config.base + (self._phase_index + progress) * self._config.delta

	def format_range(self) -> str:
		"""Format the threshold range for logging."""
		return f"{self.start_threshold:.2%} → {self.end_threshold:.2%}"
