"""Controller multi-objective harmonic-rank fitness calculator.

Mirrors `FitnessCalculatorHarmonicRank` (IDS) but operates on controller-
specific metrics:
  - err²    : accumulated squared attitude error (lower = better, ranks ascending).
              Stored as `Metrics.ce` (the controller evaluator mirrors -mean_reward
              into ce so the lower-is-better convention holds).
  - stable  : closed-loop stable_rate (higher = better, ranks descending).
              Stored as `Metrics.acc`.
  - jerk    : motor_jerk_mean (lower = better, ranks ascending). PLUMBED since
              29/05/2026 — the Rust eval (dagger_train.rs / the Metal scorer)
              measures per-step Σ(Δpwm)² and the evaluator surfaces it onto
              Metrics.motor_jerk_mean in BOTH the NEURONS (_evaluate_core) and
              MEMORY (score_genomes) paths. So with weight > 0 it IS ranked. The
              None-skip + warning below is now only a SAFETY fallback for a
              degenerate eval (e.g. a corrupted/180° genome that returns None).
              NB: jerk is RANKED in the fitness but NOT penalized in the reward
              (compute_reward keeps lambda_smooth=0) — those are separate.
  - mono    : mono_violations_total (lower = better, ranks ascending). PLUMBED —
              same as jerk (monotonicity_violations on the last output thermometer).

WHM = sum(weights) / sum(weight_i / rank_i). Lower WHM = closer to rank 1
across all active metrics. Defaults match `FitnessCalculatorController` exactly
(only err² ranked) so swapping calculators is opt-in via weight changes.

Trade-off vs FitnessCalculatorController:
  - CONTROLLER (single-objective)  : ranks purely on integrated err². Smooth
    signal, can plateau if multiple genomes tie on err² but differ on stability.
  - CONTROLLER_HARMONIC            : multi-objective rank space. Genome with
    err²=ε higher but stable_rate=100% will outrank a slightly-better-err²-but-
    0%-stable genome. Useful when stable_rate matters (paper #1 comparison
    against PID, which has stable=100%).
"""

import warnings

from wnn.ram.metrics import Metrics
from .FitnessCalculator import FitnessCalculator


class FitnessCalculatorControllerHarmonic(FitnessCalculator):
	"""WHM ranking across controller metrics with per-metric weights."""

	def __init__(
		self,
		weight_err_sq: float = 1.0,
		weight_stable: float = 0.0,
		weight_jerk:   float = 0.0,
		weight_mono:   float = 0.0,
		weight_steady: float = 0.0,
	):
		self.weight_err_sq = float(weight_err_sq)
		self.weight_stable = float(weight_stable)
		self.weight_jerk   = float(weight_jerk)
		self.weight_mono   = float(weight_mono)
		self.weight_steady = float(weight_steady)
		self._warned_jerk = False
		self._warned_mono = False
		self._warned_steady = False

	@staticmethod
	def _compute_ranks(values: list[float], ascending: bool = True) -> list[int]:
		"""Compute 1-based ranks. ascending=True → lower value gets rank 1."""
		n = len(values)
		order = sorted(range(n), key=lambda i: values[i] if ascending else -values[i])
		ranks = [0] * n
		for rank, idx in enumerate(order, start=1):
			ranks[idx] = rank
		return ranks

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		active: list[tuple[list[int], float]] = []

		# err² → ranked on ce (controller evaluator mirrors -mean_reward into ce)
		if self.weight_err_sq > 0:
			ranks = self._compute_ranks([m.ce for m in metrics_list], ascending=True)
			active.append((ranks, self.weight_err_sq))

		# stable_rate → ranked on acc, descending (higher acc = lower rank)
		if self.weight_stable > 0:
			ranks = self._compute_ranks([m.acc for m in metrics_list], ascending=False)
			active.append((ranks, self.weight_stable))

		# motor_jerk_mean — RESERVED. Skip ranking when field is None on any
		# genome; warn once per process so misconfig is visible.
		if self.weight_jerk > 0:
			vals = [m.motor_jerk_mean for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_jerk:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_jerk > 0 but "
						"Metrics.motor_jerk_mean is None — measurement not yet "
						"plumbed (compute_reward uses lambda_smooth=0). Weight "
						"ignored. To activate: set lambda_smooth > 0 in the "
						"Rust eval path and surface the running jerk total in "
						"the Metrics returned by dagger_train.",
						RuntimeWarning, stacklevel=2)
					self._warned_jerk = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_jerk))

		# mono_violations_total — RESERVED, same handling as jerk.
		if self.weight_mono > 0:
			vals = [m.mono_violations_total for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_mono:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_mono > 0 but "
						"Metrics.mono_violations_total is None — measurement not "
						"yet plumbed. Weight ignored.",
						RuntimeWarning, stacklevel=2)
					self._warned_mono = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_mono))

		# mean_steady_error_deg — the I-pressure term: mean attitude err over the
		# last 20% of steps (the settled window). Isolates the steady-state offset
		# (which only an integrator can kill) from the transient that dominates err².
		# lower = better, ranks ascending. Skip + warn-once if unplumbed (None).
		if self.weight_steady > 0:
			vals = [m.mean_steady_error_deg for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_steady:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_steady > 0 but "
						"Metrics.mean_steady_error_deg is None — steady-state-window "
						"metric not plumbed by the scorer. Weight ignored.",
						RuntimeWarning, stacklevel=2)
					self._warned_steady = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_steady))

		if not active:
			return [1.0] * n

		w_sum = sum(w for _, w in active)
		return [
			w_sum / sum(w / ranks[i] for ranks, w in active)
			for i in range(n)
		]

	@property
	def name(self) -> str:
		parts = [f"err²={self.weight_err_sq}"]
		if self.weight_stable > 0: parts.append(f"stable={self.weight_stable}")
		if self.weight_jerk   > 0: parts.append(f"jerk={self.weight_jerk}")
		if self.weight_mono   > 0: parts.append(f"mono={self.weight_mono}")
		if self.weight_steady > 0: parts.append(f"steady={self.weight_steady}")
		return f"ControllerHarmonic({', '.join(parts)})"
