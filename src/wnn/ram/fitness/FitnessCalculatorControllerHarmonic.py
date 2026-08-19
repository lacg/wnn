"""Controller multi-objective harmonic-rank fitness calculator.

Mirrors `FitnessCalculatorHarmonicRank` (IDS) but operates on controller-
specific metrics:
  - err²    : accumulated squared attitude error (lower = better, ranks ascending).
              Ranked as -ControllerMetrics.reward — reward has its OWN field since
              05/08/2026; the old "-reward mirrored into ce" hack is gone.
  - stable  : closed-loop stable_rate (higher = better, ranks descending).
              ControllerMetrics.stable_rate — no longer smuggled through `acc`.
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
from .FitnessCalculator import FitnessCalculator, compute_ranks


def _controller_reward(m) -> float:
	"""ControllerMetrics.reward, refusing anything that lacks it (a legacy pre-05/08
	cached checkpoint loads as IDSMetrics — drop it and re-evaluate, never guess)."""
	r = getattr(m, "reward", None)
	if r is None:
		raise TypeError(
			"FitnessCalculatorControllerHarmonic needs ControllerMetrics with a "
			"reward field; got legacy/IDS metrics — drop cached metrics and re-evaluate.")
	return float(r)


class FitnessCalculatorControllerHarmonic(FitnessCalculator):
	"""Weighted-rank ranking across controller metrics, harmonic or arithmetic.

	AGGREGATION (19/08/2026, Luiz). Both modes rank each metric within the handed
	population and combine the ranks with the same weights; they differ only in the
	combine step, and the difference is not cosmetic:

	  harmonic   WHM = Σw / Σ(w/rank).   w/rank is hyperbolic in rank, so the score
	             is dominated by a genome's BEST weighted rank and nearly indifferent
	             to its worst — rank 1 at weight .35 contributes .350 while rank 9 at
	             weight .15 contributes .017. It selects SPECIALISTS: arm 9's headline
	             won on steady rank-1 alone while losing the other four metrics, dead
	             last on jerk at no cost. (The old docstring's "penalizes imbalance"
	             claim was inverted — the harmonic mean is dominated by the SMALLEST
	             elements, and small rank = good.)

	  arithmetic Σ(w·rank) / Σw.   Every rank hurts in proportion to its weight, so
	             four weighted losses outweigh one weighted win. This is what the
	             weights read as meaning, and it is what stage-select uses.

	Lower is better in both. The in-stage GA keeps harmonic until the alt-weight
	sweep's round 2 lands — round 2 replicates round 1 with only the seed changed,
	so its search compass must not move — and is revisited at the ladder restart.
	"""

	def __init__(
		self,
		weight_err_sq: float = 1.0,
		weight_stable: float = 0.0,
		weight_jerk:   float = 0.0,
		weight_mono:   float = 0.0,
		weight_steady: float = 0.0,
		weight_effort: float = 0.0,
		weight_alt:    float = 0.0,
		weight_pos:    float = 0.0,
		aggregation:   str   = "harmonic",
	):
		if aggregation not in ("harmonic", "arithmetic"):
			raise ValueError(f"aggregation must be 'harmonic' or 'arithmetic', got {aggregation!r}")
		self.aggregation = aggregation
		self.weight_err_sq = float(weight_err_sq)
		self.weight_stable = float(weight_stable)
		self.weight_jerk   = float(weight_jerk)
		self.weight_mono   = float(weight_mono)
		self.weight_steady = float(weight_steady)
		self.weight_effort = float(weight_effort)
		# SCOPE C as RANK dimensions (18/08/2026). The altitude/horizontal channels
		# used to reach the search ONLY through the reward, as -λ·err² terms. A λ
		# carries the metres↔radians unit conversion, so its tuned value is bound to
		# the CAPACITY it was swept at: λ_alt=16 was swept at 128n/b30 and is correct
		# there, but at 32n it made the altitude term ~9,900x the attitude term, and
		# a genome that hovered level while tumbling at 52 deg out-ranked one flying
		# at 11 deg. A RANK is scale-free — metres never compete numerically with
		# radians — so the channel can be weighted without carrying a unit.
		self.weight_alt    = float(weight_alt)
		self.weight_pos    = float(weight_pos)
		self._warned_jerk = False
		self._warned_mono = False
		self._warned_steady = False
		self._warned_effort = False
		self._warned_alt = False
		self._warned_pos = False

	# Shared fractional tie-aware ranking (see compute_ranks in FitnessCalculator.py).
	# This class carried its own positional copy — "mirrors IDS pattern", 08122b58 —
	# until 09/08/2026; in controller populations the dominant tie is stable_rate=100%.
	_compute_ranks = staticmethod(compute_ranks)

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		active: list[tuple[list[int], float]] = []

		# err² → ranked on ce (controller evaluator mirrors -mean_reward into ce)
		if self.weight_err_sq > 0:
			ranks = self._compute_ranks(
				[-_controller_reward(m) for m in metrics_list], ascending=True)
			active.append((ranks, self.weight_err_sq))

		# stable_rate → ranked on acc, descending (higher acc = lower rank)
		if self.weight_stable > 0:
			ranks = self._compute_ranks(
				[m.stable_rate for m in metrics_list], ascending=False)
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

		# mean_effort — the Σu² allocation-efficiency term (overactuated Phase 3,
		# Luiz 12/07/2026): mean per-step Σ pwm² of the applied command. On the
		# attitude-only sim, allocation mismatch costs EFFORT (not attitude
		# error — the LQR out-gains it), so this is the term that lets the GA
		# see misallocation on planar airframes. The min-norm pinv baseline is
		# the effort optimum; lower = closer to it. Ranks ascending.
		if self.weight_effort > 0:
			vals = [m.mean_effort for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_effort:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_effort > 0 but "
						"Metrics.mean_effort is None — scorer predates the 13-metric "
						"row (ABI 9). Weight ignored.",
						RuntimeWarning, stacklevel=2)
					self._warned_effort = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_effort))

		# mean_altitude_error_m — the VERTICAL channel as its own rank dimension
		# (metres, lower = better). See the note in __init__: this exists so the
		# channel can be weighted WITHOUT a λ carrying the metres↔radians conversion
		# into the reward, where it silently scaled with capacity.
		if self.weight_alt > 0:
			vals = [getattr(m, "mean_altitude_error_m", None) for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_alt:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_alt > 0 but "
						"Metrics.mean_altitude_error_m is None — the run is not a "
						"--translation run, or the scorer predates metric row 14. "
						"Weight ignored.",
						RuntimeWarning, stacklevel=2)
					self._warned_alt = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_alt))

		# mean_position_error_m — the HORIZONTAL channel, same reasoning. Inert until
		# stage 2 is armed (--xy-offset > 0): with the channel off every genome sits
		# at the origin, the metric is a constant, and the rank is one big tie that
		# contributes nothing — harmless, but it means a non-zero weight here is only
		# meaningful once episodes actually start displaced.
		if self.weight_pos > 0:
			vals = [getattr(m, "mean_position_error_m", None) for m in metrics_list]
			if any(v is None for v in vals):
				if not self._warned_pos:
					warnings.warn(
						"FitnessCalculatorControllerHarmonic: weight_pos > 0 but "
						"Metrics.mean_position_error_m is None — the run is not a "
						"--translation run, or the scorer predates metric row 13. "
						"Weight ignored.",
						RuntimeWarning, stacklevel=2)
					self._warned_pos = True
			else:
				ranks = self._compute_ranks([float(v) for v in vals], ascending=True)
				active.append((ranks, self.weight_pos))

		if not active:
			return [1.0] * n

		w_sum = sum(w for _, w in active)
		if self.aggregation == "arithmetic":
			return [
				sum(w * ranks[i] for ranks, w in active) / w_sum
				for i in range(n)
			]
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
		if self.weight_effort > 0: parts.append(f"effort={self.weight_effort}")
		# alt is printed UNCONDITIONALLY, including at 0.00 (Luiz, 18/08). The other
		# weights hide at zero because absence and zero mean the same thing for them.
		# For alt they do not: a 0.00 arm is a deliberate CONTROL in the alt-weight
		# sweep, and a suppressed label makes it indistinguishable from a run whose
		# alt weight never reached the calculator at all — which is exactly the
		# failure this label would have caught on 18/08 and did not. Printing the
		# zero is what turns the label into evidence that the dimension was ranked.
		parts.append(f"alt={self.weight_alt}")
		# pos stays conditional while it is inert (every genome sits at the origin
		# until --xy-offset > 0, so the rank is one big tie). Make it unconditional
		# the day stage 2 arms, for the reason above.
		if self.weight_pos > 0: parts.append(f"pos={self.weight_pos}")
		# The aggregation is printed when it is not the harmonic default: two runs
		# with identical weights but different combine steps select DIFFERENT
		# genomes (arm 9: MEMORY#0 vs CONNECTIONS#0), so a label that hid this
		# would make them look like the same fitness function — the exact failure
		# the alt=0.00 label rule exists to prevent.
		name = "ControllerHarmonic" if self.aggregation == "harmonic" else "ControllerArithRank"
		return f"{name}({', '.join(parts)})"
