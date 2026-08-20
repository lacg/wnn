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
from .FitnessCalculator import FitnessCalculator


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
		zrank_clamp:   float = 3.0,
	):
		if aggregation not in ("harmonic", "arithmetic", "zscore"):
			raise ValueError(
				f"aggregation must be 'harmonic', 'arithmetic' or 'zscore', got {aggregation!r}")
		if not (zrank_clamp > 0.0):
			raise ValueError(f"zrank_clamp must be positive, got {zrank_clamp!r}")
		self.aggregation = aggregation
		# Winsorization bound for zscore (the λ_alt lesson: no single dimension
		# may capture the score, however extreme the outlier). Read only by zscore.
		self.zrank_clamp = float(zrank_clamp)
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

	# Ranking lives in ram_core::fitness since 19/08/2026 (the fractional tie fix
	# of 09/08 went with it — pinned by ranks_ties_share_average_positions in Rust).

	# Optional metric columns, in rank order: (Metrics attr, weight attr, warned
	# flag attr, why-it-can-be-None). All are COSTS (lower = better); err²/stable
	# are handled separately because they are REQUIRED (every ControllerMetrics
	# carries them) and because err² ranks on reward with higher_is_better=True.
	# Skip-with-one-warning preserves the long-standing policy: an unplumbed
	# metric drops its COLUMN loudly here, so a None can never reach the wheel —
	# the wheel REFUSES non-finite values rather than ranking around a scorer bug.
	_OPTIONAL_COLUMNS = (
		("motor_jerk_mean", "weight_jerk", "_warned_jerk",
		 "measurement not yet plumbed (compute_reward uses lambda_smooth=0), or a "
		 "degenerate eval returned None."),
		("mono_violations_total", "weight_mono", "_warned_mono",
		 "measurement not yet plumbed, or a degenerate eval returned None."),
		# steady: the I-pressure term — mean attitude err over the last 20% of
		# steps (the settled window), isolating the offset only an integrator
		# can kill from the transient that dominates err².
		("mean_steady_error_deg", "weight_steady", "_warned_steady",
		 "steady-state-window metric not plumbed by the scorer."),
		# effort: the Σu² allocation-efficiency term (overactuated Phase 3,
		# Luiz 12/07/2026) — on the attitude-only sim, allocation mismatch costs
		# EFFORT, not attitude error; the min-norm pinv baseline is the optimum.
		("mean_effort", "weight_effort", "_warned_effort",
		 "scorer predates the 13-metric row (ABI 9)."),
		# alt: the VERTICAL channel as its own scale-free dimension — exists so
		# the channel can be weighted WITHOUT a λ carrying the metres↔radians
		# conversion into the reward, where it silently scaled with capacity.
		("mean_altitude_error_m", "weight_alt", "_warned_alt",
		 "the run is not a --translation run, or the scorer predates metric row 14."),
		# pos: the HORIZONTAL channel; inert until stage 2 arms (--xy-offset > 0).
		("mean_position_error_m", "weight_pos", "_warned_pos",
		 "the run is not a --translation run, or the scorer predates metric row 13."),
	)

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		"""Reduce Metrics to domain-blind columns and let the WHEEL rank them.

		Since 19/08/2026 every number that ranks a genome is computed by
		ram_core::fitness (via ram_controller.fitness_combine) — the combine is
		results-determining logic and lives in the ABI-gated, cargo-tested wheel,
		not in the editable layer a live run can lazily import mid-edit (Luiz:
		"why is this in Python at all?"). This method keeps exactly two jobs:
		the Metrics→columns mapping, and the warn-once policy above. Columns
		carry an orientation flag and are NEVER pre-negated — a negation at one
		call site and not another is the drift class the wheel API forbids.
		"""
		n = len(metrics_list)
		if n == 0:
			return []
		if n == 1:
			return [1.0]

		flat: list[float] = []
		weights: list[float] = []
		higher: list[bool] = []

		# err² → ranked on reward, higher reward = better (reward has its OWN
		# field since 05/08/2026; the "-reward mirrored into ce" hack is gone).
		if self.weight_err_sq > 0:
			flat.extend(_controller_reward(m) for m in metrics_list)
			weights.append(self.weight_err_sq)
			higher.append(True)

		# stable_rate → higher = better.
		if self.weight_stable > 0:
			flat.extend(float(m.stable_rate) for m in metrics_list)
			weights.append(self.weight_stable)
			higher.append(True)

		for attr, weight_attr, warned_attr, why in self._OPTIONAL_COLUMNS:
			weight = getattr(self, weight_attr)
			if weight <= 0:
				continue
			vals = [getattr(m, attr, None) for m in metrics_list]
			if any(v is None for v in vals):
				if not getattr(self, warned_attr):
					warnings.warn(
						f"FitnessCalculatorControllerHarmonic: {weight_attr} > 0 but "
						f"Metrics.{attr} is None — {why} Weight ignored.",
						RuntimeWarning, stacklevel=2)
					setattr(self, warned_attr, True)
				continue
			flat.extend(float(v) for v in vals)
			weights.append(weight)
			higher.append(False)

		if not weights:
			return [1.0] * n

		# Lazy import, deliberately: this module is imported by the WORKER too
		# (the IDS calculators share the package), and the worker must not need
		# ram_controller installed to import wnn.ram.fitness.
		from wnn.control._accel import fitness_combine
		return list(fitness_combine(flat, n, weights, higher,
		                            self.aggregation, self.zrank_clamp))

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
		# the alt=0.00 label rule exists to prevent. ZRank carries no Controller
		# prefix: the combine is domain-blind ram_core math shared with IDS, and
		# the metric names in the parens already say which domain this is.
		name = {"harmonic": "ControllerHarmonic",
		        "arithmetic": "ControllerArithRank",
		        "zscore": "ZRank"}[self.aggregation]
		return f"{name}({', '.join(parts)})"
