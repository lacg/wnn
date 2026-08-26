"""
Rank-combine fitness calculator for IDS/LM genome ranking.

Reduces Metrics to domain-blind columns (CE, accuracy, F1-macro, FPR) and lets
ram_core rank them. The combine step is selectable — harmonic (the banked WHM),
arithmetic, or zscore — which is what makes the Z_RANK ablation possible.
"""

import warnings

from wnn.ram.metrics import Metrics, FitnessWeights
from .FitnessCalculator import FitnessCalculator


# Median best_ce over 18,355 unsw-nb15 BINARY iterations (IDSZ), frozen 26/08/2026.
# Equivalently ce/H(p) = 0.1937 of that task's base-rate log-loss — which is the
# portable form, and the one ram_accelerator derives per task.
FROZEN_BINARY_CE_ANCHOR = 0.133


class FitnessCalculatorHarmonicRank(FitnessCalculator):
	"""Rank-combine over the IDS metrics. Lower score = better.

	The default combine is the weighted harmonic mean of per-metric ranks:

		WHM = sum(weights) / sum(weight_i / rank_i)

	Metrics with weight=0 are excluded. Since 20/08/2026 the combine itself runs
	in ram_core (via `wnn.accel.fitness_combine`) rather than here — the same
	code the controller ranks with. This class keeps two jobs: the
	Metrics->columns mapping, and the warn-once policy for absent metrics.

	⚠️ The WHM is dominated by a genome's BEST weighted rank and nearly
	indifferent to its worst — rank 1 at weight .35 contributes 0.350 while rank
	9 at weight .15 contributes 0.017. It therefore rewards SPECIALISTS, the
	opposite of the "penalizes imbalance" claim in CLAUDE.md; the arithmetic
	combine is the one that punishes a bad metric. Ranks are also
	magnitude-blind: first by 13 points and first by 0.1 points score
	identically. `aggregation="zscore"` (winsorized robust z, clamped) answers
	both objections.

	`aggregation="harmonic"` is numerically the old Python path — the Rust
	combine uses the same fractional-tie rule (09/08/2026 fix) — so the banked
	SP100 runs stay comparable.
	"""

	def __init__(
		self,
		weight_ce:    float = 1.0,
		weight_acc:   float = 1.0,
		weight_f1:    float = 0.0,
		weight_fpr:   float = 0.0,
		aggregation:  str   = "harmonic",
		zrank_clamp:  float = 3.0,
		ce_anchor:    float | None = None,
	):
		if aggregation not in ("harmonic", "arithmetic", "zscore", "desirability"):
			raise ValueError(
				"aggregation must be 'harmonic', 'arithmetic', 'zscore' or "
				f"'desirability', got {aggregation!r}")
		if not (zrank_clamp > 0.0):
			raise ValueError(f"zrank_clamp must be positive, got {zrank_clamp!r}")
		self.weights = FitnessWeights(ce=weight_ce, acc=weight_acc, f1=weight_f1, fpr=weight_fpr)
		self.aggregation = aggregation
		self.zrank_clamp = float(zrank_clamp)
		# None keeps the FROZEN unsw-nb15 BINARY fit, so every flow queued
		# before the anchor was derivable scores bit-identically. A caller that
		# passes one has normalised it against the task's own base-rate entropy
		# (ram_accelerator ABI 9) — see _DESIRABILITY_COLUMNS.
		if ce_anchor is not None and not (float(ce_anchor) > 0.0):
			raise ValueError(f"ce_anchor must be > 0, got {ce_anchor!r}")
		self.ce_anchor = FROZEN_BINARY_CE_ANCHOR if ce_anchor is None else float(ce_anchor)
		self._warned_f1 = False
		self._warned_fpr = False

	# (metric attr, weight attr, warned attr, higher-is-better, why it may be absent)
	_COLUMNS = (
		("ce",  "ce",  None,          False, ""),
		("acc", "acc", None,          True,  ""),
		("f1",  "f1",  "_warned_f1",  True,
		 "the evaluator did not report F1 — a binary-mode run, or a scorer predating per-genome F1."),
		("fpr", "fpr", "_warned_fpr", False,
		 "the evaluator did not report FPR — a binary-mode run, or a scorer predating per-genome FPR."),
	)

	def fitness(self, metrics_list: list[Metrics]) -> list[float]:
		"""Reduce Metrics to domain-blind columns and let the WHEEL rank them."""
		n = len(metrics_list)
		if n == 0:
			return []
		if self.aggregation == "desirability":
			return self._desirability_fitness(metrics_list)
		if n == 1:
			return [1.0]

		flat:    list[float] = []
		weights: list[float] = []
		higher:  list[bool]  = []

		for attr, weight_attr, warned_attr, is_higher, why in self._COLUMNS:
			weight = getattr(self.weights, weight_attr)
			if weight <= 0:
				continue
			vals = [getattr(m, attr, None) for m in metrics_list]
			if any(v is None for v in vals):
				# ce/acc are always present; f1/fpr are optional and warn once.
				if warned_attr is not None and not getattr(self, warned_attr):
					warnings.warn(
						f"FitnessCalculatorHarmonicRank: weight_{weight_attr} > 0 but "
						f"Metrics.{attr} is None — {why} Weight ignored.",
						RuntimeWarning, stacklevel=2)
					setattr(self, warned_attr, True)
				continue
			flat.extend(float(v) for v in vals)
			weights.append(weight)
			higher.append(is_higher)

		if not weights:
			return [1.0] * n

		# Lazy import, deliberately: this module is imported by the CONTROLLER
		# too (the package is shared), and a controller process must not need
		# ram_accelerator installed to import wnn.ram.fitness.
		from wnn.accel import fitness_combine
		return list(fitness_combine(flat, n, weights, higher,
		                            self.aggregation, self.zrank_clamp))

	# Desirability column map (26/08/2026, docs/DESIRABILITY_FITNESS_SHAPES.md):
	# weight field -> (Metrics attr, shape, half-anchor). ce anchor 0.133 is the
	# MEASURED median best_ce over 18,355 unswt iterations (IDSZ) — frozen for
	# the unswt A/B; a per-dataset-family anchor param comes before any
	# cross-dataset desirability cohort.
	# The ce entry's anchor is a PLACEHOLDER — `self.ce_anchor` overrides it per
	# instance. CE is the one column here with no absolute meaning: it carries
	# units and scales with the class count and imbalance, so 0.133 (a BINARY
	# fit) is 14.3x too small on the same dataset's 10-class task, which drives
	# the column toward the h<=20 clamp and inflates its share of the score far
	# above its weight. The other three are bounded (f1, acc) or carry an
	# absolute operator threshold (fpr 0.10) and are therefore portable as-is.
	_DESIRABILITY_COLUMNS = (
		("ce",  "ce",       "exp",   FROZEN_BINARY_CE_ANCHOR),
		("acc", "acc",      "power", 0.80),
		("f1",  "f1",       "power", 0.80),
		("fpr", "fpr",      "exp",   0.10),
	)

	def _desirability_fitness(self, metrics_list: list[Metrics]) -> list[float]:
		"""score = Σ w·h = weighted half-lives of desirability lost (lower =
		better; ram_core::fitness::desirability_combine_flat, worker ABI 8).
		ABSOLUTE, not pool-relative — n == 1 is scored for real. A weighted
		metric that is None RAISES: silently dropping a column re-creates the
		"weights inert" failure this mode exists to end (the controller
		calculator makes the same promise)."""
		n = len(metrics_list)
		flat: list[float] = []
		weights: list[float] = []
		shapes: list[str] = []
		anchors: list[float] = []
		for weight_attr, attr, shape, anchor in self._DESIRABILITY_COLUMNS:
			weight = getattr(self.weights, weight_attr)
			if weight <= 0:
				continue
			vals = [getattr(m, attr, None) for m in metrics_list]
			if any(v is None for v in vals):
				raise ValueError(
					f"desirability: weight_{weight_attr} > 0 but Metrics.{attr} is "
					"None — binary-mode scorer without per-genome F1/FPR, or a "
					"legacy metrics row. Fix the scorer or zero the weight; "
					"desirability never silently drops a weighted metric.")
			flat.extend(float(v) for v in vals)
			weights.append(float(weight))
			shapes.append(shape)
			anchors.append(self.ce_anchor if weight_attr == "ce" else float(anchor))
		if not weights:
			return [1.0] * n
		from wnn.accel import desirability_fitness_combine
		return list(desirability_fitness_combine(flat, n, weights, shapes, anchors))

	@property
	def name(self) -> str:
		parts = []
		if self.weights.ce != 1.0 or self.weights.fpr > 0:
			parts.append(f"ce={self.weights.ce}")
		if self.weights.acc != 1.0 or self.weights.fpr > 0:
			parts.append(f"acc={self.weights.acc}")
		if self.weights.f1 > 0:
			parts.append(f"f1={self.weights.f1}")
		if self.weights.fpr > 0:
			parts.append(f"fpr={self.weights.fpr}")
		# A non-frozen anchor RENAMES the calculator for the same reason the
		# aggregation does: two runs with identical weights but different CE
		# scales select DIFFERENT genomes, so a label that hid the anchor would
		# make them look like the same fitness function.
		if self.aggregation == "desirability" and self.ce_anchor != FROZEN_BINARY_CE_ANCHOR:
			parts.append(f"ce_anchor={self.ce_anchor:.4f}")
		# The aggregation RENAMES the calculator rather than annotating it: two
		# runs with identical weights but different combines select DIFFERENT
		# genomes, so a label that hid the combine would make them look like the
		# same fitness function. ZRank carries no domain prefix — the combine is
		# domain-blind ram_core math shared with the controller.
		name = {"harmonic": "HarmonicRank",
		        "arithmetic": "ArithRank",
		        "zscore": "ZRank",
		        "desirability": "Desir"}[self.aggregation]
		return f"{name}({', '.join(parts)})" if parts else name
