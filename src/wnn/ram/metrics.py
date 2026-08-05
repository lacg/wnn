"""Core metric types for WNN evaluation and fitness ranking.

Provides a single source of truth for metric representation across the entire
system: evaluators, fitness calculators, GA/TS populations, validation summaries,
checkpoints, and the dashboard.

Usage:
	from wnn.ram.metrics import Metrics, MetricType, GenomeType
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class MetricType(Enum):
	"""Metric identifiers. Used for fitness weight configuration and metric access."""
	CE = "ce"       # Cross-entropy (lower = better)
	ACC = "acc"     # Accuracy (higher = better)
	F1 = "f1"       # F1-macro (higher = better)
	FPR = "fpr"     # False positive rate (lower = better)

	@property
	def lower_is_better(self) -> bool:
		return self in (MetricType.CE, MetricType.FPR)

	@property
	def higher_is_better(self) -> bool:
		return self in (MetricType.ACC, MetricType.F1)


class GenomeType(Enum):
	"""Genome selection criteria. Determines which genome is tracked/saved."""
	BEST_CE = "best_ce"
	BEST_ACC = "best_acc"
	BEST_F1 = "best_f1"
	BEST_FPR = "best_fpr"
	BEST_FITNESS = "best_fitness"

	@property
	def metric(self) -> Optional[MetricType]:
		"""The metric this genome type optimizes for, or None for composite fitness."""
		return {
			GenomeType.BEST_CE: MetricType.CE,
			GenomeType.BEST_ACC: MetricType.ACC,
			GenomeType.BEST_F1: MetricType.F1,
			GenomeType.BEST_FPR: MetricType.FPR,
			GenomeType.BEST_FITNESS: None,
		}[self]

	@property
	def label(self) -> str:
		"""Human-readable label for dashboard display."""
		return {
			GenomeType.BEST_CE: "Best CE",
			GenomeType.BEST_ACC: "Best Accuracy",
			GenomeType.BEST_F1: "Best F1-Macro",
			GenomeType.BEST_FPR: "Best FPR",
			GenomeType.BEST_FITNESS: "Best Fitness",
		}[self]


@dataclass(kw_only=True)
class Metrics:
	"""Domain-neutral base for genome-quality metrics.

	Carries ONLY what every domain shares: the composite `fitness` the calculator
	assigned (the one cross-domain ranking value — see feedback_rank_by_fitness_not_ce)
	and eval bookkeeping. Domain measurements live on the subclasses:

	- `IDSMetrics`: classification (IDS, and the LM benchmarks share the shape) —
	  ce/acc/f1/fpr. CE is real there: it is computed, and it feeds the Wa/Wb/Wc
	  fitness schemes.
	- `ControllerMetrics`: closed-loop control — reward/stable_rate/err°/steady°.

	HISTORY, so nobody reintroduces it: until 05/08/2026 `ce` and `acc` were REQUIRED
	fields on this base, so the controller was structurally obliged to fill them and
	did so with `ce = -reward`, `acc = stable_rate`. Every "CE=..." a controller log
	ever printed was negated reward wearing a CE label. That mislabeling is exactly
	what this split removes (Luiz: "remove CE from everywhere" — everywhere it was
	fake; it stays where it is real).
	"""
	fitness: Optional[float] = None
	# Best-effort per-genome wall-clock for training+eval (ms). Populated by
	# the IDS hybrid evaluator path; None for paths that don't measure.
	eval_time_ms: Optional[int] = None

	def to_dict(self) -> dict:
		d = {"kind": self._kind()}
		if self.fitness is not None:
			d["fitness"] = self.fitness
		return d

	def _kind(self) -> str:
		return "base"

	@classmethod
	def from_dict(cls, d: dict) -> 'Metrics':
		"""Factory: dispatch on the serialized kind. Legacy dicts (pre-05/08/2026)
		have no "kind" but always have "ce" — every one of those was written by an
		IDS/LM path OR by a controller path whose ce was -reward; they load as
		IDSMetrics, and the controller fitness calculators refuse them loudly with
		instructions to re-evaluate (safer than silently resurrecting a mislabel)."""
		kind = d.get("kind")
		if kind == "controller":
			return ControllerMetrics._load(d)
		if kind == "ids" or "ce" in d:
			return IDSMetrics._load(d)
		return cls(fitness=d.get("fitness"))


@dataclass(kw_only=True)
class IDSMetrics(Metrics):
	"""Classification metrics — IDS flows and the LM benchmarks (same shape).

	Fields:
		ce: Cross-entropy loss (lower = better). REAL here: computed by the
		    evaluators and consumed by the Wa/Wb/Wc fitness schemes.
		acc: Accuracy (higher = better).
		f1 / fpr: IDS-only extras. None if not computed.
	"""
	ce: float
	acc: float
	f1: Optional[float] = None
	fpr: Optional[float] = None
	threshold: Optional[float] = None
	bit_accuracy: Optional[float] = None
	stage_metrics: Optional[list['IDSMetrics']] = None  # Per-stage breakdown (multi-stage LM)

	def get(self, metric: MetricType) -> Optional[float]:
		"""Get metric value by type."""
		return {
			MetricType.CE: self.ce,
			MetricType.ACC: self.acc,
			MetricType.F1: self.f1,
			MetricType.FPR: self.fpr,
		}[metric]

	def is_better_than(self, other: 'IDSMetrics', metric: MetricType) -> bool:
		"""Compare two IDSMetrics by a specific metric type."""
		a = self.get(metric)
		b = other.get(metric)
		if a is None or b is None:
			return False
		if metric.lower_is_better:
			return a < b
		return a > b

	def has_ids_metrics(self) -> bool:
		"""Whether F1 and FPR are available (IDS mode)."""
		return self.f1 is not None and self.fpr is not None

	def _kind(self) -> str:
		return "ids"

	def to_dict(self) -> dict:
		d = super().to_dict()
		d["ce"] = self.ce
		d["acc"] = self.acc
		if self.f1 is not None:
			d["f1"] = self.f1
		if self.fpr is not None:
			d["fpr"] = self.fpr
		if self.threshold is not None:
			d["threshold"] = self.threshold
		if self.bit_accuracy is not None:
			d["bit_accuracy"] = self.bit_accuracy
		if self.stage_metrics is not None:
			d["stage_metrics"] = [sm.to_dict() for sm in self.stage_metrics]
		return d

	@classmethod
	def _load(cls, d: dict) -> 'IDSMetrics':
		stage = None
		if "stage_metrics" in d:
			stage = [cls._load(sm) for sm in d["stage_metrics"]]
		return cls(
			ce=d["ce"],
			acc=d["acc"],
			f1=d.get("f1"),
			fpr=d.get("fpr"),
			threshold=d.get("threshold"),
			fitness=d.get("fitness"),
			bit_accuracy=d.get("bit_accuracy"),
			stage_metrics=stage,
		)

	def __repr__(self) -> str:
		parts = [f"CE={self.ce:.4f}", f"Acc={self.acc:.2%}"]
		if self.f1 is not None:
			parts.append(f"F1={self.f1:.2%}")
		if self.fpr is not None:
			parts.append(f"FPR={self.fpr:.2%}")
		if self.fitness is not None:
			parts.append(f"fit={self.fitness:.4f}")
		return f"IDSMetrics({', '.join(parts)})"


@dataclass(kw_only=True)
class ControllerMetrics(Metrics):
	"""Closed-loop controller metrics. NO ce field — nothing here is a cross-entropy,
	and the fitness (ControllerHarmonic: err²/stable/jerk/mono) has no CE term.

	Fields:
		reward: closed-loop reward (higher = better). Its OWN field now — it used to
		        hide as `-ce`, which is why FitnessCalculatorController once carried a
		        `-m.ce` fallback: the composite fitness overwrites `fitness`, and the
		        reward had nowhere else to live.
		stable_rate: fraction of stable episodes (higher = better). Its OWN field —
		             it used to be stored in `acc`.
	"""
	reward: float
	stable_rate: float
	mean_attitude_error_deg: Optional[float] = None  # closed-loop mean attitude error
	motor_jerk_mean: Optional[float] = None        # mean per-step Σ(PWM_delta)² across episodes
	mono_violations_total: Optional[float] = None  # mean monotonicity violations per step
	mean_steady_error_deg: Optional[float] = None  # mean attitude err over last 20% of steps (I-pressure)
	mean_effort: Optional[float] = None            # mean per-step Σ(PWM²) — allocation-effort proxy (Σu², Phase 3)

	@property
	def acc(self) -> float:
		"""Alias for generic display code (the GA gen line labels it 'stable' on the
		controller path). Read-only ON PURPOSE: the real field is stable_rate."""
		return self.stable_rate

	def _kind(self) -> str:
		return "controller"

	def to_dict(self) -> dict:
		d = super().to_dict()
		d["reward"] = self.reward
		d["stable_rate"] = self.stable_rate
		for k in ("mean_attitude_error_deg", "motor_jerk_mean",
		          "mono_violations_total", "mean_steady_error_deg", "mean_effort"):
			v = getattr(self, k)
			if v is not None:
				d[k] = v
		return d

	@classmethod
	def _load(cls, d: dict) -> 'ControllerMetrics':
		return cls(
			reward=d["reward"],
			stable_rate=d["stable_rate"],
			fitness=d.get("fitness"),
			mean_attitude_error_deg=d.get("mean_attitude_error_deg"),
			motor_jerk_mean=d.get("motor_jerk_mean"),
			mono_violations_total=d.get("mono_violations_total"),
			mean_steady_error_deg=d.get("mean_steady_error_deg"),
			mean_effort=d.get("mean_effort"),
		)

	def __repr__(self) -> str:
		parts = [f"reward={self.reward:.4f}", f"stable={self.stable_rate:.2%}"]
		if self.mean_attitude_error_deg is not None:
			parts.append(f"err={self.mean_attitude_error_deg:.2f}°")
		if self.mean_steady_error_deg is not None:
			parts.append(f"steady={self.mean_steady_error_deg:.2f}°")
		if self.fitness is not None:
			parts.append(f"fit={self.fitness:.4f}")
		return f"ControllerMetrics({', '.join(parts)})"


@dataclass
class FitnessWeights:
	"""Configuration for the fitness calculator weights.

	Passed as a single object instead of 4 separate floats through the stack.
	"""
	ce: float = 1.0
	acc: float = 1.0
	f1: float = 0.0
	fpr: float = 0.0

	def active_metrics(self) -> list[MetricType]:
		"""Return list of metrics with non-zero weight."""
		result = []
		if self.ce > 0:
			result.append(MetricType.CE)
		if self.acc > 0:
			result.append(MetricType.ACC)
		if self.f1 > 0:
			result.append(MetricType.F1)
		if self.fpr > 0:
			result.append(MetricType.FPR)
		return result

	def total(self) -> float:
		return self.ce + self.acc + self.f1 + self.fpr

	def get(self, metric: MetricType) -> float:
		return {
			MetricType.CE: self.ce,
			MetricType.ACC: self.acc,
			MetricType.F1: self.f1,
			MetricType.FPR: self.fpr,
		}[metric]

	def to_dict(self) -> dict:
		return {"ce": self.ce, "acc": self.acc, "f1": self.f1, "fpr": self.fpr}

	@classmethod
	def from_dict(cls, d: dict) -> 'FitnessWeights':
		return cls(
			ce=d.get("ce", d.get("fitness_weight_ce", 1.0)),
			acc=d.get("acc", d.get("fitness_weight_acc", 1.0)),
			f1=d.get("f1", d.get("fitness_weight_f1", 0.0)),
			fpr=d.get("fpr", d.get("fitness_weight_fpr", 0.0)),
		)

	@classmethod
	def from_params(cls, params: dict) -> 'FitnessWeights':
		"""Extract from flow/experiment params dict."""
		return cls(
			ce=params.get("fitness_weight_ce", 1.0),
			acc=params.get("fitness_weight_acc", 1.0),
			f1=params.get("fitness_weight_f1", params.get("ids_fitness_weight_f1", 0.0)),
			fpr=params.get("fitness_weight_fpr", params.get("ids_fitness_weight_fpr", 0.0)),
		)

	def __repr__(self) -> str:
		parts = []
		if self.ce > 0:
			parts.append(f"CE={self.ce}")
		if self.acc > 0:
			parts.append(f"Acc={self.acc}")
		if self.f1 > 0:
			parts.append(f"F1={self.f1}")
		if self.fpr > 0:
			parts.append(f"FPR={self.fpr}")
		return f"FitnessWeights({', '.join(parts)})"
