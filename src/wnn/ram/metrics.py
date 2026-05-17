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


@dataclass
class Metrics:
	"""Evaluation metrics for a single genome.

	This is THE canonical representation of genome quality across the entire
	system. Passed between evaluators, fitness calculators, GA/TS populations,
	validation summaries, and checkpoints.

	Fields:
		ce: Cross-entropy loss (lower = better). Always present.
		acc: Accuracy (higher = better). Always present.
		f1: F1-macro score (higher = better). None if not computed.
		fpr: False positive rate (lower = better). None if not computed.
		threshold: Decision threshold (single-cluster IDS). None if N/A.
		fitness: Composite fitness score from the calculator. None until computed.
		bit_accuracy: Weighted bit accuracy (bitwise LM only). None if N/A.
	"""
	ce: float
	acc: float
	f1: Optional[float] = None
	fpr: Optional[float] = None
	threshold: Optional[float] = None
	fitness: Optional[float] = None
	bit_accuracy: Optional[float] = None
	stage_metrics: Optional[list['Metrics']] = None  # Per-stage breakdown (multi-stage LM)
	# Best-effort per-genome wall-clock for training+eval (ms). Populated by
	# the IDS hybrid evaluator path; None for paths that don't measure.
	eval_time_ms: Optional[int] = None

	def get(self, metric: MetricType) -> Optional[float]:
		"""Get metric value by type."""
		return {
			MetricType.CE: self.ce,
			MetricType.ACC: self.acc,
			MetricType.F1: self.f1,
			MetricType.FPR: self.fpr,
		}[metric]

	def is_better_than(self, other: 'Metrics', metric: MetricType) -> bool:
		"""Compare two Metrics by a specific metric type."""
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

	def to_dict(self) -> dict:
		"""Serialize to dict (for JSON/checkpoint storage)."""
		d = {"ce": self.ce, "acc": self.acc}
		if self.f1 is not None:
			d["f1"] = self.f1
		if self.fpr is not None:
			d["fpr"] = self.fpr
		if self.threshold is not None:
			d["threshold"] = self.threshold
		if self.fitness is not None:
			d["fitness"] = self.fitness
		if self.bit_accuracy is not None:
			d["bit_accuracy"] = self.bit_accuracy
		if self.stage_metrics is not None:
			d["stage_metrics"] = [sm.to_dict() for sm in self.stage_metrics]
		return d

	@classmethod
	def from_dict(cls, d: dict) -> 'Metrics':
		"""Deserialize from dict."""
		stage = None
		if "stage_metrics" in d:
			stage = [cls.from_dict(sm) for sm in d["stage_metrics"]]
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
		return f"Metrics({', '.join(parts)})"


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
