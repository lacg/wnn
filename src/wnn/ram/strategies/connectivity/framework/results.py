"""Optimizer result container + stop reasons."""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Generic, Optional, TypeVar

T = TypeVar('T')


class StopReason(IntEnum):
	"""Reason why optimization stopped early."""
	CONVERGENCE = auto()  # No improvement for patience iterations
	OVERFITTING = auto()  # Overfitting callback triggered early stop
	MAX_ITERATIONS = auto()  # Reached maximum iterations (not early stopped)
	SHUTDOWN = auto()  # External shutdown request (e.g., flow cancelled)


@dataclass


@dataclass
class OptimizerResult(Generic[T]):
	"""
	Unified result from optimization (GA, TS, SA).

	This is a generic result type that works with any genome type (Tensor, ClusterGenome, etc.)
	through the type parameter T.

	Naming conventions:
	- Uses 'genome' terminology (more generic than 'connections')
	- Uses 'fitness' terminology (minimization by default, lower is better)

	Attributes:
		initial_genome: Starting genome before optimization
		best_genome: Best genome found during optimization
		initial_fitness: Fitness of initial genome (lower is better)
		final_fitness: Fitness of best genome
		improvement_percent: Percentage improvement ((initial - final) / initial * 100)
		iterations_run: Number of iterations/generations run
		method_name: Name of the optimization method (e.g., "ArchitectureGA")
		history: List of (iteration, best_fitness) tuples for plotting
		early_stopped: Whether optimization stopped early (due to convergence or overfitting)
		stop_reason: Why optimization stopped (StopReason enum)
		final_population: Final population for seeding next phase (GA/TS)
		initial_accuracy: Optional accuracy at start
		final_accuracy: Optional accuracy at end
		final_threshold: Final accuracy threshold (pass to next phase for continuity)
	"""
	initial_genome: T
	best_genome: T
	initial_fitness: float
	final_fitness: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: list[tuple[int, float]] = field(default_factory=list)
	early_stopped: bool = False
	stop_reason: Optional[StopReason] = None
	# For population seeding between phases
	final_population: Optional[list[T]] = None
	# Per-genome (CE, accuracy, f1?, fpr?) matching final_population order
	population_metrics: Optional[list[tuple]] = None
	# Accuracy tracking
	initial_accuracy: Optional[float] = None
	final_accuracy: Optional[float] = None
	# Threshold continuity: pass to next phase (no hardcoded phase_index jumps)
	final_threshold: Optional[float] = None

	def __repr__(self) -> str:
		stop_str = f", stop={self.stop_reason.name}" if self.stop_reason else ""
		return (
			f"OptimizerResult("
			f"method={self.method_name}, "
			f"initial={self.initial_fitness:.4f}, "
			f"final={self.final_fitness:.4f}, "
			f"improvement={self.improvement_percent:.2f}%{stop_str})"
		)
