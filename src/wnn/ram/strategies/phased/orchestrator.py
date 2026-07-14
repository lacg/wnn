"""PhasedOrchestrator — the shared phase-sequencing skeleton (D1).

Owns the loop both strands used to duplicate: iterate phases, skip completed
ones on resume, maintain the full-population CarryState, and persist a unified
schema-2 checkpoint after each phase.

Cancellation/crash-save is NOT the orchestrator's job: each strand's GA strategy
owns the cooperative-cancel + adaptive crash-save hook (the shared
GenericGAStrategy core — IDS via ArchitectureGAStrategy, the controller via
ControllerCancelMixin), and the finished-phase state is already on disk via the
per-phase checkpoint. The old base EmergencyDump (a SIGTERM handler dumping the
carry) was fully redundant with those two and was retired (D-series cleanup).

Subclasses implement run_phase() (strategy construction + execution) and may
override the hooks for dashboard/tracker reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from wnn.ram.strategies.phased.carry import CarryState
from wnn.ram.strategies.phased.checkpoint import PhaseCheckpoint, load_checkpoint, save_checkpoint
from wnn.ram.strategies.phased.codecs import GenomeCodec


@dataclass
class PhaseSpec:
	"""One phase in the sequence. `payload` carries strand-specific config."""
	key: str
	name: str
	payload: Any = None


@dataclass
class PhaseOutcome:
	"""What a phase hands back to the orchestrator."""
	best_genome: Any = None
	final_population: Optional[list] = None
	final_threshold: Optional[float] = None
	final_fitness: Optional[float] = None
	final_accuracy: Optional[float] = None
	initial_fitness: Optional[float] = None
	initial_accuracy: Optional[float] = None
	iterations_run: int = 0
	patience: int = 0
	strategy_type: str = ""
	extra: dict = field(default_factory=dict)
	# A phase may signal "skip me" (e.g. deleted from dashboard) with None outcome


class PhasedOrchestrator(ABC):
	"""Sequencing skeleton: carry + checkpoints + emergency dump + hooks."""

	def __init__(
		self,
		checkpoint_dir: "str | Path | None",
		codec: GenomeCodec,
		log: Callable[[str], None] = print,
	):
		self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
		self._codec = codec
		self._log = log

	# ------------------------------------------------------------------ hooks
	@abstractmethod
	def run_phase(self, spec: PhaseSpec, carry: CarryState, index: int) -> Optional[PhaseOutcome]:
		"""Run one phase. Return None to skip (carry passes through unchanged)."""
		...

	def on_phase_start(self, spec: PhaseSpec, index: int) -> None: ...
	def on_phase_complete(self, spec: PhaseSpec, index: int, outcome: PhaseOutcome) -> None: ...

	# ------------------------------------------------------------ checkpoints
	def checkpoint_path(self, spec: PhaseSpec) -> Optional[Path]:
		if self._checkpoint_dir is None:
			return None
		safe = spec.name.replace(" ", "_").replace(":", "")
		return self._checkpoint_dir / f"phase_{spec.key}_{safe}.json"

	def load_phase_checkpoint(self, spec: PhaseSpec) -> Optional[PhaseCheckpoint]:
		p = self.checkpoint_path(spec)
		return load_checkpoint(p, self._codec) if p else None

	def _save_phase_checkpoint(self, spec: PhaseSpec, outcome: PhaseOutcome) -> None:
		p = self.checkpoint_path(spec)
		if p is None:
			return
		ckpt = PhaseCheckpoint(
			phase_key=spec.key,
			phase_name=spec.name,
			strategy_type=outcome.strategy_type,
			best_genome=outcome.best_genome,
			final_population=outcome.final_population,
			iterations_run=outcome.iterations_run,
			patience=outcome.patience,
			final_fitness=outcome.final_fitness,
			final_accuracy=outcome.final_accuracy,
			final_threshold=outcome.final_threshold,
			initial_fitness=outcome.initial_fitness,
			initial_accuracy=outcome.initial_accuracy,
			extra=outcome.extra,
		)
		save_checkpoint(p, ckpt, self._codec)

	# ------------------------------------------------------------------ loop
	def run_all(
		self,
		phases: list[PhaseSpec],
		carry: CarryState,
		resume_from: Optional[str] = None,
	) -> dict[str, PhaseOutcome]:
		"""Run the sequence. On resume, completed phases load from checkpoints
		and feed the carry exactly as if they had just run."""
		outcomes: dict[str, PhaseOutcome] = {}
		start_idx = 0
		if resume_from is not None:
			keys = [s.key for s in phases]
			if resume_from not in keys:
				raise ValueError(f"resume_from {resume_from!r} not in phases {keys}")
			start_idx = keys.index(resume_from)

		for index, spec in enumerate(phases):
			if index < start_idx:
				ckpt = self.load_phase_checkpoint(spec)
				if ckpt is None:
					raise ValueError(
						f"Cannot resume from {resume_from}: missing checkpoint for phase {spec.key}")
				carry.update(ckpt.best_genome, ckpt.final_population, ckpt.final_threshold)
				outcomes[spec.key] = PhaseOutcome(
					best_genome=ckpt.best_genome,
					final_population=ckpt.final_population,
					final_threshold=ckpt.final_threshold,
					final_fitness=ckpt.final_fitness,
					final_accuracy=ckpt.final_accuracy,
					iterations_run=ckpt.iterations_run,
					patience=ckpt.patience,
					strategy_type=ckpt.strategy_type,
					extra=ckpt.extra,
				)
				self._log(f"[phased] resumed phase {spec.key} from checkpoint "
				          f"(fitness={ckpt.final_fitness}, gen={ckpt.iterations_run})")
				continue

			self.on_phase_start(spec, index)
			outcome = self.run_phase(spec, carry, index)
			if outcome is None:
				self._log(f"[phased] phase {spec.key} skipped — carry passes through")
				continue
			carry.update(outcome.best_genome, outcome.final_population, outcome.final_threshold)
			self._save_phase_checkpoint(spec, outcome)
			outcomes[spec.key] = outcome
			self.on_phase_complete(spec, index, outcome)
		return outcomes
