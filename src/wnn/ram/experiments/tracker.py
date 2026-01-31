"""
Experiment tracking abstraction layer.

This module provides an interface for tracking experiment progress that WNN code
can use without depending on specific storage implementations (SQLite, files, etc.).

Usage:
    from wnn.ram.experiments.tracker import ExperimentTracker, create_tracker

    # Create a tracker (implementation chosen by configuration)
    tracker = create_tracker(db_path="/path/to/db.sqlite")

    # Use in experiment code
    exp_id = tracker.start_experiment("my-experiment", ...)
    phase_id = tracker.start_phase(exp_id, "1-GA-Neurons", ...)
    iter_id = tracker.record_iteration(phase_id, iteration_num=1, best_ce=10.5, ...)
    tracker.record_genome_evaluation(iter_id, genome_id, ce=10.5, accuracy=0.01, ...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class TrackerStatus(str, Enum):
    """Status values for flows, experiments, and phases."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class FitnessCalculatorType(str, Enum):
    """Fitness calculator types."""
    CE = "ce"
    HARMONIC_RANK = "harmonic_rank"
    WEIGHTED_HARMONIC = "weighted_harmonic"


class GenomeRole(str, Enum):
    """Role of a genome in an iteration."""
    ELITE = "elite"
    OFFSPRING = "offspring"
    INIT = "init"


class CheckpointType(str, Enum):
    """Checkpoint types."""
    AUTO = "auto"
    USER = "user"
    PHASE_END = "phase_end"
    EXPERIMENT_END = "experiment_end"


@dataclass
class TierConfig:
    """Configuration for a single tier."""
    tier: int
    clusters: int
    neurons: int
    bits: int


@dataclass
class GenomeConfig:
    """Full genome configuration with tiers."""
    tiers: list[TierConfig]

    @property
    def total_clusters(self) -> int:
        return sum(t.clusters for t in self.tiers)

    @property
    def total_neurons(self) -> int:
        return sum(t.clusters * t.neurons for t in self.tiers)

    @property
    def total_memory_bytes(self) -> int:
        return sum(t.clusters * t.neurons * (2 ** t.bits) for t in self.tiers)


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking.

    WNN code should only depend on this interface, not on specific implementations.
    This allows swapping storage backends (SQLite, files, cloud, etc.) without
    changing experiment code.
    """

    # =========================================================================
    # Flow lifecycle
    # =========================================================================

    @abstractmethod
    def create_flow(
        self,
        name: str,
        config: dict[str, Any],
        description: Optional[str] = None,
    ) -> int:
        """Create a new flow (sequence of experiments). Returns flow ID."""
        pass

    @abstractmethod
    def update_flow_status(self, flow_id: int, status: TrackerStatus) -> None:
        """Update flow status."""
        pass

    # =========================================================================
    # Experiment lifecycle
    # =========================================================================

    @abstractmethod
    def start_experiment(
        self,
        name: str,
        flow_id: Optional[int] = None,
        fitness_calculator: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK,
        fitness_weight_ce: float = 1.0,
        fitness_weight_acc: float = 1.0,
        tier_config: Optional[str] = None,
        context_size: int = 4,
        population_size: int = 50,
    ) -> int:
        """Start a new experiment. Returns experiment ID."""
        pass

    @abstractmethod
    def update_experiment_status(
        self,
        experiment_id: int,
        status: TrackerStatus,
    ) -> None:
        """Update experiment status."""
        pass

    @abstractmethod
    def update_experiment_progress(
        self,
        experiment_id: int,
        last_phase_id: Optional[int] = None,
        last_iteration: Optional[int] = None,
        checkpoint_id: Optional[int] = None,
    ) -> None:
        """Update experiment progress for resume tracking."""
        pass

    # =========================================================================
    # Phase lifecycle
    # =========================================================================

    @abstractmethod
    def start_phase(
        self,
        experiment_id: int,
        name: str,
        phase_type: str,
        sequence_order: int,
        max_iterations: int = 250,
        population_size: Optional[int] = None,
    ) -> int:
        """Start a new phase. Returns phase ID."""
        pass

    @abstractmethod
    def update_phase_status(self, phase_id: int, status: TrackerStatus) -> None:
        """Update phase status."""
        pass

    @abstractmethod
    def update_phase_progress(
        self,
        phase_id: int,
        current_iteration: int,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
    ) -> None:
        """Update phase progress."""
        pass

    # =========================================================================
    # Iteration tracking
    # =========================================================================

    @abstractmethod
    def record_iteration(
        self,
        phase_id: int,
        iteration_num: int,
        best_ce: float,
        best_accuracy: Optional[float] = None,
        avg_ce: Optional[float] = None,
        avg_accuracy: Optional[float] = None,
        elite_count: Optional[int] = None,
        offspring_count: Optional[int] = None,
        offspring_viable: Optional[int] = None,
        fitness_threshold: Optional[float] = None,
        elapsed_secs: Optional[float] = None,
    ) -> int:
        """Record an iteration/generation. Returns iteration ID."""
        pass

    # =========================================================================
    # Genome tracking
    # =========================================================================

    @abstractmethod
    def get_or_create_genome(
        self,
        experiment_id: int,
        genome_config: GenomeConfig,
    ) -> int:
        """Get or create a genome record. Returns genome ID."""
        pass

    @abstractmethod
    def record_genome_evaluation(
        self,
        iteration_id: int,
        genome_id: int,
        position: int,
        role: GenomeRole,
        ce: float,
        accuracy: float,
        elite_rank: Optional[int] = None,
        fitness_score: Optional[float] = None,
        eval_time_ms: Optional[int] = None,
    ) -> int:
        """Record a genome evaluation. Returns evaluation ID."""
        pass

    @abstractmethod
    def record_genome_evaluations_batch(
        self,
        evaluations: list[dict],
    ) -> list[int]:
        """Record multiple genome evaluations. Returns evaluation IDs."""
        pass

    # =========================================================================
    # Health check tracking
    # =========================================================================

    @abstractmethod
    def record_health_check(
        self,
        iteration_id: int,
        k: int,
        top_k_ce: float,
        top_k_accuracy: float,
        best_ce: Optional[float] = None,
        best_ce_accuracy: Optional[float] = None,
        best_acc_ce: Optional[float] = None,
        best_acc_accuracy: Optional[float] = None,
        patience_remaining: Optional[int] = None,
        patience_status: Optional[str] = None,
    ) -> int:
        """Record a health check. Returns health check ID."""
        pass

    # =========================================================================
    # Checkpoint tracking
    # =========================================================================

    @abstractmethod
    def record_checkpoint(
        self,
        experiment_id: int,
        name: str,
        file_path: str,
        checkpoint_type: CheckpointType,
        phase_id: Optional[int] = None,
        iteration_id: Optional[int] = None,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
    ) -> int:
        """Record a checkpoint. Returns checkpoint ID."""
        pass

    # =========================================================================
    # Query methods (for dashboard/UI)
    # =========================================================================

    @abstractmethod
    def get_running_experiment(self) -> Optional[dict]:
        """Get the currently running experiment."""
        pass

    @abstractmethod
    def get_experiment_phases(self, experiment_id: int) -> list[dict]:
        """Get all phases for an experiment."""
        pass

    @abstractmethod
    def get_all_iterations(self, experiment_id: int) -> list[dict]:
        """Get all iterations for an experiment (for trend chart)."""
        pass


# =============================================================================
# Implementations
# =============================================================================

class SqliteTracker(ExperimentTracker):
    """
    SQLite-based experiment tracker.

    Uses the DataLayer for actual database operations.
    """

    def __init__(self, db_path: str, logger: Optional[Callable[[str], None]] = None):
        from wnn.ram.experiments.data_layer import DataLayer
        self._db = DataLayer(db_path, logger)
        self._logger = logger or (lambda x: None)

    def create_flow(
        self,
        name: str,
        config: dict[str, Any],
        description: Optional[str] = None,
    ) -> int:
        return self._db.create_flow(name, config, description)

    def update_flow_status(self, flow_id: int, status: TrackerStatus) -> None:
        from wnn.ram.experiments.data_layer import FlowStatus
        self._db.update_flow_status(flow_id, FlowStatus(status.value))

    def start_experiment(
        self,
        name: str,
        flow_id: Optional[int] = None,
        fitness_calculator: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK,
        fitness_weight_ce: float = 1.0,
        fitness_weight_acc: float = 1.0,
        tier_config: Optional[str] = None,
        context_size: int = 4,
        population_size: int = 50,
    ) -> int:
        from wnn.ram.experiments.data_layer import FitnessCalculator, ExperimentStatus
        exp_id = self._db.create_experiment(
            name=name,
            flow_id=flow_id,
            fitness_calculator=FitnessCalculator(fitness_calculator.value),
            fitness_weight_ce=fitness_weight_ce,
            fitness_weight_acc=fitness_weight_acc,
            tier_config=tier_config,
            context_size=context_size,
            population_size=population_size,
        )
        self._db.update_experiment_status(exp_id, ExperimentStatus.RUNNING)
        return exp_id

    def update_experiment_status(
        self,
        experiment_id: int,
        status: TrackerStatus,
    ) -> None:
        from wnn.ram.experiments.data_layer import ExperimentStatus
        self._db.update_experiment_status(experiment_id, ExperimentStatus(status.value))

    def update_experiment_progress(
        self,
        experiment_id: int,
        last_phase_id: Optional[int] = None,
        last_iteration: Optional[int] = None,
        checkpoint_id: Optional[int] = None,
    ) -> None:
        self._db.update_experiment_progress(
            experiment_id, last_phase_id, last_iteration, checkpoint_id
        )

    def start_phase(
        self,
        experiment_id: int,
        name: str,
        phase_type: str,
        sequence_order: int,
        max_iterations: int = 250,
        population_size: Optional[int] = None,
    ) -> int:
        from wnn.ram.experiments.data_layer import PhaseStatus
        phase_id = self._db.create_phase(
            experiment_id, name, phase_type, sequence_order, max_iterations, population_size
        )
        self._db.update_phase_status(phase_id, PhaseStatus.RUNNING)
        return phase_id

    def update_phase_status(self, phase_id: int, status: TrackerStatus) -> None:
        from wnn.ram.experiments.data_layer import PhaseStatus
        self._db.update_phase_status(phase_id, PhaseStatus(status.value))

    def update_phase_progress(
        self,
        phase_id: int,
        current_iteration: int,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
    ) -> None:
        self._db.update_phase_progress(phase_id, current_iteration, best_ce, best_accuracy)

    def record_iteration(
        self,
        phase_id: int,
        iteration_num: int,
        best_ce: float,
        best_accuracy: Optional[float] = None,
        avg_ce: Optional[float] = None,
        avg_accuracy: Optional[float] = None,
        elite_count: Optional[int] = None,
        offspring_count: Optional[int] = None,
        offspring_viable: Optional[int] = None,
        fitness_threshold: Optional[float] = None,
        elapsed_secs: Optional[float] = None,
    ) -> int:
        return self._db.create_iteration(
            phase_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
            elite_count, offspring_count, offspring_viable, fitness_threshold, elapsed_secs
        )

    def get_or_create_genome(
        self,
        experiment_id: int,
        genome_config: GenomeConfig,
    ) -> int:
        from wnn.ram.experiments.data_layer import GenomeConfig as DLGenomeConfig, TierConfig as DLTierConfig
        dl_config = DLGenomeConfig(
            tiers=[DLTierConfig(t.tier, t.clusters, t.neurons, t.bits) for t in genome_config.tiers]
        )
        return self._db.get_or_create_genome(experiment_id, dl_config)

    def record_genome_evaluation(
        self,
        iteration_id: int,
        genome_id: int,
        position: int,
        role: GenomeRole,
        ce: float,
        accuracy: float,
        elite_rank: Optional[int] = None,
        fitness_score: Optional[float] = None,
        eval_time_ms: Optional[int] = None,
    ) -> int:
        from wnn.ram.experiments.data_layer import GenomeRole as DLRole
        return self._db.create_genome_evaluation(
            iteration_id, genome_id, position, DLRole(role.value),
            ce, accuracy, elite_rank, fitness_score, eval_time_ms
        )

    def record_genome_evaluations_batch(
        self,
        evaluations: list[dict],
    ) -> list[int]:
        return self._db.create_genome_evaluations_batch(evaluations)

    def record_health_check(
        self,
        iteration_id: int,
        k: int,
        top_k_ce: float,
        top_k_accuracy: float,
        best_ce: Optional[float] = None,
        best_ce_accuracy: Optional[float] = None,
        best_acc_ce: Optional[float] = None,
        best_acc_accuracy: Optional[float] = None,
        patience_remaining: Optional[int] = None,
        patience_status: Optional[str] = None,
    ) -> int:
        return self._db.create_health_check(
            iteration_id, k, top_k_ce, top_k_accuracy, best_ce, best_ce_accuracy,
            best_acc_ce, best_acc_accuracy, patience_remaining, patience_status
        )

    def record_checkpoint(
        self,
        experiment_id: int,
        name: str,
        file_path: str,
        checkpoint_type: CheckpointType,
        phase_id: Optional[int] = None,
        iteration_id: Optional[int] = None,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
    ) -> int:
        from wnn.ram.experiments.data_layer import CheckpointType as DLCheckpointType
        return self._db.create_checkpoint(
            experiment_id, name, file_path, DLCheckpointType(checkpoint_type.value),
            phase_id, iteration_id, best_ce, best_accuracy
        )

    def get_running_experiment(self) -> Optional[dict]:
        return self._db.get_running_experiment()

    def get_experiment_phases(self, experiment_id: int) -> list[dict]:
        return self._db.get_experiment_phases(experiment_id)

    def get_all_iterations(self, experiment_id: int) -> list[dict]:
        return self._db.get_all_iterations(experiment_id)


class NoOpTracker(ExperimentTracker):
    """
    No-op tracker for testing or when tracking is disabled.

    All methods return dummy IDs and do nothing.
    """

    def __init__(self):
        self._next_id = 1

    def _get_id(self) -> int:
        id = self._next_id
        self._next_id += 1
        return id

    def create_flow(self, name: str, config: dict, description: Optional[str] = None) -> int:
        return self._get_id()

    def update_flow_status(self, flow_id: int, status: TrackerStatus) -> None:
        pass

    def start_experiment(self, name: str, **kwargs) -> int:
        return self._get_id()

    def update_experiment_status(self, experiment_id: int, status: TrackerStatus) -> None:
        pass

    def update_experiment_progress(self, experiment_id: int, **kwargs) -> None:
        pass

    def start_phase(self, experiment_id: int, name: str, phase_type: str, sequence_order: int, **kwargs) -> int:
        return self._get_id()

    def update_phase_status(self, phase_id: int, status: TrackerStatus) -> None:
        pass

    def update_phase_progress(self, phase_id: int, current_iteration: int, **kwargs) -> None:
        pass

    def record_iteration(self, phase_id: int, iteration_num: int, best_ce: float, **kwargs) -> int:
        return self._get_id()

    def get_or_create_genome(self, experiment_id: int, genome_config: GenomeConfig) -> int:
        return self._get_id()

    def record_genome_evaluation(self, iteration_id: int, genome_id: int, position: int, role: GenomeRole, ce: float, accuracy: float, **kwargs) -> int:
        return self._get_id()

    def record_genome_evaluations_batch(self, evaluations: list[dict]) -> list[int]:
        return [self._get_id() for _ in evaluations]

    def record_health_check(self, iteration_id: int, k: int, top_k_ce: float, top_k_accuracy: float, **kwargs) -> int:
        return self._get_id()

    def record_checkpoint(self, experiment_id: int, name: str, file_path: str, checkpoint_type: CheckpointType, **kwargs) -> int:
        return self._get_id()

    def get_running_experiment(self) -> Optional[dict]:
        return None

    def get_experiment_phases(self, experiment_id: int) -> list[dict]:
        return []

    def get_all_iterations(self, experiment_id: int) -> list[dict]:
        return []


# =============================================================================
# Factory function
# =============================================================================

def create_tracker(
    db_path: Optional[str] = None,
    logger: Optional[Callable[[str], None]] = None,
    enabled: bool = True,
) -> ExperimentTracker:
    """
    Create an experiment tracker.

    Args:
        db_path: Path to SQLite database. If None, uses NoOpTracker.
        logger: Optional logging function.
        enabled: If False, returns NoOpTracker regardless of db_path.

    Returns:
        ExperimentTracker instance.
    """
    if not enabled or db_path is None:
        return NoOpTracker()

    return SqliteTracker(db_path, logger)
