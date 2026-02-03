"""
Experiment tracking abstraction layer.

This module provides an interface for tracking experiment progress that WNN code
can use without depending on specific storage implementations (SQLite, files, etc.).

Data model: Flow → Experiments → Iterations
- Flow: A sequence of experiments (e.g., "4ngram_baseline")
- Experiment: One config spec (e.g., "Phase 1a: GA Neurons Only"), with sequence_order
- Iteration: One GA generation or TS step within an experiment

Usage:
    from wnn.ram.experiments.tracker import ExperimentTracker, create_tracker

    # Create a tracker (implementation chosen by configuration)
    tracker = create_tracker(db_path="/path/to/db.sqlite")

    # Use in experiment code
    exp_id = tracker.start_experiment("Phase 1a: GA Neurons Only", flow_id=1, sequence_order=0, ...)
    iter_id = tracker.record_iteration(exp_id, iteration_num=1, best_ce=10.5, ...)
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
    WEIGHTED_HARMONIC = "weighted_harmonic"  # Legacy - same as HARMONIC_RANK
    NORMALIZED = "normalized"
    NORMALIZED_HARMONIC = "normalized_harmonic"


class GenomeRole(str, Enum):
    """Role of a genome in an iteration."""
    # GA roles
    ELITE = "elite"
    OFFSPRING = "offspring"
    INIT = "init"
    # TS roles
    TOP_K = "top_k"  # Top-k neighbors in TS cache (like GA elites)
    NEIGHBOR = "neighbor"  # Other evaluated neighbors
    CURRENT = "current"  # Current best genome being refined


class CheckpointType(str, Enum):
    """Checkpoint types."""
    AUTO = "auto"
    USER = "user"
    EXPERIMENT_END = "experiment_end"


@dataclass
class TierConfig:
    """Configuration for a single tier.

    Tracks which cluster indices belong to this tier, not just counts.
    - start_cluster/end_cluster: Cluster index range (inclusive start, exclusive end)
    - For non-contiguous tiers, multiple TierConfig entries with same tier number
      will be created, each with different cluster ranges.
    """
    tier: int
    clusters: int
    neurons: int
    bits: int
    # Cluster index range (optional for backwards compatibility)
    start_cluster: int | None = None  # First cluster index in this tier (inclusive)
    end_cluster: int | None = None    # Last cluster index + 1 (exclusive)


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

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        import json
        result = []
        for t in self.tiers:
            entry = {"tier": t.tier, "clusters": t.clusters, "neurons": t.neurons, "bits": t.bits}
            # Include cluster range if available
            if t.start_cluster is not None:
                entry["start_cluster"] = t.start_cluster
            if t.end_cluster is not None:
                entry["end_cluster"] = t.end_cluster
            result.append(entry)
        return json.dumps(result)

    def compute_hash(self) -> str:
        """Compute deterministic hash for deduplication."""
        import hashlib
        # Sort tiers by (neurons, bits, clusters) for consistent hashing
        sorted_tiers = sorted(self.tiers, key=lambda t: (t.neurons, t.bits, t.clusters))
        key = "|".join(f"{t.clusters},{t.neurons},{t.bits}" for t in sorted_tiers)
        return hashlib.sha256(key.encode()).hexdigest()[:16]


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
    def create_pending_experiment(
        self,
        name: str,
        flow_id: int,
        sequence_order: int,
        phase_type: Optional[str] = None,
        max_iterations: int = 250,
    ) -> int:
        """Create an experiment with pending status. Returns experiment ID."""
        pass

    @abstractmethod
    def get_experiment_by_flow_sequence(
        self,
        flow_id: int,
        sequence_order: int,
    ) -> Optional[dict]:
        """Get experiment by flow_id and sequence_order. Returns experiment dict or None."""
        pass

    @abstractmethod
    def start_experiment(
        self,
        name: str,
        flow_id: Optional[int] = None,
        sequence_order: Optional[int] = None,
        fitness_calculator: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED,
        fitness_weight_ce: float = 1.0,
        fitness_weight_acc: float = 1.0,
        tier_config: Optional[str] = None,
        context_size: int = 4,
        population_size: int = 50,
        phase_type: Optional[str] = None,
        max_iterations: int = 250,
    ) -> int:
        """Start an experiment (finds existing or creates, sets to running). Returns experiment ID."""
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
        current_iteration: Optional[int] = None,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
        checkpoint_id: Optional[int] = None,
    ) -> None:
        """Update experiment progress."""
        pass

    # =========================================================================
    # Iteration tracking
    # =========================================================================

    @abstractmethod
    def record_iteration(
        self,
        experiment_id: int,
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
        baseline_ce: Optional[float] = None,
        delta_baseline: Optional[float] = None,
        delta_previous: Optional[float] = None,
        patience_counter: Optional[int] = None,
        patience_max: Optional[int] = None,
        candidates_total: Optional[int] = None,
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

    def create_pending_experiment(
        self,
        name: str,
        flow_id: int,
        sequence_order: int,
        phase_type: Optional[str] = None,
        max_iterations: int = 250,
    ) -> int:
        """Create an experiment with pending status."""
        exp_id = self._db.create_experiment(
            name=name,
            flow_id=flow_id,
            sequence_order=sequence_order,
            phase_type=phase_type,
            max_iterations=max_iterations,
        )
        # Status defaults to 'pending' in DB
        return exp_id

    def get_experiment_by_flow_sequence(
        self,
        flow_id: int,
        sequence_order: int,
    ) -> Optional[dict]:
        """Get experiment by flow_id and sequence_order."""
        return self._db.get_experiment_by_flow_sequence(flow_id, sequence_order)

    def start_experiment(
        self,
        name: str,
        flow_id: Optional[int] = None,
        sequence_order: Optional[int] = None,
        fitness_calculator: FitnessCalculatorType = FitnessCalculatorType.NORMALIZED,
        fitness_weight_ce: float = 1.0,
        fitness_weight_acc: float = 1.0,
        tier_config: Optional[str] = None,
        context_size: int = 4,
        population_size: int = 50,
        phase_type: Optional[str] = None,
        max_iterations: int = 250,
    ) -> int:
        from wnn.ram.experiments.data_layer import FitnessCalculator, ExperimentStatus

        # Try to find existing experiment by flow_id and sequence_order
        exp_id = None
        if flow_id is not None and sequence_order is not None:
            existing = self._db.get_experiment_by_flow_sequence(flow_id, sequence_order)
            if existing:
                exp_id = existing['id']

        # Create if not found
        if exp_id is None:
            exp_id = self._db.create_experiment(
                name=name,
                flow_id=flow_id,
                sequence_order=sequence_order,
                fitness_calculator=FitnessCalculator(fitness_calculator.name.lower()),
                fitness_weight_ce=fitness_weight_ce,
                fitness_weight_acc=fitness_weight_acc,
                tier_config=tier_config,
                context_size=context_size,
                population_size=population_size,
                phase_type=phase_type,
                max_iterations=max_iterations,
            )

        # Set to running
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
        current_iteration: Optional[int] = None,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
        checkpoint_id: Optional[int] = None,
    ) -> None:
        self._db.update_experiment_progress(
            experiment_id, current_iteration, best_ce, best_accuracy, checkpoint_id
        )

    def record_iteration(
        self,
        experiment_id: int,
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
        baseline_ce: Optional[float] = None,
        delta_baseline: Optional[float] = None,
        delta_previous: Optional[float] = None,
        patience_counter: Optional[int] = None,
        patience_max: Optional[int] = None,
        candidates_total: Optional[int] = None,
    ) -> int:
        return self._db.create_iteration(
            experiment_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
            elite_count, offspring_count, offspring_viable, fitness_threshold, elapsed_secs,
            baseline_ce, delta_baseline, delta_previous, patience_counter, patience_max, candidates_total
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

    def create_pending_experiment(self, name: str, flow_id: int, sequence_order: int, **kwargs) -> int:
        return self._get_id()

    def get_experiment_by_flow_sequence(self, flow_id: int, sequence_order: int) -> Optional[dict]:
        return None

    def start_experiment(self, name: str, **kwargs) -> int:
        return self._get_id()

    def update_experiment_status(self, experiment_id: int, status: TrackerStatus) -> None:
        pass

    def update_experiment_progress(self, experiment_id: int, **kwargs) -> None:
        pass

    def record_iteration(self, experiment_id: int, iteration_num: int, best_ce: float, **kwargs) -> int:
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
