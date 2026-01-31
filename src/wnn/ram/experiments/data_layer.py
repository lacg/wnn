"""
SQLite data layer for WNN experiment tracking.

This module provides direct database access for experiment data, replacing
the HTTP-based dashboard_client.

Usage:
    from wnn.ram.experiments.data_layer import DataLayer

    db = DataLayer("/path/to/database.db")

    # Create a flow
    flow_id = db.create_flow("My Flow", {"experiments": [...]})

    # Create an experiment
    exp_id = db.create_experiment(
        flow_id=flow_id,
        name="1-GA-Neurons",
        fitness_calculator="harmonic_rank",
    )

    # Record iterations and genome evaluations
    iter_id = db.create_iteration(phase_id, iteration_num=1, best_ce=10.5)
    db.create_genome_evaluation(iter_id, genome_id, ce=10.5, accuracy=0.01)
"""

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class FlowStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class FitnessCalculator(str, Enum):
    CE = "ce"
    HARMONIC_RANK = "harmonic_rank"
    WEIGHTED_HARMONIC = "weighted_harmonic"


class GenomeRole(str, Enum):
    # GA roles
    ELITE = "elite"
    OFFSPRING = "offspring"
    INIT = "init"
    # TS roles
    TOP_K = "top_k"  # Top-k neighbors in TS cache (like GA elites)
    NEIGHBOR = "neighbor"  # Other evaluated neighbors
    CURRENT = "current"  # Current best genome being refined


class CheckpointType(str, Enum):
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

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "clusters": self.clusters,
            "neurons": self.neurons,
            "bits": self.bits,
        }


@dataclass
class GenomeConfig:
    """Full genome configuration with tiers."""
    tiers: list[TierConfig]

    def to_json(self) -> str:
        return json.dumps([t.to_dict() for t in self.tiers])

    def compute_hash(self) -> str:
        """Compute unique hash for this genome configuration."""
        data = json.dumps([t.to_dict() for t in self.tiers], sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @property
    def total_clusters(self) -> int:
        return sum(t.clusters for t in self.tiers)

    @property
    def total_neurons(self) -> int:
        return sum(t.clusters * t.neurons for t in self.tiers)

    @property
    def total_memory_bytes(self) -> int:
        # Each neuron has 2^bits addresses, 1 byte each
        return sum(t.clusters * t.neurons * (2 ** t.bits) for t in self.tiers)


def _now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DataLayer:
    """
    SQLite data layer for experiment tracking.

    Thread-safe via connection pooling (one connection per thread).
    Uses WAL mode for concurrent read/write access.
    """

    def __init__(
        self,
        db_path: str | Path,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._db_path = str(db_path)
        self._logger = logger or (lambda x: None)
        self._local = threading.local()

        # Initialize database (creates tables if needed)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()

        # Create tables (matches Rust schema)
        conn.executescript("""
            -- Flows
            CREATE TABLE IF NOT EXISTS flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                config_json TEXT NOT NULL DEFAULT '{}',
                seed_checkpoint_id INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            -- Experiments
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id INTEGER REFERENCES flows(id),
                sequence_order INTEGER,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                fitness_calculator TEXT NOT NULL DEFAULT 'harmonic_rank',
                fitness_weight_ce REAL DEFAULT 1.0,
                fitness_weight_acc REAL DEFAULT 1.0,
                tier_config TEXT,
                context_size INTEGER DEFAULT 4,
                population_size INTEGER DEFAULT 50,
                pid INTEGER,
                last_phase_id INTEGER,
                last_iteration INTEGER,
                resume_checkpoint_id INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                paused_at TEXT
            );

            -- Phases
            CREATE TABLE IF NOT EXISTS phases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                name TEXT NOT NULL,
                phase_type TEXT NOT NULL,
                sequence_order INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                max_iterations INTEGER NOT NULL DEFAULT 250,
                population_size INTEGER,
                current_iteration INTEGER DEFAULT 0,
                best_ce REAL,
                best_accuracy REAL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            );

            -- Iterations
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase_id INTEGER NOT NULL REFERENCES phases(id),
                iteration_num INTEGER NOT NULL,
                best_ce REAL NOT NULL,
                best_accuracy REAL,
                avg_ce REAL,
                avg_accuracy REAL,  -- Population/top-k average accuracy
                elite_count INTEGER,  -- GA: elites kept, TS: top-k cache size
                offspring_count INTEGER,
                offspring_viable INTEGER,
                fitness_threshold REAL,
                elapsed_secs REAL,
                created_at TEXT NOT NULL,
                UNIQUE(phase_id, iteration_num)
            );

            -- Genomes
            CREATE TABLE IF NOT EXISTS genomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                config_hash TEXT NOT NULL,
                tiers_json TEXT NOT NULL,
                total_clusters INTEGER NOT NULL,
                total_neurons INTEGER NOT NULL,
                total_memory_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_id, config_hash)
            );

            -- Genome evaluations
            CREATE TABLE IF NOT EXISTS genome_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id INTEGER NOT NULL REFERENCES iterations(id),
                genome_id INTEGER NOT NULL REFERENCES genomes(id),
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                elite_rank INTEGER,
                ce REAL NOT NULL,
                accuracy REAL NOT NULL,
                fitness_score REAL,
                eval_time_ms INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE(iteration_id, position)
            );

            -- Health checks
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id INTEGER NOT NULL REFERENCES iterations(id),
                k INTEGER NOT NULL,
                top_k_ce REAL NOT NULL,
                top_k_accuracy REAL NOT NULL,
                best_ce REAL,
                best_ce_accuracy REAL,
                best_acc_ce REAL,
                best_acc_accuracy REAL,
                patience_remaining INTEGER,
                patience_status TEXT,
                created_at TEXT NOT NULL
            );

            -- Checkpoints
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                phase_id INTEGER,
                iteration_id INTEGER,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                checkpoint_type TEXT NOT NULL,
                best_ce REAL,
                best_accuracy REAL,
                created_at TEXT NOT NULL
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_experiments_flow ON experiments(flow_id);
            CREATE INDEX IF NOT EXISTS idx_phases_experiment ON phases(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_iterations_phase ON iterations(phase_id);
            CREATE INDEX IF NOT EXISTS idx_genome_evals_iteration ON genome_evaluations(iteration_id);
            CREATE INDEX IF NOT EXISTS idx_genomes_experiment ON genomes(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_iterations_created ON iterations(created_at);
            CREATE INDEX IF NOT EXISTS idx_genome_evals_created ON genome_evaluations(created_at);
        """)
        conn.commit()

    def close(self):
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    # =========================================================================
    # Flow methods
    # =========================================================================

    def create_flow(
        self,
        name: str,
        config: dict[str, Any],
        description: Optional[str] = None,
        seed_checkpoint_id: Optional[int] = None,
    ) -> int:
        """Create a new flow."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO flows (name, description, config_json, seed_checkpoint_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, json.dumps(config), seed_checkpoint_id, _now_iso()),
            )
            flow_id = cursor.lastrowid
            self._logger(f"Created flow {flow_id}: {name}")
            return flow_id

    def get_flow(self, flow_id: int) -> Optional[dict]:
        """Get flow by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM flows WHERE id = ?", (flow_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_flow_status(self, flow_id: int, status: FlowStatus) -> None:
        """Update flow status."""
        now = _now_iso()
        with self._transaction() as conn:
            if status == FlowStatus.RUNNING:
                conn.execute(
                    "UPDATE flows SET status = ?, started_at = ? WHERE id = ?",
                    (status.value, now, flow_id),
                )
            elif status in (FlowStatus.COMPLETED, FlowStatus.FAILED, FlowStatus.CANCELLED):
                conn.execute(
                    "UPDATE flows SET status = ?, completed_at = ? WHERE id = ?",
                    (status.value, now, flow_id),
                )
            else:
                conn.execute(
                    "UPDATE flows SET status = ? WHERE id = ?",
                    (status.value, flow_id),
                )
            self._logger(f"Flow {flow_id} -> {status.value}")

    # =========================================================================
    # Experiment methods
    # =========================================================================

    def create_experiment(
        self,
        name: str,
        flow_id: Optional[int] = None,
        sequence_order: Optional[int] = None,
        fitness_calculator: FitnessCalculator = FitnessCalculator.HARMONIC_RANK,
        fitness_weight_ce: float = 1.0,
        fitness_weight_acc: float = 1.0,
        tier_config: Optional[str] = None,
        context_size: int = 4,
        population_size: int = 50,
    ) -> int:
        """Create a new experiment."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO experiments
                   (flow_id, sequence_order, name, fitness_calculator, fitness_weight_ce,
                    fitness_weight_acc, tier_config, context_size, population_size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    flow_id, sequence_order, name, fitness_calculator.value,
                    fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                    population_size, _now_iso(),
                ),
            )
            exp_id = cursor.lastrowid
            self._logger(f"Created experiment {exp_id}: {name}")
            return exp_id

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        """Get experiment by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_experiment_status(
        self,
        experiment_id: int,
        status: ExperimentStatus,
        pid: Optional[int] = None,
    ) -> None:
        """Update experiment status."""
        now = _now_iso()
        with self._transaction() as conn:
            if status == ExperimentStatus.RUNNING:
                conn.execute(
                    "UPDATE experiments SET status = ?, started_at = ?, pid = ? WHERE id = ?",
                    (status.value, now, pid or os.getpid(), experiment_id),
                )
            elif status == ExperimentStatus.PAUSED:
                conn.execute(
                    "UPDATE experiments SET status = ?, paused_at = ? WHERE id = ?",
                    (status.value, now, experiment_id),
                )
            elif status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
                conn.execute(
                    "UPDATE experiments SET status = ?, ended_at = ?, pid = NULL WHERE id = ?",
                    (status.value, now, experiment_id),
                )
            else:
                conn.execute(
                    "UPDATE experiments SET status = ? WHERE id = ?",
                    (status.value, experiment_id),
                )
            self._logger(f"Experiment {experiment_id} -> {status.value}")

    def update_experiment_progress(
        self,
        experiment_id: int,
        last_phase_id: Optional[int] = None,
        last_iteration: Optional[int] = None,
        resume_checkpoint_id: Optional[int] = None,
    ) -> None:
        """Update experiment progress (for resume tracking)."""
        with self._transaction() as conn:
            updates = []
            params = []
            if last_phase_id is not None:
                updates.append("last_phase_id = ?")
                params.append(last_phase_id)
            if last_iteration is not None:
                updates.append("last_iteration = ?")
                params.append(last_iteration)
            if resume_checkpoint_id is not None:
                updates.append("resume_checkpoint_id = ?")
                params.append(resume_checkpoint_id)
            if updates:
                params.append(experiment_id)
                conn.execute(
                    f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
                    params,
                )

    # =========================================================================
    # Phase methods
    # =========================================================================

    def create_phase(
        self,
        experiment_id: int,
        name: str,
        phase_type: str,
        sequence_order: int,
        max_iterations: int = 250,
        population_size: Optional[int] = None,
    ) -> int:
        """Create a new phase."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO phases
                   (experiment_id, name, phase_type, sequence_order, max_iterations, population_size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, name, phase_type, sequence_order, max_iterations, population_size, _now_iso()),
            )
            phase_id = cursor.lastrowid
            self._logger(f"Created phase {phase_id}: {name}")
            return phase_id

    def get_phase(self, phase_id: int) -> Optional[dict]:
        """Get phase by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM phases WHERE id = ?", (phase_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_phase_status(self, phase_id: int, status: PhaseStatus) -> None:
        """Update phase status."""
        now = _now_iso()
        with self._transaction() as conn:
            if status == PhaseStatus.RUNNING:
                conn.execute(
                    "UPDATE phases SET status = ?, started_at = ? WHERE id = ?",
                    (status.value, now, phase_id),
                )
            elif status in (PhaseStatus.COMPLETED, PhaseStatus.FAILED, PhaseStatus.SKIPPED):
                conn.execute(
                    "UPDATE phases SET status = ?, ended_at = ? WHERE id = ?",
                    (status.value, now, phase_id),
                )
            else:
                conn.execute(
                    "UPDATE phases SET status = ? WHERE id = ?",
                    (status.value, phase_id),
                )
            self._logger(f"Phase {phase_id} -> {status.value}")

    def update_phase_progress(
        self,
        phase_id: int,
        current_iteration: int,
        best_ce: Optional[float] = None,
        best_accuracy: Optional[float] = None,
    ) -> None:
        """Update phase progress."""
        with self._transaction() as conn:
            conn.execute(
                """UPDATE phases
                   SET current_iteration = ?, best_ce = COALESCE(?, best_ce),
                       best_accuracy = COALESCE(?, best_accuracy)
                   WHERE id = ?""",
                (current_iteration, best_ce, best_accuracy, phase_id),
            )

    # =========================================================================
    # Iteration methods
    # =========================================================================

    def create_iteration(
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
        baseline_ce: Optional[float] = None,
        delta_baseline: Optional[float] = None,
        delta_previous: Optional[float] = None,
        patience_counter: Optional[int] = None,
        patience_max: Optional[int] = None,
        candidates_total: Optional[int] = None,
    ) -> int:
        """Create a new iteration record."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO iterations
                   (phase_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                    elite_count, offspring_count, offspring_viable, fitness_threshold,
                    elapsed_secs, baseline_ce, delta_baseline, delta_previous,
                    patience_counter, patience_max, candidates_total, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    phase_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                    elite_count, offspring_count, offspring_viable, fitness_threshold,
                    elapsed_secs, baseline_ce, delta_baseline, delta_previous,
                    patience_counter, patience_max, candidates_total, _now_iso(),
                ),
            )
            iteration_id = cursor.lastrowid

            # Also update phase with latest progress for live view
            conn.execute(
                """UPDATE phases
                   SET current_iteration = ?,
                       best_ce = CASE WHEN ? < COALESCE(best_ce, 999999) THEN ? ELSE best_ce END,
                       best_accuracy = CASE WHEN ? > COALESCE(best_accuracy, 0) THEN ? ELSE best_accuracy END
                   WHERE id = ?""",
                (iteration_num, best_ce, best_ce, best_accuracy, best_accuracy, phase_id),
            )
            return iteration_id

    def get_iteration(self, iteration_id: int) -> Optional[dict]:
        """Get iteration by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM iterations WHERE id = ?", (iteration_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_iterations(self, phase_id: int, limit: int = 100) -> list[dict]:
        """Get iterations for a phase."""
        rows = self._get_conn().execute(
            """SELECT * FROM iterations WHERE phase_id = ?
               ORDER BY iteration_num DESC LIMIT ?""",
            (phase_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Genome methods
    # =========================================================================

    def get_or_create_genome(
        self,
        experiment_id: int,
        genome_config: GenomeConfig,
    ) -> int:
        """Get existing genome or create new one (deduplicated by config hash)."""
        config_hash = genome_config.compute_hash()

        # Try to find existing
        row = self._get_conn().execute(
            "SELECT id FROM genomes WHERE experiment_id = ? AND config_hash = ?",
            (experiment_id, config_hash),
        ).fetchone()

        if row:
            return row["id"]

        # Create new
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO genomes
                   (experiment_id, config_hash, tiers_json, total_clusters,
                    total_neurons, total_memory_bytes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id, config_hash, genome_config.to_json(),
                    genome_config.total_clusters, genome_config.total_neurons,
                    genome_config.total_memory_bytes, _now_iso(),
                ),
            )
            return cursor.lastrowid

    def get_genome(self, genome_id: int) -> Optional[dict]:
        """Get genome by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM genomes WHERE id = ?", (genome_id,)
        ).fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Genome evaluation methods
    # =========================================================================

    def create_genome_evaluation(
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
        """Create a genome evaluation record."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO genome_evaluations
                   (iteration_id, genome_id, position, role, elite_rank, ce, accuracy,
                    fitness_score, eval_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    iteration_id, genome_id, position, role.value, elite_rank,
                    ce, accuracy, fitness_score, eval_time_ms, _now_iso(),
                ),
            )
            return cursor.lastrowid

    def create_genome_evaluations_batch(
        self,
        evaluations: list[dict],
    ) -> list[int]:
        """Create multiple genome evaluations in a single transaction."""
        now = _now_iso()
        ids = []
        with self._transaction() as conn:
            for eval_data in evaluations:
                # Extract role value if it's an enum
                role = eval_data["role"]
                role_value = role.value if hasattr(role, 'value') else role
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO genome_evaluations
                       (iteration_id, genome_id, position, role, elite_rank, ce, accuracy,
                        fitness_score, eval_time_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eval_data["iteration_id"],
                        eval_data["genome_id"],
                        eval_data["position"],
                        role_value,
                        eval_data.get("elite_rank"),
                        eval_data["ce"],
                        eval_data["accuracy"],
                        eval_data.get("fitness_score"),
                        eval_data.get("eval_time_ms"),
                        now,
                    ),
                )
                ids.append(cursor.lastrowid)
        return ids

    def get_iteration_evaluations(self, iteration_id: int) -> list[dict]:
        """Get all genome evaluations for an iteration."""
        rows = self._get_conn().execute(
            """SELECT ge.*, g.tiers_json, g.total_clusters, g.total_neurons
               FROM genome_evaluations ge
               JOIN genomes g ON ge.genome_id = g.id
               WHERE ge.iteration_id = ?
               ORDER BY ge.position""",
            (iteration_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Health check methods
    # =========================================================================

    def create_health_check(
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
        """Create a health check record."""
        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO health_checks
                   (iteration_id, k, top_k_ce, top_k_accuracy, best_ce, best_ce_accuracy,
                    best_acc_ce, best_acc_accuracy, patience_remaining, patience_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    iteration_id, k, top_k_ce, top_k_accuracy, best_ce, best_ce_accuracy,
                    best_acc_ce, best_acc_accuracy, patience_remaining, patience_status, _now_iso(),
                ),
            )
            return cursor.lastrowid

    # =========================================================================
    # Checkpoint methods
    # =========================================================================

    def create_checkpoint(
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
        """Create a checkpoint record."""
        file_size = None
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)

        with self._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO checkpoints
                   (experiment_id, phase_id, iteration_id, name, file_path, file_size_bytes,
                    checkpoint_type, best_ce, best_accuracy, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id, phase_id, iteration_id, name, file_path, file_size,
                    checkpoint_type.value, best_ce, best_accuracy, _now_iso(),
                ),
            )
            checkpoint_id = cursor.lastrowid
            self._logger(f"Created checkpoint {checkpoint_id}: {name}")
            return checkpoint_id

    def get_checkpoint(self, checkpoint_id: int) -> Optional[dict]:
        """Get checkpoint by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_latest_checkpoint(self, experiment_id: int) -> Optional[dict]:
        """Get the most recent checkpoint for an experiment."""
        row = self._get_conn().execute(
            """SELECT * FROM checkpoints
               WHERE experiment_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    # =========================================================================
    # Query methods for dashboard
    # =========================================================================

    def get_running_experiment(self) -> Optional[dict]:
        """Get the currently running experiment."""
        row = self._get_conn().execute(
            "SELECT * FROM experiments WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_experiment_phases(self, experiment_id: int) -> list[dict]:
        """Get all phases for an experiment."""
        rows = self._get_conn().execute(
            "SELECT * FROM phases WHERE experiment_id = ? ORDER BY sequence_order",
            (experiment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_current_phase(self, experiment_id: int) -> Optional[dict]:
        """Get the current (running) phase for an experiment."""
        row = self._get_conn().execute(
            """SELECT * FROM phases
               WHERE experiment_id = ? AND status = 'running'
               ORDER BY sequence_order DESC LIMIT 1""",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_all_iterations(self, experiment_id: int) -> list[dict]:
        """Get all iterations for an experiment (for trend chart)."""
        rows = self._get_conn().execute(
            """SELECT i.*, p.name as phase_name, p.phase_type
               FROM iterations i
               JOIN phases p ON i.phase_id = p.id
               WHERE p.experiment_id = ?
               ORDER BY p.sequence_order, i.iteration_num""",
            (experiment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_changes_since(
        self,
        table: str,
        since_id: int,
        limit: int = 100,
    ) -> list[dict]:
        """Get records added since a given ID (for change detection)."""
        if table not in ("iterations", "genome_evaluations", "health_checks"):
            raise ValueError(f"Invalid table: {table}")

        rows = self._get_conn().execute(
            f"SELECT * FROM {table} WHERE id > ? ORDER BY id LIMIT ?",
            (since_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
