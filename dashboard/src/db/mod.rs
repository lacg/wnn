//! Database operations for experiment tracking
//!
//! This module uses a unified schema (no V1/V2 distinction).
//! Tables: flows, experiments, phases, iterations, genomes, genome_evaluations, health_checks, checkpoints

use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use tracing;

pub type DbPool = Pool<Sqlite>;

/// Initialize database with schema
pub async fn init_db(database_url: &str) -> Result<DbPool> {
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    // Create tables
    sqlx::query(SCHEMA).execute(&pool).await?;

    // Run migrations for existing databases
    run_migrations(&pool).await?;

    Ok(pool)
}

/// Run migrations for schema changes
async fn run_migrations(pool: &DbPool) -> Result<()> {
    // Migration: Add pid to flows for stop/restart functionality
    let _ = sqlx::query("ALTER TABLE flows ADD COLUMN pid INTEGER")
        .execute(pool)
        .await;

    // Migration: Add additional columns to iterations
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN baseline_ce REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN delta_baseline REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN delta_previous REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN patience_counter INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN patience_max INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN candidates_total INTEGER")
        .execute(pool)
        .await;

    Ok(())
}

const SCHEMA: &str = r#"
-- ============================================================================
-- FLOWS: A sequence of experiments (multi-pass search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Configuration
    config_json TEXT NOT NULL DEFAULT '{}',

    -- Seed checkpoint (optional starting point)
    seed_checkpoint_id INTEGER REFERENCES checkpoints(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    completed_at TEXT
);

-- ============================================================================
-- EXPERIMENTS: A single optimization run (6-phase search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows(id),
    sequence_order INTEGER,

    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Configuration
    fitness_calculator TEXT NOT NULL DEFAULT 'harmonic_rank',
    fitness_weight_ce REAL DEFAULT 1.0,
    fitness_weight_acc REAL DEFAULT 1.0,
    tier_config TEXT,
    context_size INTEGER DEFAULT 4,
    population_size INTEGER DEFAULT 50,

    -- Process tracking
    pid INTEGER,

    -- Resume state
    last_phase_id INTEGER REFERENCES phases(id),
    last_iteration INTEGER,
    resume_checkpoint_id INTEGER REFERENCES checkpoints(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    ended_at TEXT,
    paused_at TEXT
);

-- ============================================================================
-- PHASES: A phase within an experiment (GA-Neurons, TS-Bits, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),

    name TEXT NOT NULL,
    phase_type TEXT NOT NULL,
    -- ga_neurons, ts_neurons, ga_bits, ts_bits, ga_connections, ts_connections
    sequence_order INTEGER NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending',

    -- Configuration
    max_iterations INTEGER NOT NULL DEFAULT 250,
    population_size INTEGER,

    -- Progress
    current_iteration INTEGER DEFAULT 0,

    -- Best results in this phase
    best_ce REAL,
    best_accuracy REAL,

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    ended_at TEXT
);

-- ============================================================================
-- ITERATIONS: A generation/iteration within a phase
-- ============================================================================
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    iteration_num INTEGER NOT NULL,

    -- Summary metrics (best of this iteration)
    best_ce REAL NOT NULL,
    best_accuracy REAL,
    avg_ce REAL,
    avg_accuracy REAL,

    -- Population info
    elite_count INTEGER,
    offspring_count INTEGER,
    offspring_viable INTEGER,

    -- Fitness threshold (progressive filtering)
    fitness_threshold REAL,

    -- Timing
    elapsed_secs REAL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE(phase_id, iteration_num)
);

-- ============================================================================
-- GENOMES: Unique genome configurations (deduplicated by config hash)
-- ============================================================================
CREATE TABLE IF NOT EXISTS genomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),

    -- Configuration identity (for deduplication)
    config_hash TEXT NOT NULL,

    -- Per-tier configuration as JSON
    tiers_json TEXT NOT NULL,
    -- Example: [{"tier": 0, "clusters": 100, "neurons": 15, "bits": 20}, ...]

    -- Aggregates (computed from tiers)
    total_clusters INTEGER NOT NULL,
    total_neurons INTEGER NOT NULL,
    total_memory_bytes INTEGER NOT NULL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE(experiment_id, config_hash)
);

-- ============================================================================
-- GENOME_EVALUATIONS: Per-iteration evaluation results
-- ============================================================================
CREATE TABLE IF NOT EXISTS genome_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),
    genome_id INTEGER NOT NULL REFERENCES genomes(id),

    -- Position in generation
    position INTEGER NOT NULL,

    -- Role in this iteration
    role TEXT NOT NULL,
    -- 'elite', 'offspring', 'init'
    elite_rank INTEGER,

    -- Evaluation results
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    fitness_score REAL,

    -- Timing
    eval_time_ms INTEGER,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    -- Prevent duplicate evaluations for same iteration+position
    UNIQUE(iteration_id, position)
);

-- ============================================================================
-- HEALTH_CHECKS: Periodic full validation
-- ============================================================================
CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations(id),

    -- Top-K ensemble metrics
    k INTEGER NOT NULL,
    top_k_ce REAL NOT NULL,
    top_k_accuracy REAL NOT NULL,

    -- Best individual metrics
    best_ce REAL,
    best_ce_accuracy REAL,
    best_acc_ce REAL,
    best_acc_accuracy REAL,

    -- Patience tracking
    patience_remaining INTEGER,
    patience_status TEXT,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- CHECKPOINTS: Saved state for resume
-- ============================================================================
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    phase_id INTEGER REFERENCES phases(id),
    iteration_id INTEGER REFERENCES iterations(id),

    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,

    checkpoint_type TEXT NOT NULL,
    -- 'auto', 'user', 'phase_end', 'experiment_end'

    -- Metrics snapshot
    best_ce REAL,
    best_accuracy REAL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- INDEXES for efficient queries
-- ============================================================================

-- For polling new records (change detection)
CREATE INDEX IF NOT EXISTS idx_iterations_created ON iterations(created_at);
CREATE INDEX IF NOT EXISTS idx_genome_evals_created ON genome_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_health_checks_created ON health_checks(created_at);

-- For lookups
CREATE INDEX IF NOT EXISTS idx_experiments_flow ON experiments(flow_id);
CREATE INDEX IF NOT EXISTS idx_phases_experiment ON phases(experiment_id);
CREATE INDEX IF NOT EXISTS idx_iterations_phase ON iterations(phase_id);
CREATE INDEX IF NOT EXISTS idx_genome_evals_iteration ON genome_evaluations(iteration_id);
CREATE INDEX IF NOT EXISTS idx_genomes_experiment ON genomes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);

-- For finding latest records per entity
CREATE INDEX IF NOT EXISTS idx_iterations_phase_num ON iterations(phase_id, iteration_num DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_flows_status ON flows(status);
"#;


// Query helpers
pub mod queries {
    use super::*;
    use crate::models::*;
    use chrono::{DateTime, Utc};
    use sqlx::Row;

    // =============================================================================
    // Flow queries
    // =============================================================================

    pub async fn list_flows(pool: &DbPool, status: Option<&str>, limit: i32, offset: i32) -> Result<Vec<Flow>> {
        let rows = if let Some(status_filter) = status {
            sqlx::query(
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid
                   FROM flows WHERE status = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?"#,
            )
            .bind(status_filter)
            .bind(limit)
            .bind(offset)
            .fetch_all(pool)
            .await?
        } else {
            sqlx::query(
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid
                   FROM flows
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?"#,
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(pool)
            .await?
        };

        let mut flows = Vec::with_capacity(rows.len());
        for row in rows {
            flows.push(row_to_flow(&row)?);
        }
        Ok(flows)
    }

    pub async fn get_flow(pool: &DbPool, id: i64) -> Result<Option<Flow>> {
        let row = sqlx::query(
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid
               FROM flows WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_flow(&r)?)),
            None => Ok(None),
        }
    }

    pub async fn create_flow(
        pool: &DbPool,
        name: &str,
        description: Option<&str>,
        config: &FlowConfig,
        seed_checkpoint_id: Option<i64>,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let config_json = serde_json::to_string(config)?;

        let result = sqlx::query(
            r#"INSERT INTO flows (name, description, config_json, created_at, status, seed_checkpoint_id)
               VALUES (?, ?, ?, ?, 'pending', ?)"#,
        )
        .bind(name)
        .bind(description)
        .bind(&config_json)
        .bind(&now)
        .bind(seed_checkpoint_id)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn update_flow(
        pool: &DbPool,
        id: i64,
        name: Option<&str>,
        description: Option<&str>,
        status: Option<&str>,
        config: Option<&serde_json::Value>,
        seed_checkpoint_id: Option<Option<i64>>,
    ) -> Result<bool> {
        // Build dynamic update query using raw SQL with proper binding
        let mut set_clauses = Vec::new();

        if name.is_some() {
            set_clauses.push("name = ?1");
        }
        if description.is_some() {
            set_clauses.push("description = ?2");
        }
        if status.is_some() {
            set_clauses.push("status = ?3");
            // Update timestamps based on status
            if status == Some("running") {
                set_clauses.push("started_at = ?4");
                // Clear completed_at when re-running a flow (fixes timestamp corruption)
                set_clauses.push("completed_at = NULL");
            } else if status == Some("completed") || status == Some("failed") || status == Some("cancelled") {
                set_clauses.push("completed_at = ?4");
            }
        }
        if config.is_some() {
            set_clauses.push("config_json = ?5");
        }
        if seed_checkpoint_id.is_some() {
            set_clauses.push("seed_checkpoint_id = ?6");
        }

        if set_clauses.is_empty() {
            return Ok(false);
        }

        let query = format!(
            "UPDATE flows SET {} WHERE id = ?7",
            set_clauses.join(", ")
        );

        let now = Utc::now().to_rfc3339();
        let config_json = config.map(|c| serde_json::to_string(c).unwrap_or_default());
        let seed_id = seed_checkpoint_id.flatten();

        let result = sqlx::query(&query)
            .bind(name.unwrap_or(""))
            .bind(description.unwrap_or(""))
            .bind(status.unwrap_or(""))
            .bind(&now)
            .bind(config_json.as_deref().unwrap_or(""))
            .bind(seed_id)
            .bind(id)
            .execute(pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    pub async fn delete_flow(pool: &DbPool, id: i64) -> Result<bool> {
        // Stop any running process first
        stop_flow_process(pool, id).await?;

        // Delete all associated data (experiments, phases, iterations, checkpoints, junction table)
        delete_flow_data(pool, id).await?;

        // Delete the flow itself
        let result = sqlx::query("DELETE FROM flows WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Stop a running flow process by sending SIGTERM.
    /// This is reusable by delete_flow, update_flow_for_restart, etc.
    ///
    /// Always updates status to 'cancelled', even if PID is missing.
    /// The worker will check flow status and stop gracefully.
    pub async fn stop_flow_process(pool: &DbPool, flow_id: i64) -> Result<()> {
        // Get the flow's PID
        let pid: Option<i64> = sqlx::query_scalar(
            "SELECT pid FROM flows WHERE id = ?"
        )
        .bind(flow_id)
        .fetch_optional(pool)
        .await?
        .flatten();

        // Try to send SIGTERM if we have a PID
        if let Some(pid) = pid {
            #[cfg(unix)]
            {
                // Send SIGTERM to gracefully stop the process
                let result = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
                if result == 0 {
                    tracing::info!("Sent SIGTERM to flow {} (PID {})", flow_id, pid);
                } else {
                    tracing::warn!("Failed to send SIGTERM to flow {} (PID {})", flow_id, pid);
                }
            }
        } else {
            tracing::warn!("No PID registered for flow {}, marking as cancelled (worker will check status)", flow_id);
        }

        // Always update status to cancelled and clear PID
        // Even without a PID, this allows the worker to detect cancellation
        // by checking flow status periodically
        sqlx::query(
            "UPDATE flows SET pid = NULL, status = 'cancelled' WHERE id = ?"
        )
        .bind(flow_id)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Delete all data associated with a flow (experiments, phases, iterations, checkpoints)
    /// This is reused by both delete_flow and update_flow_for_restart
    ///
    /// Uses flow_id foreign keys for clean cascade deletion:
    /// experiments.flow_id -> phases -> iterations -> genome_evaluations
    async fn delete_flow_data(pool: &DbPool, flow_id: i64) -> Result<()> {
        // Clear seed_checkpoint_id from flow FIRST (to remove FK dependency)
        sqlx::query("UPDATE flows SET seed_checkpoint_id = NULL WHERE id = ?")
            .bind(flow_id)
            .execute(pool)
            .await?;

        // Get all experiments for this flow
        let exp_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM experiments WHERE flow_id = ?"
        )
        .bind(flow_id)
        .fetch_all(pool)
        .await?;

        for exp_id in &exp_ids {
            delete_experiment_data(pool, *exp_id).await?;
        }

        // Delete experiments by flow_id
        sqlx::query("DELETE FROM experiments WHERE flow_id = ?")
            .bind(flow_id)
            .execute(pool)
            .await?;

        Ok(())
    }

    /// Delete all data for an experiment (phases, iterations, genome_evaluations, genomes, checkpoints)
    async fn delete_experiment_data(pool: &DbPool, exp_id: i64) -> Result<()> {
        // Get phase IDs for this experiment
        let phase_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM phases WHERE experiment_id = ?"
        )
        .bind(exp_id)
        .fetch_all(pool)
        .await?;

        for phase_id in &phase_ids {
            // Delete health checks for iterations of this phase
            sqlx::query(
                "DELETE FROM health_checks WHERE iteration_id IN (SELECT id FROM iterations WHERE phase_id = ?)"
            )
            .bind(phase_id)
            .execute(pool)
            .await?;

            // Delete genome evaluations for iterations of this phase
            sqlx::query(
                "DELETE FROM genome_evaluations WHERE iteration_id IN (SELECT id FROM iterations WHERE phase_id = ?)"
            )
            .bind(phase_id)
            .execute(pool)
            .await?;

            // Delete iterations for this phase
            sqlx::query("DELETE FROM iterations WHERE phase_id = ?")
                .bind(phase_id)
                .execute(pool)
                .await?;
        }

        // Delete phases
        sqlx::query("DELETE FROM phases WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete genome_evaluations that reference genomes from this experiment
        sqlx::query(
            "DELETE FROM genome_evaluations WHERE genome_id IN (SELECT id FROM genomes WHERE experiment_id = ?)"
        )
        .bind(exp_id)
        .execute(pool)
        .await?;

        // Delete genomes
        sqlx::query("DELETE FROM genomes WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete checkpoints
        sqlx::query("DELETE FROM checkpoints WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        Ok(())
    }

    pub async fn list_flow_experiments(pool: &DbPool, flow_id: i64) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments WHERE flow_id = ?
               ORDER BY sequence_order"#,
        )
        .bind(flow_id)
        .fetch_all(pool)
        .await?;

        let mut experiments = Vec::with_capacity(rows.len());
        for row in rows {
            experiments.push(row_to_experiment(&row)?);
        }
        Ok(experiments)
    }

    fn row_to_flow(row: &sqlx::sqlite::SqliteRow) -> Result<Flow> {
        let status_str: String = row.get("status");
        let config_json: String = row.get("config_json");

        Ok(Flow {
            id: row.get("id"),
            name: row.get("name"),
            description: row.get("description"),
            config: serde_json::from_str(&config_json)?,
            created_at: parse_datetime(row.get("created_at"))?,
            started_at: row.get::<Option<String>, _>("started_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            completed_at: row.get::<Option<String>, _>("completed_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            status: parse_flow_status(&status_str),
            seed_checkpoint_id: row.get("seed_checkpoint_id"),
            pid: row.get("pid"),
        })
    }

    /// Update flow PID (called by worker when starting a flow)
    pub async fn update_flow_pid(pool: &DbPool, id: i64, pid: Option<i64>) -> Result<bool> {
        let result = sqlx::query("UPDATE flows SET pid = ? WHERE id = ?")
            .bind(pid)
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Update flow for restart: set status to queued, clear pid, optionally clear seed
    /// If from_beginning is true, also deletes all linked experiments and their data
    pub async fn update_flow_for_restart(
        pool: &DbPool,
        id: i64,
        clear_seed: Option<Option<i64>>,
    ) -> Result<bool> {
        // If clearing seed (restart from beginning), also delete old experiment data
        if clear_seed.is_some() {
            delete_flow_data(pool, id).await?;
        }

        if let Some(seed_id) = clear_seed {
            // Clear both pid and seed checkpoint
            let result = sqlx::query(
                "UPDATE flows SET status = 'queued', pid = NULL, seed_checkpoint_id = ?, started_at = NULL, completed_at = NULL WHERE id = ?"
            )
            .bind(seed_id)
            .bind(id)
            .execute(pool)
            .await?;
            Ok(result.rows_affected() > 0)
        } else {
            // Just reset status and pid
            let result = sqlx::query(
                "UPDATE flows SET status = 'queued', pid = NULL, started_at = NULL, completed_at = NULL WHERE id = ?"
            )
            .bind(id)
            .execute(pool)
            .await?;
            Ok(result.rows_affected() > 0)
        }
    }

    // =============================================================================
    // Checkpoint queries
    // =============================================================================

    pub async fn list_checkpoints(
        pool: &DbPool,
        experiment_id: Option<i64>,
        checkpoint_type: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Checkpoint>> {
        let mut query = String::from(
            r#"SELECT id, experiment_id, phase_id, iteration_id, name, file_path, file_size_bytes,
                      checkpoint_type, best_ce, best_accuracy, created_at
               FROM checkpoints WHERE 1=1"#,
        );

        if experiment_id.is_some() {
            query.push_str(" AND experiment_id = ?");
        }
        if checkpoint_type.is_some() {
            query.push_str(" AND checkpoint_type = ?");
        }
        query.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");

        let mut q = sqlx::query(&query);

        if let Some(exp_id) = experiment_id {
            q = q.bind(exp_id);
        }
        if let Some(cp_type) = checkpoint_type {
            q = q.bind(cp_type);
        }
        q = q.bind(limit).bind(offset);

        let rows = q.fetch_all(pool).await?;

        let mut checkpoints = Vec::with_capacity(rows.len());
        for row in rows {
            checkpoints.push(row_to_checkpoint(&row)?);
        }
        Ok(checkpoints)
    }

    pub async fn get_checkpoint(pool: &DbPool, id: i64) -> Result<Option<Checkpoint>> {
        let row = sqlx::query(
            r#"SELECT id, experiment_id, phase_id, iteration_id, name, file_path, file_size_bytes,
                      checkpoint_type, best_ce, best_accuracy, created_at
               FROM checkpoints WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_checkpoint(&r)?)),
            None => Ok(None),
        }
    }

    pub async fn create_checkpoint(
        pool: &DbPool,
        experiment_id: i64,
        name: &str,
        file_path: &str,
        checkpoint_type: &str,
        file_size_bytes: Option<i64>,
        phase_id: Option<i64>,
        iteration_id: Option<i64>,
        best_ce: Option<f64>,
        best_accuracy: Option<f64>,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();

        let result = sqlx::query(
            r#"INSERT INTO checkpoints
               (experiment_id, phase_id, iteration_id, name, file_path, file_size_bytes,
                checkpoint_type, best_ce, best_accuracy, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(experiment_id)
        .bind(phase_id)
        .bind(iteration_id)
        .bind(name)
        .bind(file_path)
        .bind(file_size_bytes)
        .bind(checkpoint_type)
        .bind(best_ce)
        .bind(best_accuracy)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn delete_checkpoint(pool: &DbPool, id: i64) -> Result<(bool, Option<String>)> {
        let row = sqlx::query("SELECT file_path FROM checkpoints WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await?;

        let Some(row) = row else {
            return Ok((false, None));
        };

        let file_path: String = row.get("file_path");

        // Delete checkpoint
        let result = sqlx::query("DELETE FROM checkpoints WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        Ok((result.rows_affected() > 0, Some(file_path)))
    }

    fn row_to_checkpoint(row: &sqlx::sqlite::SqliteRow) -> Result<Checkpoint> {
        let checkpoint_type_str: String = row.get("checkpoint_type");

        Ok(Checkpoint {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            phase_id: row.get("phase_id"),
            iteration_id: row.get("iteration_id"),
            name: row.get("name"),
            file_path: row.get("file_path"),
            file_size_bytes: row.get("file_size_bytes"),
            checkpoint_type: parse_checkpoint_type(&checkpoint_type_str),
            best_ce: row.get("best_ce"),
            best_accuracy: row.get("best_accuracy"),
            created_at: parse_datetime(row.get("created_at"))?,
        })
    }

    fn parse_checkpoint_type(s: &str) -> CheckpointType {
        match s {
            "auto" => CheckpointType::Auto,
            "user" => CheckpointType::User,
            "phase_end" => CheckpointType::PhaseEnd,
            "experiment_end" => CheckpointType::ExperimentEnd,
            _ => CheckpointType::Auto,
        }
    }

    // =============================================================================
    // Experiment queries (new unified schema)
    // =============================================================================

    /// Get the currently running experiment
    /// Prioritizes experiments linked to flows, then most recent
    pub async fn get_running_experiment(pool: &DbPool) -> Result<Option<Experiment>> {
        let row = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments WHERE status = 'running'
               ORDER BY (CASE WHEN flow_id IS NOT NULL THEN 0 ELSE 1 END), id DESC
               LIMIT 1"#,
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_experiment(&r)?)),
            None => Ok(None),
        }
    }

    /// Get an experiment by ID
    pub async fn get_experiment(pool: &DbPool, id: i64) -> Result<Option<Experiment>> {
        let row = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_experiment(&r)?)),
            None => Ok(None),
        }
    }

    /// Create a new experiment
    pub async fn create_experiment(
        pool: &DbPool,
        name: &str,
        flow_id: Option<i64>,
        config: &serde_json::Value,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();

        // Extract config values with defaults
        // Check both "fitness_calculator" and "fitness_calculator_type" for compatibility
        let fitness_calculator = config.get("fitness_calculator")
            .or_else(|| config.get("fitness_calculator_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("normalized");
        let fitness_weight_ce = config.get("fitness_weight_ce")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let fitness_weight_acc = config.get("fitness_weight_acc")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let tier_config = config.get("tier_config")
            .map(|v| v.to_string());
        let context_size = config.get("context_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as i32;
        let population_size = config.get("population_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(50) as i32;

        let result = sqlx::query(
            r#"INSERT INTO experiments (
                name, flow_id, status, fitness_calculator, fitness_weight_ce, fitness_weight_acc,
                tier_config, context_size, population_size, created_at, started_at
            ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(name)
        .bind(flow_id)
        .bind(fitness_calculator)
        .bind(fitness_weight_ce)
        .bind(fitness_weight_acc)
        .bind(&tier_config)
        .bind(context_size)
        .bind(population_size)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// List all experiments
    pub async fn list_experiments(pool: &DbPool, limit: i32, offset: i32) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?"#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?;

        let mut experiments = Vec::with_capacity(rows.len());
        for row in rows {
            experiments.push(row_to_experiment(&row)?);
        }
        Ok(experiments)
    }

    /// Get phases for an experiment
    pub async fn get_experiment_phases(pool: &DbPool, experiment_id: i64) -> Result<Vec<Phase>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, name, phase_type, sequence_order, status,
                      max_iterations, population_size, current_iteration, best_ce, best_accuracy,
                      created_at, started_at, ended_at
               FROM phases WHERE experiment_id = ? ORDER BY sequence_order"#,
        )
        .bind(experiment_id)
        .fetch_all(pool)
        .await?;

        let mut phases = Vec::with_capacity(rows.len());
        for row in rows {
            phases.push(row_to_phase(&row)?);
        }
        Ok(phases)
    }

    /// Get iterations for a phase
    pub async fn get_phase_iterations(pool: &DbPool, phase_id: i64) -> Result<Vec<Iteration>> {
        let rows = sqlx::query(
            r#"SELECT id, phase_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                      elite_count, offspring_count, offspring_viable, fitness_threshold,
                      elapsed_secs, baseline_ce, delta_baseline, delta_previous,
                      patience_counter, patience_max, candidates_total, created_at
               FROM iterations WHERE phase_id = ? ORDER BY iteration_num"#,
        )
        .bind(phase_id)
        .fetch_all(pool)
        .await?;

        let mut iterations = Vec::with_capacity(rows.len());
        for row in rows {
            iterations.push(row_to_iteration(&row)?);
        }
        Ok(iterations)
    }

    /// Get the current phase (running or most recent) for an experiment
    pub async fn get_current_phase(pool: &DbPool, experiment_id: i64) -> Result<Option<Phase>> {
        let row = sqlx::query(
            r#"SELECT id, experiment_id, name, phase_type, sequence_order, status,
                      max_iterations, population_size, current_iteration, best_ce, best_accuracy,
                      created_at, started_at, ended_at
               FROM phases WHERE experiment_id = ?
               ORDER BY CASE WHEN status = 'running' THEN 0 ELSE 1 END, sequence_order DESC
               LIMIT 1"#,
        )
        .bind(experiment_id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_phase(&r)?)),
            None => Ok(None),
        }
    }

    /// Get recent iterations for an experiment (across all phases)
    pub async fn get_recent_iterations(pool: &DbPool, experiment_id: i64, limit: i32) -> Result<Vec<Iteration>> {
        let rows = sqlx::query(
            r#"SELECT i.id, i.phase_id, i.iteration_num, i.best_ce, i.best_accuracy, i.avg_ce,
                      i.avg_accuracy, i.elite_count, i.offspring_count, i.offspring_viable,
                      i.fitness_threshold, i.elapsed_secs, i.baseline_ce, i.delta_baseline,
                      i.delta_previous, i.patience_counter, i.patience_max, i.candidates_total,
                      i.created_at
               FROM iterations i
               JOIN phases p ON i.phase_id = p.id
               WHERE p.experiment_id = ?
               ORDER BY i.created_at DESC
               LIMIT ?"#,
        )
        .bind(experiment_id)
        .bind(limit)
        .fetch_all(pool)
        .await?;

        let mut iterations = Vec::with_capacity(rows.len());
        for row in rows {
            iterations.push(row_to_iteration(&row)?);
        }
        // Reverse to get chronological order
        iterations.reverse();
        Ok(iterations)
    }

    fn row_to_experiment(row: &sqlx::sqlite::SqliteRow) -> Result<Experiment> {
        let status_str: String = row.get("status");
        let fitness_calc_str: String = row.get("fitness_calculator");

        Ok(Experiment {
            id: row.get("id"),
            flow_id: row.get("flow_id"),
            sequence_order: row.get("sequence_order"),
            name: row.get("name"),
            status: parse_experiment_status(&status_str),
            fitness_calculator: parse_fitness_calculator(&fitness_calc_str),
            fitness_weight_ce: row.get("fitness_weight_ce"),
            fitness_weight_acc: row.get("fitness_weight_acc"),
            tier_config: row.get("tier_config"),
            context_size: row.get("context_size"),
            population_size: row.get("population_size"),
            pid: row.get("pid"),
            last_phase_id: row.get("last_phase_id"),
            last_iteration: row.get("last_iteration"),
            resume_checkpoint_id: row.get("resume_checkpoint_id"),
            created_at: parse_datetime(row.get("created_at"))?,
            started_at: row.get::<Option<String>, _>("started_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            ended_at: row.get::<Option<String>, _>("ended_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            paused_at: row.get::<Option<String>, _>("paused_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
        })
    }

    fn row_to_phase(row: &sqlx::sqlite::SqliteRow) -> Result<Phase> {
        let status_str: String = row.get("status");

        Ok(Phase {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            name: row.get("name"),
            phase_type: row.get("phase_type"),
            sequence_order: row.get("sequence_order"),
            status: parse_phase_status(&status_str),
            max_iterations: row.get("max_iterations"),
            population_size: row.get("population_size"),
            current_iteration: row.get("current_iteration"),
            best_ce: row.get("best_ce"),
            best_accuracy: row.get("best_accuracy"),
            created_at: parse_datetime(row.get("created_at"))?,
            started_at: row.get::<Option<String>, _>("started_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            ended_at: row.get::<Option<String>, _>("ended_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
        })
    }

    fn row_to_iteration(row: &sqlx::sqlite::SqliteRow) -> Result<Iteration> {
        Ok(Iteration {
            id: row.get("id"),
            phase_id: row.get("phase_id"),
            iteration_num: row.get("iteration_num"),
            best_ce: row.get("best_ce"),
            best_accuracy: row.get("best_accuracy"),
            avg_ce: row.get("avg_ce"),
            avg_accuracy: row.get("avg_accuracy"),
            elite_count: row.get("elite_count"),
            offspring_count: row.get("offspring_count"),
            offspring_viable: row.get("offspring_viable"),
            fitness_threshold: row.get("fitness_threshold"),
            elapsed_secs: row.get("elapsed_secs"),
            baseline_ce: row.get("baseline_ce"),
            delta_baseline: row.get("delta_baseline"),
            delta_previous: row.get("delta_previous"),
            patience_counter: row.get("patience_counter"),
            patience_max: row.get("patience_max"),
            candidates_total: row.get("candidates_total"),
            created_at: parse_datetime(row.get("created_at"))?,
        })
    }

    /// Get genome evaluations for an iteration
    pub async fn get_genome_evaluations(pool: &DbPool, iteration_id: i64) -> Result<Vec<GenomeEvaluation>> {
        let rows = sqlx::query(
            r#"SELECT ge.id, ge.iteration_id, ge.genome_id, ge.position, ge.role,
                      ge.elite_rank, ge.ce, ge.accuracy, ge.fitness_score, ge.eval_time_ms,
                      ge.created_at
               FROM genome_evaluations ge
               WHERE ge.iteration_id = ?
               ORDER BY ge.position"#,
        )
        .bind(iteration_id)
        .fetch_all(pool)
        .await?;

        let mut evaluations = Vec::with_capacity(rows.len());
        for row in rows {
            evaluations.push(GenomeEvaluation {
                id: row.get("id"),
                iteration_id: row.get("iteration_id"),
                genome_id: row.get("genome_id"),
                position: row.get("position"),
                role: parse_genome_role(row.get::<String, _>("role").as_str()),
                elite_rank: row.get("elite_rank"),
                ce: row.get("ce"),
                accuracy: row.get("accuracy"),
                fitness_score: row.get("fitness_score"),
                eval_time_ms: row.get("eval_time_ms"),
                created_at: parse_datetime(row.get("created_at"))?,
            });
        }
        Ok(evaluations)
    }

    fn parse_genome_role(s: &str) -> GenomeRole {
        match s {
            "elite" => GenomeRole::Elite,
            "offspring" => GenomeRole::Offspring,
            "init" => GenomeRole::Init,
            "top_k" => GenomeRole::TopK,
            "neighbor" => GenomeRole::Neighbor,
            "current" => GenomeRole::Current,
            _ => GenomeRole::Offspring,
        }
    }

    fn parse_experiment_status(s: &str) -> ExperimentStatus {
        match s {
            "pending" => ExperimentStatus::Pending,
            "queued" => ExperimentStatus::Queued,
            "running" => ExperimentStatus::Running,
            "paused" => ExperimentStatus::Paused,
            "completed" => ExperimentStatus::Completed,
            "failed" => ExperimentStatus::Failed,
            "cancelled" => ExperimentStatus::Cancelled,
            _ => ExperimentStatus::Pending,
        }
    }

    fn parse_phase_status(s: &str) -> PhaseStatus {
        match s {
            "pending" => PhaseStatus::Pending,
            "running" => PhaseStatus::Running,
            "paused" => PhaseStatus::Paused,
            "completed" => PhaseStatus::Completed,
            "skipped" => PhaseStatus::Skipped,
            "failed" => PhaseStatus::Failed,
            _ => PhaseStatus::Pending,
        }
    }

    fn parse_fitness_calculator(s: &str) -> FitnessCalculator {
        match s {
            "ce" => FitnessCalculator::Ce,
            "harmonic_rank" => FitnessCalculator::HarmonicRank,
            "weighted_harmonic" => FitnessCalculator::WeightedHarmonic,
            _ => FitnessCalculator::HarmonicRank,
        }
    }

    // =============================================================================
    // Helper functions
    // =============================================================================

    fn parse_datetime(s: String) -> Result<DateTime<Utc>> {
        Ok(DateTime::parse_from_rfc3339(&s)?.with_timezone(&Utc))
    }

    fn parse_flow_status(s: &str) -> FlowStatus {
        match s {
            "pending" => FlowStatus::Pending,
            "queued" => FlowStatus::Queued,
            "running" => FlowStatus::Running,
            "completed" => FlowStatus::Completed,
            "failed" => FlowStatus::Failed,
            "cancelled" => FlowStatus::Cancelled,
            _ => FlowStatus::Pending,
        }
    }
}
