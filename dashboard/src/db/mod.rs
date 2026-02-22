//! Database operations for experiment tracking
//!
//! Simplified schema (Phase layer removed):
//! Tables: flows, experiments, iterations, genomes, genome_evaluations, health_checks, checkpoints

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

/// Run migrations for schema changes (legacy databases only)
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

    // Migration: Add gating columns to experiments for UI-driven gating analysis
    let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN gating_status TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN gating_results TEXT")
        .execute(pool)
        .await;

    // Migration: Add genome_stats_json to checkpoints for per-tier statistics
    let _ = sqlx::query("ALTER TABLE checkpoints ADD COLUMN genome_stats_json TEXT")
        .execute(pool)
        .await;

    // Migration: Add architecture_type to experiments (tiered | bitwise)
    let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN architecture_type TEXT DEFAULT 'tiered'")
        .execute(pool)
        .await;

    // Migration: Add bitwise-specific fields to genomes
    let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN architecture_type TEXT DEFAULT 'tiered'")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN connections_json TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN hf_config_json TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN hf_export_path TEXT")
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
-- EXPERIMENTS: A single optimization run (GA/TS search)
-- Each experiment in a flow represents one optimization stage (e.g., GA-Neurons, TS-Bits)
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows(id),
    sequence_order INTEGER,

    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Experiment type (was phase_type before simplification)
    phase_type TEXT,
    -- ga_neurons, ts_neurons, ga_bits, ts_bits, ga_connections, ts_connections

    -- Configuration
    fitness_calculator TEXT NOT NULL DEFAULT 'harmonic_rank',
    fitness_weight_ce REAL DEFAULT 1.0,
    fitness_weight_acc REAL DEFAULT 1.0,
    tier_config TEXT,
    context_size INTEGER DEFAULT 4,
    population_size INTEGER DEFAULT 50,
    max_iterations INTEGER DEFAULT 250,

    -- Process tracking
    pid INTEGER,

    -- Progress
    current_iteration INTEGER DEFAULT 0,
    best_ce REAL,
    best_accuracy REAL,

    -- Resume state
    last_iteration INTEGER,
    resume_checkpoint_id INTEGER REFERENCES checkpoints(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    ended_at TEXT,
    paused_at TEXT
);

-- ============================================================================
-- ITERATIONS: A generation/iteration within an experiment
-- ============================================================================
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
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

    -- Delta tracking
    baseline_ce REAL,
    delta_baseline REAL,
    delta_previous REAL,

    -- Patience tracking
    patience_counter INTEGER,
    patience_max INTEGER,
    candidates_total INTEGER,

    -- Timing
    elapsed_secs REAL,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    UNIQUE(experiment_id, iteration_num)
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
-- VALIDATION_SUMMARIES: Full-dataset validation results per genome
-- Each record = one genome validated at a checkpoint (init/final of an experiment)
-- Deduplication: if genome_hash already exists, skip expensive validation and reuse cached values
-- ============================================================================
CREATE TABLE IF NOT EXISTS validation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows(id),
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    validation_point TEXT NOT NULL,   -- 'init' or 'final'
    genome_type TEXT NOT NULL,        -- 'best_ce', 'best_acc', 'best_fitness'
    genome_hash TEXT NOT NULL,        -- Config hash for deduplication
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    -- One record per genome type per checkpoint
    UNIQUE(experiment_id, validation_point, genome_type)
);

-- ============================================================================
-- CHECKPOINTS: Saved state for resume
-- ============================================================================
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    iteration_id INTEGER REFERENCES iterations(id),

    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,

    checkpoint_type TEXT NOT NULL,
    -- 'auto', 'user', 'experiment_end'

    -- Metrics snapshot
    best_ce REAL,
    best_accuracy REAL,

    -- Genome statistics (includes per-tier stats)
    genome_stats_json TEXT,

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- GATING_RUNS: Gating analysis runs per experiment
-- ============================================================================
CREATE TABLE IF NOT EXISTS gating_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),

    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, running, completed, failed

    -- Configuration used for this run
    config_json TEXT,
    -- { neurons_per_gate, bits_per_neuron, threshold, ... }

    -- Results
    genomes_tested INTEGER,
    results_json TEXT,
    -- Array of { genome_type, ce, acc, gated_ce, gated_acc, gating_config }
    error TEXT,

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_gating_runs_experiment ON gating_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_gating_runs_status ON gating_runs(status);

-- ============================================================================
-- INDEXES for efficient queries
-- ============================================================================

-- For polling new records (change detection)
CREATE INDEX IF NOT EXISTS idx_iterations_created ON iterations(created_at);
CREATE INDEX IF NOT EXISTS idx_genome_evals_created ON genome_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_health_checks_created ON health_checks(created_at);

-- For lookups
CREATE INDEX IF NOT EXISTS idx_experiments_flow ON experiments(flow_id);
CREATE INDEX IF NOT EXISTS idx_iterations_experiment ON iterations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_genome_evals_iteration ON genome_evaluations(iteration_id);
CREATE INDEX IF NOT EXISTS idx_genomes_experiment ON genomes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_experiment ON validation_summaries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_genome ON validation_summaries(genome_hash);
CREATE INDEX IF NOT EXISTS idx_validation_summaries_flow ON validation_summaries(flow_id);

-- For finding latest records per entity
CREATE INDEX IF NOT EXISTS idx_iterations_exp_num ON iterations(experiment_id, iteration_num DESC);
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
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat
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
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat
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
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat
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

    /// Create a new flow
    ///
    /// Experiments are passed separately (not in FlowConfig) - they get stored in the experiments table.
    /// This follows normalized design: Flow 1:N Experiments via FK, not embedded JSON.
    pub async fn create_flow(
        pool: &DbPool,
        name: &str,
        description: Option<&str>,
        config: &FlowConfig,
        experiments: &[ExperimentSpec],
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

        let flow_id = result.last_insert_rowid();

        // Create pending experiments for each experiment spec
        for (idx, exp_spec) in experiments.iter().enumerate() {
            // Derive phase_type from experiment spec
            // Check for special phase types (e.g., grid_search for bitwise Phase 1)
            let phase_type = if let Some(pt) = exp_spec.params.get("phase_type").and_then(|v| v.as_str()) {
                pt.to_string()
            } else {
                let opt_target = if exp_spec.optimize_bits {
                    "bits"
                } else if exp_spec.optimize_neurons {
                    "neurons"
                } else {
                    "connections"
                };
                match exp_spec.experiment_type {
                    crate::models::ExperimentType::GridSearch => "grid_search".to_string(),
                    _ => {
                        let exp_type = match exp_spec.experiment_type {
                            crate::models::ExperimentType::Ga => "ga",
                            crate::models::ExperimentType::Ts => "ts",
                            crate::models::ExperimentType::Neurogenesis => "neurogenesis",
                            crate::models::ExperimentType::Synaptogenesis => "synaptogenesis",
                            crate::models::ExperimentType::Axonogenesis => "axonogenesis",
                            crate::models::ExperimentType::GridSearch => unreachable!(),
                        };
                        format!("{}_{}", exp_type, opt_target)
                    }
                }
            };

            // Get max_iterations: grid_search is always 1; others from params
            let max_iterations = if exp_spec.experiment_type == crate::models::ExperimentType::GridSearch {
                Some(1) // Grid search is a single step — always 1
            } else {
                exp_spec.params.get("generations")
                    .or_else(|| exp_spec.params.get("iterations"))
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .or_else(|| {
                        match exp_spec.experiment_type {
                            crate::models::ExperimentType::GridSearch => unreachable!(),
                            crate::models::ExperimentType::Ga => {
                                config.params.get("ga_generations")
                                    .and_then(|v| v.as_i64())
                                    .map(|v| v as i32)
                            }
                            crate::models::ExperimentType::Ts => {
                                config.params.get("ts_iterations")
                                    .and_then(|v| v.as_i64())
                                    .map(|v| v as i32)
                            }
                            crate::models::ExperimentType::Neurogenesis | crate::models::ExperimentType::Synaptogenesis | crate::models::ExperimentType::Axonogenesis => {
                                exp_spec.params.get("iterations")
                                    .and_then(|v| v.as_i64())
                                    .map(|v| v as i32)
                            }
                        }
                    })
            };

            create_pending_experiment(
                pool,
                &exp_spec.name,
                flow_id,
                idx as i32,
                Some(&phase_type),
                max_iterations,
                config,
            ).await?;
        }

        Ok(flow_id)
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

        // When flow config changes, recompute max_iterations for pending experiments
        if let Some(config_val) = config {
            if let Ok(flow_config) = serde_json::from_value::<crate::models::FlowConfig>(config_val.clone()) {
                let pending_experiments: Vec<(i64, Option<String>)> = sqlx::query_as(
                    "SELECT id, phase_type FROM experiments WHERE flow_id = ? AND status = 'pending'"
                )
                .bind(id)
                .fetch_all(pool)
                .await?;

                // Propagate flow config values to pending experiments
                let new_pop_size = flow_config.params.get("population_size")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32);

                for (exp_id, phase_type) in &pending_experiments {
                    if let Some(max_iters) = compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config) {
                        sqlx::query("UPDATE experiments SET max_iterations = ? WHERE id = ?")
                            .bind(max_iters)
                            .bind(exp_id)
                            .execute(pool)
                            .await?;
                    }
                    if let Some(pop) = new_pop_size {
                        sqlx::query("UPDATE experiments SET population_size = ? WHERE id = ?")
                            .bind(pop)
                            .bind(exp_id)
                            .execute(pool)
                            .await?;
                    }
                }
            }
        }

        // Cascade status changes when flow fails/cancelled
        // Mark any running experiments as failed/cancelled too
        if status == Some("failed") || status == Some("cancelled") {
            let cascade_status = status.unwrap();

            // Update running experiments for this flow
            sqlx::query(
                "UPDATE experiments SET status = ?, ended_at = ?
                 WHERE flow_id = ? AND status = 'running'"
            )
            .bind(cascade_status)
            .bind(&now)
            .bind(id)
            .execute(pool)
            .await?;
        }

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

        // Also cancel all running experiments linked to this flow
        sqlx::query(
            "UPDATE experiments SET status = 'cancelled', ended_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE flow_id = ? AND status = 'running'"
        )
        .bind(flow_id)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Delete all data associated with a flow (experiments, iterations, checkpoints)
    /// This is reused by both delete_flow and update_flow_for_restart
    ///
    /// Uses flow_id foreign keys for clean cascade deletion:
    /// experiments.flow_id -> iterations -> genome_evaluations
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

    /// Clear display data for an experiment (iterations, genome_evaluations, genomes, validation_summaries)
    /// but KEEP checkpoints so resume can seed from them.
    async fn clear_experiment_display_data(pool: &DbPool, exp_id: i64) -> Result<()> {
        // Delete validation summaries
        sqlx::query("DELETE FROM validation_summaries WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete health checks for iterations
        sqlx::query(
            "DELETE FROM health_checks WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
        )
        .bind(exp_id)
        .execute(pool)
        .await?;

        // Delete genome evaluations for iterations
        sqlx::query(
            "DELETE FROM genome_evaluations WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
        )
        .bind(exp_id)
        .execute(pool)
        .await?;

        // Delete iterations
        sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete genome_evaluations that reference genomes
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

        Ok(())
    }

    /// Delete all data for an experiment (iterations, genome_evaluations, genomes, checkpoints, validation_summaries, gating_runs)
    async fn delete_experiment_data(pool: &DbPool, exp_id: i64) -> Result<()> {
        // Delete gating runs for this experiment
        sqlx::query("DELETE FROM gating_runs WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete validation summaries for this experiment
        sqlx::query("DELETE FROM validation_summaries WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete health checks for iterations of this experiment
        sqlx::query(
            "DELETE FROM health_checks WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
        )
        .bind(exp_id)
        .execute(pool)
        .await?;

        // Delete genome evaluations for iterations of this experiment
        sqlx::query(
            "DELETE FROM genome_evaluations WHERE iteration_id IN (SELECT id FROM iterations WHERE experiment_id = ?)"
        )
        .bind(exp_id)
        .execute(pool)
        .await?;

        // Delete iterations for this experiment
        sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
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

        // Get checkpoint file paths before deleting records
        let checkpoint_paths: Vec<String> = sqlx::query_scalar(
            "SELECT file_path FROM checkpoints WHERE experiment_id = ?"
        )
        .bind(exp_id)
        .fetch_all(pool)
        .await?;

        // Delete checkpoint records
        sqlx::query("DELETE FROM checkpoints WHERE experiment_id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Delete checkpoint files from disk (best-effort)
        for path in checkpoint_paths {
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::warn!("Failed to delete checkpoint file {}: {}", path, e);
            } else {
                tracing::info!("Deleted checkpoint file: {}", path);
            }
        }

        Ok(())
    }

    /// Delete a single experiment (must be pending status).
    /// Cascades to all child data, then re-numbers remaining experiments.
    pub async fn delete_experiment(pool: &DbPool, exp_id: i64) -> Result<bool> {
        // Verify experiment exists and get its flow_id and status
        let row = sqlx::query("SELECT flow_id, status FROM experiments WHERE id = ?")
            .bind(exp_id)
            .fetch_optional(pool)
            .await?;

        let Some(row) = row else { return Ok(false) };
        let status: String = row.get("status");
        let flow_id: Option<i64> = row.get("flow_id");

        if status != "pending" {
            anyhow::bail!("Can only delete pending experiments (current status: {})", status);
        }

        // Delete all child data
        delete_experiment_data(pool, exp_id).await?;

        // Delete the experiment itself
        sqlx::query("DELETE FROM experiments WHERE id = ?")
            .bind(exp_id)
            .execute(pool)
            .await?;

        // Re-number remaining experiments for this flow to close gaps
        if let Some(fid) = flow_id {
            let remaining_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
            )
            .bind(fid)
            .fetch_all(pool)
            .await?;

            for (idx, eid) in remaining_ids.iter().enumerate() {
                sqlx::query("UPDATE experiments SET sequence_order = ? WHERE id = ?")
                    .bind(idx as i32)
                    .bind(eid)
                    .execute(pool)
                    .await?;
            }
        }

        Ok(true)
    }

    /// Reorder experiments within a flow.
    /// `experiment_ids` must contain all experiment IDs for the flow, in the desired order.
    pub async fn reorder_experiments(pool: &DbPool, flow_id: i64, experiment_ids: &[i64]) -> Result<bool> {
        // Get all experiment IDs for this flow
        let existing_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
        )
        .bind(flow_id)
        .fetch_all(pool)
        .await?;

        // Validate: same count and same set of IDs
        if existing_ids.len() != experiment_ids.len() {
            anyhow::bail!(
                "Expected {} experiment IDs, got {}",
                existing_ids.len(),
                experiment_ids.len()
            );
        }

        let mut expected: Vec<i64> = existing_ids.clone();
        expected.sort();
        let mut provided: Vec<i64> = experiment_ids.to_vec();
        provided.sort();
        if expected != provided {
            anyhow::bail!("Provided experiment IDs don't match the flow's experiments");
        }

        // Update sequence_order for each experiment
        for (idx, eid) in experiment_ids.iter().enumerate() {
            sqlx::query("UPDATE experiments SET sequence_order = ? WHERE id = ?")
                .bind(idx as i32)
                .bind(eid)
                .execute(pool)
                .await?;
        }

        Ok(true)
    }

    pub async fn list_flow_experiments(pool: &DbPool, flow_id: i64) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at,
                      phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                      architecture_type, gating_status, gating_results
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
            last_heartbeat: row.get::<Option<String>, _>("last_heartbeat")
                .map(|s| parse_datetime(s))
                .transpose()?,
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

    /// Update flow heartbeat (called periodically by worker)
    pub async fn update_flow_heartbeat(pool: &DbPool, id: i64) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query("UPDATE flows SET last_heartbeat = ? WHERE id = ?")
            .bind(&now)
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Find stale running flows (no heartbeat in the last N seconds)
    /// Returns flows that should be re-queued
    #[allow(dead_code)]
    pub async fn find_stale_running_flows(pool: &DbPool, stale_seconds: i64) -> Result<Vec<Flow>> {
        let cutoff = (Utc::now() - chrono::Duration::seconds(stale_seconds)).to_rfc3339();
        let rows = sqlx::query(
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat
               FROM flows
               WHERE status = 'running'
               AND (last_heartbeat IS NULL OR last_heartbeat < ?)
               ORDER BY created_at ASC"#,
        )
        .bind(&cutoff)
        .fetch_all(pool)
        .await?;

        let mut flows = Vec::with_capacity(rows.len());
        for row in rows {
            flows.push(row_to_flow(&row)?);
        }
        Ok(flows)
    }

    /// Re-queue a stale flow (reset status and clear pid/heartbeat)
    #[allow(dead_code)]
    pub async fn requeue_stale_flow(pool: &DbPool, id: i64) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE flows SET status = 'queued', pid = NULL, last_heartbeat = NULL WHERE id = ? AND status = 'running'"
        )
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Update flow for restart: set status to queued, clear pid, optionally clear seed
    /// If from_beginning is true, deletes all linked experiments and their data,
    /// then recreates fresh pending experiments from the saved metadata
    pub async fn update_flow_for_restart(
        pool: &DbPool,
        id: i64,
        clear_seed: Option<Option<i64>>,
    ) -> Result<bool> {
        // If clearing seed (restart from beginning), snapshot experiments, delete, then recreate
        if clear_seed.is_some() {
            // Snapshot experiment metadata before deletion
            let saved_experiments: Vec<(String, i32, Option<String>, i32)> = sqlx::query_as(
                "SELECT name, sequence_order, phase_type, max_iterations FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
            )
            .bind(id)
            .fetch_all(pool)
            .await?;

            // Delete all experiment data and experiments
            delete_flow_data(pool, id).await?;

            // Recreate fresh pending experiments from snapshot
            if !saved_experiments.is_empty() {
                // Get the flow config for create_pending_experiment
                let config_json: String = sqlx::query_scalar(
                    "SELECT config_json FROM flows WHERE id = ?"
                )
                .bind(id)
                .fetch_one(pool)
                .await?;
                let flow_config: crate::models::FlowConfig = serde_json::from_str(&config_json)?;

                for (name, sequence_order, phase_type, _old_max_iterations) in &saved_experiments {
                    // Recompute max_iterations from current flow config instead of using stale DB values
                    let max_iterations = compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config);
                    create_pending_experiment(
                        pool,
                        name,
                        id,
                        *sequence_order,
                        phase_type.as_deref(),
                        max_iterations,
                        &flow_config,
                    ).await?;
                }
            }
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
            // Resume: clear display data (iterations, genomes, etc.) for non-completed experiments
            // so the dashboard shows a clean slate. Checkpoints are preserved for seeding.
            let non_completed_exp_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM experiments WHERE flow_id = ? AND status != 'completed'"
            )
            .bind(id)
            .fetch_all(pool)
            .await?;

            for exp_id in &non_completed_exp_ids {
                clear_experiment_display_data(pool, *exp_id).await?;
            }

            // Reset non-completed experiments' timestamps
            sqlx::query(
                "UPDATE experiments SET started_at = NULL, ended_at = NULL, status = 'pending' WHERE flow_id = ? AND status != 'completed'"
            )
            .bind(id)
            .execute(pool)
            .await?;

            // Recompute max_iterations from current flow config for ALL experiments
            // (flow config may have been edited since experiments were created)
            let config_json: String = sqlx::query_scalar(
                "SELECT config_json FROM flows WHERE id = ?"
            )
            .bind(id)
            .fetch_one(pool)
            .await?;
            let flow_config: crate::models::FlowConfig = serde_json::from_str(&config_json)?;

            let all_experiments: Vec<(i64, Option<String>)> = sqlx::query_as(
                "SELECT id, phase_type FROM experiments WHERE flow_id = ?"
            )
            .bind(id)
            .fetch_all(pool)
            .await?;

            for (exp_id, phase_type) in &all_experiments {
                if let Some(max_iters) = compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config) {
                    sqlx::query("UPDATE experiments SET max_iterations = ? WHERE id = ?")
                        .bind(max_iters)
                        .bind(exp_id)
                        .execute(pool)
                        .await?;
                }
            }

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
            r#"SELECT c.id, c.experiment_id, c.iteration_id, c.name, c.file_path, c.file_size_bytes,
                      c.checkpoint_type, c.best_ce, c.best_accuracy, c.genome_stats_json, c.created_at,
                      e.flow_id, f.name as flow_name
               FROM checkpoints c
               LEFT JOIN experiments e ON c.experiment_id = e.id
               LEFT JOIN flows f ON e.flow_id = f.id
               WHERE 1=1"#,
        );

        if experiment_id.is_some() {
            query.push_str(" AND c.experiment_id = ?");
        }
        if checkpoint_type.is_some() {
            query.push_str(" AND c.checkpoint_type = ?");
        }
        query.push_str(" ORDER BY c.created_at DESC LIMIT ? OFFSET ?");

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
            checkpoints.push(row_to_checkpoint_with_flow(&row)?);
        }
        Ok(checkpoints)
    }

    pub async fn get_checkpoint(pool: &DbPool, id: i64) -> Result<Option<Checkpoint>> {
        let row = sqlx::query(
            r#"SELECT id, experiment_id, iteration_id, name, file_path, file_size_bytes,
                      checkpoint_type, best_ce, best_accuracy, genome_stats_json, created_at
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
        iteration_id: Option<i64>,
        best_ce: Option<f64>,
        best_accuracy: Option<f64>,
        genome_stats: Option<&serde_json::Value>,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let genome_stats_json = genome_stats.map(|v| serde_json::to_string(v).unwrap_or_default());

        let result = sqlx::query(
            r#"INSERT INTO checkpoints
               (experiment_id, iteration_id, name, file_path, file_size_bytes,
                checkpoint_type, best_ce, best_accuracy, genome_stats_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(experiment_id)
        .bind(iteration_id)
        .bind(name)
        .bind(file_path)
        .bind(file_size_bytes)
        .bind(checkpoint_type)
        .bind(best_ce)
        .bind(best_accuracy)
        .bind(&genome_stats_json)
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
        let genome_stats_json: Option<String> = row.get("genome_stats_json");
        let genome_stats = genome_stats_json
            .and_then(|s| serde_json::from_str(&s).ok());

        Ok(Checkpoint {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            iteration_id: row.get("iteration_id"),
            name: row.get("name"),
            file_path: row.get("file_path"),
            file_size_bytes: row.get("file_size_bytes"),
            checkpoint_type: parse_checkpoint_type(&checkpoint_type_str),
            best_ce: row.get("best_ce"),
            best_accuracy: row.get("best_accuracy"),
            genome_stats,
            created_at: parse_datetime(row.get("created_at"))?,
            flow_id: None,
            flow_name: None,
        })
    }

    fn row_to_checkpoint_with_flow(row: &sqlx::sqlite::SqliteRow) -> Result<Checkpoint> {
        let checkpoint_type_str: String = row.get("checkpoint_type");
        let genome_stats_json: Option<String> = row.get("genome_stats_json");
        let genome_stats = genome_stats_json
            .and_then(|s| serde_json::from_str(&s).ok());

        Ok(Checkpoint {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            iteration_id: row.get("iteration_id"),
            name: row.get("name"),
            file_path: row.get("file_path"),
            file_size_bytes: row.get("file_size_bytes"),
            checkpoint_type: parse_checkpoint_type(&checkpoint_type_str),
            best_ce: row.get("best_ce"),
            best_accuracy: row.get("best_accuracy"),
            genome_stats,
            created_at: parse_datetime(row.get("created_at"))?,
            flow_id: row.get("flow_id"),
            flow_name: row.get("flow_name"),
        })
    }

    fn parse_checkpoint_type(s: &str) -> CheckpointType {
        match s {
            "auto" => CheckpointType::Auto,
            "user" => CheckpointType::User,
            "experiment_end" => CheckpointType::ExperimentEnd,
            _ => CheckpointType::Auto,
        }
    }

    // =============================================================================
    // Validation Summary queries
    // =============================================================================

    /// Get validation summaries for an experiment
    pub async fn get_validation_summaries(
        pool: &DbPool,
        experiment_id: i64,
    ) -> Result<Vec<ValidationSummary>> {
        let rows = sqlx::query(
            r#"SELECT id, flow_id, experiment_id, validation_point, genome_type,
                      genome_hash, ce, accuracy, created_at
               FROM validation_summaries
               WHERE experiment_id = ?
               ORDER BY validation_point, genome_type"#,
        )
        .bind(experiment_id)
        .fetch_all(pool)
        .await?;

        let mut summaries = Vec::with_capacity(rows.len());
        for row in rows {
            summaries.push(row_to_validation_summary(&row)?);
        }
        Ok(summaries)
    }

    /// Get validation summaries for a flow (all experiments)
    pub async fn get_flow_validation_summaries(
        pool: &DbPool,
        flow_id: i64,
    ) -> Result<Vec<ValidationSummary>> {
        let rows = sqlx::query(
            r#"SELECT vs.id, vs.flow_id, vs.experiment_id, vs.validation_point, vs.genome_type,
                      vs.genome_hash, vs.ce, vs.accuracy, vs.created_at
               FROM validation_summaries vs
               JOIN experiments e ON vs.experiment_id = e.id
               WHERE e.flow_id = ?
               ORDER BY e.sequence_order, vs.validation_point, vs.genome_type"#,
        )
        .bind(flow_id)
        .fetch_all(pool)
        .await?;

        let mut summaries = Vec::with_capacity(rows.len());
        for row in rows {
            summaries.push(row_to_validation_summary(&row)?);
        }
        Ok(summaries)
    }

    /// Check if a genome has already been validated (by genome_hash)
    /// Returns the cached CE and accuracy if found
    pub async fn get_cached_validation(
        pool: &DbPool,
        genome_hash: &str,
    ) -> Result<Option<(f64, f64)>> {
        let row = sqlx::query(
            r#"SELECT ce, accuracy FROM validation_summaries WHERE genome_hash = ? LIMIT 1"#,
        )
        .bind(genome_hash)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some((r.get("ce"), r.get("accuracy")))),
            None => Ok(None),
        }
    }

    /// Create a validation summary (upsert by experiment_id + validation_point + genome_type)
    pub async fn upsert_validation_summary(
        pool: &DbPool,
        flow_id: Option<i64>,
        experiment_id: i64,
        validation_point: &str,
        genome_type: &str,
        genome_hash: &str,
        ce: f64,
        accuracy: f64,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();

        let result = sqlx::query(
            r#"INSERT INTO validation_summaries
               (flow_id, experiment_id, validation_point, genome_type, genome_hash, ce, accuracy, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, validation_point, genome_type) DO UPDATE SET
                 flow_id = excluded.flow_id,
                 genome_hash = excluded.genome_hash,
                 ce = excluded.ce,
                 accuracy = excluded.accuracy,
                 created_at = excluded.created_at"#,
        )
        .bind(flow_id)
        .bind(experiment_id)
        .bind(validation_point)
        .bind(genome_type)
        .bind(genome_hash)
        .bind(ce)
        .bind(accuracy)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    fn row_to_validation_summary(row: &sqlx::sqlite::SqliteRow) -> Result<ValidationSummary> {
        let validation_point_str: String = row.get("validation_point");
        let genome_type_str: String = row.get("genome_type");

        Ok(ValidationSummary {
            id: row.get("id"),
            flow_id: row.get("flow_id"),
            experiment_id: row.get("experiment_id"),
            validation_point: parse_validation_point(&validation_point_str),
            genome_type: parse_genome_validation_type(&genome_type_str),
            genome_hash: row.get("genome_hash"),
            ce: row.get("ce"),
            accuracy: row.get("accuracy"),
            created_at: parse_datetime(row.get("created_at"))?,
        })
    }

    fn parse_validation_point(s: &str) -> ValidationPoint {
        match s {
            "init" => ValidationPoint::Init,
            "final" => ValidationPoint::Final,
            _ => ValidationPoint::Final,
        }
    }

    fn parse_genome_validation_type(s: &str) -> GenomeValidationType {
        match s {
            "best_ce" => GenomeValidationType::BestCe,
            "best_acc" => GenomeValidationType::BestAcc,
            "best_fitness" => GenomeValidationType::BestFitness,
            _ => GenomeValidationType::BestCe,
        }
    }

    // =============================================================================
    // Experiment queries (new unified schema)
    // =============================================================================

    /// Get the currently running experiment
    /// Only returns experiments that are truly running:
    /// - Experiment has status='running' AND
    /// - Either has no flow (standalone) OR its flow is also 'running'
    /// This prevents orphan experiments (where flow was cancelled but experiment wasn't updated)
    pub async fn get_running_experiment(pool: &DbPool) -> Result<Option<Experiment>> {
        let row = sqlx::query(
            r#"SELECT e.id, e.flow_id, e.sequence_order, e.name, e.status, e.fitness_calculator,
                      e.fitness_weight_ce, e.fitness_weight_acc, e.tier_config, e.context_size,
                      e.population_size, e.pid, e.last_iteration, e.resume_checkpoint_id,
                      e.created_at, e.started_at, e.ended_at, e.paused_at,
                      e.phase_type, e.max_iterations, e.current_iteration, e.best_ce, e.best_accuracy,
                      e.architecture_type
               FROM experiments e
               LEFT JOIN flows f ON e.flow_id = f.id
               WHERE e.status = 'running'
                 AND (e.flow_id IS NULL OR f.status = 'running')
               ORDER BY (CASE WHEN e.flow_id IS NOT NULL THEN 0 ELSE 1 END), e.id DESC
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
                      population_size, pid, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at,
                      phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                      architecture_type, gating_status, gating_results
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

        let architecture_type = config.get("architecture_type")
            .and_then(|v| v.as_str())
            .unwrap_or("tiered");

        let result = sqlx::query(
            r#"INSERT INTO experiments (
                name, flow_id, status, fitness_calculator, fitness_weight_ce, fitness_weight_acc,
                tier_config, context_size, population_size, architecture_type, created_at, started_at
            ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(name)
        .bind(flow_id)
        .bind(fitness_calculator)
        .bind(fitness_weight_ce)
        .bind(fitness_weight_acc)
        .bind(&tier_config)
        .bind(context_size)
        .bind(population_size)
        .bind(architecture_type)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Create a new experiment with pending status (for flow creation)
    /// Now includes flow config fields (tier_config, fitness settings, etc.)
    pub async fn create_pending_experiment(
        pool: &DbPool,
        name: &str,
        flow_id: i64,
        sequence_order: i32,
        phase_type: Option<&str>,
        max_iterations: Option<i32>,
        flow_config: &crate::models::FlowConfig,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();

        // Extract config values from flow params
        let tier_config = flow_config.params.get("tier_config")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let fitness_calculator = flow_config.params.get("fitness_calculator")
            .and_then(|v| v.as_str())
            .unwrap_or("harmonic_rank");
        let fitness_weight_ce = flow_config.params.get("fitness_weight_ce")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let fitness_weight_acc = flow_config.params.get("fitness_weight_acc")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let context_size = flow_config.params.get("context_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as i32;
        let population_size = flow_config.params.get("population_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(50) as i32;

        let architecture_type = flow_config.params.get("architecture_type")
            .and_then(|v| v.as_str())
            .unwrap_or("tiered");

        let result = sqlx::query(
            r#"INSERT INTO experiments (
                name, flow_id, sequence_order, status, phase_type, max_iterations,
                tier_config, fitness_calculator, fitness_weight_ce, fitness_weight_acc,
                context_size, population_size, architecture_type, created_at
            ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(name)
        .bind(flow_id)
        .bind(sequence_order)
        .bind(phase_type)
        .bind(max_iterations.unwrap_or(250))
        .bind(&tier_config)
        .bind(fitness_calculator)
        .bind(fitness_weight_ce)
        .bind(fitness_weight_acc)
        .bind(context_size)
        .bind(population_size)
        .bind(architecture_type)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Update an experiment
    pub async fn update_experiment(
        pool: &DbPool,
        id: i64,
        name: Option<&str>,
        status: Option<&str>,
        best_ce: Option<f64>,
        best_accuracy: Option<f64>,
        current_iteration: Option<i32>,
        max_iterations: Option<i32>,
    ) -> Result<bool> {
        let now = Utc::now().to_rfc3339();

        // Build dynamic update query
        let mut set_clauses = Vec::new();
        let mut binds: Vec<String> = Vec::new();

        if let Some(n) = name {
            set_clauses.push("name = ?");
            binds.push(n.to_string());
        }
        if let Some(s) = status {
            set_clauses.push("status = ?");
            binds.push(s.to_string());
            // Update timestamps based on status
            match s {
                "running" => {
                    set_clauses.push("started_at = ?");
                    binds.push(now.clone());
                    // Clear stale ended_at and metrics from previous runs
                    set_clauses.push("ended_at = NULL");
                    set_clauses.push("current_iteration = 0");
                    set_clauses.push("best_ce = NULL");
                    set_clauses.push("best_accuracy = NULL");
                    set_clauses.push("last_iteration = NULL");
                    // Clean stale data from previous runs of this experiment.
                    // Order: genome_evaluations/health_checks (FK→iterations) first,
                    // then iterations, genomes, validation_summaries (FK→experiments).
                    sqlx::query(
                        "DELETE FROM genome_evaluations WHERE iteration_id IN \
                         (SELECT id FROM iterations WHERE experiment_id = ?)"
                    ).bind(id).execute(pool).await?;
                    sqlx::query(
                        "DELETE FROM health_checks WHERE iteration_id IN \
                         (SELECT id FROM iterations WHERE experiment_id = ?)"
                    ).bind(id).execute(pool).await?;
                    sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
                        .bind(id).execute(pool).await?;
                    sqlx::query("DELETE FROM genomes WHERE experiment_id = ?")
                        .bind(id).execute(pool).await?;
                    sqlx::query("DELETE FROM validation_summaries WHERE experiment_id = ?")
                        .bind(id).execute(pool).await?;
                }
                "completed" | "failed" | "cancelled" => {
                    set_clauses.push("ended_at = ?");
                    binds.push(now.clone());
                }
                _ => {}
            }
        }
        if best_ce.is_some() {
            set_clauses.push("best_ce = ?");
        }
        if best_accuracy.is_some() {
            set_clauses.push("best_accuracy = ?");
        }
        if current_iteration.is_some() {
            set_clauses.push("current_iteration = ?");
        }
        if max_iterations.is_some() {
            set_clauses.push("max_iterations = ?");
        }

        if set_clauses.is_empty() {
            return Ok(false);
        }

        let query = format!(
            "UPDATE experiments SET {} WHERE id = ?",
            set_clauses.join(", ")
        );

        let mut q = sqlx::query(&query);

        // Bind string values
        for b in &binds {
            q = q.bind(b);
        }
        // Bind optional numeric values
        if let Some(ce) = best_ce {
            q = q.bind(ce);
        }
        if let Some(acc) = best_accuracy {
            q = q.bind(acc);
        }
        if let Some(iter) = current_iteration {
            q = q.bind(iter);
        }
        if let Some(max) = max_iterations {
            q = q.bind(max);
        }
        // Bind ID last
        q = q.bind(id);

        let result = q.execute(pool).await?;
        Ok(result.rows_affected() > 0)
    }

    /// List all experiments
    pub async fn list_experiments(pool: &DbPool, limit: i32, offset: i32) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at,
                      phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                      architecture_type, gating_status, gating_results
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

    /// Link an experiment to a flow
    pub async fn link_experiment_to_flow(
        pool: &DbPool,
        flow_id: i64,
        experiment_id: i64,
        sequence_order: i32,
    ) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE experiments SET flow_id = ?, sequence_order = ? WHERE id = ?",
        )
        .bind(flow_id)
        .bind(sequence_order)
        .bind(experiment_id)
        .execute(pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Get iterations for an experiment
    #[allow(dead_code)]
    pub async fn get_experiment_iterations(pool: &DbPool, experiment_id: i64) -> Result<Vec<Iteration>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                      elite_count, offspring_count, offspring_viable, fitness_threshold,
                      elapsed_secs, baseline_ce, delta_baseline, delta_previous,
                      patience_counter, patience_max, candidates_total, created_at
               FROM iterations WHERE experiment_id = ? ORDER BY iteration_num"#,
        )
        .bind(experiment_id)
        .fetch_all(pool)
        .await?;

        let mut iterations = Vec::with_capacity(rows.len());
        for row in rows {
            iterations.push(row_to_iteration(&row)?);
        }
        Ok(iterations)
    }

    /// Get recent iterations for an experiment
    pub async fn get_recent_iterations(pool: &DbPool, experiment_id: i64, limit: i32) -> Result<Vec<Iteration>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, iteration_num, best_ce, best_accuracy, avg_ce,
                      avg_accuracy, elite_count, offspring_count, offspring_viable,
                      fitness_threshold, elapsed_secs, baseline_ce, delta_baseline,
                      delta_previous, patience_counter, patience_max, candidates_total,
                      created_at
               FROM iterations
               WHERE experiment_id = ?
               ORDER BY created_at DESC
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

        // Parse gating fields (optional, may not exist in older databases)
        let gating_status: Option<GatingStatus> = row.try_get::<Option<String>, _>("gating_status")
            .ok()
            .flatten()
            .map(|s| parse_gating_status(&s));
        let gating_results: Option<GatingResults> = row.try_get::<Option<String>, _>("gating_results")
            .ok()
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok());

        let architecture_type: ArchitectureType = row.try_get::<Option<String>, _>("architecture_type")
            .ok()
            .flatten()
            .map(|s| parse_architecture_type(&s))
            .unwrap_or_default();

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
            phase_type: row.try_get("phase_type").ok(),
            max_iterations: row.try_get("max_iterations").ok(),
            current_iteration: row.try_get("current_iteration").ok(),
            best_ce: row.try_get("best_ce").ok(),
            best_accuracy: row.try_get("best_accuracy").ok(),
            architecture_type,
            gating_status,
            gating_results,
        })
    }

    fn parse_gating_status(s: &str) -> GatingStatus {
        match s {
            "pending" => GatingStatus::Pending,
            "running" => GatingStatus::Running,
            "completed" => GatingStatus::Completed,
            "failed" => GatingStatus::Failed,
            _ => GatingStatus::Pending,
        }
    }

    fn row_to_iteration(row: &sqlx::sqlite::SqliteRow) -> Result<Iteration> {
        Ok(Iteration {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
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
        // Try RFC 3339 first (standard format: 2026-02-02T05:48:49Z or 2026-02-02T05:48:49+00:00)
        if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
            return Ok(dt.with_timezone(&Utc));
        }

        // Try ISO 8601 with space instead of T (legacy format: 2026-02-02 05:48:49)
        // Also handles dates with/without microseconds
        use chrono::NaiveDateTime;
        let formats = [
            "%Y-%m-%d %H:%M:%S%.f",  // With optional fractional seconds
            "%Y-%m-%d %H:%M:%S",      // Without fractional seconds
            "%Y-%m-%dT%H:%M:%S%.f",   // T separator with fractional (no timezone)
            "%Y-%m-%dT%H:%M:%S",      // T separator without fractional (no timezone)
        ];

        for fmt in formats {
            if let Ok(naive) = NaiveDateTime::parse_from_str(&s, fmt) {
                return Ok(naive.and_utc());
            }
        }

        // If all parsing fails, return an error with context
        Err(anyhow::anyhow!("Failed to parse datetime: '{}'", s))
    }

    fn parse_architecture_type(s: &str) -> ArchitectureType {
        match s {
            "bitwise" => ArchitectureType::Bitwise,
            "multi_stage" | "multistage" => ArchitectureType::MultiStage,
            _ => ArchitectureType::Tiered,
        }
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

    /// Compute max_iterations from phase_type and flow config.
    /// Returns None if phase_type is unknown or config doesn't have the needed params.
    fn compute_max_iterations_from_phase_type(
        phase_type: Option<&str>,
        config: &crate::models::FlowConfig,
    ) -> Option<i32> {
        let pt = phase_type?;
        if pt == "grid_search" {
            return Some(1);
        }
        if pt.starts_with("neurogenesis") || pt.starts_with("synaptogenesis") || pt.starts_with("axonogenesis") {
            return config.params.get("adaptation_iterations")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32);
        }
        if pt.starts_with("ga") {
            return config.params.get("ga_generations")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32);
        }
        if pt.starts_with("ts") {
            return config.params.get("ts_iterations")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32);
        }
        None
    }

    // =============================================================================
    // Gating Run queries
    // =============================================================================

    fn gating_status_to_str(status: &GatingStatus) -> &'static str {
        match status {
            GatingStatus::Pending => "pending",
            GatingStatus::Running => "running",
            GatingStatus::Completed => "completed",
            GatingStatus::Failed => "failed",
        }
    }

    fn str_to_gating_status(s: &str) -> GatingStatus {
        match s {
            "pending" => GatingStatus::Pending,
            "running" => GatingStatus::Running,
            "completed" => GatingStatus::Completed,
            "failed" => GatingStatus::Failed,
            _ => GatingStatus::Pending,
        }
    }

    fn row_to_gating_run(row: &sqlx::sqlite::SqliteRow) -> Result<GatingRun> {
        let status_str: String = row.try_get("status")?;
        let config_json: Option<String> = row.try_get("config_json")?;
        let results_json: Option<String> = row.try_get("results_json")?;
        let created_at_str: String = row.try_get("created_at")?;
        let started_at_str: Option<String> = row.try_get("started_at")?;
        let completed_at_str: Option<String> = row.try_get("completed_at")?;

        Ok(GatingRun {
            id: row.try_get("id")?,
            experiment_id: row.try_get("experiment_id")?,
            status: str_to_gating_status(&status_str),
            config: config_json.and_then(|s| serde_json::from_str(&s).ok()),
            genomes_tested: row.try_get("genomes_tested")?,
            results: results_json.and_then(|s| serde_json::from_str(&s).ok()),
            error: row.try_get("error")?,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            started_at: started_at_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            completed_at: completed_at_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
        })
    }

    /// Create a new gating run for an experiment
    pub async fn create_gating_run(
        pool: &DbPool,
        experiment_id: i64,
        config: Option<&GatingConfig>,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let config_json = config.map(|c| serde_json::to_string(c).unwrap_or_default());

        let result = sqlx::query(
            r#"INSERT INTO gating_runs (experiment_id, status, config_json, created_at)
               VALUES (?, 'pending', ?, ?)"#
        )
        .bind(experiment_id)
        .bind(&config_json)
        .bind(&now)
        .execute(pool)
        .await?;

        // Also update the experiment's gating_status for backward compat
        let _ = sqlx::query("UPDATE experiments SET gating_status = 'pending' WHERE id = ?")
            .bind(experiment_id)
            .execute(pool)
            .await;

        Ok(result.last_insert_rowid())
    }

    /// Get a specific gating run
    pub async fn get_gating_run(pool: &DbPool, id: i64) -> Result<Option<GatingRun>> {
        let row = sqlx::query(
            r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                      results_json, error, created_at, started_at, completed_at
               FROM gating_runs WHERE id = ?"#
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_gating_run(&r)?)),
            None => Ok(None),
        }
    }

    /// List gating runs for an experiment
    pub async fn list_gating_runs(pool: &DbPool, experiment_id: i64) -> Result<Vec<GatingRun>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                      results_json, error, created_at, started_at, completed_at
               FROM gating_runs WHERE experiment_id = ?
               ORDER BY created_at DESC"#
        )
        .bind(experiment_id)
        .fetch_all(pool)
        .await?;

        let mut runs = Vec::with_capacity(rows.len());
        for row in rows {
            runs.push(row_to_gating_run(&row)?);
        }
        Ok(runs)
    }

    /// Update gating run status
    pub async fn update_gating_run_status(
        pool: &DbPool,
        id: i64,
        status: &GatingStatus,
    ) -> Result<Option<GatingRun>> {
        let status_str = gating_status_to_str(status);
        let now = Utc::now().to_rfc3339();

        // Set started_at when transitioning to running
        let started_clause = if *status == GatingStatus::Running {
            ", started_at = ?"
        } else {
            ""
        };

        // Set completed_at when transitioning to completed or failed
        let completed_clause = if *status == GatingStatus::Completed || *status == GatingStatus::Failed {
            ", completed_at = ?"
        } else {
            ""
        };

        let query = format!(
            "UPDATE gating_runs SET status = ?{}{} WHERE id = ?",
            started_clause, completed_clause
        );

        let mut q = sqlx::query(&query).bind(status_str);
        if *status == GatingStatus::Running {
            q = q.bind(&now);
        }
        if *status == GatingStatus::Completed || *status == GatingStatus::Failed {
            q = q.bind(&now);
        }
        q = q.bind(id);

        let result = q.execute(pool).await?;

        if result.rows_affected() == 0 {
            return Ok(None);
        }

        // Also update experiment's gating_status for backward compat
        let _ = sqlx::query(
            "UPDATE experiments SET gating_status = ? WHERE id = (SELECT experiment_id FROM gating_runs WHERE id = ?)"
        )
        .bind(status_str)
        .bind(id)
        .execute(pool)
        .await;

        get_gating_run(pool, id).await
    }

    /// Update gating run with results
    pub async fn update_gating_run_results(
        pool: &DbPool,
        id: i64,
        genomes_tested: i32,
        results: &[GatingResult],
        error: Option<&str>,
    ) -> Result<Option<GatingRun>> {
        let now = Utc::now().to_rfc3339();
        let results_json = serde_json::to_string(results)?;
        let status = if error.is_some() { "failed" } else { "completed" };

        let result = sqlx::query(
            r#"UPDATE gating_runs
               SET status = ?, genomes_tested = ?, results_json = ?, error = ?, completed_at = ?
               WHERE id = ?"#
        )
        .bind(status)
        .bind(genomes_tested)
        .bind(&results_json)
        .bind(error)
        .bind(&now)
        .bind(id)
        .execute(pool)
        .await?;

        if result.rows_affected() == 0 {
            return Ok(None);
        }

        // Also update experiment's gating_status and results for backward compat
        let gating_results = GatingResults {
            completed_at: Some(Utc::now()),
            genomes_tested: genomes_tested as usize,
            results: results.to_vec(),
            error: error.map(|s| s.to_string()),
        };
        let exp_results_json = serde_json::to_string(&gating_results)?;

        let _ = sqlx::query(
            "UPDATE experiments SET gating_status = ?, gating_results = ? WHERE id = (SELECT experiment_id FROM gating_runs WHERE id = ?)"
        )
        .bind(status)
        .bind(&exp_results_json)
        .bind(id)
        .execute(pool)
        .await;

        get_gating_run(pool, id).await
    }

    /// Get pending gating runs (for worker polling)
    pub async fn get_pending_gating_runs(pool: &DbPool) -> Result<Vec<GatingRun>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                      results_json, error, created_at, started_at, completed_at
               FROM gating_runs WHERE status = 'pending'
               ORDER BY created_at ASC"#
        )
        .fetch_all(pool)
        .await?;

        let mut runs = Vec::with_capacity(rows.len());
        for row in rows {
            runs.push(row_to_gating_run(&row)?);
        }
        Ok(runs)
    }

}
