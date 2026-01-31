//! Database operations for experiment tracking

use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};

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
    // Migration 1: Add pid and checkpoint_phase to experiments (legacy)
    let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN pid INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN checkpoint_phase TEXT")
        .execute(pool)
        .await;

    // Migration 2: Create v2 schema (new data model)
    sqlx::query(SCHEMA_V2).execute(pool).await?;

    // Migration 3: Add avg_accuracy to iterations_v2
    let _ = sqlx::query("ALTER TABLE iterations_v2 ADD COLUMN avg_accuracy REAL")
        .execute(pool)
        .await;

    // Migration 4: Add pid to flows for stop/restart functionality
    let _ = sqlx::query("ALTER TABLE flows ADD COLUMN pid INTEGER")
        .execute(pool)
        .await;

    Ok(())
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    config_json TEXT,
    pid INTEGER,
    checkpoint_phase TEXT
);

CREATE TABLE IF NOT EXISTS phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    name TEXT NOT NULL,
    phase_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    iteration_num INTEGER NOT NULL,
    best_ce REAL NOT NULL,
    avg_ce REAL,
    best_accuracy REAL,
    elapsed_secs REAL NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    top_k_ce REAL NOT NULL,
    top_k_accuracy REAL NOT NULL,
    best_ce REAL,
    best_ce_accuracy REAL,
    best_acc_ce REAL,
    best_acc_accuracy REAL,
    k INTEGER NOT NULL DEFAULT 10,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phase_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases(id),
    metric_type TEXT NOT NULL,
    ce REAL NOT NULL,
    accuracy REAL NOT NULL,
    memory_bytes INTEGER,
    improvement_pct REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_phases_experiment ON phases(experiment_id);
CREATE INDEX IF NOT EXISTS idx_iterations_phase ON iterations(phase_id);
CREATE INDEX IF NOT EXISTS idx_health_checks_phase ON health_checks(phase_id);

-- Flows (sequences of experiments)
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    seed_checkpoint_id INTEGER REFERENCES checkpoints(id)
);

-- Checkpoints (first-class, with file paths)
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    created_at TEXT NOT NULL,
    final_fitness REAL,
    final_accuracy REAL,
    iterations_run INTEGER,
    genome_stats_json TEXT,
    is_final BOOLEAN DEFAULT FALSE,
    reference_count INTEGER DEFAULT 0
);

-- Flow-Experiment mapping
CREATE TABLE IF NOT EXISTS flow_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER NOT NULL REFERENCES flows(id),
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    sequence_order INTEGER NOT NULL
);

-- Checkpoint references (for safe deletion)
CREATE TABLE IF NOT EXISTS checkpoint_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id),
    referencing_experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    reference_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_flows_status ON flows(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_is_final ON checkpoints(is_final);
CREATE INDEX IF NOT EXISTS idx_flow_experiments_flow ON flow_experiments(flow_id);
CREATE INDEX IF NOT EXISTS idx_flow_experiments_experiment ON flow_experiments(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_references_checkpoint ON checkpoint_references(checkpoint_id);
"#;

// =============================================================================
// V2 SCHEMA: New data model with DB as source of truth
// =============================================================================
const SCHEMA_V2: &str = r#"
-- ============================================================================
-- FLOWS_V2: A sequence of experiments (multi-pass search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS flows_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending, queued, running, paused, completed, failed, cancelled

    -- Configuration
    config_json TEXT NOT NULL DEFAULT '{}',

    -- Seed checkpoint (optional starting point)
    seed_checkpoint_id INTEGER REFERENCES checkpoints_v2(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    completed_at TEXT
);

-- ============================================================================
-- EXPERIMENTS_V2: A single optimization run (6-phase search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id INTEGER REFERENCES flows_v2(id),
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
    last_phase_id INTEGER REFERENCES phases_v2(id),
    last_iteration INTEGER,
    resume_checkpoint_id INTEGER REFERENCES checkpoints_v2(id),

    -- Timing
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    started_at TEXT,
    ended_at TEXT,
    paused_at TEXT
);

-- ============================================================================
-- PHASES_V2: A phase within an experiment (GA-Neurons, TS-Bits, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS phases_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments_v2(id),

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
-- ITERATIONS_V2: A generation/iteration within a phase
-- ============================================================================
CREATE TABLE IF NOT EXISTS iterations_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id INTEGER NOT NULL REFERENCES phases_v2(id),
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
-- GENOMES_V2: Unique genome configurations (deduplicated by config hash)
-- ============================================================================
CREATE TABLE IF NOT EXISTS genomes_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments_v2(id),

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
-- GENOME_EVALUATIONS_V2: Per-iteration evaluation results
-- ============================================================================
CREATE TABLE IF NOT EXISTS genome_evaluations_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations_v2(id),
    genome_id INTEGER NOT NULL REFERENCES genomes_v2(id),

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

    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================================
-- HEALTH_CHECKS_V2: Periodic full validation
-- ============================================================================
CREATE TABLE IF NOT EXISTS health_checks_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL REFERENCES iterations_v2(id),

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
-- CHECKPOINTS_V2: Saved state for resume
-- ============================================================================
CREATE TABLE IF NOT EXISTS checkpoints_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments_v2(id),
    phase_id INTEGER REFERENCES phases_v2(id),
    iteration_id INTEGER REFERENCES iterations_v2(id),

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
CREATE INDEX IF NOT EXISTS idx_iterations_v2_created ON iterations_v2(created_at);
CREATE INDEX IF NOT EXISTS idx_genome_evals_v2_created ON genome_evaluations_v2(created_at);
CREATE INDEX IF NOT EXISTS idx_health_checks_v2_created ON health_checks_v2(created_at);

-- For lookups
CREATE INDEX IF NOT EXISTS idx_experiments_v2_flow ON experiments_v2(flow_id);
CREATE INDEX IF NOT EXISTS idx_phases_v2_experiment ON phases_v2(experiment_id);
CREATE INDEX IF NOT EXISTS idx_iterations_v2_phase ON iterations_v2(phase_id);
CREATE INDEX IF NOT EXISTS idx_genome_evals_v2_iteration ON genome_evaluations_v2(iteration_id);
CREATE INDEX IF NOT EXISTS idx_genomes_v2_experiment ON genomes_v2(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_v2_experiment ON checkpoints_v2(experiment_id);

-- For finding latest records per entity
CREATE INDEX IF NOT EXISTS idx_iterations_v2_phase_num ON iterations_v2(phase_id, iteration_num DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_v2_status ON experiments_v2(status);
CREATE INDEX IF NOT EXISTS idx_flows_v2_status ON flows_v2(status);
"#;

// Query helpers
pub mod queries {
    use super::*;
    use crate::models::*;
    use chrono::{DateTime, Utc};
    use sqlx::Row;

    // =============================================================================
    // Experiment queries
    // =============================================================================

    pub async fn list_experiments(pool: &DbPool, limit: i32, offset: i32) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT id, name, log_path, started_at, ended_at, status, config_json, pid, checkpoint_phase
               FROM experiments
               ORDER BY started_at DESC
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

    pub async fn get_experiment(pool: &DbPool, id: i64) -> Result<Option<Experiment>> {
        let row = sqlx::query(
            r#"SELECT id, name, log_path, started_at, ended_at, status, config_json, pid, checkpoint_phase
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

    pub async fn create_experiment(pool: &DbPool, name: &str, log_path: &str, config: &ExperimentConfig) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let config_json = serde_json::to_string(config)?;

        let result = sqlx::query(
            r#"INSERT INTO experiments (name, log_path, started_at, status, config_json)
               VALUES (?, ?, ?, 'running', ?)"#,
        )
        .bind(name)
        .bind(log_path)
        .bind(&now)
        .bind(&config_json)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    fn row_to_experiment(row: &sqlx::sqlite::SqliteRow) -> Result<Experiment> {
        let status_str: String = row.get("status");
        let config_json: Option<String> = row.get("config_json");

        Ok(Experiment {
            id: row.get("id"),
            name: row.get("name"),
            log_path: row.get("log_path"),
            started_at: parse_datetime(row.get("started_at"))?,
            ended_at: row.get::<Option<String>, _>("ended_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            status: parse_experiment_status(&status_str),
            config: config_json
                .map(|s| serde_json::from_str(&s))
                .transpose()?
                .unwrap_or_default(),
            pid: row.get("pid"),
            checkpoint_phase: row.get("checkpoint_phase"),
        })
    }

    pub async fn get_phases(pool: &DbPool, experiment_id: i64) -> Result<Vec<Phase>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, name, phase_type, started_at, ended_at, status
               FROM phases WHERE experiment_id = ? ORDER BY id"#,
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

    pub async fn create_phase(
        pool: &DbPool,
        experiment_id: i64,
        name: &str,
        phase_type: &str,
    ) -> Result<Phase> {
        let now = chrono::Utc::now().to_rfc3339();

        let result = sqlx::query(
            r#"INSERT INTO phases (experiment_id, name, phase_type, started_at, status)
               VALUES (?, ?, ?, ?, 'running')
               RETURNING id, experiment_id, name, phase_type, started_at, ended_at, status"#,
        )
        .bind(experiment_id)
        .bind(name)
        .bind(phase_type)
        .bind(&now)
        .fetch_one(pool)
        .await?;

        row_to_phase(&result)
    }

    pub async fn update_phase(
        pool: &DbPool,
        phase_id: i64,
        status: Option<&str>,
        ended_at: Option<&str>,
    ) -> Result<Phase> {
        // Build dynamic update
        let mut updates = Vec::new();
        if status.is_some() {
            updates.push("status = ?");
        }
        if ended_at.is_some() {
            updates.push("ended_at = ?");
        }

        if updates.is_empty() {
            // Nothing to update, just return current phase
            let row = sqlx::query(
                "SELECT id, experiment_id, name, phase_type, started_at, ended_at, status FROM phases WHERE id = ?"
            )
            .bind(phase_id)
            .fetch_one(pool)
            .await?;
            return row_to_phase(&row);
        }

        let query = format!(
            "UPDATE phases SET {} WHERE id = ? RETURNING id, experiment_id, name, phase_type, started_at, ended_at, status",
            updates.join(", ")
        );

        let mut q = sqlx::query(&query);
        if let Some(s) = status {
            q = q.bind(s);
        }
        if let Some(e) = ended_at {
            q = q.bind(e);
        }
        q = q.bind(phase_id);

        let result = q.fetch_one(pool).await?;
        row_to_phase(&result)
    }

    fn row_to_phase(row: &sqlx::sqlite::SqliteRow) -> Result<Phase> {
        let phase_type_str: String = row.get("phase_type");
        let status_str: String = row.get("status");

        Ok(Phase {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            name: row.get("name"),
            phase_type: parse_phase_type(&phase_type_str),
            started_at: parse_datetime(row.get("started_at"))?,
            ended_at: row.get::<Option<String>, _>("ended_at")
                .map(|s| parse_datetime(s))
                .transpose()?,
            status: parse_phase_status(&status_str),
        })
    }

    pub async fn get_iterations(pool: &DbPool, phase_id: i64) -> Result<Vec<Iteration>> {
        let rows = sqlx::query(
            r#"SELECT id, phase_id, iteration_num, best_ce, avg_ce, best_accuracy, elapsed_secs, timestamp
               FROM iterations WHERE phase_id = ? ORDER BY iteration_num"#,
        )
        .bind(phase_id)
        .fetch_all(pool)
        .await?;

        let mut iterations = Vec::with_capacity(rows.len());
        for row in rows {
            iterations.push(Iteration {
                id: row.get("id"),
                phase_id: row.get("phase_id"),
                iteration_num: row.get("iteration_num"),
                best_ce: row.get("best_ce"),
                avg_ce: row.get("avg_ce"),
                best_accuracy: row.get("best_accuracy"),
                elapsed_secs: row.get("elapsed_secs"),
                timestamp: parse_datetime(row.get("timestamp"))?,
            });
        }
        Ok(iterations)
    }

    pub async fn insert_iteration(pool: &DbPool, iteration: &Iteration) -> Result<i64> {
        let result = sqlx::query(
            r#"INSERT INTO iterations (phase_id, iteration_num, best_ce, avg_ce, best_accuracy, elapsed_secs, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(iteration.phase_id)
        .bind(iteration.iteration_num)
        .bind(iteration.best_ce)
        .bind(iteration.avg_ce)
        .bind(iteration.best_accuracy)
        .bind(iteration.elapsed_secs)
        .bind(iteration.timestamp.to_rfc3339())
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

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
        // First delete flow_experiments mappings
        sqlx::query("DELETE FROM flow_experiments WHERE flow_id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        // Then delete the flow
        let result = sqlx::query("DELETE FROM flows WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    pub async fn list_flow_experiments(pool: &DbPool, flow_id: i64) -> Result<Vec<Experiment>> {
        let rows = sqlx::query(
            r#"SELECT e.id, e.name, e.log_path, e.started_at, e.ended_at, e.status, e.config_json, e.pid, e.checkpoint_phase
               FROM experiments e
               JOIN flow_experiments fe ON e.id = fe.experiment_id
               WHERE fe.flow_id = ?
               ORDER BY fe.sequence_order"#,
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

    pub async fn add_experiment_to_flow(
        pool: &DbPool,
        flow_id: i64,
        experiment_id: i64,
        sequence_order: i32,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO flow_experiments (flow_id, experiment_id, sequence_order) VALUES (?, ?, ?)"
        )
        .bind(flow_id)
        .bind(experiment_id)
        .bind(sequence_order)
        .execute(pool)
        .await?;
        Ok(())
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
            // Delete V2 data following the chain: experiments_v2 -> phases_v2 -> iterations_v2 -> genome_evaluations_v2

            // Get V2 experiment IDs for this flow
            let v2_exp_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT id FROM experiments_v2 WHERE flow_id = ?"
            )
            .bind(id)
            .fetch_all(pool)
            .await?;

            for exp_id in &v2_exp_ids {
                // Get phase IDs for this experiment
                let phase_ids: Vec<i64> = sqlx::query_scalar(
                    "SELECT id FROM phases_v2 WHERE experiment_id = ?"
                )
                .bind(exp_id)
                .fetch_all(pool)
                .await?;

                for phase_id in &phase_ids {
                    // Delete genome evaluations for iterations of this phase
                    sqlx::query(
                        "DELETE FROM genome_evaluations_v2 WHERE iteration_id IN (SELECT id FROM iterations_v2 WHERE phase_id = ?)"
                    )
                    .bind(phase_id)
                    .execute(pool)
                    .await?;

                    // Delete iterations for this phase
                    sqlx::query("DELETE FROM iterations_v2 WHERE phase_id = ?")
                        .bind(phase_id)
                        .execute(pool)
                        .await?;
                }

                // Delete phases for this experiment
                sqlx::query("DELETE FROM phases_v2 WHERE experiment_id = ?")
                    .bind(exp_id)
                    .execute(pool)
                    .await?;
            }

            // Delete V2 experiments
            sqlx::query("DELETE FROM experiments_v2 WHERE flow_id = ?")
                .bind(id)
                .execute(pool)
                .await?;

            // Also clean up V1 data (flow_experiments and experiments)
            let v1_exp_ids: Vec<i64> = sqlx::query_scalar(
                "SELECT experiment_id FROM flow_experiments WHERE flow_id = ?"
            )
            .bind(id)
            .fetch_all(pool)
            .await?;

            // Delete flow_experiments mappings
            sqlx::query("DELETE FROM flow_experiments WHERE flow_id = ?")
                .bind(id)
                .execute(pool)
                .await?;

            // Delete V1 experiments
            for exp_id in &v1_exp_ids {
                sqlx::query("DELETE FROM experiments WHERE id = ?")
                    .bind(exp_id)
                    .execute(pool)
                    .await?;
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
        is_final: Option<bool>,
        limit: i32,
        offset: i32,
    ) -> Result<Vec<Checkpoint>> {
        let mut query = String::from(
            r#"SELECT id, experiment_id, name, file_path, file_size_bytes, created_at,
                      final_fitness, final_accuracy, iterations_run, genome_stats_json, is_final, reference_count
               FROM checkpoints WHERE 1=1"#,
        );

        if experiment_id.is_some() {
            query.push_str(" AND experiment_id = ?");
        }
        if is_final.is_some() {
            query.push_str(" AND is_final = ?");
        }
        query.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");

        let mut q = sqlx::query(&query);

        if let Some(exp_id) = experiment_id {
            q = q.bind(exp_id);
        }
        if let Some(final_only) = is_final {
            q = q.bind(final_only);
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
            r#"SELECT id, experiment_id, name, file_path, file_size_bytes, created_at,
                      final_fitness, final_accuracy, iterations_run, genome_stats_json, is_final, reference_count
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
        file_size_bytes: Option<i64>,
        final_fitness: Option<f64>,
        final_accuracy: Option<f64>,
        iterations_run: Option<i32>,
        genome_stats: Option<&GenomeStats>,
        is_final: bool,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let genome_stats_json = genome_stats.map(|s| serde_json::to_string(s)).transpose()?;

        let result = sqlx::query(
            r#"INSERT INTO checkpoints
               (experiment_id, name, file_path, file_size_bytes, created_at, final_fitness, final_accuracy,
                iterations_run, genome_stats_json, is_final, reference_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"#,
        )
        .bind(experiment_id)
        .bind(name)
        .bind(file_path)
        .bind(file_size_bytes)
        .bind(&now)
        .bind(final_fitness)
        .bind(final_accuracy)
        .bind(iterations_run)
        .bind(&genome_stats_json)
        .bind(is_final)
        .execute(pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn delete_checkpoint(pool: &DbPool, id: i64, force: bool) -> Result<(bool, Option<String>)> {
        // Check reference count
        let row = sqlx::query("SELECT reference_count, file_path FROM checkpoints WHERE id = ?")
            .bind(id)
            .fetch_optional(pool)
            .await?;

        let Some(row) = row else {
            return Ok((false, None));
        };

        let ref_count: i32 = row.get("reference_count");
        let file_path: String = row.get("file_path");

        if ref_count > 0 && !force {
            return Err(anyhow::anyhow!(
                "Checkpoint has {} references. Use force=true to delete anyway.",
                ref_count
            ));
        }

        // Delete references first
        sqlx::query("DELETE FROM checkpoint_references WHERE checkpoint_id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        // Delete checkpoint
        let result = sqlx::query("DELETE FROM checkpoints WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;

        Ok((result.rows_affected() > 0, Some(file_path)))
    }

    pub async fn increment_checkpoint_references(pool: &DbPool, checkpoint_id: i64) -> Result<()> {
        sqlx::query("UPDATE checkpoints SET reference_count = reference_count + 1 WHERE id = ?")
            .bind(checkpoint_id)
            .execute(pool)
            .await?;
        Ok(())
    }

    fn row_to_checkpoint(row: &sqlx::sqlite::SqliteRow) -> Result<Checkpoint> {
        let genome_stats_json: Option<String> = row.get("genome_stats_json");

        Ok(Checkpoint {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            name: row.get("name"),
            file_path: row.get("file_path"),
            file_size_bytes: row.get("file_size_bytes"),
            created_at: parse_datetime(row.get("created_at"))?,
            final_fitness: row.get("final_fitness"),
            final_accuracy: row.get("final_accuracy"),
            iterations_run: row.get("iterations_run"),
            genome_stats: genome_stats_json
                .map(|s| serde_json::from_str(&s))
                .transpose()?,
            is_final: row.get("is_final"),
            reference_count: row.get("reference_count"),
        })
    }

    // =============================================================================
    // V2 Queries (new data model - DB as source of truth)
    // =============================================================================

    /// Get the currently running experiment from v2 tables
    pub async fn get_running_experiment_v2(pool: &DbPool) -> Result<Option<ExperimentV2>> {
        let row = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments_v2 WHERE status = 'running' LIMIT 1"#,
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_experiment_v2(&r)?)),
            None => Ok(None),
        }
    }

    /// Get an experiment by ID from v2 tables
    pub async fn get_experiment_v2(pool: &DbPool, id: i64) -> Result<Option<ExperimentV2>> {
        let row = sqlx::query(
            r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                      fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                      population_size, pid, last_phase_id, last_iteration, resume_checkpoint_id,
                      created_at, started_at, ended_at, paused_at
               FROM experiments_v2 WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_experiment_v2(&r)?)),
            None => Ok(None),
        }
    }

    /// Get phases for an experiment from v2 tables
    pub async fn get_phases_v2(pool: &DbPool, experiment_id: i64) -> Result<Vec<PhaseV2>> {
        let rows = sqlx::query(
            r#"SELECT id, experiment_id, name, phase_type, sequence_order, status,
                      max_iterations, population_size, current_iteration, best_ce, best_accuracy,
                      created_at, started_at, ended_at
               FROM phases_v2 WHERE experiment_id = ? ORDER BY sequence_order"#,
        )
        .bind(experiment_id)
        .fetch_all(pool)
        .await?;

        let mut phases = Vec::with_capacity(rows.len());
        for row in rows {
            phases.push(row_to_phase_v2(&row)?);
        }
        Ok(phases)
    }

    /// Get iterations for a phase from v2 tables
    pub async fn get_iterations_v2(pool: &DbPool, phase_id: i64) -> Result<Vec<IterationV2>> {
        let rows = sqlx::query(
            r#"SELECT id, phase_id, iteration_num, best_ce, best_accuracy, avg_ce, avg_accuracy,
                      elite_count, offspring_count, offspring_viable, fitness_threshold,
                      elapsed_secs, created_at
               FROM iterations_v2 WHERE phase_id = ? ORDER BY iteration_num"#,
        )
        .bind(phase_id)
        .fetch_all(pool)
        .await?;

        let mut iterations = Vec::with_capacity(rows.len());
        for row in rows {
            iterations.push(row_to_iteration_v2(&row)?);
        }
        Ok(iterations)
    }

    /// Get the current phase (running or most recent) for an experiment
    pub async fn get_current_phase_v2(pool: &DbPool, experiment_id: i64) -> Result<Option<PhaseV2>> {
        let row = sqlx::query(
            r#"SELECT id, experiment_id, name, phase_type, sequence_order, status,
                      max_iterations, population_size, current_iteration, best_ce, best_accuracy,
                      created_at, started_at, ended_at
               FROM phases_v2 WHERE experiment_id = ?
               ORDER BY CASE WHEN status = 'running' THEN 0 ELSE 1 END, sequence_order DESC
               LIMIT 1"#,
        )
        .bind(experiment_id)
        .fetch_optional(pool)
        .await?;

        match row {
            Some(r) => Ok(Some(row_to_phase_v2(&r)?)),
            None => Ok(None),
        }
    }

    /// Get recent iterations for an experiment (across all phases)
    pub async fn get_recent_iterations_v2(pool: &DbPool, experiment_id: i64, limit: i32) -> Result<Vec<IterationV2>> {
        let rows = sqlx::query(
            r#"SELECT i.id, i.phase_id, i.iteration_num, i.best_ce, i.best_accuracy, i.avg_ce,
                      i.avg_accuracy, i.elite_count, i.offspring_count, i.offspring_viable,
                      i.fitness_threshold, i.elapsed_secs, i.created_at
               FROM iterations_v2 i
               JOIN phases_v2 p ON i.phase_id = p.id
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
            iterations.push(row_to_iteration_v2(&row)?);
        }
        // Reverse to get chronological order
        iterations.reverse();
        Ok(iterations)
    }

    fn row_to_experiment_v2(row: &sqlx::sqlite::SqliteRow) -> Result<ExperimentV2> {
        let status_str: String = row.get("status");
        let fitness_calc_str: String = row.get("fitness_calculator");

        Ok(ExperimentV2 {
            id: row.get("id"),
            flow_id: row.get("flow_id"),
            sequence_order: row.get("sequence_order"),
            name: row.get("name"),
            status: parse_experiment_status_v2(&status_str),
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

    fn row_to_phase_v2(row: &sqlx::sqlite::SqliteRow) -> Result<PhaseV2> {
        let status_str: String = row.get("status");

        Ok(PhaseV2 {
            id: row.get("id"),
            experiment_id: row.get("experiment_id"),
            name: row.get("name"),
            phase_type: row.get("phase_type"),
            sequence_order: row.get("sequence_order"),
            status: parse_phase_status_v2(&status_str),
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

    fn row_to_iteration_v2(row: &sqlx::sqlite::SqliteRow) -> Result<IterationV2> {
        Ok(IterationV2 {
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
            created_at: parse_datetime(row.get("created_at"))?,
        })
    }

    /// Get genome evaluations for an iteration
    pub async fn get_genome_evaluations_v2(pool: &DbPool, iteration_id: i64) -> Result<Vec<GenomeEvaluationV2>> {
        let rows = sqlx::query(
            r#"SELECT ge.id, ge.iteration_id, ge.genome_id, ge.position, ge.role,
                      ge.elite_rank, ge.ce, ge.accuracy, ge.fitness_score, ge.eval_time_ms,
                      ge.created_at
               FROM genome_evaluations_v2 ge
               WHERE ge.iteration_id = ?
               ORDER BY ge.position"#,
        )
        .bind(iteration_id)
        .fetch_all(pool)
        .await?;

        let mut evaluations = Vec::with_capacity(rows.len());
        for row in rows {
            evaluations.push(GenomeEvaluationV2 {
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

    fn parse_experiment_status_v2(s: &str) -> ExperimentStatusV2 {
        match s {
            "pending" => ExperimentStatusV2::Pending,
            "queued" => ExperimentStatusV2::Queued,
            "running" => ExperimentStatusV2::Running,
            "paused" => ExperimentStatusV2::Paused,
            "completed" => ExperimentStatusV2::Completed,
            "failed" => ExperimentStatusV2::Failed,
            "cancelled" => ExperimentStatusV2::Cancelled,
            _ => ExperimentStatusV2::Pending,
        }
    }

    fn parse_phase_status_v2(s: &str) -> PhaseStatusV2 {
        match s {
            "pending" => PhaseStatusV2::Pending,
            "running" => PhaseStatusV2::Running,
            "paused" => PhaseStatusV2::Paused,
            "completed" => PhaseStatusV2::Completed,
            "skipped" => PhaseStatusV2::Skipped,
            "failed" => PhaseStatusV2::Failed,
            _ => PhaseStatusV2::Pending,
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

    fn parse_experiment_status(s: &str) -> ExperimentStatus {
        match s {
            "running" => ExperimentStatus::Running,
            "completed" => ExperimentStatus::Completed,
            "failed" => ExperimentStatus::Failed,
            "cancelled" => ExperimentStatus::Cancelled,
            _ => ExperimentStatus::Running,
        }
    }

    fn parse_phase_type(s: &str) -> PhaseType {
        match s {
            "ga_neurons" => PhaseType::GaNeurons,
            "ts_neurons" => PhaseType::TsNeurons,
            "ga_bits" => PhaseType::GaBits,
            "ts_bits" => PhaseType::TsBits,
            "ga_connections" => PhaseType::GaConnections,
            "ts_connections" => PhaseType::TsConnections,
            _ => PhaseType::GaNeurons,
        }
    }

    fn parse_phase_status(s: &str) -> PhaseStatus {
        match s {
            "running" => PhaseStatus::Running,
            "completed" => PhaseStatus::Completed,
            "skipped" => PhaseStatus::Skipped,
            _ => PhaseStatus::Running,
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
}
