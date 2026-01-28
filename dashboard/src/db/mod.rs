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

    Ok(pool)
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    config_json TEXT
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

// Query helpers will go here
pub mod queries {
    use super::*;
    use crate::models::*;

    pub async fn get_experiment(pool: &DbPool, id: i64) -> Result<Option<Experiment>> {
        // TODO: Implement
        Ok(None)
    }

    pub async fn get_phases(pool: &DbPool, experiment_id: i64) -> Result<Vec<Phase>> {
        // TODO: Implement
        Ok(vec![])
    }

    pub async fn get_iterations(pool: &DbPool, phase_id: i64) -> Result<Vec<Iteration>> {
        // TODO: Implement
        Ok(vec![])
    }

    pub async fn insert_iteration(pool: &DbPool, iteration: &Iteration) -> Result<i64> {
        // TODO: Implement
        Ok(0)
    }
}
