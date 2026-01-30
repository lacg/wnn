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
            r#"SELECT id, name, log_path, started_at, ended_at, status, config_json
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
            r#"SELECT id, name, log_path, started_at, ended_at, status, config_json
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
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id
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
                r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id
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
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id
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
            r#"SELECT e.id, e.name, e.log_path, e.started_at, e.ended_at, e.status, e.config_json
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
        })
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
