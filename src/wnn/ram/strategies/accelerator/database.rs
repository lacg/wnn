//! SQLite database operations for the v2 data model.
//!
//! This module provides functions for writing genome evaluations and iterations
//! directly to SQLite from Rust, enabling fast batch writes during GA/TS optimization.
//!
//! Uses WAL mode for concurrent access with the Python data layer and dashboard.

use rusqlite::{params, Connection, Result as SqliteResult};
use std::path::Path;
use std::sync::Mutex;

/// Thread-safe database connection wrapper.
pub struct Database {
    conn: Mutex<Connection>,
}

impl Database {
    /// Open a connection to the SQLite database.
    ///
    /// Uses WAL mode for concurrent read/write access.
    pub fn open<P: AsRef<Path>>(path: P) -> SqliteResult<Self> {
        let conn = Connection::open(path)?;

        // Configure for concurrent access
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             PRAGMA busy_timeout=30000;",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Create an iteration record.
    ///
    /// Returns the new iteration ID.
    pub fn create_iteration(
        &self,
        phase_id: i64,
        iteration_num: i32,
        best_ce: f64,
        best_accuracy: Option<f64>,
        avg_ce: Option<f64>,
        elite_count: Option<i32>,
        offspring_count: Option<i32>,
        offspring_viable: Option<i32>,
        fitness_threshold: Option<f64>,
        elapsed_secs: Option<f64>,
    ) -> SqliteResult<i64> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        conn.execute(
            "INSERT INTO iterations_v2
             (phase_id, iteration_num, best_ce, best_accuracy, avg_ce,
              elite_count, offspring_count, offspring_viable, fitness_threshold,
              elapsed_secs, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                phase_id,
                iteration_num,
                best_ce,
                best_accuracy,
                avg_ce,
                elite_count,
                offspring_count,
                offspring_viable,
                fitness_threshold,
                elapsed_secs,
                now,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get or create a genome record.
    ///
    /// Genomes are deduplicated by config_hash within an experiment.
    /// Returns the genome ID.
    pub fn get_or_create_genome(
        &self,
        experiment_id: i64,
        config_hash: &str,
        tiers_json: &str,
        total_clusters: i32,
        total_neurons: i32,
        total_memory_bytes: i64,
    ) -> SqliteResult<i64> {
        let conn = self.conn.lock().unwrap();

        // Try to find existing genome
        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM genomes_v2 WHERE experiment_id = ?1 AND config_hash = ?2",
                params![experiment_id, config_hash],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            return Ok(id);
        }

        // Create new genome
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        conn.execute(
            "INSERT INTO genomes_v2
             (experiment_id, config_hash, tiers_json, total_clusters, total_neurons,
              total_memory_bytes, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                experiment_id,
                config_hash,
                tiers_json,
                total_clusters,
                total_neurons,
                total_memory_bytes,
                now,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Create a genome evaluation record.
    ///
    /// Returns the evaluation ID.
    pub fn create_genome_evaluation(
        &self,
        iteration_id: i64,
        genome_id: i64,
        position: i32,
        role: &str,
        elite_rank: Option<i32>,
        ce: f64,
        accuracy: f64,
        fitness_score: Option<f64>,
        eval_time_ms: Option<i32>,
    ) -> SqliteResult<i64> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        conn.execute(
            "INSERT INTO genome_evaluations_v2
             (iteration_id, genome_id, position, role, elite_rank, ce, accuracy,
              fitness_score, eval_time_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                iteration_id,
                genome_id,
                position,
                role,
                elite_rank,
                ce,
                accuracy,
                fitness_score,
                eval_time_ms,
                now,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Create multiple genome evaluations in a single transaction.
    ///
    /// This is significantly faster than individual inserts.
    pub fn create_genome_evaluations_batch(
        &self,
        evaluations: &[GenomeEvaluation],
    ) -> SqliteResult<Vec<i64>> {
        let mut conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let tx = conn.transaction()?;
        let mut ids = Vec::with_capacity(evaluations.len());

        {
            let mut stmt = tx.prepare(
                "INSERT INTO genome_evaluations_v2
                 (iteration_id, genome_id, position, role, elite_rank, ce, accuracy,
                  fitness_score, eval_time_ms, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;

            for eval in evaluations {
                stmt.execute(params![
                    eval.iteration_id,
                    eval.genome_id,
                    eval.position,
                    eval.role,
                    eval.elite_rank,
                    eval.ce,
                    eval.accuracy,
                    eval.fitness_score,
                    eval.eval_time_ms,
                    now,
                ])?;
                ids.push(tx.last_insert_rowid());
            }
        }

        tx.commit()?;
        Ok(ids)
    }

    /// Create a health check record.
    pub fn create_health_check(
        &self,
        iteration_id: i64,
        k: i32,
        top_k_ce: f64,
        top_k_accuracy: f64,
        best_ce: Option<f64>,
        best_ce_accuracy: Option<f64>,
        best_acc_ce: Option<f64>,
        best_acc_accuracy: Option<f64>,
        patience_remaining: Option<i32>,
        patience_status: Option<&str>,
    ) -> SqliteResult<i64> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        conn.execute(
            "INSERT INTO health_checks_v2
             (iteration_id, k, top_k_ce, top_k_accuracy, best_ce, best_ce_accuracy,
              best_acc_ce, best_acc_accuracy, patience_remaining, patience_status, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                iteration_id,
                k,
                top_k_ce,
                top_k_accuracy,
                best_ce,
                best_ce_accuracy,
                best_acc_ce,
                best_acc_accuracy,
                patience_remaining,
                patience_status,
                now,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Update phase progress (current iteration and best metrics).
    pub fn update_phase_progress(
        &self,
        phase_id: i64,
        current_iteration: i32,
        best_ce: Option<f64>,
        best_accuracy: Option<f64>,
    ) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE phases_v2
             SET current_iteration = ?1,
                 best_ce = COALESCE(?2, best_ce),
                 best_accuracy = COALESCE(?3, best_accuracy)
             WHERE id = ?4",
            params![current_iteration, best_ce, best_accuracy, phase_id],
        )?;

        Ok(())
    }
}

/// Genome evaluation data for batch inserts.
#[derive(Debug, Clone)]
pub struct GenomeEvaluation {
    pub iteration_id: i64,
    pub genome_id: i64,
    pub position: i32,
    pub role: String,
    pub elite_rank: Option<i32>,
    pub ce: f64,
    pub accuracy: f64,
    pub fitness_score: Option<f64>,
    pub eval_time_ms: Option<i32>,
}

impl GenomeEvaluation {
    pub fn new(
        iteration_id: i64,
        genome_id: i64,
        position: i32,
        role: &str,
        ce: f64,
        accuracy: f64,
    ) -> Self {
        Self {
            iteration_id,
            genome_id,
            position,
            role: role.to_string(),
            elite_rank: None,
            ce,
            accuracy,
            fitness_score: None,
            eval_time_ms: None,
        }
    }

    pub fn with_elite_rank(mut self, rank: i32) -> Self {
        self.elite_rank = Some(rank);
        self.role = "elite".to_string();
        self
    }

    pub fn with_fitness_score(mut self, score: f64) -> Self {
        self.fitness_score = Some(score);
        self
    }

    pub fn with_eval_time(mut self, time_ms: i32) -> Self {
        self.eval_time_ms = Some(time_ms);
        self
    }
}

/// Compute config hash for genome deduplication.
///
/// Takes a flattened array of tier configs: [(tier, clusters, neurons, bits), ...]
pub fn compute_config_hash(tier_configs: &[(i32, i32, i32, i32)]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for (tier, clusters, neurons, bits) in tier_configs {
        tier.hash(&mut hasher);
        clusters.hash(&mut hasher);
        neurons.hash(&mut hasher);
        bits.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Convert tier configs to JSON string.
pub fn tier_configs_to_json(tier_configs: &[(i32, i32, i32, i32)]) -> String {
    let tiers: Vec<_> = tier_configs
        .iter()
        .map(|(tier, clusters, neurons, bits)| {
            format!(
                r#"{{"tier":{},"clusters":{},"neurons":{},"bits":{}}}"#,
                tier, clusters, neurons, bits
            )
        })
        .collect();
    format!("[{}]", tiers.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_config_hash() {
        let config1 = vec![(0, 100, 15, 20), (1, 400, 10, 12), (2, 49757, 5, 8)];
        let config2 = vec![(0, 100, 15, 20), (1, 400, 10, 12), (2, 49757, 5, 8)];
        let config3 = vec![(0, 100, 15, 21), (1, 400, 10, 12), (2, 49757, 5, 8)];

        assert_eq!(compute_config_hash(&config1), compute_config_hash(&config2));
        assert_ne!(compute_config_hash(&config1), compute_config_hash(&config3));
    }

    #[test]
    fn test_tier_configs_to_json() {
        let config = vec![(0, 100, 15, 20), (1, 400, 10, 12)];
        let json = tier_configs_to_json(&config);
        assert!(json.contains(r#""tier":0"#));
        assert!(json.contains(r#""clusters":100"#));
    }
}
