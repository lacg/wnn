//! Schema migrations for legacy databases (split from db/mod.rs).

use super::*;

/// Run migrations for schema changes (legacy databases only)
pub(crate) async fn run_migrations(pool: &DbPool) -> Result<()>
{
	// Index for the WS snapshot hot path: recent iterations per experiment
	// ordered by created_at (P4 — the existing indexes cover iteration_num only)
	let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_iterations_exp_created ON iterations(experiment_id, created_at DESC)"
    )
        .execute(pool)
        .await;

	// Migration: Add pid to flows for stop/restart functionality
	let _ = sqlx::query("ALTER TABLE flows ADD COLUMN pid INTEGER")
		.execute(pool)
		.await;

	// Migration: Add last_heartbeat to flows (stale-flow detection). The live
	// DB gained this column out-of-band; without this migration a FRESH
	// database 500s on every flow SELECT (they all include the column).
	let _ = sqlx::query("ALTER TABLE flows ADD COLUMN last_heartbeat TEXT")
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

	// Migration: Add status_message to experiments for real-time progress tracking
	let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN status_message TEXT")
		.execute(pool)
		.await;

	// Migration: Add status_message to flows for real-time flow progress tracking
	let _ = sqlx::query("ALTER TABLE flows ADD COLUMN status_message TEXT")
		.execute(pool)
		.await;

	// Migration: Add bitwise-specific fields to genomes
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN architecture_type TEXT DEFAULT 'tiered'")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN connections_json TEXT")
		.execute(pool)
		.await;
	// Migration: real sparse footprint primitive (docs/sparse_footprint_fix.md).
	// NULL until measured at eval-time / backfilled; total_memory_bytes is the
	// deprecated dense 2^bits fiction (caps at i64::MAX for high-bit genomes).
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN materialized_cells INTEGER")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN hf_config_json TEXT")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN hf_export_path TEXT")
		.execute(pool)
		.await;

	// Migration: Create combined_validations table for multi-stage end-to-end metrics
	let _ = sqlx::query(
		r#"CREATE TABLE IF NOT EXISTS combined_validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flow_id INTEGER NOT NULL REFERENCES flows(id),
            genome_type TEXT NOT NULL,
            combined_ce REAL NOT NULL,
            combined_accuracy REAL NOT NULL,
            per_stage_ce_json TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(flow_id, genome_type)
        )"#,
	)
	.execute(pool)
	.await;

	// Migration: Add per_stage_acc_json to combined_validations
	let _ = sqlx::query("ALTER TABLE combined_validations ADD COLUMN per_stage_acc_json TEXT")
		.execute(pool)
		.await;

	// Migration: Add params_json to experiments for per-experiment params (lambda_sweep etc.)
	let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN params_json TEXT")
		.execute(pool)
		.await;

	// Migration: Add unigram_lambda to combined_validations
	let _ = sqlx::query("ALTER TABLE combined_validations ADD COLUMN unigram_lambda REAL")
		.execute(pool)
		.await;

	// Migration: Add extra_metrics_json to experiments (IDS metrics: F1, FPR, confusion matrix)
	let _ = sqlx::query("ALTER TABLE experiments ADD COLUMN extra_metrics_json TEXT")
		.execute(pool)
		.await;

	// Migration: Add IDS metrics to iterations (per-iteration F1 and FPR tracking)
	let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN best_f1 REAL")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN best_fpr REAL")
		.execute(pool)
		.await;
	// Migration: per-iteration controller attitude error (degrees); NULL for IDS/LM
	let _ = sqlx::query("ALTER TABLE iterations ADD COLUMN mean_attitude_error_deg REAL")
		.execute(pool)
		.await;

	// Migration: pause-request flag on flows (set by API, polled by worker between gens)
	// Worker sees `pause_requested=1` at the end of a GA generation, saves checkpoint,
	// sets flow.status='paused', and moves on to the next queued flow.
	let _ = sqlx::query("ALTER TABLE flows ADD COLUMN pause_requested INTEGER NOT NULL DEFAULT 0")
		.execute(pool)
		.await;

	// Migration: Add IDS metrics to validation_summaries (F1-macro and FPR)
	let _ = sqlx::query("ALTER TABLE validation_summaries ADD COLUMN f1_macro REAL")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE validation_summaries ADD COLUMN fpr REAL")
		.execute(pool)
		.await;

	// Migration: Add three-threshold metadata to validation_summaries (JSON)
	let _ = sqlx::query("ALTER TABLE validation_summaries ADD COLUMN threshold_metadata TEXT")
		.execute(pool)
		.await;

	// Migration: Add IDS metrics to genome_evaluations (per-genome F1-macro and FPR)
	let _ = sqlx::query("ALTER TABLE genome_evaluations ADD COLUMN f1_macro REAL")
		.execute(pool)
		.await;
	let _ = sqlx::query("ALTER TABLE genome_evaluations ADD COLUMN fpr REAL")
		.execute(pool)
		.await;

	// Migration: Add genome_hash (connection-inclusive) to genomes for leaderboard identity
	let _ = sqlx::query("ALTER TABLE genomes ADD COLUMN genome_hash TEXT")
		.execute(pool)
		.await;

	// Migration: Create best_genomes table for leaderboard
	let _ = sqlx::query(
		r#"CREATE TABLE IF NOT EXISTS best_genomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            stage TEXT NOT NULL,
            metric TEXT NOT NULL,
            genome_id INTEGER NOT NULL REFERENCES genomes(id),
            genome_hash TEXT NOT NULL,
            rank INTEGER,
            ce REAL NOT NULL,
            accuracy REAL NOT NULL,
            f1_macro REAL,
            fpr REAL,
            flow_id INTEGER,
            experiment_id INTEGER,
            hf_repo_id TEXT,
            hf_exported_at TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE(task_type, stage, metric, genome_hash)
        )"#,
	)
	.execute(pool)
	.await;

	let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_best_genomes_ranking ON best_genomes(task_type, stage, metric, rank)"
    )
    .execute(pool)
    .await;

	Ok(())
}
