//! Database operations for experiment tracking
//!
//! Simplified schema (Phase layer removed):
//! Tables: flows, experiments, iterations, genomes, genome_evaluations, health_checks, checkpoints

use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::Row;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use tracing;

use crate::models::*;

mod best_genomes;
mod checkpoints;
mod experiments;
mod flow_lifecycle;
mod flows;
mod gating;
mod iterations;
mod migrations;
mod parse;
mod schema;
mod validations;

pub type DbPool = Pool<Sqlite>;

/// Initialize database with schema
pub async fn init_db(database_url: &str) -> Result<DbPool> {
    // Explicit pragma posture (P4, 12/06/2026). sqlx defaults left journal
    // mode untouched: in rollback-journal mode ANY write blocked ALL readers,
    // and with per-WebSocket 500ms polling + worker writes that meant stalls
    // then SQLITE_BUSY 500s. WAL lets readers proceed during writes;
    // busy_timeout 30s rides out long writer transactions instead of erroring.
    use std::str::FromStr;
    let options = sqlx::sqlite::SqliteConnectOptions::from_str(database_url)?
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
        .busy_timeout(std::time::Duration::from_secs(30))
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    // Create tables
    sqlx::query(schema::SCHEMA).execute(&pool).await?;

    // Run migrations for existing databases
    migrations::run_migrations(&pool).await?;

    Ok(pool)
}

// Query helpers — split into per-resource submodules (12/06/2026); the
// external path `crate::db::queries::X` is preserved via these re-exports.
pub mod queries {
    pub use super::best_genomes::*;
    pub use super::checkpoints::*;
    pub use super::experiments::*;
    pub use super::flow_lifecycle::*;
    pub use super::flows::*;
    pub use super::gating::*;
    pub use super::iterations::*;
    pub use super::validations::*;
    pub(crate) use super::parse::*;
}
