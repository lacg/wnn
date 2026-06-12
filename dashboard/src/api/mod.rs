//! HTTP API routes for the dashboard
//!
//! Unified API (no V1/V2 distinction) - database as source of truth

use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, patch, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::broadcast;

use tokio::sync::RwLock;

use crate::db::DbPool;
use crate::models::*;

mod best_genomes;
mod checkpoints;
mod experiments;
mod flow_lifecycle;
mod flows;
mod gating;
mod live_progress;
mod snapshot;
mod watch;
mod ws;

use best_genomes::*;
use checkpoints::*;
use experiments::*;
use flow_lifecycle::*;
use flows::*;
use gating::*;
use live_progress::*;
use snapshot::*;
use watch::*;
use ws::*;

pub use ws::start_snapshot_poller;

/// In-memory live progress entry (no DB persistence).
pub(crate) struct LiveProgressEntry {
    data: serde_json::Value,
    updated_at: std::time::Instant,
}

pub struct AppState {
    pub db: DbPool,
    pub ws_tx: broadcast::Sender<WsMessage>,
    pub current_log_path: RwLock<Option<String>>,
    pub live_progress: std::sync::Mutex<std::collections::HashMap<i64, LiveProgressEntry>>,
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Experiments
        .route("/api/experiments", get(list_experiments).post(create_experiment))
        .route("/api/experiments/current", get(get_current_experiment))
        .route("/api/experiments/:id", get(get_experiment).patch(update_experiment).delete(delete_experiment))
        .route("/api/experiments/:id/iterations", get(get_experiment_iterations))
        .route("/api/experiments/:id/summaries", get(get_validation_summaries).post(create_validation_summary))
        // Gating runs
        .route("/api/experiments/:id/gating", get(list_gating_runs).post(create_gating_run))
        .route("/api/experiments/:id/gating/:gating_id", get(get_gating_run).patch(update_gating_run))
        .route("/api/gating/pending", get(list_pending_gating_runs))
        // Iterations
        .route("/api/iterations/:id/genomes", get(get_iteration_genomes))
        // Snapshot
        .route("/api/snapshot", get(get_snapshot))
        // Flows
        .route("/api/flows", get(list_flows).post(create_flow))
        .route("/api/flows/:id", get(get_flow).patch(update_flow).delete(delete_flow))
        .route("/api/flows/:id/experiments", get(list_flow_experiments).post(add_experiment_to_flow))
        .route("/api/flows/:id/experiments/link", post(link_experiment_to_flow))
        .route("/api/flows/:id/experiments/reorder", axum::routing::put(reorder_flow_experiments))
        .route("/api/flows/:id/stop", post(stop_flow))
        .route("/api/flows/:id/restart", post(restart_flow))
        .route("/api/flows/:id/pause", post(pause_flow))
        .route("/api/flows/:id/resume", post(resume_flow))
        .route("/api/flows/:id/pid", patch(update_flow_pid))
        .route("/api/flows/:id/heartbeat", post(update_flow_heartbeat))
        .route("/api/flows/:id/validations", get(get_flow_validations))
        .route("/api/flows/:id/combined-validations", get(get_combined_validations).post(create_combined_validation))
        .route("/api/flows/:id/run-gating", post(run_flow_gating))
        // Validations
        .route("/api/validations/check", get(check_cached_validation))
        // Checkpoints
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .route("/api/checkpoints/:id", get(get_checkpoint).delete(delete_checkpoint))
        .route("/api/checkpoints/:id/download", get(download_checkpoint))
        .route("/api/checkpoints/:id/export-hf", post(export_checkpoint_hf))
        // Best Genomes (Leaderboard)
        .route("/api/best-genomes", get(list_best_genomes).post(submit_best_genomes))
        .route("/api/best-genomes/recalculate", post(recalculate_best_genomes))
        .route("/api/best-genomes/:id", get(get_best_genome).delete(delete_best_genome))
        .route("/api/best-genomes/:id/download", get(download_best_genome))
        .route("/api/best-genomes/:id/export-hf", post(export_best_genome_hf))
        // Live progress (in-memory, no DB)
        .route("/api/experiments/:id/live-progress", get(get_live_progress).post(post_live_progress).delete(clear_live_progress))
        // Worker log watching
        .route("/api/watch", post(set_watch_log).get(get_watch_log))
        // WebSocket (database polling)
        .route("/ws", get(websocket_handler))
        .with_state(state)
}
