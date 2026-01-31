//! HTTP API routes for the dashboard

use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, patch, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::db::DbPool;
use crate::models::*;
use crate::watcher::{LogWatcher, WatcherHandle};

/// Shared dashboard state updated by watcher, read by WebSocket clients
#[derive(Debug, Default)]
pub struct DashboardState {
    pub phases: Vec<Phase>,
    pub current_phase: Option<Phase>,
    pub iterations: Vec<Iteration>,
    pub best_ce: f64,
    pub best_ce_acc: f64,
    pub best_acc: f64,
    pub best_acc_ce: f64,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            best_ce: f64::INFINITY,
            best_acc_ce: f64::INFINITY,
            ..Default::default()
        }
    }

    pub fn to_snapshot(&self) -> DashboardSnapshot {
        DashboardSnapshot {
            phases: self.phases.clone(),
            current_phase: self.current_phase.clone(),
            iterations: self.iterations.clone(), // Send ALL iterations
            best_ce: self.best_ce,
            best_ce_acc: self.best_ce_acc,
            best_acc: self.best_acc,
            best_acc_ce: self.best_acc_ce,
        }
    }

    pub fn update_from_message(&mut self, msg: &WsMessage) {
        match msg {
            WsMessage::PhaseStarted(phase) => {
                self.phases.push(phase.clone());
                self.current_phase = Some(phase.clone());
                self.iterations.clear();
            }
            WsMessage::IterationUpdate(iter) => {
                self.iterations.push(iter.clone());
                // Keep last 500 iterations
                if self.iterations.len() > 500 {
                    self.iterations.remove(0);
                }
                // Update best metrics
                if iter.best_ce < self.best_ce {
                    self.best_ce = iter.best_ce;
                    self.best_ce_acc = iter.best_accuracy.unwrap_or(0.0);
                }
                if let Some(acc) = iter.best_accuracy {
                    if acc > self.best_acc {
                        self.best_acc = acc;
                        self.best_acc_ce = iter.best_ce;
                    }
                }
            }
            WsMessage::PhaseCompleted { phase, .. } => {
                // Update phase status in list
                if let Some(p) = self.phases.iter_mut().find(|p| p.id == phase.id) {
                    p.status = PhaseStatus::Completed;
                    p.ended_at = phase.ended_at;
                }
            }
            _ => {}
        }
    }
}

pub struct AppState {
    pub db: DbPool,
    pub ws_tx: broadcast::Sender<WsMessage>,
    pub dashboard: Arc<RwLock<DashboardState>>,
    pub watcher_handle: Arc<Mutex<Option<WatcherHandle>>>,
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Experiments
        .route("/api/experiments", get(list_experiments))
        .route("/api/experiments/:id", get(get_experiment))
        .route("/api/experiments", post(create_experiment))
        // Phases
        .route("/api/experiments/:id/phases", get(list_phases).post(create_phase))
        .route("/api/phases/:id", patch(update_phase))
        .route("/api/phases/:id/iterations", get(list_iterations))
        // Flows
        .route("/api/flows", get(list_flows).post(create_flow))
        .route("/api/flows/:id", get(get_flow).patch(update_flow).delete(delete_flow))
        .route("/api/flows/:id/experiments", get(list_flow_experiments).post(add_experiment_to_flow))
        .route("/api/flows/:id/stop", post(stop_flow))
        .route("/api/flows/:id/restart", post(restart_flow))
        .route("/api/flows/:id/pid", patch(update_flow_pid))
        // Checkpoints
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .route("/api/checkpoints/:id", get(get_checkpoint).delete(delete_checkpoint))
        .route("/api/checkpoints/:id/download", get(download_checkpoint))
        // Real-time (v1 - log parsing)
        .route("/ws", get(websocket_handler))
        // Log watcher control
        .route("/api/watch", post(watch_log_file))
        // =============================================================================
        // V2 API: Database as source of truth (no log parsing)
        // =============================================================================
        .route("/api/v2/experiments/current", get(get_current_experiment_v2))
        .route("/api/v2/experiments/:id", get(get_experiment_v2_handler))
        .route("/api/v2/experiments/:id/phases", get(get_phases_v2_handler))
        .route("/api/v2/experiments/:id/iterations", get(get_recent_iterations_v2_handler))
        .route("/api/v2/phases/:id/iterations", get(get_iterations_v2_handler))
        .route("/api/v2/iterations/:id/genomes", get(get_iteration_genomes_v2_handler))
        .route("/api/v2/snapshot", get(get_snapshot_v2))
        // Real-time (v2 - DB polling)
        .route("/ws/v2", get(websocket_handler_v2))
        .with_state(state)
}

async fn list_experiments(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match crate::db::queries::list_experiments(&state.db, 100, 0).await {
        Ok(experiments) => (StatusCode::OK, Json(experiments)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn get_experiment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_experiment(&state.db, id).await {
        Ok(Some(exp)) => (StatusCode::OK, Json(exp)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Experiment not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateExperimentRequest {
    pub name: String,
    pub log_path: String,
    #[serde(default)]
    pub config: ExperimentConfig,
}

async fn create_experiment(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateExperimentRequest>,
) -> impl IntoResponse {
    match crate::db::queries::create_experiment(&state.db, &req.name, &req.log_path, &req.config).await {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn list_phases(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_phases(&state.db, experiment_id).await {
        Ok(phases) => (StatusCode::OK, Json(phases)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn list_iterations(
    State(state): State<Arc<AppState>>,
    Path(phase_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_iterations(&state.db, phase_id).await {
        Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreatePhaseRequest {
    pub name: String,
    pub phase_type: String,
}

async fn create_phase(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
    Json(req): Json<CreatePhaseRequest>,
) -> impl IntoResponse {
    match crate::db::queries::create_phase(&state.db, experiment_id, &req.name, &req.phase_type).await {
        Ok(phase) => (StatusCode::CREATED, Json(phase)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdatePhaseRequest {
    pub status: Option<String>,
    pub ended_at: Option<String>,
}

async fn update_phase(
    State(state): State<Arc<AppState>>,
    Path(phase_id): Path<i64>,
    Json(req): Json<UpdatePhaseRequest>,
) -> impl IntoResponse {
    match crate::db::queries::update_phase(&state.db, phase_id, req.status.as_deref(), req.ended_at.as_deref()).await {
        Ok(phase) => (StatusCode::OK, Json(phase)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Flow handlers
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ListFlowsQuery {
    pub status: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

async fn list_flows(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListFlowsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let offset = query.offset.unwrap_or(0);

    match crate::db::queries::list_flows(&state.db, query.status.as_deref(), limit, offset).await {
        Ok(flows) => (StatusCode::OK, Json(flows)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateFlowRequest {
    pub name: String,
    pub description: Option<String>,
    pub config: FlowConfig,
    pub seed_checkpoint_id: Option<i64>,
}

async fn create_flow(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateFlowRequest>,
) -> impl IntoResponse {
    match crate::db::queries::create_flow(
        &state.db,
        &req.name,
        req.description.as_deref(),
        &req.config,
        req.seed_checkpoint_id,
    ).await {
        Ok(id) => {
            // Fetch the created flow to return it
            match crate::db::queries::get_flow(&state.db, id).await {
                Ok(Some(flow)) => (StatusCode::CREATED, Json(flow)).into_response(),
                _ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn get_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_flow(&state.db, id).await {
        Ok(Some(flow)) => (StatusCode::OK, Json(flow)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateFlowRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<FlowStatus>,
    pub config: Option<serde_json::Value>,
    pub seed_checkpoint_id: Option<Option<i64>>, // None = don't update, Some(None) = clear, Some(Some(id)) = set
}

async fn update_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateFlowRequest>,
) -> impl IntoResponse {
    let status_str = req.status.as_ref().map(|s| match s {
        FlowStatus::Pending => "pending",
        FlowStatus::Queued => "queued",
        FlowStatus::Running => "running",
        FlowStatus::Completed => "completed",
        FlowStatus::Failed => "failed",
        FlowStatus::Cancelled => "cancelled",
    });

    match crate::db::queries::update_flow(
        &state.db,
        id,
        req.name.as_deref(),
        req.description.as_deref(),
        status_str,
        req.config.as_ref(),
        req.seed_checkpoint_id,
    ).await {
        Ok(true) => {
            // Fetch and return updated flow
            match crate::db::queries::get_flow(&state.db, id).await {
                Ok(Some(flow)) => {
                    // Broadcast status changes
                    if req.status.is_some() {
                        match &flow.status {
                            FlowStatus::Running => {
                                let _ = state.ws_tx.send(WsMessage::FlowStarted(flow.clone()));
                            }
                            FlowStatus::Completed => {
                                let _ = state.ws_tx.send(WsMessage::FlowCompleted(flow.clone()));
                            }
                            FlowStatus::Failed => {
                                let _ = state.ws_tx.send(WsMessage::FlowFailed {
                                    flow: flow.clone(),
                                    error: "Flow failed".to_string(),
                                });
                            }
                            _ => {}
                        }
                    }
                    (StatusCode::OK, Json(flow)).into_response()
                }
                _ => (StatusCode::OK, Json(serde_json::json!({"updated": true}))).into_response(),
            }
        }
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn delete_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::delete_flow(&state.db, id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn list_flow_experiments(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::list_flow_experiments(&state.db, flow_id).await {
        Ok(experiments) => (StatusCode::OK, Json(experiments)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct AddExperimentToFlowRequest {
    pub experiment_id: i64,
    pub sequence_order: Option<i32>,
}

async fn add_experiment_to_flow(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
    Json(req): Json<AddExperimentToFlowRequest>,
) -> impl IntoResponse {
    match crate::db::queries::add_experiment_to_flow(
        &state.db,
        flow_id,
        req.experiment_id,
        req.sequence_order.unwrap_or(0),
    ).await {
        Ok(_) => (StatusCode::OK, Json(serde_json::json!({"success": true}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Stop a running flow by sending SIGTERM to the worker process
async fn stop_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    // Get the flow to check PID
    let flow = match crate::db::queries::get_flow(&state.db, id).await {
        Ok(Some(f)) => f,
        Ok(None) => return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    // Check if flow is running
    if flow.status != FlowStatus::Running {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Flow is not running"})),
        ).into_response();
    }

    // Send SIGTERM if PID exists
    if let Some(pid) = flow.pid {
        #[cfg(unix)]
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
        tracing::info!("Sent SIGTERM to flow {} (PID {})", id, pid);
    }

    // Update status to cancelled
    match crate::db::queries::update_flow(
        &state.db,
        id,
        None,
        None,
        Some("cancelled"),
        None,
        None,
    ).await {
        Ok(_) => {
            // Broadcast cancellation
            if let Ok(Some(updated_flow)) = crate::db::queries::get_flow(&state.db, id).await {
                let _ = state.ws_tx.send(WsMessage::FlowCancelled(updated_flow.clone()));
                (StatusCode::OK, Json(updated_flow)).into_response()
            } else {
                (StatusCode::OK, Json(serde_json::json!({"stopped": true}))).into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct RestartFlowRequest {
    #[serde(default)]
    pub from_beginning: bool,  // If true, restart from scratch; if false, resume from checkpoint
}

/// Restart a flow by setting status to queued
async fn restart_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<RestartFlowRequest>,
) -> impl IntoResponse {
    // Get the flow
    let flow = match crate::db::queries::get_flow(&state.db, id).await {
        Ok(Some(f)) => f,
        Ok(None) => return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    // Check if flow can be restarted
    if flow.status == FlowStatus::Running || flow.status == FlowStatus::Queued {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Flow is already running or queued"})),
        ).into_response();
    }

    // If restarting from beginning, clear the seed checkpoint and delete checkpoint files
    let seed_checkpoint_id = if req.from_beginning {
        // Delete checkpoint directory for this flow
        let safe_name = flow.name.to_lowercase().replace(" ", "_").replace("/", "_");
        let checkpoint_dir = std::path::Path::new("checkpoints").join(&safe_name);
        if checkpoint_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&checkpoint_dir) {
                tracing::warn!("Failed to delete checkpoint directory {:?}: {}", checkpoint_dir, e);
            } else {
                tracing::info!("Deleted checkpoint directory: {:?}", checkpoint_dir);
            }
        }
        Some(None) // Clear checkpoint reference in DB
    } else {
        None // Keep existing
    };

    // Set status to queued (and optionally clear PID)
    match crate::db::queries::update_flow_for_restart(&state.db, id, seed_checkpoint_id).await {
        Ok(_) => {
            if let Ok(Some(updated_flow)) = crate::db::queries::get_flow(&state.db, id).await {
                let _ = state.ws_tx.send(WsMessage::FlowQueued(updated_flow.clone()));
                (StatusCode::OK, Json(updated_flow)).into_response()
            } else {
                (StatusCode::OK, Json(serde_json::json!({"restarted": true}))).into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateFlowPidRequest {
    pub pid: Option<i64>,
}

/// Update the PID of a flow (called by worker when starting)
async fn update_flow_pid(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateFlowPidRequest>,
) -> impl IntoResponse {
    match crate::db::queries::update_flow_pid(&state.db, id, req.pid).await {
        Ok(true) => (StatusCode::OK, Json(serde_json::json!({"success": true}))).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Checkpoint handlers
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ListCheckpointsQuery {
    pub experiment_id: Option<i64>,
    pub is_final: Option<bool>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

async fn list_checkpoints(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListCheckpointsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50);
    let offset = query.offset.unwrap_or(0);

    match crate::db::queries::list_checkpoints(
        &state.db,
        query.experiment_id,
        query.is_final,
        limit,
        offset,
    ).await {
        Ok(checkpoints) => (StatusCode::OK, Json(checkpoints)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateCheckpointRequest {
    pub experiment_id: i64,
    pub name: String,
    pub file_path: String,
    pub file_size_bytes: Option<i64>,
    pub final_fitness: Option<f64>,
    pub final_accuracy: Option<f64>,
    pub iterations_run: Option<i32>,
    pub genome_stats: Option<GenomeStats>,
    pub is_final: bool,
}

async fn create_checkpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCheckpointRequest>,
) -> impl IntoResponse {
    match crate::db::queries::create_checkpoint(
        &state.db,
        req.experiment_id,
        &req.name,
        &req.file_path,
        req.file_size_bytes,
        req.final_fitness,
        req.final_accuracy,
        req.iterations_run,
        req.genome_stats.as_ref(),
        req.is_final,
    ).await {
        Ok(id) => {
            // Fetch the created checkpoint to return and broadcast
            match crate::db::queries::get_checkpoint(&state.db, id).await {
                Ok(Some(checkpoint)) => {
                    let _ = state.ws_tx.send(WsMessage::CheckpointCreated(checkpoint.clone()));
                    (StatusCode::CREATED, Json(checkpoint)).into_response()
                }
                _ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn get_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_checkpoint(&state.db, id).await {
        Ok(Some(checkpoint)) => (StatusCode::OK, Json(checkpoint)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Checkpoint not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn delete_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Query(query): Query<DeleteCheckpointQuery>,
) -> impl IntoResponse {
    let force = query.force.unwrap_or(false);

    match crate::db::queries::delete_checkpoint(&state.db, id, force).await {
        Ok((true, Some(file_path))) => {
            // Try to delete the file (best-effort)
            if let Err(e) = std::fs::remove_file(&file_path) {
                tracing::warn!("Failed to delete checkpoint file {}: {}", file_path, e);
            }

            // Broadcast deletion
            let _ = state.ws_tx.send(WsMessage::CheckpointDeleted { id });

            StatusCode::NO_CONTENT.into_response()
        }
        Ok((true, None)) => {
            let _ = state.ws_tx.send(WsMessage::CheckpointDeleted { id });
            StatusCode::NO_CONTENT.into_response()
        }
        Ok((false, _)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Checkpoint not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct DeleteCheckpointQuery {
    pub force: Option<bool>,
}

async fn download_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    use axum::body::Body;
    use axum::http::header;
    use tokio_util::io::ReaderStream;

    // Get checkpoint from database
    let checkpoint = match crate::db::queries::get_checkpoint(&state.db, id).await {
        Ok(Some(c)) => c,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Checkpoint not found"})),
            ).into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Open the file
    let file = match tokio::fs::File::open(&checkpoint.file_path).await {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Checkpoint file not found: {}", e)
                })),
            ).into_response();
        }
    };

    // Get filename from path
    let filename = std::path::Path::new(&checkpoint.file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("checkpoint.json.gz");

    // Determine content type
    let content_type = if filename.ends_with(".gz") {
        "application/gzip"
    } else if filename.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    };

    // Create streaming body
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    // Build response with headers
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CONTENT_DISPOSITION,
                &format!("attachment; filename=\"{}\"", filename),
            ),
        ],
        body,
    ).into_response()
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(
    mut socket: axum::extract::ws::WebSocket,
    state: Arc<AppState>,
) {
    use axum::extract::ws::Message;

    // Send current snapshot to new client
    {
        let dashboard = state.dashboard.read().await;
        let snapshot = WsMessage::Snapshot(dashboard.to_snapshot());
        let json = serde_json::to_string(&snapshot).unwrap();
        if socket.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    // Subscribe to new updates
    let mut rx = state.ws_tx.subscribe();

    // Forward updates to client
    while let Ok(msg) = rx.recv().await {
        let json = serde_json::to_string(&msg).unwrap();
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }
}

// === Log Watcher Control ===

#[derive(Debug, Deserialize)]
pub struct WatchLogRequest {
    pub log_path: String,
}

/// Switch the log watcher to a new file
async fn watch_log_file(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WatchLogRequest>,
) -> impl IntoResponse {
    // Stop the current watcher if any
    {
        let mut handle_guard = state.watcher_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            handle.stop();
            tracing::info!("Stopped previous log watcher");
        }
    }

    // Clear the dashboard state for the new log file
    {
        let mut dash = state.dashboard.write().await;
        *dash = DashboardState::new();
    }

    // Start a new watcher
    let watcher = LogWatcher::new(
        req.log_path.clone(),
        state.ws_tx.clone(),
        state.dashboard.clone(),
    );

    match watcher.start(false).await {
        Ok(handle) => {
            *state.watcher_handle.lock().await = Some(handle);
            tracing::info!("Now watching log file: {}", req.log_path);
            (StatusCode::OK, Json(serde_json::json!({
                "status": "watching",
                "log_path": req.log_path
            }))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// V2 API Handlers (Database as source of truth)
// =============================================================================

/// Get the currently running experiment from v2 tables
async fn get_current_experiment_v2(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match crate::db::queries::get_running_experiment_v2(&state.db).await {
        Ok(Some(exp)) => (StatusCode::OK, Json(exp)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "No running experiment"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get an experiment by ID from v2 tables
async fn get_experiment_v2_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_experiment_v2(&state.db, id).await {
        Ok(Some(exp)) => (StatusCode::OK, Json(exp)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Experiment not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get phases for an experiment from v2 tables
async fn get_phases_v2_handler(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_phases_v2(&state.db, experiment_id).await {
        Ok(phases) => (StatusCode::OK, Json(phases)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get iterations for a phase from v2 tables
async fn get_iterations_v2_handler(
    State(state): State<Arc<AppState>>,
    Path(phase_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_iterations_v2(&state.db, phase_id).await {
        Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get genome evaluations for an iteration
async fn get_iteration_genomes_v2_handler(
    State(state): State<Arc<AppState>>,
    Path(iteration_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_genome_evaluations_v2(&state.db, iteration_id).await {
        Ok(evaluations) => (StatusCode::OK, Json(evaluations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct RecentIterationsQuery {
    pub limit: Option<i32>,
}

/// Get recent iterations for an experiment (across all phases)
async fn get_recent_iterations_v2_handler(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
    Query(query): Query<RecentIterationsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(100);
    match crate::db::queries::get_recent_iterations_v2(&state.db, experiment_id, limit).await {
        Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get full dashboard snapshot from v2 tables
async fn get_snapshot_v2(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Get running experiment
    let experiment = match crate::db::queries::get_running_experiment_v2(&state.db).await {
        Ok(exp) => exp,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    let Some(exp) = experiment else {
        // No running experiment - return empty snapshot
        return (StatusCode::OK, Json(DashboardSnapshotV2::default())).into_response();
    };

    let exp_id = exp.id;

    // Get phases
    let phases = match crate::db::queries::get_phases_v2(&state.db, exp_id).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Get current phase
    let current_phase = match crate::db::queries::get_current_phase_v2(&state.db, exp_id).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Get recent iterations
    let iterations = match crate::db::queries::get_recent_iterations_v2(&state.db, exp_id, 500).await {
        Ok(i) => i,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Calculate best metrics from phases
    let mut best_ce = f64::INFINITY;
    let mut best_accuracy = 0.0;
    for phase in &phases {
        if let Some(ce) = phase.best_ce {
            if ce < best_ce {
                best_ce = ce;
            }
        }
        if let Some(acc) = phase.best_accuracy {
            if acc > best_accuracy {
                best_accuracy = acc;
            }
        }
    }

    let snapshot = DashboardSnapshotV2 {
        current_experiment: Some(exp),
        current_phase,
        phases,
        iterations,
        best_ce: if best_ce.is_infinite() { 0.0 } else { best_ce },
        best_accuracy,
    };

    (StatusCode::OK, Json(snapshot)).into_response()
}

/// V2 WebSocket handler - polls database for updates
async fn websocket_handler_v2(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket_v2(socket, state))
}

async fn handle_socket_v2(
    mut socket: axum::extract::ws::WebSocket,
    state: Arc<AppState>,
) {
    use axum::extract::ws::Message;
    use tokio::time::{interval, Duration};

    // Send initial snapshot
    let snapshot = build_snapshot_v2(&state.db).await;
    if let Ok(json) = serde_json::to_string(&WsMessageV2::Snapshot(snapshot)) {
        if socket.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    // Poll for updates every 500ms
    let mut poll_interval = interval(Duration::from_millis(500));
    let mut last_iteration_id: Option<i64> = None;

    loop {
        poll_interval.tick().await;

        // Check for new iterations
        if let Ok(Some(exp)) = crate::db::queries::get_running_experiment_v2(&state.db).await {
            if let Ok(iterations) = crate::db::queries::get_recent_iterations_v2(&state.db, exp.id, 10).await {
                // Send new iterations since last poll
                for iter in iterations.iter().rev() {
                    if last_iteration_id.map_or(true, |last_id| iter.id > last_id) {
                        let msg = WsMessageV2::IterationCompleted(iter.clone());
                        if let Ok(json) = serde_json::to_string(&msg) {
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                return;
                            }
                        }
                        last_iteration_id = Some(iter.id);
                    }
                }
            }
        }
    }
}

async fn build_snapshot_v2(db: &DbPool) -> DashboardSnapshotV2 {
    let experiment = crate::db::queries::get_running_experiment_v2(db).await.ok().flatten();

    let Some(exp) = experiment else {
        return DashboardSnapshotV2::default();
    };

    let exp_id = exp.id;
    let phases = crate::db::queries::get_phases_v2(db, exp_id).await.unwrap_or_default();
    let current_phase = crate::db::queries::get_current_phase_v2(db, exp_id).await.ok().flatten();
    let iterations = crate::db::queries::get_recent_iterations_v2(db, exp_id, 500).await.unwrap_or_default();

    // Calculate best metrics
    let mut best_ce = f64::INFINITY;
    let mut best_accuracy = 0.0;
    for phase in &phases {
        if let Some(ce) = phase.best_ce {
            if ce < best_ce {
                best_ce = ce;
            }
        }
        if let Some(acc) = phase.best_accuracy {
            if acc > best_accuracy {
                best_accuracy = acc;
            }
        }
    }

    DashboardSnapshotV2 {
        current_experiment: Some(exp),
        current_phase,
        phases,
        iterations,
        best_ce: if best_ce.is_infinite() { 0.0 } else { best_ce },
        best_accuracy,
    }
}
