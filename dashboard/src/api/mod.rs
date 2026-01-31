//! HTTP API routes for the dashboard
//!
//! Unified API (no V1/V2 distinction) - database as source of truth

use axum::{
    extract::{Path, Query, State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, patch, post},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::db::DbPool;
use crate::models::*;

pub struct AppState {
    pub db: DbPool,
    pub ws_tx: broadcast::Sender<WsMessage>,
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Experiments
        .route("/api/experiments", get(list_experiments))
        .route("/api/experiments/current", get(get_current_experiment))
        .route("/api/experiments/:id", get(get_experiment))
        .route("/api/experiments/:id/phases", get(get_experiment_phases))
        .route("/api/experiments/:id/iterations", get(get_experiment_iterations))
        // Phases
        .route("/api/phases/:id/iterations", get(get_phase_iterations))
        // Iterations
        .route("/api/iterations/:id/genomes", get(get_iteration_genomes))
        // Snapshot
        .route("/api/snapshot", get(get_snapshot))
        // Flows
        .route("/api/flows", get(list_flows).post(create_flow))
        .route("/api/flows/:id", get(get_flow).patch(update_flow).delete(delete_flow))
        .route("/api/flows/:id/experiments", get(list_flow_experiments))
        .route("/api/flows/:id/stop", post(stop_flow))
        .route("/api/flows/:id/restart", post(restart_flow))
        .route("/api/flows/:id/pid", patch(update_flow_pid))
        // Checkpoints
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .route("/api/checkpoints/:id", get(get_checkpoint).delete(delete_checkpoint))
        .route("/api/checkpoints/:id/download", get(download_checkpoint))
        // WebSocket (database polling)
        .route("/ws", get(websocket_handler))
        .with_state(state)
}

// =============================================================================
// Experiment handlers
// =============================================================================

async fn list_experiments(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match crate::db::queries::list_experiments(&state.db, 100, 0).await {
        Ok(experiments) => (StatusCode::OK, Json(experiments)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn get_current_experiment(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match crate::db::queries::get_running_experiment(&state.db).await {
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

async fn get_experiment_phases(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_experiment_phases(&state.db, experiment_id).await {
        Ok(phases) => (StatusCode::OK, Json(phases)).into_response(),
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

async fn get_experiment_iterations(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
    Query(query): Query<RecentIterationsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(100);
    match crate::db::queries::get_recent_iterations(&state.db, experiment_id, limit).await {
        Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Phase handlers
// =============================================================================

async fn get_phase_iterations(
    State(state): State<Arc<AppState>>,
    Path(phase_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_phase_iterations(&state.db, phase_id).await {
        Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Iteration handlers
// =============================================================================

async fn get_iteration_genomes(
    State(state): State<Arc<AppState>>,
    Path(iteration_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_genome_evaluations(&state.db, iteration_id).await {
        Ok(evaluations) => (StatusCode::OK, Json(evaluations)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Snapshot handler
// =============================================================================

async fn get_snapshot(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Get running experiment
    let experiment = match crate::db::queries::get_running_experiment(&state.db).await {
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
        return (StatusCode::OK, Json(DashboardSnapshot::default())).into_response();
    };

    let exp_id = exp.id;

    // Get phases
    let phases = match crate::db::queries::get_experiment_phases(&state.db, exp_id).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Get current phase
    let current_phase = match crate::db::queries::get_current_phase(&state.db, exp_id).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Get recent iterations
    let iterations = match crate::db::queries::get_recent_iterations(&state.db, exp_id, 500).await {
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

    let snapshot = DashboardSnapshot {
        current_experiment: Some(exp),
        current_phase,
        phases,
        iterations,
        best_ce: if best_ce.is_infinite() { 0.0 } else { best_ce },
        best_accuracy,
    };

    (StatusCode::OK, Json(snapshot)).into_response()
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
        FlowStatus::Paused => "paused",
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

/// Stop a running flow by sending SIGTERM to the worker process
async fn stop_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    // Get the flow to check status
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

    // Use shared stop function (sends SIGTERM, updates status to cancelled)
    if let Err(e) = crate::db::queries::stop_flow_process(&state.db, id).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response();
    }

    // Broadcast cancellation and return updated flow
    match crate::db::queries::get_flow(&state.db, id).await {
        Ok(Some(updated_flow)) => {
            let _ = state.ws_tx.send(WsMessage::FlowCancelled(updated_flow.clone()));
            (StatusCode::OK, Json(updated_flow)).into_response()
        }
        Ok(None) => (StatusCode::OK, Json(serde_json::json!({"stopped": true}))).into_response(),
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
        // Try both relative (from dashboard) and parent (project root) paths
        let safe_name = flow.name.to_lowercase().replace(" ", "_").replace("/", "_");

        // Try parent directory first (project root checkpoints)
        let parent_checkpoint_dir = std::path::Path::new("../checkpoints").join(&safe_name);
        let local_checkpoint_dir = std::path::Path::new("checkpoints").join(&safe_name);

        for checkpoint_dir in [&parent_checkpoint_dir, &local_checkpoint_dir] {
            if checkpoint_dir.exists() {
                if let Err(e) = std::fs::remove_dir_all(checkpoint_dir) {
                    tracing::warn!("Failed to delete checkpoint directory {:?}: {}", checkpoint_dir, e);
                } else {
                    tracing::info!("Deleted checkpoint directory: {:?}", checkpoint_dir);
                }
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
    pub checkpoint_type: Option<String>,
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
        query.checkpoint_type.as_deref(),
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
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub checkpoint_type: Option<String>,
    pub phase_id: Option<i64>,
    pub iteration_id: Option<i64>,
}

async fn create_checkpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCheckpointRequest>,
) -> impl IntoResponse {
    let checkpoint_type = req.checkpoint_type.as_deref().unwrap_or("auto");
    match crate::db::queries::create_checkpoint(
        &state.db,
        req.experiment_id,
        &req.name,
        &req.file_path,
        checkpoint_type,
        req.file_size_bytes,
        req.phase_id,
        req.iteration_id,
        req.best_ce,
        req.best_accuracy,
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
) -> impl IntoResponse {
    match crate::db::queries::delete_checkpoint(&state.db, id).await {
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

// =============================================================================
// WebSocket handler (database polling)
// =============================================================================

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
    use tokio::time::{interval, Duration};

    // Send initial snapshot and track current state
    let snapshot = build_snapshot(&state.db).await;
    let mut last_experiment_id = snapshot.current_experiment.as_ref().map(|e| e.id);
    let mut last_phase_id = snapshot.current_phase.as_ref().map(|p| p.id);
    let mut last_phase_status = snapshot.current_phase.as_ref().map(|p| p.status.clone());

    // Track the last iteration we've sent to avoid re-sending iterations from snapshot
    let mut last_iteration_id: Option<i64> = snapshot.iterations.last().map(|i| i.id);

    if let Ok(json) = serde_json::to_string(&WsMessage::Snapshot(snapshot)) {
        if socket.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    // Subscribe to broadcast channel for flow events
    let mut rx = state.ws_tx.subscribe();

    // Poll for updates every 500ms
    let mut poll_interval = interval(Duration::from_millis(500));

    loop {
        tokio::select! {
            // Handle broadcast messages (flow status updates)
            result = rx.recv() => {
                match result {
                    Ok(msg) => {
                        // On flow state change, send a fresh snapshot
                        let should_send_snapshot = matches!(&msg,
                            WsMessage::FlowStarted(_) |
                            WsMessage::FlowCompleted(_) |
                            WsMessage::FlowFailed { .. } |
                            WsMessage::FlowCancelled(_)
                        );

                        // Forward flow-related messages to client
                        let json_result = match &msg {
                            WsMessage::FlowStarted(_) |
                            WsMessage::FlowQueued(_) |
                            WsMessage::FlowCompleted(_) |
                            WsMessage::FlowFailed { .. } |
                            WsMessage::FlowCancelled(_) => {
                                serde_json::to_string(&msg)
                            }
                            _ => continue, // Skip non-flow messages
                        };
                        if let Ok(json) = json_result {
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                return;
                            }
                        }

                        // Send fresh snapshot after flow state change
                        if should_send_snapshot {
                            let snapshot = build_snapshot(&state.db).await;
                            last_experiment_id = snapshot.current_experiment.as_ref().map(|e| e.id);
                            last_phase_id = snapshot.current_phase.as_ref().map(|p| p.id);
                            last_phase_status = snapshot.current_phase.as_ref().map(|p| p.status.clone());
                            if let Ok(json) = serde_json::to_string(&WsMessage::Snapshot(snapshot)) {
                                if socket.send(Message::Text(json.into())).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // Missed some messages, continue
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return;
                    }
                }
            }

            // DB polling for iterations and state changes
            _ = poll_interval.tick() => {
                // Check for experiment/phase changes
                let snapshot = build_snapshot(&state.db).await;
                let current_exp_id = snapshot.current_experiment.as_ref().map(|e| e.id);
                let current_phase_id = snapshot.current_phase.as_ref().map(|p| p.id);
                let current_phase_status = snapshot.current_phase.as_ref().map(|p| p.status.clone());

                // Send fresh snapshot if experiment or phase changed
                if current_exp_id != last_experiment_id ||
                   current_phase_id != last_phase_id ||
                   current_phase_status != last_phase_status {
                    last_experiment_id = current_exp_id;
                    last_phase_id = current_phase_id;
                    last_phase_status = current_phase_status;
                    if let Ok(json) = serde_json::to_string(&WsMessage::Snapshot(snapshot)) {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            return;
                        }
                    }
                } else if let Some(ref exp) = snapshot.current_experiment {
                    // Just send new iterations
                    if let Ok(iterations) = crate::db::queries::get_recent_iterations(&state.db, exp.id, 10).await {
                        for iter in iterations.iter().rev() {
                            if last_iteration_id.map_or(true, |last_id| iter.id > last_id) {
                                let msg = WsMessage::IterationCompleted(iter.clone());
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
    }
}

async fn build_snapshot(db: &DbPool) -> DashboardSnapshot {
    let experiment = crate::db::queries::get_running_experiment(db).await.ok().flatten();

    let Some(exp) = experiment else {
        return DashboardSnapshot::default();
    };

    let exp_id = exp.id;
    let phases = crate::db::queries::get_experiment_phases(db, exp_id).await.unwrap_or_default();
    let current_phase = crate::db::queries::get_current_phase(db, exp_id).await.ok().flatten();
    let iterations = crate::db::queries::get_recent_iterations(db, exp_id, 500).await.unwrap_or_default();

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

    DashboardSnapshot {
        current_experiment: Some(exp),
        current_phase,
        phases,
        iterations,
        best_ce: if best_ce.is_infinite() { 0.0 } else { best_ce },
        best_accuracy,
    }
}
