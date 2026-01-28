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
use tokio::sync::{broadcast, RwLock};

use crate::db::DbPool;
use crate::models::*;

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
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Experiments
        .route("/api/experiments", get(list_experiments))
        .route("/api/experiments/:id", get(get_experiment))
        .route("/api/experiments", post(create_experiment))
        // Phases
        .route("/api/experiments/:id/phases", get(list_phases))
        .route("/api/phases/:id/iterations", get(list_iterations))
        // Flows
        .route("/api/flows", get(list_flows).post(create_flow))
        .route("/api/flows/:id", get(get_flow).patch(update_flow).delete(delete_flow))
        .route("/api/flows/:id/experiments", get(list_flow_experiments))
        // Checkpoints
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .route("/api/checkpoints/:id", get(get_checkpoint).delete(delete_checkpoint))
        .route("/api/checkpoints/:id/download", get(download_checkpoint))
        // Real-time
        .route("/ws", get(websocket_handler))
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
}

async fn update_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateFlowRequest>,
) -> impl IntoResponse {
    let status_str = req.status.as_ref().map(|s| match s {
        FlowStatus::Pending => "pending",
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
