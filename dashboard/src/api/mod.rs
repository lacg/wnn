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

async fn list_experiments(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(Vec::<Experiment>::new())
}

async fn get_experiment(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
) -> impl IntoResponse {
    Json(Option::<Experiment>::None)
}

async fn create_experiment(
    State(_state): State<Arc<AppState>>,
    Json(_config): Json<ExperimentConfig>,
) -> impl IntoResponse {
    Json(serde_json::json!({"id": 1}))
}

async fn list_phases(
    State(_state): State<Arc<AppState>>,
    Path(_experiment_id): Path<i64>,
) -> impl IntoResponse {
    Json(Vec::<Phase>::new())
}

async fn list_iterations(
    State(_state): State<Arc<AppState>>,
    Path(_phase_id): Path<i64>,
) -> impl IntoResponse {
    Json(Vec::<Iteration>::new())
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
    State(_state): State<Arc<AppState>>,
    Query(_query): Query<ListFlowsQuery>,
) -> impl IntoResponse {
    // TODO: Implement with actual database query
    Json(Vec::<Flow>::new())
}

#[derive(Debug, Deserialize)]
pub struct CreateFlowRequest {
    pub name: String,
    pub description: Option<String>,
    pub config: FlowConfig,
    pub seed_checkpoint_id: Option<i64>,
}

async fn create_flow(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<CreateFlowRequest>,
) -> impl IntoResponse {
    // TODO: Implement with actual database insert
    let flow = Flow {
        id: 1,
        name: req.name,
        description: req.description,
        config: req.config,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        status: FlowStatus::Pending,
        seed_checkpoint_id: req.seed_checkpoint_id,
    };
    (StatusCode::CREATED, Json(flow))
}

async fn get_flow(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
) -> impl IntoResponse {
    // TODO: Implement with actual database query
    (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Flow not found"})))
}

#[derive(Debug, Deserialize)]
pub struct UpdateFlowRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<FlowStatus>,
}

async fn update_flow(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
    Json(_req): Json<UpdateFlowRequest>,
) -> impl IntoResponse {
    // TODO: Implement with actual database update
    (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Flow not found"})))
}

async fn delete_flow(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
) -> impl IntoResponse {
    // TODO: Implement with actual database delete
    StatusCode::NO_CONTENT
}

async fn list_flow_experiments(
    State(_state): State<Arc<AppState>>,
    Path(_flow_id): Path<i64>,
) -> impl IntoResponse {
    // TODO: Implement with actual database query
    Json(Vec::<Experiment>::new())
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
    State(_state): State<Arc<AppState>>,
    Query(_query): Query<ListCheckpointsQuery>,
) -> impl IntoResponse {
    // TODO: Implement with actual database query
    Json(Vec::<Checkpoint>::new())
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
    // TODO: Implement with actual database insert
    let checkpoint = Checkpoint {
        id: 1,
        experiment_id: req.experiment_id,
        name: req.name,
        file_path: req.file_path,
        file_size_bytes: req.file_size_bytes,
        created_at: chrono::Utc::now(),
        final_fitness: req.final_fitness,
        final_accuracy: req.final_accuracy,
        iterations_run: req.iterations_run,
        genome_stats: req.genome_stats,
        is_final: req.is_final,
        reference_count: 0,
    };

    // Broadcast checkpoint creation
    let _ = state.ws_tx.send(WsMessage::CheckpointCreated(checkpoint.clone()));

    (StatusCode::CREATED, Json(checkpoint))
}

async fn get_checkpoint(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
) -> impl IntoResponse {
    // TODO: Implement with actual database query
    (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Checkpoint not found"})))
}

async fn delete_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Query(query): Query<DeleteCheckpointQuery>,
) -> impl IntoResponse {
    // TODO: Check reference_count > 0 and reject if force=false
    // TODO: Delete from database and file system

    // Broadcast deletion
    let _ = state.ws_tx.send(WsMessage::CheckpointDeleted { id });

    StatusCode::NO_CONTENT
}

#[derive(Debug, Deserialize)]
pub struct DeleteCheckpointQuery {
    pub force: Option<bool>,
}

async fn download_checkpoint(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<i64>,
) -> impl IntoResponse {
    // TODO: Implement file streaming
    // Get checkpoint from database, read file_path, stream contents
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "Checkpoint not found"})),
    )
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
