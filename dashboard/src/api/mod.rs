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

pub struct AppState {
    pub db: DbPool,
    pub ws_tx: broadcast::Sender<WsMessage>,
    pub current_log_path: RwLock<Option<String>>,
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        // Experiments
        .route("/api/experiments", get(list_experiments).post(create_experiment))
        .route("/api/experiments/current", get(get_current_experiment))
        .route("/api/experiments/:id", get(get_experiment).patch(update_experiment))
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
        .route("/api/flows/:id/stop", post(stop_flow))
        .route("/api/flows/:id/restart", post(restart_flow))
        .route("/api/flows/:id/pid", patch(update_flow_pid))
        .route("/api/flows/:id/heartbeat", post(update_flow_heartbeat))
        .route("/api/flows/:id/validations", get(get_flow_validations))
        .route("/api/flows/:id/run-gating", post(run_flow_gating))
        // Validations
        .route("/api/validations/check", get(check_cached_validation))
        // Checkpoints
        .route("/api/checkpoints", get(list_checkpoints).post(create_checkpoint))
        .route("/api/checkpoints/:id", get(get_checkpoint).delete(delete_checkpoint))
        .route("/api/checkpoints/:id/download", get(download_checkpoint))
        .route("/api/checkpoints/:id/export-hf", post(export_checkpoint_hf))
        // Worker log watching
        .route("/api/watch", post(set_watch_log).get(get_watch_log))
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

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CreateExperimentRequest {
    pub name: String,
    pub flow_id: Option<i64>,
    #[serde(default)]
    pub config: serde_json::Value,
    pub log_path: Option<String>,
}

async fn create_experiment(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateExperimentRequest>,
) -> impl IntoResponse {
    match crate::db::queries::create_experiment(
        &state.db,
        &req.name,
        req.flow_id,
        &req.config,
    ).await {
        Ok(id) => {
            match crate::db::queries::get_experiment(&state.db, id).await {
                Ok(Some(exp)) => (StatusCode::CREATED, Json(exp)).into_response(),
                _ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
            }
        }
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

#[derive(Debug, Deserialize)]
pub struct UpdateExperimentRequest {
    pub name: Option<String>,
    pub status: Option<String>,
    pub best_ce: Option<f64>,
    pub best_accuracy: Option<f64>,
    pub current_iteration: Option<i32>,
    pub max_iterations: Option<i32>,
}

async fn update_experiment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateExperimentRequest>,
) -> impl IntoResponse {
    match crate::db::queries::update_experiment(
        &state.db,
        id,
        req.name.as_deref(),
        req.status.as_deref(),
        req.best_ce,
        req.best_accuracy,
        req.current_iteration,
        req.max_iterations,
    ).await {
        Ok(true) => {
            // Fetch and return updated experiment
            match crate::db::queries::get_experiment(&state.db, id).await {
                Ok(Some(exp)) => {
                    // Broadcast status change
                    let _ = state.ws_tx.send(WsMessage::ExperimentStatusChanged(exp.clone()));
                    (StatusCode::OK, Json(exp)).into_response()
                }
                _ => (StatusCode::OK, Json(serde_json::json!({"updated": true}))).into_response(),
            }
        }
        Ok(false) => (
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

async fn get_validation_summaries(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_validation_summaries(&state.db, experiment_id).await {
        Ok(summaries) => (StatusCode::OK, Json(summaries)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateValidationSummaryRequest {
    pub flow_id: Option<i64>,
    pub validation_point: String,  // 'init' or 'final'
    pub genome_type: String,       // 'best_ce', 'best_acc', 'best_fitness'
    pub genome_hash: String,
    pub ce: f64,
    pub accuracy: f64,
}

async fn create_validation_summary(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
    Json(req): Json<CreateValidationSummaryRequest>,
) -> impl IntoResponse {
    match crate::db::queries::upsert_validation_summary(
        &state.db,
        req.flow_id,
        experiment_id,
        &req.validation_point,
        &req.genome_type,
        &req.genome_hash,
        req.ce,
        req.accuracy,
    ).await {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CheckCachedValidationQuery {
    pub genome_hash: String,
}

async fn check_cached_validation(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CheckCachedValidationQuery>,
) -> impl IntoResponse {
    match crate::db::queries::get_cached_validation(&state.db, &query.genome_hash).await {
        Ok(Some((ce, accuracy))) => (StatusCode::OK, Json(serde_json::json!({
            "found": true,
            "ce": ce,
            "accuracy": accuracy
        }))).into_response(),
        Ok(None) => (StatusCode::OK, Json(serde_json::json!({
            "found": false
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

async fn get_flow_validations(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::get_flow_validation_summaries(&state.db, flow_id).await {
        Ok(summaries) => (StatusCode::OK, Json(summaries)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

// =============================================================================
// Gating Run handlers
// =============================================================================

/// Create a new gating run for an experiment
async fn create_gating_run(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    // Verify experiment exists and is completed
    let experiment = match crate::db::queries::get_experiment(&state.db, experiment_id).await {
        Ok(Some(exp)) => exp,
        Ok(None) => return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Experiment not found"})),
        ).into_response(),
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    // Check experiment is completed
    if experiment.status != ExperimentStatus::Completed {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Experiment must be completed to run gating analysis"})),
        ).into_response();
    }

    // Check if there's already a pending or running gating run
    if let Ok(runs) = crate::db::queries::list_gating_runs(&state.db, experiment_id).await {
        for run in &runs {
            if run.status == GatingStatus::Pending || run.status == GatingStatus::Running {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "Gating analysis already in progress",
                        "gating_run_id": run.id
                    })),
                ).into_response();
            }
        }
    }

    // Create new gating run
    match crate::db::queries::create_gating_run(&state.db, experiment_id, None).await {
        Ok(id) => {
            // Fetch and return the created run
            match crate::db::queries::get_gating_run(&state.db, id).await {
                Ok(Some(run)) => {
                    // Broadcast gating run created
                    let _ = state.ws_tx.send(WsMessage::GatingRunCreated(run.clone()));
                    (StatusCode::CREATED, Json(run)).into_response()
                }
                _ => (StatusCode::CREATED, Json(serde_json::json!({
                    "id": id,
                    "experiment_id": experiment_id,
                    "status": "pending"
                }))).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// List all gating runs for an experiment
async fn list_gating_runs(
    State(state): State<Arc<AppState>>,
    Path(experiment_id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::list_gating_runs(&state.db, experiment_id).await {
        Ok(runs) => (StatusCode::OK, Json(runs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Get a specific gating run
async fn get_gating_run(
    State(state): State<Arc<AppState>>,
    Path((experiment_id, gating_id)): Path<(i64, i64)>,
) -> impl IntoResponse {
    match crate::db::queries::get_gating_run(&state.db, gating_id).await {
        Ok(Some(run)) => {
            // Verify it belongs to the experiment
            if run.experiment_id != experiment_id {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": "Gating run not found for this experiment"})),
                ).into_response();
            }
            (StatusCode::OK, Json(run)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Gating run not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Request to update a gating run
#[derive(Debug, Deserialize)]
pub struct UpdateGatingRunRequest {
    pub status: Option<GatingStatus>,
    pub genomes_tested: Option<i32>,
    pub results: Option<Vec<GatingResult>>,
    pub error: Option<String>,
}

/// Update a gating run (status or results)
async fn update_gating_run(
    State(state): State<Arc<AppState>>,
    Path((experiment_id, gating_id)): Path<(i64, i64)>,
    Json(req): Json<UpdateGatingRunRequest>,
) -> impl IntoResponse {
    // First verify the gating run exists and belongs to this experiment
    let run = match crate::db::queries::get_gating_run(&state.db, gating_id).await {
        Ok(Some(r)) => r,
        Ok(None) => return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Gating run not found"})),
        ).into_response(),
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    if run.experiment_id != experiment_id {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Gating run not found for this experiment"})),
        ).into_response();
    }

    // If results are provided, update with results (completes the run)
    if let Some(ref results) = req.results {
        let genomes_tested = req.genomes_tested.unwrap_or(results.len() as i32);
        match crate::db::queries::update_gating_run_results(
            &state.db,
            gating_id,
            genomes_tested,
            results,
            req.error.as_deref(),
        ).await {
            Ok(Some(updated_run)) => {
                let _ = state.ws_tx.send(WsMessage::GatingRunUpdated(updated_run.clone()));
                return (StatusCode::OK, Json(updated_run)).into_response();
            }
            Ok(None) => return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Gating run not found"})),
            ).into_response(),
            Err(e) => return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response(),
        }
    }

    // If only status is provided, update status
    if let Some(ref status) = req.status {
        match crate::db::queries::update_gating_run_status(&state.db, gating_id, status).await {
            Ok(Some(updated_run)) => {
                let _ = state.ws_tx.send(WsMessage::GatingRunUpdated(updated_run.clone()));
                return (StatusCode::OK, Json(updated_run)).into_response();
            }
            Ok(None) => return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Gating run not found"})),
            ).into_response(),
            Err(e) => return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response(),
        }
    }

    // Nothing to update
    (StatusCode::OK, Json(run)).into_response()
}

/// List all pending gating runs (for worker polling)
async fn list_pending_gating_runs(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match crate::db::queries::get_pending_gating_runs(&state.db).await {
        Ok(runs) => (StatusCode::OK, Json(runs)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Trigger gating analysis for all completed experiments in a flow
async fn run_flow_gating(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
) -> impl IntoResponse {
    // Get all experiments for this flow
    let experiments = match crate::db::queries::list_flow_experiments(&state.db, flow_id).await {
        Ok(exps) => exps,
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    if experiments.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "No experiments found for this flow"})),
        ).into_response();
    }

    // Queue gating for each completed experiment
    let mut queued = Vec::new();
    let mut skipped = Vec::new();
    let mut errors = Vec::new();

    for exp in experiments {
        // Skip non-completed experiments
        if exp.status != ExperimentStatus::Completed {
            skipped.push(serde_json::json!({
                "experiment_id": exp.id,
                "reason": "not_completed"
            }));
            continue;
        }

        // Check if there's already a pending or running gating run
        if let Ok(runs) = crate::db::queries::list_gating_runs(&state.db, exp.id).await {
            let has_active = runs.iter().any(|r| r.status == GatingStatus::Pending || r.status == GatingStatus::Running);
            if has_active {
                skipped.push(serde_json::json!({
                    "experiment_id": exp.id,
                    "reason": "gating_in_progress"
                }));
                continue;
            }
        }

        // Create new gating run
        match crate::db::queries::create_gating_run(&state.db, exp.id, None).await {
            Ok(id) => {
                // Broadcast
                if let Ok(Some(run)) = crate::db::queries::get_gating_run(&state.db, id).await {
                    let _ = state.ws_tx.send(WsMessage::GatingRunCreated(run));
                }
                queued.push(serde_json::json!({
                    "experiment_id": exp.id,
                    "gating_run_id": id
                }));
            }
            Err(e) => errors.push(serde_json::json!({
                "experiment_id": exp.id,
                "error": e.to_string()
            })),
        }
    }

    (StatusCode::OK, Json(serde_json::json!({
        "flow_id": flow_id,
        "queued": queued,
        "skipped": skipped,
        "errors": errors,
        "message": format!("{} experiments queued for gating analysis", queued.len())
    }))).into_response()
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

    // Get recent iterations directly from experiment
    let iterations = match crate::db::queries::get_recent_iterations(&state.db, exp_id, 500).await {
        Ok(i) => i,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Use experiment-level metrics directly
    let best_ce = exp.best_ce.unwrap_or(0.0);
    let best_accuracy = exp.best_accuracy.unwrap_or(0.0);

    let snapshot = DashboardSnapshot {
        current_experiment: Some(exp),
        iterations,
        best_ce,
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

/// Request to create a new flow
/// Experiments are passed separately (normalized design: Flow 1:N Experiments via FK)
#[derive(Debug, Deserialize)]
pub struct CreateFlowRequest {
    pub name: String,
    pub description: Option<String>,
    /// Flow-level configuration (template name, shared params)
    #[serde(default)]
    pub config: FlowConfig,
    /// Experiments to create with the flow (stored in experiments table, not config)
    #[serde(default)]
    pub experiments: Vec<ExperimentSpec>,
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
        &req.experiments,
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

#[derive(Debug, Deserialize)]
pub struct LinkExperimentRequest {
    pub experiment_id: i64,
    #[serde(default)]
    pub sequence_order: i32,
}

async fn link_experiment_to_flow(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
    Json(req): Json<LinkExperimentRequest>,
) -> impl IntoResponse {
    match crate::db::queries::link_experiment_to_flow(
        &state.db,
        flow_id,
        req.experiment_id,
        req.sequence_order,
    ).await {
        Ok(true) => (StatusCode::OK, Json(serde_json::json!({"linked": true}))).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "Experiment not found"}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}

/// Add a new experiment to a flow
/// This creates the experiment in the experiments table with pending status
#[derive(Debug, Deserialize)]
pub struct AddExperimentRequest {
    pub experiment: ExperimentSpec,
    /// If not specified, will be appended as the last experiment
    pub sequence_order: Option<i32>,
}

async fn add_experiment_to_flow(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
    Json(req): Json<AddExperimentRequest>,
) -> impl IntoResponse {
    // Verify flow exists
    let flow = match crate::db::queries::get_flow(&state.db, flow_id).await {
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

    // Check flow is not already running
    if flow.status == FlowStatus::Running {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Cannot add experiments to a running flow"})),
        ).into_response();
    }

    // Get current experiments to determine sequence_order if not specified
    let existing = match crate::db::queries::list_flow_experiments(&state.db, flow_id).await {
        Ok(e) => e,
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    };

    let sequence_order = req.sequence_order.unwrap_or(existing.len() as i32);

    // Derive phase_type from experiment spec
    let exp_spec = &req.experiment;
    let opt_target = if exp_spec.optimize_bits {
        "bits"
    } else if exp_spec.optimize_neurons {
        "neurons"
    } else {
        "connections"
    };
    let phase_type = match exp_spec.experiment_type {
        ExperimentType::GridSearch => "grid_search".to_string(),
        _ => {
            let exp_type = match exp_spec.experiment_type {
                ExperimentType::Ga => "ga",
                ExperimentType::Ts => "ts",
                ExperimentType::GridSearch => unreachable!(),
            };
            format!("{}_{}", exp_type, opt_target)
        }
    };

    // Get max_iterations: first from experiment params, then from flow config
    let max_iterations = exp_spec.params.get("generations")
        .or_else(|| exp_spec.params.get("iterations"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .or_else(|| {
            match exp_spec.experiment_type {
                ExperimentType::GridSearch => {
                    Some(1) // Grid search is a single step
                }
                ExperimentType::Ga => {
                    flow.config.params.get("ga_generations")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                }
                ExperimentType::Ts => {
                    flow.config.params.get("ts_iterations")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                }
            }
        });

    // Create the experiment (use flow's config for tier_config etc.)
    match crate::db::queries::create_pending_experiment(
        &state.db,
        &exp_spec.name,
        flow_id,
        sequence_order,
        Some(&phase_type),
        max_iterations,
        &flow.config,
    ).await {
        Ok(id) => {
            // Fetch the created experiment
            match crate::db::queries::get_experiment(&state.db, id).await {
                Ok(Some(exp)) => (StatusCode::CREATED, Json(exp)).into_response(),
                _ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
            }
        }
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
    #[serde(default)]
    pub start_from_experiment: Option<usize>,  // If set, skip experiments before this index (0-based)
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

    // If flow is running or queued, stop it first
    if flow.status == FlowStatus::Running || flow.status == FlowStatus::Queued {
        if let Err(e) = crate::db::queries::stop_flow_process(&state.db, id).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to stop flow: {}", e)})),
            ).into_response();
        }
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

    // If start_from_experiment is set, update the flow config
    if let Some(start_idx) = req.start_from_experiment {
        // Get current config and add start_from_experiment to params
        if let Ok(Some(mut current_flow)) = crate::db::queries::get_flow(&state.db, id).await {
            // FlowConfig has params: HashMap<String, Value>
            current_flow.config.params.insert(
                "start_from_experiment".to_string(),
                serde_json::json!(start_idx)
            );
            // Serialize FlowConfig to serde_json::Value for update
            if let Ok(config_json) = serde_json::to_value(&current_flow.config) {
                let _ = crate::db::queries::update_flow(&state.db, id, None, None, None, Some(&config_json), None).await;
            }
        }
    }

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

/// Update the heartbeat of a flow (called periodically by worker)
async fn update_flow_heartbeat(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match crate::db::queries::update_flow_heartbeat(&state.db, id).await {
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
    pub iteration_id: Option<i64>,
    pub genome_stats: Option<serde_json::Value>,
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
        req.iteration_id,
        req.best_ce,
        req.best_accuracy,
        req.genome_stats.as_ref(),
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
// HuggingFace export handler
// =============================================================================

#[derive(Debug, Deserialize)]
struct ExportHfRequest {
    #[serde(default = "default_export_dir")]
    output_dir: String,
}

fn default_export_dir() -> String {
    "exports".to_string()
}

/// Export a checkpoint's genome data for HuggingFace model creation.
///
/// Returns the checkpoint metadata + experiment info needed for HF export.
/// The actual export (training memory + serializing safetensors) happens in Python.
async fn export_checkpoint_hf(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(req): Json<ExportHfRequest>,
) -> impl IntoResponse {
    // Get checkpoint
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

    // Get the experiment for architecture context
    let experiment = match crate::db::queries::get_experiment(&state.db, checkpoint.experiment_id).await {
        Ok(Some(e)) => e,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Experiment not found"})),
            ).into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            ).into_response();
        }
    };

    // Return all the data needed for Python-side HF export
    (StatusCode::OK, Json(serde_json::json!({
        "checkpoint_id": checkpoint.id,
        "checkpoint_path": checkpoint.file_path,
        "checkpoint_name": checkpoint.name,
        "best_ce": checkpoint.best_ce,
        "best_accuracy": checkpoint.best_accuracy,
        "genome_stats": checkpoint.genome_stats,
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "architecture_type": experiment.architecture_type,
        "tier_config": experiment.tier_config,
        "context_size": experiment.context_size,
        "output_dir": req.output_dir,
    }))).into_response()
}

// =============================================================================
// Worker log watch
// =============================================================================

#[derive(Deserialize)]
struct WatchLogRequest {
    log_path: String,
}

async fn set_watch_log(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WatchLogRequest>,
) -> impl IntoResponse {
    *state.current_log_path.write().await = Some(req.log_path.clone());
    (StatusCode::OK, Json(serde_json::json!({"log_path": req.log_path}))).into_response()
}

async fn get_watch_log(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let path = state.current_log_path.read().await.clone();
    (StatusCode::OK, Json(serde_json::json!({"log_path": path}))).into_response()
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
    let mut last_experiment_status = snapshot.current_experiment.as_ref().map(|e| e.status.clone());

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

                        // Forward flow-related and gating messages to client
                        let json_result = match &msg {
                            WsMessage::FlowStarted(_) |
                            WsMessage::FlowQueued(_) |
                            WsMessage::FlowCompleted(_) |
                            WsMessage::FlowFailed { .. } |
                            WsMessage::FlowCancelled(_) |
                            WsMessage::GatingRunCreated(_) |
                            WsMessage::GatingRunUpdated(_) => {
                                serde_json::to_string(&msg)
                            }
                            _ => continue, // Skip other messages
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
                            last_experiment_status = snapshot.current_experiment.as_ref().map(|e| e.status.clone());
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
                // Check for experiment changes
                let snapshot = build_snapshot(&state.db).await;
                let current_exp_id = snapshot.current_experiment.as_ref().map(|e| e.id);
                let current_exp_status = snapshot.current_experiment.as_ref().map(|e| e.status.clone());

                // Send fresh snapshot if experiment changed
                if current_exp_id != last_experiment_id ||
                   current_exp_status != last_experiment_status {
                    last_experiment_id = current_exp_id;
                    last_experiment_status = current_exp_status;
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
    let iterations = crate::db::queries::get_recent_iterations(db, exp_id, 500).await.unwrap_or_default();

    // Use experiment-level metrics directly
    let best_ce = exp.best_ce.unwrap_or(0.0);
    let best_accuracy = exp.best_accuracy.unwrap_or(0.0);

    DashboardSnapshot {
        current_experiment: Some(exp),
        iterations,
        best_ce,
        best_accuracy,
    }
}
