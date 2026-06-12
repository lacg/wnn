//! Gating run handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Gating Run handlers
// =============================================================================

/// Create a new gating run for an experiment
pub(crate) async fn create_gating_run(
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
pub(crate) async fn list_gating_runs(
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
pub(crate) async fn get_gating_run(
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
pub(crate) async fn update_gating_run(
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
pub(crate) async fn list_pending_gating_runs(
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
pub(crate) async fn run_flow_gating(
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
