//! Flow CRUD and experiment-linking handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Flow handlers
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ListFlowsQuery {
    pub status: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

pub(crate) async fn list_flows(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListFlowsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(50).clamp(1, 100_000);
    let offset = query.offset.unwrap_or(0).max(0);

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
    /// Escape hatch for deliberately creating an empty flow (experiments
    /// added later via /experiments). Without it, an empty experiments list
    /// is rejected — the worker marks 0-experiment flows completed instantly
    /// with zero work (CLAUDE.md Rule 2), so it is almost always a client bug
    /// (e.g. the pre-12/06 dashboard_client nested `experiments` inside
    /// `config` where serde silently dropped it).
    #[serde(default)]
    pub allow_empty_experiments: bool,
}

pub(crate) async fn create_flow(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateFlowRequest>,
) -> impl IntoResponse {
    if req.name.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Flow name must not be empty"})),
        ).into_response();
    }
    if req.experiments.is_empty() && !req.allow_empty_experiments {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Flow has no experiments — the worker would mark it completed instantly with zero work (Rule 2). \
                          Pass experiments at top level (NOT nested inside config), or set allow_empty_experiments=true."
            })),
        ).into_response();
    }
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

pub(crate) async fn get_flow(
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
    pub status_message: Option<String>,
}

pub(crate) async fn update_flow(
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
        req.status_message.as_deref(),
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
            Json(serde_json::json!({"error": "Flow not found (or a concurrent status change won the race)"})),
        ).into_response(),
        Err(e) => {
            let msg = e.to_string();
            // State-machine rejections are client errors, not server faults
            let code = if msg.contains("invalid status transition") {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (code, Json(serde_json::json!({"error": msg}))).into_response()
        }
    }
}

pub(crate) async fn delete_flow(
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

pub(crate) async fn list_flow_experiments(
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

pub(crate) async fn link_experiment_to_flow(
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

pub(crate) async fn add_experiment_to_flow(
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

    // Use explicit phase_type if provided, otherwise derive from experiment spec
    let exp_spec = &req.experiment;
    let phase_type = if let Some(ref pt) = exp_spec.phase_type {
        pt.clone()
    } else {
        let opt_target = if exp_spec.optimize_bits {
            "bits"
        } else if exp_spec.optimize_neurons {
            "neurons"
        } else {
            "connections"
        };
        match exp_spec.experiment_type {
            ExperimentType::GridSearch => "grid_search".to_string(),
            ExperimentType::LambdaSweep => "lambda_sweep".to_string(),
            _ => {
                let exp_type = match exp_spec.experiment_type {
                    ExperimentType::Ga => "ga",
                    ExperimentType::Ts => "ts",
                    ExperimentType::Neurogenesis => "neurogenesis",
                    ExperimentType::Synaptogenesis => "synaptogenesis",
                    ExperimentType::Axonogenesis => "axonogenesis",
                    ExperimentType::GridSearch | ExperimentType::LambdaSweep => unreachable!(),
                };
                format!("{}_{}", exp_type, opt_target)
            }
        }
    };

    // Get max_iterations: first from experiment params, then from flow config
    let max_iterations = exp_spec.params.get("generations")
        .or_else(|| exp_spec.params.get("iterations"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .or_else(|| {
            match exp_spec.experiment_type {
                ExperimentType::GridSearch | ExperimentType::LambdaSweep => {
                    Some(1) // Grid search / lambda sweep is a single step
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
                ExperimentType::Neurogenesis | ExperimentType::Synaptogenesis | ExperimentType::Axonogenesis => {
                    exp_spec.params.get("iterations")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                }
            }
        });

    // Create the experiment (use flow's config for tier_config etc.)
    let exp_params = if exp_spec.params.is_empty() { None } else { Some(&exp_spec.params) };
    match crate::db::queries::create_pending_experiment(
        &state.db,
        &exp_spec.name,
        flow_id,
        sequence_order,
        Some(&phase_type),
        max_iterations,
        &flow.config,
        exp_params,
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

/// Reorder experiments within a flow
#[derive(Debug, Deserialize)]
pub struct ReorderExperimentsRequest {
    pub experiment_ids: Vec<i64>,
}

pub(crate) async fn reorder_flow_experiments(
    State(state): State<Arc<AppState>>,
    Path(flow_id): Path<i64>,
    Json(req): Json<ReorderExperimentsRequest>,
) -> impl IntoResponse {
    // Verify flow exists and is not running
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

    if flow.status == FlowStatus::Running {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Cannot reorder experiments while flow is running"})),
        ).into_response();
    }

    match crate::db::queries::reorder_experiments(&state.db, flow_id, &req.experiment_ids).await {
        Ok(true) => {
            // Return updated experiment list
            match crate::db::queries::list_flow_experiments(&state.db, flow_id).await {
                Ok(experiments) => (StatusCode::OK, Json(experiments)).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ).into_response(),
            }
        }
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Flow not found"})),
        ).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        ).into_response(),
    }
}
