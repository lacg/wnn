//! Experiment, validation-summary, combined-validation, and iteration handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Experiment handlers
// =============================================================================

pub(crate) async fn list_experiments(State(state): State<Arc<AppState>>) -> impl IntoResponse
{
	match crate::db::queries::list_experiments(&state.db, 100, 0).await
	{
		Ok(experiments) => (StatusCode::OK, Json(experiments)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CreateExperimentRequest
{
	pub name: String,
	pub flow_id: Option<i64>,
	#[serde(default)]
	pub config: serde_json::Value,
	pub log_path: Option<String>,
}

pub(crate) async fn create_experiment(
	State(state): State<Arc<AppState>>,
	Json(req): Json<CreateExperimentRequest>,
) -> impl IntoResponse
{
	match crate::db::queries::create_experiment(&state.db, &req.name, req.flow_id, &req.config).await
	{
		Ok(id) => match crate::db::queries::get_experiment(&state.db, id).await
		{
			Ok(Some(exp)) => (StatusCode::CREATED, Json(exp)).into_response(),
			_ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
		},
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_current_experiment(State(state): State<Arc<AppState>>)
-> impl IntoResponse
{
	match crate::db::queries::get_running_experiment(&state.db).await
	{
		Ok(Some(exp)) => (StatusCode::OK, Json(exp)).into_response(),
		Ok(None) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "No running experiment"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_experiment(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_experiment(&state.db, id).await
	{
		Ok(Some(exp)) => (StatusCode::OK, Json(exp)).into_response(),
		Ok(None) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Experiment not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
pub struct UpdateExperimentRequest
{
	pub name: Option<String>,
	/// Typed (snake_case) — a typo'd status used to be stored verbatim and
	/// silently parsed back as Pending; now it's a 422 at the boundary.
	pub status: Option<crate::models::ExperimentStatus>,
	pub best_ce: Option<f64>,
	pub best_accuracy: Option<f64>,
	pub current_iteration: Option<i32>,
	pub max_iterations: Option<i32>,
	/// 'tiered' | 'bitwise' | 'multi_stage' | 'ids' | 'controller' — drives
	/// the dashboard's per-architecture column selection. Was silently
	/// dropped pre-P5 (the client sent the legacy name cluster_type).
	pub architecture_type: Option<String>,
}

pub(crate) async fn update_experiment(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
	Json(req): Json<UpdateExperimentRequest>,
) -> impl IntoResponse
{
	// Serialize the typed status back to its canonical snake_case string
	let status_str = req.status.as_ref().map(|s| {
		serde_json::to_value(s)
			.ok()
			.and_then(|v| v.as_str().map(String::from))
			.unwrap_or_default()
	});
	match crate::db::queries::update_experiment(
		&state.db,
		id,
		req.name.as_deref(),
		status_str.as_deref(),
		req.best_ce,
		req.best_accuracy,
		req.current_iteration,
		req.max_iterations,
		req.architecture_type.as_deref(),
	)
	.await
	{
		Ok(true) =>
		{
			// Fetch and return updated experiment
			match crate::db::queries::get_experiment(&state.db, id).await
			{
				Ok(Some(exp)) =>
				{
					// Broadcast status change
					let _ = state
						.ws_tx
						.send(WsMessage::ExperimentStatusChanged(exp.clone()));
					(StatusCode::OK, Json(exp)).into_response()
				}
				_ => (StatusCode::OK, Json(serde_json::json!({"updated": true}))).into_response(),
			}
		}
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Experiment not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn delete_experiment(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::delete_experiment(&state.db, id).await
	{
		Ok(true) => StatusCode::NO_CONTENT.into_response(),
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Experiment not found"})),
		)
			.into_response(),
		Err(e) =>
		{
			let msg = e.to_string();
			let status = if msg.contains("Can only delete pending")
			{
				StatusCode::BAD_REQUEST
			}
			else
			{
				StatusCode::INTERNAL_SERVER_ERROR
			};
			(status, Json(serde_json::json!({"error": msg}))).into_response()
		}
	}
}

#[derive(Debug, Deserialize)]
pub struct RecentIterationsQuery
{
	pub limit: Option<i32>,
}

pub(crate) async fn get_experiment_iterations(
	State(state): State<Arc<AppState>>,
	Path(experiment_id): Path<i64>,
	Query(query): Query<RecentIterationsQuery>,
) -> impl IntoResponse
{
	let limit = query.limit.unwrap_or(100).clamp(1, 100_000);
	match crate::db::queries::get_recent_iterations(&state.db, experiment_id, limit).await
	{
		Ok(iterations) => (StatusCode::OK, Json(iterations)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_validation_summaries(
	State(state): State<Arc<AppState>>,
	Path(experiment_id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_validation_summaries(&state.db, experiment_id).await
	{
		Ok(summaries) => (StatusCode::OK, Json(summaries)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
pub struct CreateValidationSummaryRequest
{
	pub flow_id: Option<i64>,
	pub validation_point: String, // 'init' or 'final'
	pub genome_type: String,      // 'best_ce', 'best_acc', 'best_fitness'
	pub genome_hash: String,
	pub ce: f64,
	pub accuracy: f64,
	pub f1_macro: Option<f64>,
	pub fpr: Option<f64>,
	pub threshold_metadata: Option<String>,
}

pub(crate) async fn create_validation_summary(
	State(state): State<Arc<AppState>>,
	Path(experiment_id): Path<i64>,
	Json(req): Json<CreateValidationSummaryRequest>,
) -> impl IntoResponse
{
	match crate::db::queries::upsert_validation_summary(
		&state.db,
		req.flow_id,
		experiment_id,
		&req.validation_point,
		&req.genome_type,
		&req.genome_hash,
		req.ce,
		req.accuracy,
		req.f1_macro,
		req.fpr,
		req.threshold_metadata.as_deref(),
	)
	.await
	{
		Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
pub struct CheckCachedValidationQuery
{
	pub genome_hash: String,
	pub dataset_key: Option<String>,
}

pub(crate) async fn check_cached_validation(
	State(state): State<Arc<AppState>>,
	Query(query): Query<CheckCachedValidationQuery>,
) -> impl IntoResponse
{
	match crate::db::queries::get_cached_validation(
		&state.db,
		&query.genome_hash,
		query.dataset_key.as_deref(),
	)
	.await
	{
		Ok(Some((ce, accuracy, f1_macro, fpr, threshold_metadata))) => (
			StatusCode::OK,
			Json(serde_json::json!({
					"found": true,
					"ce": ce,
					"accuracy": accuracy,
					"f1_macro": f1_macro,
					"fpr": fpr,
					"threshold_metadata": threshold_metadata
			})),
		)
			.into_response(),
		Ok(None) => (
			StatusCode::OK,
			Json(serde_json::json!({
					"found": false
			})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_flow_validations(
	State(state): State<Arc<AppState>>,
	Path(flow_id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_flow_validation_summaries(&state.db, flow_id).await
	{
		Ok(summaries) => (StatusCode::OK, Json(summaries)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

// =============================================================================
// Combined Validation handlers (multi-stage end-to-end metrics)
// =============================================================================

pub(crate) async fn get_combined_validations(
	State(state): State<Arc<AppState>>,
	Path(flow_id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_combined_validations(&state.db, flow_id).await
	{
		Ok(validations) => (StatusCode::OK, Json(validations)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
pub struct CreateCombinedValidationRequest
{
	pub genome_type: String, // 'best_ce', 'best_acc', 'best_fitness'
	pub combined_ce: f64,
	pub combined_accuracy: f64,
	pub per_stage_ce: Option<Vec<f64>>,
	pub per_stage_acc: Option<Vec<f64>>,
	pub unigram_lambda: Option<f64>,
}

pub(crate) async fn create_combined_validation(
	State(state): State<Arc<AppState>>,
	Path(flow_id): Path<i64>,
	Json(req): Json<CreateCombinedValidationRequest>,
) -> impl IntoResponse
{
	let per_stage_ce_slice = req.per_stage_ce.as_deref();
	let per_stage_acc_slice = req.per_stage_acc.as_deref();
	match crate::db::queries::upsert_combined_validation(
		&state.db,
		flow_id,
		&req.genome_type,
		req.combined_ce,
		req.combined_accuracy,
		per_stage_ce_slice,
		per_stage_acc_slice,
		req.unigram_lambda,
	)
	.await
	{
		Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

// =============================================================================
// Iteration handlers
// =============================================================================

pub(crate) async fn get_iteration_genomes(
	State(state): State<Arc<AppState>>,
	Path(iteration_id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_genome_evaluations(&state.db, iteration_id).await
	{
		Ok(evaluations) => (StatusCode::OK, Json(evaluations)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}
