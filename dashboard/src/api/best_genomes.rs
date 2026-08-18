//! Best-genome (leaderboard) handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Best Genomes (Leaderboard) handlers
// =============================================================================

#[derive(Debug, Deserialize)]
pub(crate) struct ListBestGenomesQuery
{
	task_type: Option<String>,
	stage: Option<String>,
	metric: Option<String>,
	limit: Option<i32>,
	offset: Option<i32>,
	feature_selection: Option<String>,
	n_bits: Option<i32>,
	ids_dataset: Option<String>,
	ids_split: Option<String>,
}

pub(crate) async fn list_best_genomes(
	State(state): State<Arc<AppState>>,
	Query(query): Query<ListBestGenomesQuery>,
) -> impl IntoResponse
{
	let limit = query.limit.unwrap_or(50).clamp(1, 100_000);
	let offset = query.offset.unwrap_or(0).max(0);

	match crate::db::queries::list_best_genomes(
		&state.db,
		query.task_type.as_deref(),
		query.stage.as_deref(),
		query.metric.as_deref(),
		limit,
		offset,
		query.feature_selection.as_deref(),
		query.n_bits,
		query.ids_dataset.as_deref(),
		query.ids_split.as_deref(),
	)
	.await
	{
		Ok(genomes) => (StatusCode::OK, Json(genomes)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_best_genome(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_best_genome(&state.db, id).await
	{
		Ok(Some(genome)) => (StatusCode::OK, Json(genome)).into_response(),
		Ok(None) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Best genome not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn download_best_genome(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_best_genome_data(&state.db, id).await
	{
		Ok(Some(data)) => (StatusCode::OK, Json(data)).into_response(),
		Ok(None) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Best genome data not found"})),
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
pub(crate) struct SubmitBestGenomeEntry
{
	task_type: String,
	stage: String,
	metric: String,
	genome_hash: String,
	ce: f64,
	accuracy: f64,
	f1_macro: Option<f64>,
	fpr: Option<f64>,
	flow_id: Option<i64>,
	experiment_id: Option<i64>,
	genome_data: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SubmitBestGenomesRequest
{
	genomes: Vec<SubmitBestGenomeEntry>,
}

pub(crate) async fn submit_best_genomes(
	State(state): State<Arc<AppState>>,
	Json(req): Json<SubmitBestGenomesRequest>,
) -> impl IntoResponse
{
	let mut accepted = Vec::new();
	let mut rejected = Vec::new();

	for entry in &req.genomes
	{
		match crate::db::queries::submit_best_genome(
			&state.db,
			&entry.task_type,
			&entry.stage,
			&entry.metric,
			&entry.genome_hash,
			entry.ce,
			entry.accuracy,
			entry.f1_macro,
			entry.fpr,
			entry.flow_id,
			entry.experiment_id,
			entry.genome_data.as_ref(),
		)
		.await
		{
			Ok(Some(id)) =>
			{
				accepted.push(serde_json::json!({"genome_hash": entry.genome_hash, "id": id}))
			}
			Ok(None) => rejected
				.push(serde_json::json!({"genome_hash": entry.genome_hash, "reason": "genome not found"})),
			Err(e) => rejected
				.push(serde_json::json!({"genome_hash": entry.genome_hash, "reason": e.to_string()})),
		}
	}

	(
		StatusCode::OK,
		Json(serde_json::json!({
				"accepted": accepted,
				"rejected": rejected,
		})),
	)
		.into_response()
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecalculateRequest
{
	task_type: String,
	stage: String,
	metric: String,
	#[serde(default = "default_max_entries")]
	max_entries: i32,
}

fn default_max_entries() -> i32
{
	150
}

pub(crate) async fn recalculate_best_genomes(
	State(state): State<Arc<AppState>>,
	Json(req): Json<RecalculateRequest>,
) -> impl IntoResponse
{
	match crate::db::queries::recalculate_rankings(
		&state.db,
		&req.task_type,
		&req.stage,
		&req.metric,
		req.max_entries,
	)
	.await
	{
		Ok(count) => (StatusCode::OK, Json(serde_json::json!({"ranked": count}))).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn delete_best_genome(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::delete_best_genome(&state.db, id).await
	{
		Ok(true) => StatusCode::NO_CONTENT.into_response(),
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Best genome not found"})),
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
pub(crate) struct BestGenomeExportHfRequest
{
	repo_id: String,
}

pub(crate) async fn export_best_genome_hf(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
	Json(req): Json<BestGenomeExportHfRequest>,
) -> impl IntoResponse
{
	match crate::db::queries::update_best_genome_hf(&state.db, id, &req.repo_id).await
	{
		Ok(true) => match crate::db::queries::get_best_genome(&state.db, id).await
		{
			Ok(Some(genome)) => (StatusCode::OK, Json(genome)).into_response(),
			_ => (
				StatusCode::OK,
				Json(serde_json::json!({"id": id, "hf_repo_id": req.repo_id})),
			)
				.into_response(),
		},
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Best genome not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}
