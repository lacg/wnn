//! Checkpoint handlers incl. download and HF export (split from api/mod.rs).

use super::*;

// =============================================================================
// Checkpoint handlers
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct ListCheckpointsQuery
{
	pub experiment_id: Option<i64>,
	pub checkpoint_type: Option<String>,
	pub limit: Option<i32>,
	pub offset: Option<i32>,
}

pub(crate) async fn list_checkpoints(
	State(state): State<Arc<AppState>>,
	Query(query): Query<ListCheckpointsQuery>,
) -> impl IntoResponse
{
	let limit = query.limit.unwrap_or(50).clamp(1, 100_000);
	let offset = query.offset.unwrap_or(0).max(0);

	match crate::db::queries::list_checkpoints(
		&state.db,
		query.experiment_id,
		query.checkpoint_type.as_deref(),
		limit,
		offset,
	)
	.await
	{
		Ok(checkpoints) => (StatusCode::OK, Json(checkpoints)).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

/// Validate a checkpoint file path: must be relative, live under the
/// `checkpoints/` directory, and contain no parent-dir (`..`) components.
/// Live-data audit (12/06/2026): every existing row is `checkpoints/...`-
/// relative, so this rejects nothing legitimate while closing the
/// arbitrary-file read/delete surface via POSTed paths.
fn checkpoint_path_is_safe(file_path: &str) -> bool
{
	let p = std::path::Path::new(file_path);
	// No parent-dir components anywhere, relative or absolute.
	if p
		.components()
		.any(|c| matches!(c, std::path::Component::ParentDir))
	{
		return false;
	}
	if p.is_relative()
	{
		return p.starts_with("checkpoints");
	}
	// Absolute paths (phased_search registers these) must live under one of
	// the known checkpoint roots: project-root checkpoints/ (the dashboard
	// runs from dashboard/, so that's ../checkpoints) or a local one.
	for root in ["../checkpoints", "checkpoints"]
	{
		if let Ok(canon_root) = std::fs::canonicalize(root)
		{
			if p.starts_with(&canon_root)
			{
				return true;
			}
		}
	}
	false
}

#[derive(Debug, Deserialize)]
pub struct CreateCheckpointRequest
{
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

pub(crate) async fn create_checkpoint(
	State(state): State<Arc<AppState>>,
	Json(req): Json<CreateCheckpointRequest>,
) -> impl IntoResponse
{
	if !checkpoint_path_is_safe(&req.file_path)
	{
		return (
			StatusCode::BAD_REQUEST,
			Json(serde_json::json!({
					"error": format!(
							"file_path must be a relative path under checkpoints/ with no '..' components, got {:?}",
							req.file_path
					)
			})),
		)
			.into_response();
	}
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
	)
	.await
	{
		Ok(id) =>
		{
			// Fetch the created checkpoint to return and broadcast
			match crate::db::queries::get_checkpoint(&state.db, id).await
			{
				Ok(Some(checkpoint)) =>
				{
					let _ = state
						.ws_tx
						.send(WsMessage::CheckpointCreated(checkpoint.clone()));
					(StatusCode::CREATED, Json(checkpoint)).into_response()
				}
				_ => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
			}
		}
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn get_checkpoint(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::get_checkpoint(&state.db, id).await
	{
		Ok(Some(checkpoint)) => (StatusCode::OK, Json(checkpoint)).into_response(),
		Ok(None) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Checkpoint not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn delete_checkpoint(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::delete_checkpoint(&state.db, id).await
	{
		Ok((true, Some(file_path))) =>
		{
			// Try to delete the file (best-effort). Re-validate the stored
			// path: rows created before validation existed could point anywhere.
			if !checkpoint_path_is_safe(&file_path)
			{
				tracing::warn!(
					"Checkpoint {} file_path {:?} is outside the checkpoints root — DB row deleted, file left untouched",
					id,
					file_path
				);
			}
			else if let Err(e) = std::fs::remove_file(&file_path)
			{
				tracing::warn!("Failed to delete checkpoint file {}: {}", file_path, e);
			}

			// Broadcast deletion
			let _ = state.ws_tx.send(WsMessage::CheckpointDeleted { id });

			StatusCode::NO_CONTENT.into_response()
		}
		Ok((true, None)) =>
		{
			let _ = state.ws_tx.send(WsMessage::CheckpointDeleted { id });
			StatusCode::NO_CONTENT.into_response()
		}
		Ok((false, _)) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Checkpoint not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::CONFLICT,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

pub(crate) async fn download_checkpoint(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	use axum::body::Body;
	use axum::http::header;
	use tokio_util::io::ReaderStream;

	// Get checkpoint from database
	let checkpoint = match crate::db::queries::get_checkpoint(&state.db, id).await
	{
		Ok(Some(c)) => c,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Checkpoint not found"})),
			)
				.into_response();
		}
		Err(e) =>
		{
			return (
				StatusCode::INTERNAL_SERVER_ERROR,
				Json(serde_json::json!({"error": e.to_string()})),
			)
				.into_response();
		}
	};

	// Re-validate the stored path before serving it: rows created before
	// validation existed could point anywhere on disk.
	if !checkpoint_path_is_safe(&checkpoint.file_path)
	{
		return (
			StatusCode::FORBIDDEN,
			Json(serde_json::json!({
					"error": "Checkpoint file_path is outside the checkpoints root"
			})),
		)
			.into_response();
	}

	// Open the file
	let file = match tokio::fs::File::open(&checkpoint.file_path).await
	{
		Ok(f) => f,
		Err(e) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({
						"error": format!("Checkpoint file not found: {}", e)
				})),
			)
				.into_response();
		}
	};

	// Get filename from path
	let filename = std::path::Path::new(&checkpoint.file_path)
		.file_name()
		.and_then(|n| n.to_str())
		.unwrap_or("checkpoint.json.gz");

	// Determine content type
	let content_type = if filename.ends_with(".gz")
	{
		"application/gzip"
	}
	else if filename.ends_with(".json")
	{
		"application/json"
	}
	else
	{
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
	)
		.into_response()
}

// =============================================================================
// HuggingFace export handler
// =============================================================================

#[derive(Debug, Deserialize)]
pub(crate) struct ExportHfRequest
{
	#[serde(default = "default_export_dir")]
	output_dir: String,
}

fn default_export_dir() -> String
{
	"exports".to_string()
}

/// Export a checkpoint's genome data for HuggingFace model creation.
///
/// Returns the checkpoint metadata + experiment info needed for HF export.
/// The actual export (training memory + serializing safetensors) happens in Python.
pub(crate) async fn export_checkpoint_hf(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
	Json(req): Json<ExportHfRequest>,
) -> impl IntoResponse
{
	// Get checkpoint
	let checkpoint = match crate::db::queries::get_checkpoint(&state.db, id).await
	{
		Ok(Some(c)) => c,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Checkpoint not found"})),
			)
				.into_response();
		}
		Err(e) =>
		{
			return (
				StatusCode::INTERNAL_SERVER_ERROR,
				Json(serde_json::json!({"error": e.to_string()})),
			)
				.into_response();
		}
	};

	// Get the experiment for architecture context
	let experiment = match crate::db::queries::get_experiment(&state.db, checkpoint.experiment_id)
		.await
	{
		Ok(Some(e)) => e,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Experiment not found"})),
			)
				.into_response();
		}
		Err(e) =>
		{
			return (
				StatusCode::INTERNAL_SERVER_ERROR,
				Json(serde_json::json!({"error": e.to_string()})),
			)
				.into_response();
		}
	};

	// Return all the data needed for Python-side HF export
	(
		StatusCode::OK,
		Json(serde_json::json!({
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
		})),
	)
		.into_response()
}
