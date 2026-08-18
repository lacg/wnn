//! Worker log watch handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Worker log watch
// =============================================================================

#[derive(Deserialize)]
pub(crate) struct WatchLogRequest
{
	log_path: String,
}

pub(crate) async fn set_watch_log(
	State(state): State<Arc<AppState>>,
	Json(req): Json<WatchLogRequest>,
) -> impl IntoResponse
{
	*state.current_log_path.write().await = Some(req.log_path.clone());
	(
		StatusCode::OK,
		Json(serde_json::json!({"log_path": req.log_path})),
	)
		.into_response()
}

pub(crate) async fn get_watch_log(State(state): State<Arc<AppState>>) -> impl IntoResponse
{
	let path = state.current_log_path.read().await.clone();
	(StatusCode::OK, Json(serde_json::json!({"log_path": path}))).into_response()
}
