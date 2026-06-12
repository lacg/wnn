//! In-memory live progress handlers (split from api/mod.rs).

use super::*;

// =============================================================================
// Live Progress handlers (in-memory, no DB)
// =============================================================================

pub(crate) async fn post_live_progress(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(data): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut map = state.live_progress.lock().unwrap();
    map.insert(id, LiveProgressEntry {
        data,
        updated_at: std::time::Instant::now(),
    });
    StatusCode::OK
}

pub(crate) async fn get_live_progress(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let map = state.live_progress.lock().unwrap();
    match map.get(&id) {
        Some(entry) if entry.updated_at.elapsed() < std::time::Duration::from_secs(300) => {
            (StatusCode::OK, Json(entry.data.clone())).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

pub(crate) async fn clear_live_progress(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let mut map = state.live_progress.lock().unwrap();
    map.remove(&id);
    StatusCode::OK
}
