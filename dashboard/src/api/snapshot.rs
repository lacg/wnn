//! Dashboard snapshot handler + builder (split from api/mod.rs).

use super::*;

// =============================================================================
// Snapshot handler
// =============================================================================

pub(crate) async fn get_snapshot(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

pub(crate) async fn build_snapshot(db: &DbPool) -> DashboardSnapshot {
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
