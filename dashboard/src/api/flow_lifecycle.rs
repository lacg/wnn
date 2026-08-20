//! Flow lifecycle handlers: stop/restart/pause/resume/pid/heartbeat (split from api/mod.rs).

use super::*;

/// Stop a running flow by sending SIGTERM to the worker process
pub(crate) async fn stop_flow(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	// Get the flow to check status
	let flow = match crate::db::queries::get_flow(&state.db, id).await
	{
		Ok(Some(f)) => f,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Flow not found"})),
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

	// Check if flow is running or queued
	if flow.status != FlowStatus::Running && flow.status != FlowStatus::Queued
	{
		return (
			StatusCode::BAD_REQUEST,
			Json(serde_json::json!({"error": "Flow is not running or queued"})),
		)
			.into_response();
	}

	// Use shared stop function (sends SIGTERM, updates status to cancelled)
	if let Err(e) = crate::db::queries::stop_flow_process(&state.db, id).await
	{
		return (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response();
	}

	// Broadcast cancellation and return updated flow
	match crate::db::queries::get_flow(&state.db, id).await
	{
		Ok(Some(updated_flow)) =>
		{
			let _ = state
				.ws_tx
				.send(WsMessage::FlowCancelled(updated_flow.clone()));
			(StatusCode::OK, Json(updated_flow)).into_response()
		}
		Ok(None) => (StatusCode::OK, Json(serde_json::json!({"stopped": true}))).into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

#[derive(Debug, Deserialize)]
pub struct RestartFlowRequest
{
	#[serde(default)]
	pub from_beginning: bool, // If true, restart from scratch; if false, resume from checkpoint
	#[serde(default)]
	pub start_from_experiment: Option<usize>, // If set, skip experiments before this index (0-based)
}

/// Restart a flow by setting status to queued
pub(crate) async fn restart_flow(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
	Json(req): Json<RestartFlowRequest>,
) -> impl IntoResponse
{
	// Get the flow
	let flow = match crate::db::queries::get_flow(&state.db, id).await
	{
		Ok(Some(f)) => f,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Flow not found"})),
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

	// If flow is running or queued, stop it first
	if flow.status == FlowStatus::Running || flow.status == FlowStatus::Queued
	{
		if let Err(e) = crate::db::queries::stop_flow_process(&state.db, id).await
		{
			return (
				StatusCode::INTERNAL_SERVER_ERROR,
				Json(serde_json::json!({"error": format!("Failed to stop flow: {}", e)})),
			)
				.into_response();
		}
	}

	// If restarting from beginning, clear the seed checkpoint and delete checkpoint files
	let seed_checkpoint_id = if req.from_beginning
	{
		// Delete checkpoint directory for this flow.
		// SECURITY: the directory name is derived from the user-supplied flow
		// name; restrict it to [a-z0-9_-] so sequences like ".." can never
		// escape the checkpoints root (remove_dir_all on a traversal path
		// would delete arbitrary directories). Benign names produce the same
		// result as the worker's `lower().replace(" ","_").replace("/","_")`.
		let safe_name: String = flow
			.name
			.to_lowercase()
			.chars()
			.map(|c| {
				if c.is_ascii_alphanumeric() || c == '-' || c == '_'
				{
					c
				}
				else
				{
					'_'
				}
			})
			.collect();

		if safe_name.trim_matches('_').is_empty()
		{
			tracing::warn!(
				"Flow name {:?} sanitizes to nothing safe — skipping checkpoint-dir deletion",
				flow.name
			);
		}
		else
		{
			// Try parent directory first (project root checkpoints)
			let parent_checkpoint_dir = std::path::Path::new("../checkpoints").join(&safe_name);
			let local_checkpoint_dir = std::path::Path::new("checkpoints").join(&safe_name);

			for checkpoint_dir in [&parent_checkpoint_dir, &local_checkpoint_dir]
			{
				if checkpoint_dir.exists()
				{
					if let Err(e) = std::fs::remove_dir_all(checkpoint_dir)
					{
						tracing::warn!(
							"Failed to delete checkpoint directory {:?}: {}",
							checkpoint_dir,
							e
						);
					}
					else
					{
						tracing::info!("Deleted checkpoint directory: {:?}", checkpoint_dir);
					}
				}
			}
		}
		Some(None) // Clear checkpoint reference in DB
	}
	else
	{
		None // Keep existing
	};

	// If start_from_experiment is set, update the flow config
	if let Some(start_idx) = req.start_from_experiment
	{
		// Get current config and add start_from_experiment to params
		if let Ok(Some(mut current_flow)) = crate::db::queries::get_flow(&state.db, id).await
		{
			// FlowConfig has params: HashMap<String, Value>
			current_flow.config.params.insert(
				"start_from_experiment".to_string(),
				serde_json::json!(start_idx),
			);
			// Serialize FlowConfig to serde_json::Value for update
			if let Ok(config_json) = serde_json::to_value(&current_flow.config)
			{
				let _ = crate::db::queries::update_flow(
					&state.db,
					id,
					None,
					None,
					None,
					Some(&config_json),
					None,
					None,
				)
				.await;
			}
		}
	}

	// Set status to queued (and optionally clear PID)
	match crate::db::queries::update_flow_for_restart(&state.db, id, seed_checkpoint_id).await
	{
		Ok(_) =>
		{
			if let Ok(Some(updated_flow)) = crate::db::queries::get_flow(&state.db, id).await
			{
				let _ = state
					.ws_tx
					.send(WsMessage::FlowQueued(updated_flow.clone()));
				(StatusCode::OK, Json(updated_flow)).into_response()
			}
			else
			{
				(StatusCode::OK, Json(serde_json::json!({"restarted": true}))).into_response()
			}
		}
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

/// Pause a flow. What that means depends on whether it has started.
///
/// RUNNING — cooperative. Sets `flows.pause_requested = 1` and leaves the
/// status alone. The Python worker polls this flag between generations:
/// when it observes pause_requested=1, it saves a per-gen checkpoint, sets
/// `flows.status = 'paused'`, and moves on (doesn't park the worker). Only
/// the worker can pause a running flow, because only the worker can
/// checkpoint it.
///
/// QUEUED — immediate. Sets the flag AND `status = 'paused'` in one
/// statement. There is nothing to checkpoint, and the flag ALONE does not
/// gate admission: the worker selects on `status = 'queued'` only
/// (`_get_next_queued_flow`) and never reads pause_requested until the run
/// is already under way. Flagging without flipping the status therefore let
/// a "paused" flow start anyway and burn a whole grid phase before pausing
/// itself at the first generation boundary — measured 19/08/2026 after 357
/// flows were flagged and none were actually gated. Mirror of `resume_flow`,
/// which flips `paused -> queued`.
pub(crate) async fn pause_flow(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	// Verify flow exists + is in a state that can be paused
	let flow = match crate::db::queries::get_flow(&state.db, id).await
	{
		Ok(Some(f)) => f,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Flow not found"})),
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

	// Only running/queued flows are meaningful to pause. Already-paused is idempotent.
	if flow.status != FlowStatus::Running
		&& flow.status != FlowStatus::Queued
		&& flow.status != FlowStatus::Paused
	{
		return (
			StatusCode::BAD_REQUEST,
			Json(
				serde_json::json!({"error": format!("Flow is not running/queued (status={:?})", flow.status)}),
			),
		)
			.into_response();
	}

	// A queued flow is gated here and now; a running one waits for the worker.
	let gated_now = flow.status == FlowStatus::Queued;
	let sql = if gated_now
	{
		"UPDATE flows SET pause_requested = 1, status = 'paused' WHERE id = ?"
	}
	else
	{
		"UPDATE flows SET pause_requested = 1 WHERE id = ?"
	};

	if let Err(e) = sqlx::query(sql).bind(id).execute(&state.db).await
	{
		return (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response();
	}

	(
		StatusCode::OK,
		Json(serde_json::json!({
				"id": id,
				"pause_requested": true,
				// Tells the caller whether the flow is ALREADY gated or merely
				// flagged — the distinction that was invisible before.
				"status": if gated_now { "paused" } else { "pause_requested" },
		})),
	)
		.into_response()
}

/// Resume a paused flow.
///
/// Clears `flows.pause_requested`, flips status `paused → queued`, and
/// clears `paused_at` so the worker picks the flow up again on the next
/// poll. Resume re-enters the normal id-DESC queue (no front-jump).
pub(crate) async fn resume_flow(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	// Verify flow exists
	let flow = match crate::db::queries::get_flow(&state.db, id).await
	{
		Ok(Some(f)) => f,
		Ok(None) =>
		{
			return (
				StatusCode::NOT_FOUND,
				Json(serde_json::json!({"error": "Flow not found"})),
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

	// Only paused flows are meaningful to resume.
	if flow.status != FlowStatus::Paused
	{
		return (
			StatusCode::BAD_REQUEST,
			Json(serde_json::json!({"error": format!("Flow is not paused (status={:?})", flow.status)})),
		)
			.into_response();
	}

	if let Err(e) =
		sqlx::query("UPDATE flows SET pause_requested = 0, status = 'queued' WHERE id = ?")
			.bind(id)
			.execute(&state.db)
			.await
	{
		return (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response();
	}

	// Broadcast queued event so the dashboard updates
	if let Ok(Some(updated_flow)) = crate::db::queries::get_flow(&state.db, id).await
	{
		let _ = state
			.ws_tx
			.send(WsMessage::FlowQueued(updated_flow.clone()));
		(StatusCode::OK, Json(updated_flow)).into_response()
	}
	else
	{
		(
			StatusCode::OK,
			Json(serde_json::json!({
					"id": id,
					"status": "queued",
					"pause_requested": false,
			})),
		)
			.into_response()
	}
}

#[derive(Debug, Deserialize)]
pub struct UpdateFlowPidRequest
{
	pub pid: Option<i64>,
}

/// Update the PID of a flow (called by worker when starting)
pub(crate) async fn update_flow_pid(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
	Json(req): Json<UpdateFlowPidRequest>,
) -> impl IntoResponse
{
	match crate::db::queries::update_flow_pid(&state.db, id, req.pid).await
	{
		Ok(true) => (StatusCode::OK, Json(serde_json::json!({"success": true}))).into_response(),
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Flow not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}

/// Update the heartbeat of a flow (called periodically by worker)
pub(crate) async fn update_flow_heartbeat(
	State(state): State<Arc<AppState>>,
	Path(id): Path<i64>,
) -> impl IntoResponse
{
	match crate::db::queries::update_flow_heartbeat(&state.db, id).await
	{
		Ok(true) => (StatusCode::OK, Json(serde_json::json!({"success": true}))).into_response(),
		Ok(false) => (
			StatusCode::NOT_FOUND,
			Json(serde_json::json!({"error": "Flow not found"})),
		)
			.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(serde_json::json!({"error": e.to_string()})),
		)
			.into_response(),
	}
}
