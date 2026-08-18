//! WebSocket handler and shared snapshot poller (split from api/mod.rs).

use super::*;

// =============================================================================
// WebSocket handler (database polling)
// =============================================================================

pub(crate) async fn websocket_handler(
	ws: WebSocketUpgrade,
	State(state): State<Arc<AppState>>,
) -> impl IntoResponse
{
	ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Shared snapshot poller (P4, 12/06/2026): polls the DB once per 500ms and
/// broadcasts Snapshot / IterationCompleted over the existing ws_tx channel.
/// Previously EVERY WebSocket client ran this poll privately — N clients
/// meant 2N queries/sec fetching up to 500 rows each.
pub fn start_snapshot_poller(state: Arc<AppState>)
{
	tokio::spawn(async move {
		let mut tick = tokio::time::interval(std::time::Duration::from_millis(500));
		let mut last_experiment_id: Option<i64> = None;
		let mut last_experiment_status: Option<crate::models::ExperimentStatus> = None;
		let mut last_iteration_id: Option<i64> = None;
		loop
		{
			tick.tick().await;
			// No subscribers → skip the queries entirely
			if state.ws_tx.receiver_count() == 0
			{
				continue;
			}
			let snapshot = build_snapshot(&state.db).await;
			let current_exp_id = snapshot.current_experiment.as_ref().map(|e| e.id);
			let current_exp_status = snapshot
				.current_experiment
				.as_ref()
				.map(|e| e.status.clone());

			if current_exp_id != last_experiment_id || current_exp_status != last_experiment_status
			{
				last_experiment_id = current_exp_id;
				last_experiment_status = current_exp_status;
				last_iteration_id = snapshot.iterations.last().map(|i| i.id);
				let _ = state.ws_tx.send(WsMessage::Snapshot(snapshot));
			}
			else if let Some(ref exp) = snapshot.current_experiment
			{
				if let Ok(iterations) =
					crate::db::queries::get_recent_iterations(&state.db, exp.id, 10).await
				{
					for iter in iterations.iter().rev()
					{
						if last_iteration_id.map_or(true, |last_id| iter.id > last_id)
						{
							last_iteration_id = Some(iter.id);
							let _ = state
								.ws_tx
								.send(WsMessage::IterationCompleted(iter.clone()));
						}
					}
				}
			}
		}
	});
}

pub(crate) async fn handle_socket(mut socket: axum::extract::ws::WebSocket, state: Arc<AppState>)
{
	use axum::extract::ws::Message;

	// Initial snapshot for this client (one query per connect); live updates
	// come from the shared snapshot poller via the broadcast channel.
	let snapshot = build_snapshot(&state.db).await;
	if let Ok(json) = serde_json::to_string(&WsMessage::Snapshot(snapshot))
	{
		if socket.send(Message::Text(json.into())).await.is_err()
		{
			return;
		}
	}

	let mut rx = state.ws_tx.subscribe();

	loop
	{
		match rx.recv().await
		{
			Ok(msg) =>
			{
				let json_result = match &msg
				{
					WsMessage::Snapshot(_)
					| WsMessage::IterationCompleted(_)
					| WsMessage::FlowStarted(_)
					| WsMessage::FlowQueued(_)
					| WsMessage::FlowCompleted(_)
					| WsMessage::FlowFailed { .. }
					| WsMessage::FlowCancelled(_)
					| WsMessage::GatingRunCreated(_)
					| WsMessage::GatingRunUpdated(_) => serde_json::to_string(&msg),
					_ => continue, // Skip other messages
				};
				if let Ok(json) = json_result
				{
					if socket.send(Message::Text(json.into())).await.is_err()
					{
						return;
					}
				}
			}
			Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) =>
			{
				// Missed some messages; the next Snapshot resyncs the client
			}
			Err(tokio::sync::broadcast::error::RecvError::Closed) =>
			{
				return;
			}
		}
	}
}
