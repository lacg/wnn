//! Flow CRUD queries (split from db/mod.rs `queries`).

use super::queries::*;
use super::*;

// =============================================================================
// Flow queries
// =============================================================================

pub async fn list_flows(
	pool: &DbPool,
	status: Option<&str>,
	limit: i32,
	offset: i32,
) -> Result<Vec<Flow>>
{
	let rows = if let Some(status_filter) = status
	{
		sqlx::query(
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat, status_message, pause_requested
               FROM flows WHERE status = ?
               ORDER BY id DESC
               LIMIT ? OFFSET ?"#,
        )
        .bind(status_filter)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
	}
	else
	{
		sqlx::query(
            r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat, status_message, pause_requested
               FROM flows
               ORDER BY id DESC
               LIMIT ? OFFSET ?"#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?
	};

	let mut flows = Vec::with_capacity(rows.len());
	for row in rows
	{
		flows.push(row_to_flow(&row)?);
	}
	Ok(flows)
}

pub async fn get_flow(pool: &DbPool, id: i64) -> Result<Option<Flow>>
{
	let row = sqlx::query(
        r#"SELECT id, name, description, config_json, created_at, started_at, completed_at, status, seed_checkpoint_id, pid, last_heartbeat, status_message, pause_requested
           FROM flows WHERE id = ?"#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

	match row
	{
		Some(r) => Ok(Some(row_to_flow(&r)?)),
		None => Ok(None),
	}
}

/// Create a new flow
///
/// Experiments are passed separately (not in FlowConfig) - they get stored in the experiments table.
/// This follows normalized design: Flow 1:N Experiments via FK, not embedded JSON.
pub async fn create_flow(
	pool: &DbPool,
	name: &str,
	description: Option<&str>,
	config: &FlowConfig,
	experiments: &[ExperimentSpec],
	seed_checkpoint_id: Option<i64>,
) -> Result<i64>
{
	let now = Utc::now().to_rfc3339();
	// Unified seed registry: every flow gets a recorded seed. If the config has none,
	// auto-generate a UTC-timestamp seed (YYYYMMDDHHMMSS) so the flow is reproducible
	// and self-documenting — mirrors wnn.seeds for the controller scripts.
	let mut cfg = config.clone();
	let (seed, seed_source) = match cfg.params.get("seed").and_then(|v| v.as_i64())
	{
		Some(s) => (s, "explicit"),
		None =>
		{
			let s = Utc::now()
				.format("%Y%m%d%H%M%S")
				.to_string()
				.parse::<i64>()
				.unwrap_or(0);
			cfg.params.insert("seed".to_string(), serde_json::json!(s));
			(s, "timestamp")
		}
	};
	let config_json = serde_json::to_string(&cfg)?;

	// One transaction for flow + seed registry + ALL experiments: a crash
	// mid-way used to leave a flow with 0/partial experiments — exactly
	// the Rule-2 "completes instantly, does nothing" trap (P3).
	let mut tx = pool.begin().await?;

	let result = sqlx::query(
		r#"INSERT INTO flows (name, description, config_json, created_at, status, seed_checkpoint_id)
           VALUES (?, ?, ?, ?, 'pending', ?)"#,
	)
	.bind(name)
	.bind(description)
	.bind(&config_json)
	.bind(&now)
	.bind(seed_checkpoint_id)
	.execute(&mut *tx)
	.await?;

	let flow_id = result.last_insert_rowid();

	// Record this flow's seed in the shared seed_runs registry (controller scripts
	// record via wnn.seeds). For IDS/tiered/bitwise one seed drives the split + K-fold,
	// so train/test/val all = base; the 3-way is realised by the 80/20 + K-fold split.
	// Best-effort: a registry hiccup must never fail flow creation.
	let arch = cfg
		.params
		.get("architecture_type")
		.and_then(|v| v.as_str())
		.unwrap_or("tiered")
		.to_string();
	let _ = sqlx::query(
		r#"CREATE TABLE IF NOT EXISTS seed_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, script TEXT NOT NULL,
            source TEXT NOT NULL, base INTEGER NOT NULL, run_index INTEGER NOT NULL,
            train_seed INTEGER NOT NULL, test_seed INTEGER NOT NULL, val_seed INTEGER NOT NULL,
            extra_json TEXT)"#,
	)
	.execute(&mut *tx)
	.await;
	let _ = sqlx::query(
        r#"INSERT INTO seed_runs (created_at, script, source, base, run_index, train_seed, test_seed, val_seed, extra_json)
           VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)"#,
    )
    .bind(&now)
    .bind(format!("flow:{}:{}", flow_id, arch))
    .bind(seed_source)
    .bind(seed).bind(seed).bind(seed).bind(seed)
    .bind(serde_json::json!({"flow_id": flow_id, "name": name}).to_string())
    .execute(&mut *tx).await;

	// Create pending experiments for each experiment spec
	for (idx, exp_spec) in experiments.iter().enumerate()
	{
		// Use explicit phase_type if provided, otherwise derive
		let phase_type = if let Some(ref pt) = exp_spec.phase_type
		{
			pt.clone()
		}
		else if let Some(pt) = exp_spec.params.get("phase_type").and_then(|v| v.as_str())
		{
			pt.to_string()
		}
		else
		{
			let opt_target = if exp_spec.optimize_bits
			{
				"bits"
			}
			else if exp_spec.optimize_neurons
			{
				"neurons"
			}
			else
			{
				"connections"
			};
			match exp_spec.experiment_type
			{
				crate::models::ExperimentType::GridSearch => "grid_search".to_string(),
				crate::models::ExperimentType::LambdaSweep => "lambda_sweep".to_string(),
				_ =>
				{
					let exp_type = match exp_spec.experiment_type
					{
						crate::models::ExperimentType::Ga => "ga",
						crate::models::ExperimentType::Ts => "ts",
						crate::models::ExperimentType::Neurogenesis => "neurogenesis",
						crate::models::ExperimentType::Synaptogenesis => "synaptogenesis",
						crate::models::ExperimentType::Axonogenesis => "axonogenesis",
						crate::models::ExperimentType::GridSearch
						| crate::models::ExperimentType::LambdaSweep => unreachable!(),
					};
					format!("{}_{}", exp_type, opt_target)
				}
			}
		};

		// Get max_iterations: grid_search is always 1; others from params
		let max_iterations = if matches!(
			exp_spec.experiment_type,
			crate::models::ExperimentType::GridSearch | crate::models::ExperimentType::LambdaSweep
		)
		{
			Some(1) // Grid search / lambda sweep is a single step — always 1
		}
		else
		{
			exp_spec
				.params
				.get("generations")
				.or_else(|| exp_spec.params.get("iterations"))
				.and_then(|v| v.as_i64())
				.map(|v| v as i32)
				.or_else(|| match exp_spec.experiment_type
				{
					crate::models::ExperimentType::GridSearch
					| crate::models::ExperimentType::LambdaSweep => unreachable!(),
					crate::models::ExperimentType::Ga => config
						.params
						.get("ga_generations")
						.and_then(|v| v.as_i64())
						.map(|v| v as i32),
					crate::models::ExperimentType::Ts => config
						.params
						.get("ts_iterations")
						.and_then(|v| v.as_i64())
						.map(|v| v as i32),
					crate::models::ExperimentType::Neurogenesis
					| crate::models::ExperimentType::Synaptogenesis
					| crate::models::ExperimentType::Axonogenesis => exp_spec
						.params
						.get("iterations")
						.and_then(|v| v.as_i64())
						.map(|v| v as i32),
				})
		};

		let exp_params = if exp_spec.params.is_empty()
		{
			None
		}
		else
		{
			Some(&exp_spec.params)
		};
		create_pending_experiment(
			&mut *tx,
			&exp_spec.name,
			flow_id,
			idx as i32,
			Some(&phase_type),
			max_iterations,
			config,
			exp_params,
		)
		.await?;
	}

	tx.commit().await?;
	Ok(flow_id)
}

/// Server-side flow status state machine (P3, 12/06/2026).
///
/// Terminal states are only re-entered through dedicated endpoints:
/// completed → anything must go through POST /restart (which resets child
/// data coherently via update_flow_for_restart, not this PATCH); failed/
/// cancelled may be re-queued but never jump straight to running (the
/// worker only picks up 'queued'). Same→same is idempotent and allowed.
/// Before this, ANY→ANY was accepted: a stray →running PATCH on a
/// completed flow silently destroyed started_at/completed_at.
fn flow_transition_allowed(from: &str, to: &str) -> bool
{
	if from == to
	{
		return true;
	}
	matches!(
		(from, to),
		("pending", "queued")
			| ("pending", "running")
			| ("pending", "cancelled")
			| ("pending", "failed")
			| ("queued", "running")
			| ("queued", "paused")
			| ("queued", "cancelled")
			| ("queued", "failed")
			| ("running", "completed")
			| ("running", "failed")
			| ("running", "cancelled")
			| ("running", "paused")
			| ("running", "queued")
			| ("paused", "queued")
			| ("paused", "running")
			| ("paused", "cancelled")
			| ("paused", "failed")
			| ("failed", "queued")
			| ("cancelled", "queued")
	)
}

pub async fn update_flow(
	pool: &DbPool,
	id: i64,
	name: Option<&str>,
	description: Option<&str>,
	status: Option<&str>,
	config: Option<&serde_json::Value>,
	seed_checkpoint_id: Option<Option<i64>>,
	status_message: Option<&str>,
) -> Result<bool>
{
	// Validate the status transition against the current state BEFORE
	// building the update (P3: ANY→ANY was previously accepted). The
	// observed status is also re-asserted in the UPDATE's WHERE clause so
	// a concurrent transition between this read and the write makes the
	// update a no-op instead of clobbering the newer state (TOCTOU).
	let mut observed_status: Option<String> = None;
	if let Some(new_status) = status
	{
		let current: Option<String> = sqlx::query_scalar("SELECT status FROM flows WHERE id = ?")
			.bind(id)
			.fetch_optional(pool)
			.await?;
		let Some(current) = current
		else
		{
			return Ok(false); // flow doesn't exist
		};
		if !flow_transition_allowed(&current, new_status)
		{
			anyhow::bail!(
				"invalid status transition: {} -> {} (flow {}); use POST /restart to re-run a terminal flow",
				current,
				new_status,
				id
			);
		}
		observed_status = Some(current);
	}

	// Build dynamic update query using raw SQL with proper binding
	let mut set_clauses = Vec::new();

	if name.is_some()
	{
		set_clauses.push("name = ?1");
	}
	if description.is_some()
	{
		set_clauses.push("description = ?2");
	}
	if status.is_some()
	{
		set_clauses.push("status = ?3");
		// Update timestamps based on status
		if status == Some("running")
		{
			set_clauses.push("started_at = ?4");
			// Clear completed_at when re-running a flow (fixes timestamp corruption)
			set_clauses.push("completed_at = NULL");
		}
		else if status == Some("completed") || status == Some("failed") || status == Some("cancelled")
		{
			set_clauses.push("completed_at = ?4");
		}
	}
	if config.is_some()
	{
		set_clauses.push("config_json = ?5");
	}
	if seed_checkpoint_id.is_some()
	{
		set_clauses.push("seed_checkpoint_id = ?6");
	}
	if status_message.is_some()
	{
		set_clauses.push("status_message = ?7");
	}

	if set_clauses.is_empty()
	{
		return Ok(false);
	}

	// Re-assert the validated status in the WHERE clause (no-op on race)
	let query = if observed_status.is_some()
	{
		format!(
			"UPDATE flows SET {} WHERE id = ?8 AND status = ?9",
			set_clauses.join(", ")
		)
	}
	else
	{
		format!("UPDATE flows SET {} WHERE id = ?8", set_clauses.join(", "))
	};

	let now = Utc::now().to_rfc3339();
	let config_json = config.map(|c| serde_json::to_string(c).unwrap_or_default());
	let seed_id = seed_checkpoint_id.flatten();

	let result = sqlx::query(&query)
		.bind(name.unwrap_or(""))
		.bind(description.unwrap_or(""))
		.bind(status.unwrap_or(""))
		.bind(&now)
		.bind(config_json.as_deref().unwrap_or(""))
		.bind(seed_id)
		.bind(status_message.unwrap_or(""))
		.bind(id)
		.bind(observed_status.as_deref().unwrap_or(""))
		.execute(pool)
		.await?;

	// When flow config changes, recompute max_iterations for pending experiments
	if let Some(config_val) = config
	{
		if let Ok(flow_config) = serde_json::from_value::<crate::models::FlowConfig>(config_val.clone())
		{
			let pending_experiments: Vec<(i64, Option<String>)> = sqlx::query_as(
				"SELECT id, phase_type FROM experiments WHERE flow_id = ? AND status = 'pending'",
			)
			.bind(id)
			.fetch_all(pool)
			.await?;

			// Propagate flow config values to pending experiments
			let new_pop_size = flow_config
				.params
				.get("population_size")
				.and_then(|v| v.as_i64())
				.map(|v| v as i32);

			for (exp_id, phase_type) in &pending_experiments
			{
				if let Some(max_iters) =
					compute_max_iterations_from_phase_type(phase_type.as_deref(), &flow_config)
				{
					sqlx::query("UPDATE experiments SET max_iterations = ? WHERE id = ?")
						.bind(max_iters)
						.bind(exp_id)
						.execute(pool)
						.await?;
				}
				if let Some(pop) = new_pop_size
				{
					sqlx::query("UPDATE experiments SET population_size = ? WHERE id = ?")
						.bind(pop)
						.bind(exp_id)
						.execute(pool)
						.await?;
				}
			}
		}
	}

	// Cascade status changes when flow fails/cancelled
	// Mark any running experiments as failed/cancelled too
	// (only when the flow update actually applied — not on a lost race)
	if result.rows_affected() > 0 && (status == Some("failed") || status == Some("cancelled"))
	{
		let cascade_status = status.unwrap();

		// Update running experiments for this flow
		sqlx::query(
			"UPDATE experiments SET status = ?, ended_at = ?
             WHERE flow_id = ? AND status = 'running'",
		)
		.bind(cascade_status)
		.bind(&now)
		.bind(id)
		.execute(pool)
		.await?;
	}

	Ok(result.rows_affected() > 0)
}

pub(crate) fn row_to_flow(row: &sqlx::sqlite::SqliteRow) -> Result<Flow>
{
	let status_str: String = row.get("status");
	let config_json: String = row.get("config_json");

	Ok(Flow {
		id: row.get("id"),
		name: row.get("name"),
		description: row.get("description"),
		config: serde_json::from_str(&config_json)?,
		created_at: parse_datetime(row.get("created_at"))?,
		started_at: row
			.get::<Option<String>, _>("started_at")
			.map(|s| parse_datetime(s))
			.transpose()?,
		completed_at: row
			.get::<Option<String>, _>("completed_at")
			.map(|s| parse_datetime(s))
			.transpose()?,
		status: parse_flow_status(&status_str),
		seed_checkpoint_id: row.get("seed_checkpoint_id"),
		pid: row.get("pid"),
		last_heartbeat: row
			.get::<Option<String>, _>("last_heartbeat")
			.map(|s| parse_datetime(s))
			.transpose()?,
		status_message: row.get("status_message"),
		pause_requested: row.get::<Option<i64>, _>("pause_requested").unwrap_or(0),
	})
}
