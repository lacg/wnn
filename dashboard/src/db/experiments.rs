//! Experiment CRUD queries (split from db/mod.rs `queries`).

use super::*;
use super::queries::*;

/// Delete a single experiment (must be pending status).
/// Cascades to all child data, then re-numbers remaining experiments.
pub async fn delete_experiment(pool: &DbPool, exp_id: i64) -> Result<bool> {
    // Verify experiment exists and get its flow_id and status
    let row = sqlx::query("SELECT flow_id, status FROM experiments WHERE id = ?")
        .bind(exp_id)
        .fetch_optional(pool)
        .await?;

    let Some(row) = row else { return Ok(false) };
    let status: String = row.get("status");
    let flow_id: Option<i64> = row.get("flow_id");

    if status != "pending" {
        anyhow::bail!("Can only delete pending experiments (current status: {})", status);
    }

    // Delete all child data
    delete_experiment_data(pool, exp_id).await?;

    // Delete the experiment itself
    sqlx::query("DELETE FROM experiments WHERE id = ?")
        .bind(exp_id)
        .execute(pool)
        .await?;

    // Re-number remaining experiments for this flow to close gaps
    if let Some(fid) = flow_id {
        let remaining_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
        )
        .bind(fid)
        .fetch_all(pool)
        .await?;

        for (idx, eid) in remaining_ids.iter().enumerate() {
            sqlx::query("UPDATE experiments SET sequence_order = ? WHERE id = ?")
                .bind(idx as i32)
                .bind(eid)
                .execute(pool)
                .await?;
        }
    }

    Ok(true)
}

/// Reorder experiments within a flow.
/// `experiment_ids` must contain all experiment IDs for the flow, in the desired order.
pub async fn reorder_experiments(pool: &DbPool, flow_id: i64, experiment_ids: &[i64]) -> Result<bool> {
    // Get all experiment IDs for this flow
    let existing_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT id FROM experiments WHERE flow_id = ? ORDER BY sequence_order"
    )
    .bind(flow_id)
    .fetch_all(pool)
    .await?;

    // Validate: same count and same set of IDs
    if existing_ids.len() != experiment_ids.len() {
        anyhow::bail!(
            "Expected {} experiment IDs, got {}",
            existing_ids.len(),
            experiment_ids.len()
        );
    }

    let mut expected: Vec<i64> = existing_ids.clone();
    expected.sort();
    let mut provided: Vec<i64> = experiment_ids.to_vec();
    provided.sort();
    if expected != provided {
        anyhow::bail!("Provided experiment IDs don't match the flow's experiments");
    }

    // Update sequence_order for each experiment
    for (idx, eid) in experiment_ids.iter().enumerate() {
        sqlx::query("UPDATE experiments SET sequence_order = ? WHERE id = ?")
            .bind(idx as i32)
            .bind(eid)
            .execute(pool)
            .await?;
    }

    Ok(true)
}

pub async fn list_flow_experiments(pool: &DbPool, flow_id: i64) -> Result<Vec<Experiment>> {
    let rows = sqlx::query(
        r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                  fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                  population_size, pid, last_iteration, resume_checkpoint_id,
                  created_at, started_at, ended_at, paused_at,
                  phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                  status_message, architecture_type, gating_status, gating_results,
                  params_json, extra_metrics_json
           FROM experiments WHERE flow_id = ?
           ORDER BY sequence_order"#,
    )
    .bind(flow_id)
    .fetch_all(pool)
    .await?;

    let mut experiments = Vec::with_capacity(rows.len());
    for row in rows {
        experiments.push(row_to_experiment(&row)?);
    }
    Ok(experiments)
}

// =============================================================================
// Experiment queries (new unified schema)
// =============================================================================

/// Get the currently running experiment
/// Only returns experiments that are truly running:
/// - Experiment has status='running' AND
/// - Either has no flow (standalone) OR its flow is also 'running'
/// This prevents orphan experiments (where flow was cancelled but experiment wasn't updated)
pub async fn get_running_experiment(pool: &DbPool) -> Result<Option<Experiment>> {
    let row = sqlx::query(
        r#"SELECT e.id, e.flow_id, e.sequence_order, e.name, e.status, e.fitness_calculator,
                  e.fitness_weight_ce, e.fitness_weight_acc, e.tier_config, e.context_size,
                  e.population_size, e.pid, e.last_iteration, e.resume_checkpoint_id,
                  e.created_at, e.started_at, e.ended_at, e.paused_at,
                  e.phase_type, e.max_iterations, e.current_iteration, e.best_ce, e.best_accuracy,
                  e.status_message, e.architecture_type
           FROM experiments e
           LEFT JOIN flows f ON e.flow_id = f.id
           WHERE e.status = 'running'
             AND (e.flow_id IS NULL OR f.status = 'running')
           ORDER BY (CASE WHEN e.flow_id IS NOT NULL THEN 0 ELSE 1 END), e.id DESC
           LIMIT 1"#,
    )
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(row_to_experiment(&r)?)),
        None => Ok(None),
    }
}

/// Get an experiment by ID
pub async fn get_experiment(pool: &DbPool, id: i64) -> Result<Option<Experiment>> {
    let row = sqlx::query(
        r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                  fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                  population_size, pid, last_iteration, resume_checkpoint_id,
                  created_at, started_at, ended_at, paused_at,
                  phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                  status_message, architecture_type, gating_status, gating_results,
                  params_json, extra_metrics_json
           FROM experiments WHERE id = ?"#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(row_to_experiment(&r)?)),
        None => Ok(None),
    }
}

/// Create a new experiment
pub async fn create_experiment(
    pool: &DbPool,
    name: &str,
    flow_id: Option<i64>,
    config: &serde_json::Value,
) -> Result<i64> {
    let now = Utc::now().to_rfc3339();

    // Extract config values with defaults
    // Check both "fitness_calculator" and "fitness_calculator_type" for compatibility
    let fitness_calculator = config.get("fitness_calculator")
        .or_else(|| config.get("fitness_calculator_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("normalized");
    let fitness_weight_ce = config.get("fitness_weight_ce")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let fitness_weight_acc = config.get("fitness_weight_acc")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let tier_config = config.get("tier_config")
        .map(|v| v.to_string());
    let context_size = config.get("context_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    let population_size = config.get("population_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(50) as i32;

    let architecture_type = config.get("architecture_type")
        .and_then(|v| v.as_str())
        .unwrap_or("tiered");

    let result = sqlx::query(
        r#"INSERT INTO experiments (
            name, flow_id, status, fitness_calculator, fitness_weight_ce, fitness_weight_acc,
            tier_config, context_size, population_size, architecture_type, created_at, started_at
        ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
    )
    .bind(name)
    .bind(flow_id)
    .bind(fitness_calculator)
    .bind(fitness_weight_ce)
    .bind(fitness_weight_acc)
    .bind(&tier_config)
    .bind(context_size)
    .bind(population_size)
    .bind(architecture_type)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok(result.last_insert_rowid())
}

/// Create a new experiment with pending status (for flow creation)
/// Now includes flow config fields (tier_config, fitness settings, etc.)
pub async fn create_pending_experiment<'e, E>(
    executor: E,
    name: &str,
    flow_id: i64,
    sequence_order: i32,
    phase_type: Option<&str>,
    max_iterations: Option<i32>,
    flow_config: &crate::models::FlowConfig,
    exp_params: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<i64>
where
    E: sqlx::SqliteExecutor<'e>,
{
    let now = Utc::now().to_rfc3339();

    // Extract config values from flow params
    let tier_config = flow_config.params.get("tier_config")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let fitness_calculator = flow_config.params.get("fitness_calculator")
        .and_then(|v| v.as_str())
        .unwrap_or("harmonic_rank");
    let fitness_weight_ce = flow_config.params.get("fitness_weight_ce")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let fitness_weight_acc = flow_config.params.get("fitness_weight_acc")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let context_size = flow_config.params.get("context_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(4) as i32;
    let population_size = flow_config.params.get("population_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(50) as i32;

    let architecture_type = flow_config.params.get("architecture_type")
        .and_then(|v| v.as_str())
        .unwrap_or("tiered");

    let params_json = exp_params
        .filter(|p| !p.is_empty())
        .map(|p| serde_json::to_string(p).unwrap_or_default());

    let result = sqlx::query(
        r#"INSERT INTO experiments (
            name, flow_id, sequence_order, status, phase_type, max_iterations,
            tier_config, fitness_calculator, fitness_weight_ce, fitness_weight_acc,
            context_size, population_size, architecture_type, params_json, created_at
        ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
    )
    .bind(name)
    .bind(flow_id)
    .bind(sequence_order)
    .bind(phase_type)
    .bind(max_iterations.unwrap_or(250))
    .bind(&tier_config)
    .bind(fitness_calculator)
    .bind(fitness_weight_ce)
    .bind(fitness_weight_acc)
    .bind(context_size)
    .bind(population_size)
    .bind(architecture_type)
    .bind(&params_json)
    .bind(&now)
    .execute(executor)
    .await?;

    Ok(result.last_insert_rowid())
}

/// Update an experiment
pub async fn update_experiment(
    pool: &DbPool,
    id: i64,
    name: Option<&str>,
    status: Option<&str>,
    best_ce: Option<f64>,
    best_accuracy: Option<f64>,
    current_iteration: Option<i32>,
    max_iterations: Option<i32>,
    architecture_type: Option<&str>,
) -> Result<bool> {
    let now = Utc::now().to_rfc3339();

    // Build dynamic update query
    let mut set_clauses = Vec::new();
    let mut binds: Vec<String> = Vec::new();

    if let Some(at) = architecture_type {
        set_clauses.push("architecture_type = ?");
        binds.push(at.to_string());
    }

    if let Some(n) = name {
        set_clauses.push("name = ?");
        binds.push(n.to_string());
    }
    if let Some(s) = status {
        set_clauses.push("status = ?");
        binds.push(s.to_string());
        // Update timestamps based on status
        match s {
            "running" => {
                set_clauses.push("started_at = ?");
                binds.push(now.clone());
                set_clauses.push("ended_at = NULL");
                // Wipe ONLY on a genuinely fresh start. A re-pickup of a
                // crashed/paused run has prior progress and MUST be preserved
                // so the worker's checkpoint-resume can continue —
                // unconditional wiping here is what destroyed flow 4042's
                // gen 0-75 data. "Fresh" means current_iteration <= 0 AND no
                // iteration rows exist: current_iteration is only advanced by
                // separate worker PATCHes, so a crash during gen 0 (rows
                // inserted, counter not yet PATCHed) used to pass the old
                // counter-only guard and silently delete real rows (P3 fix).
                let prior_iter: i32 = sqlx::query_scalar(
                    "SELECT COALESCE(current_iteration, 0) FROM experiments WHERE id = ?"
                ).bind(id).fetch_optional(pool).await?.unwrap_or(0);
                let has_iteration_rows: i32 = sqlx::query_scalar(
                    "SELECT EXISTS(SELECT 1 FROM iterations WHERE experiment_id = ?)"
                ).bind(id).fetch_one(pool).await?;
                if prior_iter <= 0 && has_iteration_rows == 0 {
                    // Fresh start: clear stale metrics + any partial rows.
                    // validation_summaries are KEPT — they double as the
                    // cross-flow validation cache and are upserted on re-run.
                    set_clauses.push("current_iteration = 0");
                    set_clauses.push("best_ce = NULL");
                    set_clauses.push("best_accuracy = NULL");
                    set_clauses.push("last_iteration = NULL");
                    // One transaction; best_genomes refs first (FK), then
                    // genome-children, then genomes.
                    let mut tx = pool.begin().await?;
                    sqlx::query(
                        "DELETE FROM genome_evaluations WHERE iteration_id IN \
                         (SELECT id FROM iterations WHERE experiment_id = ?)"
                    ).bind(id).execute(&mut *tx).await?;
                    sqlx::query(
                        "DELETE FROM health_checks WHERE iteration_id IN \
                         (SELECT id FROM iterations WHERE experiment_id = ?)"
                    ).bind(id).execute(&mut *tx).await?;
                    sqlx::query("DELETE FROM iterations WHERE experiment_id = ?")
                        .bind(id).execute(&mut *tx).await?;
                    sqlx::query(
                        "DELETE FROM best_genomes WHERE genome_id IN \
                         (SELECT id FROM genomes WHERE experiment_id = ?)"
                    ).bind(id).execute(&mut *tx).await?;
                    sqlx::query(
                        "DELETE FROM genome_evaluations WHERE genome_id IN \
                         (SELECT id FROM genomes WHERE experiment_id = ?)"
                    ).bind(id).execute(&mut *tx).await?;
                    sqlx::query("DELETE FROM genomes WHERE experiment_id = ?")
                        .bind(id).execute(&mut *tx).await?;
                    tx.commit().await?;
                }
                // else: resume — keep iterations/genomes/best/current_iteration intact.
            }
            "completed" | "failed" | "cancelled" => {
                set_clauses.push("ended_at = ?");
                binds.push(now.clone());
            }
            _ => {}
        }
    }
    if best_ce.is_some() {
        set_clauses.push("best_ce = ?");
    }
    if best_accuracy.is_some() {
        set_clauses.push("best_accuracy = ?");
    }
    if current_iteration.is_some() {
        set_clauses.push("current_iteration = ?");
    }
    if max_iterations.is_some() {
        set_clauses.push("max_iterations = ?");
    }

    if set_clauses.is_empty() {
        return Ok(false);
    }

    let query = format!(
        "UPDATE experiments SET {} WHERE id = ?",
        set_clauses.join(", ")
    );

    let mut q = sqlx::query(&query);

    // Bind string values
    for b in &binds {
        q = q.bind(b);
    }
    // Bind optional numeric values
    if let Some(ce) = best_ce {
        q = q.bind(ce);
    }
    if let Some(acc) = best_accuracy {
        q = q.bind(acc);
    }
    if let Some(iter) = current_iteration {
        q = q.bind(iter);
    }
    if let Some(max) = max_iterations {
        q = q.bind(max);
    }
    // Bind ID last
    q = q.bind(id);

    let result = q.execute(pool).await?;
    Ok(result.rows_affected() > 0)
}

/// List all experiments
pub async fn list_experiments(pool: &DbPool, limit: i32, offset: i32) -> Result<Vec<Experiment>> {
    let rows = sqlx::query(
        r#"SELECT id, flow_id, sequence_order, name, status, fitness_calculator,
                  fitness_weight_ce, fitness_weight_acc, tier_config, context_size,
                  population_size, pid, last_iteration, resume_checkpoint_id,
                  created_at, started_at, ended_at, paused_at,
                  phase_type, max_iterations, current_iteration, best_ce, best_accuracy,
                  status_message, architecture_type, gating_status, gating_results,
                  params_json, extra_metrics_json
           FROM experiments
           ORDER BY created_at DESC
           LIMIT ? OFFSET ?"#,
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await?;

    let mut experiments = Vec::with_capacity(rows.len());
    for row in rows {
        experiments.push(row_to_experiment(&row)?);
    }
    Ok(experiments)
}

/// Link an experiment to a flow
pub async fn link_experiment_to_flow(
    pool: &DbPool,
    flow_id: i64,
    experiment_id: i64,
    sequence_order: i32,
) -> Result<bool> {
    let result = sqlx::query(
        "UPDATE experiments SET flow_id = ?, sequence_order = ? WHERE id = ?",
    )
    .bind(flow_id)
    .bind(sequence_order)
    .bind(experiment_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

fn row_to_experiment(row: &sqlx::sqlite::SqliteRow) -> Result<Experiment> {
    let status_str: String = row.get("status");
    let fitness_calc_str: String = row.get("fitness_calculator");

    // Parse gating fields (optional, may not exist in older databases)
    let gating_status: Option<GatingStatus> = row.try_get::<Option<String>, _>("gating_status")
        .ok()
        .flatten()
        .map(|s| parse_gating_status(&s));
    let gating_results: Option<GatingResults> = row.try_get::<Option<String>, _>("gating_results")
        .ok()
        .flatten()
        .and_then(|s| serde_json::from_str(&s).ok());

    let architecture_type: ArchitectureType = row.try_get::<Option<String>, _>("architecture_type")
        .ok()
        .flatten()
        .map(|s| parse_architecture_type(&s))
        .unwrap_or_default();

    Ok(Experiment {
        id: row.get("id"),
        flow_id: row.get("flow_id"),
        sequence_order: row.get("sequence_order"),
        name: row.get("name"),
        status: parse_experiment_status(&status_str),
        fitness_calculator: parse_fitness_calculator(&fitness_calc_str),
        fitness_weight_ce: row.get("fitness_weight_ce"),
        fitness_weight_acc: row.get("fitness_weight_acc"),
        tier_config: row.get("tier_config"),
        context_size: row.get("context_size"),
        population_size: row.get("population_size"),
        pid: row.get("pid"),
        last_iteration: row.get("last_iteration"),
        resume_checkpoint_id: row.get("resume_checkpoint_id"),
        created_at: parse_datetime(row.get("created_at"))?,
        started_at: row.get::<Option<String>, _>("started_at")
            .map(|s| parse_datetime(s))
            .transpose()?,
        ended_at: row.get::<Option<String>, _>("ended_at")
            .map(|s| parse_datetime(s))
            .transpose()?,
        paused_at: row.get::<Option<String>, _>("paused_at")
            .map(|s| parse_datetime(s))
            .transpose()?,
        phase_type: row.try_get("phase_type").ok(),
        max_iterations: row.try_get("max_iterations").ok(),
        current_iteration: row.try_get("current_iteration").ok(),
        best_ce: row.try_get("best_ce").ok(),
        best_accuracy: row.try_get("best_accuracy").ok(),
        status_message: row.try_get::<Option<String>, _>("status_message").ok().flatten(),
        architecture_type,
        gating_status,
        gating_results,
        params: row.try_get::<Option<String>, _>("params_json")
            .ok()
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok()),
        extra_metrics: row.try_get::<Option<String>, _>("extra_metrics_json")
            .ok()
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}
