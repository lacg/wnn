//! Gating run queries (split from db/mod.rs `queries`).

use super::*;

// =============================================================================
// Gating Run queries
// =============================================================================

fn gating_status_to_str(status: &GatingStatus) -> &'static str {
    match status {
        GatingStatus::Pending => "pending",
        GatingStatus::Running => "running",
        GatingStatus::Completed => "completed",
        GatingStatus::Failed => "failed",
    }
}

fn str_to_gating_status(s: &str) -> GatingStatus {
    match s {
        "pending" => GatingStatus::Pending,
        "running" => GatingStatus::Running,
        "completed" => GatingStatus::Completed,
        "failed" => GatingStatus::Failed,
        _ => GatingStatus::Pending,
    }
}

fn row_to_gating_run(row: &sqlx::sqlite::SqliteRow) -> Result<GatingRun> {
    let status_str: String = row.try_get("status")?;
    let config_json: Option<String> = row.try_get("config_json")?;
    let results_json: Option<String> = row.try_get("results_json")?;
    let created_at_str: String = row.try_get("created_at")?;
    let started_at_str: Option<String> = row.try_get("started_at")?;
    let completed_at_str: Option<String> = row.try_get("completed_at")?;

    Ok(GatingRun {
        id: row.try_get("id")?,
        experiment_id: row.try_get("experiment_id")?,
        status: str_to_gating_status(&status_str),
        config: config_json.and_then(|s| serde_json::from_str(&s).ok()),
        genomes_tested: row.try_get("genomes_tested")?,
        results: results_json.and_then(|s| serde_json::from_str(&s).ok()),
        error: row.try_get("error")?,
        created_at: DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        started_at: started_at_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
        completed_at: completed_at_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
    })
}

/// Create a new gating run for an experiment
pub async fn create_gating_run(
    pool: &DbPool,
    experiment_id: i64,
    config: Option<&GatingConfig>,
) -> Result<i64> {
    let now = Utc::now().to_rfc3339();
    let config_json = config.map(|c| serde_json::to_string(c).unwrap_or_default());

    let result = sqlx::query(
        r#"INSERT INTO gating_runs (experiment_id, status, config_json, created_at)
           VALUES (?, 'pending', ?, ?)"#
    )
    .bind(experiment_id)
    .bind(&config_json)
    .bind(&now)
    .execute(pool)
    .await?;

    // Also update the experiment's gating_status for backward compat
    let _ = sqlx::query("UPDATE experiments SET gating_status = 'pending' WHERE id = ?")
        .bind(experiment_id)
        .execute(pool)
        .await;

    Ok(result.last_insert_rowid())
}

/// Get a specific gating run
pub async fn get_gating_run(pool: &DbPool, id: i64) -> Result<Option<GatingRun>> {
    let row = sqlx::query(
        r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                  results_json, error, created_at, started_at, completed_at
           FROM gating_runs WHERE id = ?"#
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(row_to_gating_run(&r)?)),
        None => Ok(None),
    }
}

/// List gating runs for an experiment
pub async fn list_gating_runs(pool: &DbPool, experiment_id: i64) -> Result<Vec<GatingRun>> {
    let rows = sqlx::query(
        r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                  results_json, error, created_at, started_at, completed_at
           FROM gating_runs WHERE experiment_id = ?
           ORDER BY created_at DESC"#
    )
    .bind(experiment_id)
    .fetch_all(pool)
    .await?;

    let mut runs = Vec::with_capacity(rows.len());
    for row in rows {
        runs.push(row_to_gating_run(&row)?);
    }
    Ok(runs)
}

/// Update gating run status
pub async fn update_gating_run_status(
    pool: &DbPool,
    id: i64,
    status: &GatingStatus,
) -> Result<Option<GatingRun>> {
    let status_str = gating_status_to_str(status);
    let now = Utc::now().to_rfc3339();

    // Set started_at when transitioning to running
    let started_clause = if *status == GatingStatus::Running {
        ", started_at = ?"
    } else {
        ""
    };

    // Set completed_at when transitioning to completed or failed
    let completed_clause = if *status == GatingStatus::Completed || *status == GatingStatus::Failed {
        ", completed_at = ?"
    } else {
        ""
    };

    let query = format!(
        "UPDATE gating_runs SET status = ?{}{} WHERE id = ?",
        started_clause, completed_clause
    );

    let mut q = sqlx::query(&query).bind(status_str);
    if *status == GatingStatus::Running {
        q = q.bind(&now);
    }
    if *status == GatingStatus::Completed || *status == GatingStatus::Failed {
        q = q.bind(&now);
    }
    q = q.bind(id);

    let result = q.execute(pool).await?;

    if result.rows_affected() == 0 {
        return Ok(None);
    }

    // Also update experiment's gating_status for backward compat
    let _ = sqlx::query(
        "UPDATE experiments SET gating_status = ? WHERE id = (SELECT experiment_id FROM gating_runs WHERE id = ?)"
    )
    .bind(status_str)
    .bind(id)
    .execute(pool)
    .await;

    get_gating_run(pool, id).await
}

/// Update gating run with results
pub async fn update_gating_run_results(
    pool: &DbPool,
    id: i64,
    genomes_tested: i32,
    results: &[GatingResult],
    error: Option<&str>,
) -> Result<Option<GatingRun>> {
    let now = Utc::now().to_rfc3339();
    let results_json = serde_json::to_string(results)?;
    let status = if error.is_some() { "failed" } else { "completed" };

    let result = sqlx::query(
        r#"UPDATE gating_runs
           SET status = ?, genomes_tested = ?, results_json = ?, error = ?, completed_at = ?
           WHERE id = ?"#
    )
    .bind(status)
    .bind(genomes_tested)
    .bind(&results_json)
    .bind(error)
    .bind(&now)
    .bind(id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        return Ok(None);
    }

    // Also update experiment's gating_status and results for backward compat
    let gating_results = GatingResults {
        completed_at: Some(Utc::now()),
        genomes_tested: genomes_tested as usize,
        results: results.to_vec(),
        error: error.map(|s| s.to_string()),
    };
    let exp_results_json = serde_json::to_string(&gating_results)?;

    let _ = sqlx::query(
        "UPDATE experiments SET gating_status = ?, gating_results = ? WHERE id = (SELECT experiment_id FROM gating_runs WHERE id = ?)"
    )
    .bind(status)
    .bind(&exp_results_json)
    .bind(id)
    .execute(pool)
    .await;

    get_gating_run(pool, id).await
}

/// Get pending gating runs (for worker polling)
pub async fn get_pending_gating_runs(pool: &DbPool) -> Result<Vec<GatingRun>> {
    let rows = sqlx::query(
        r#"SELECT id, experiment_id, status, config_json, genomes_tested,
                  results_json, error, created_at, started_at, completed_at
           FROM gating_runs WHERE status = 'pending'
           ORDER BY created_at ASC"#
    )
    .fetch_all(pool)
    .await?;

    let mut runs = Vec::with_capacity(rows.len());
    for row in rows {
        runs.push(row_to_gating_run(&row)?);
    }
    Ok(runs)
}
